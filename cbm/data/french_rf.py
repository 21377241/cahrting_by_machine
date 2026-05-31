"""Ken French 月度无风险利率（RF）加载器。

论文超额收益：excess_ret = stock_ret - RF_t（1 个月 T-bill，来自 French Data Library）。

缓存路径：``data/french_rf_monthly.parquet``
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import polars as pl
from loguru import logger

FRENCH_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_CSV.zip"
)
DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "french_rf_monthly.parquet"


def _parse_french_factors_csv(text: str) -> pl.DataFrame:
    """解析 F-F_Research_Data_Factors.csv（RF 列为百分数）。"""
    rows: list[tuple[int, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"^\d{6},", line):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        yyyymm = int(parts[0])
        rf_pct = float(parts[4])
        rows.append((yyyymm, rf_pct / 100.0))  # 转为小数（与 CRSP MthRet 一致）

    if not rows:
        raise ValueError("French factors 文件中未解析到任何 RF 行")

    df = pl.DataFrame({"ym": [r[0] for r in rows], "rate": [r[1] for r in rows]})
    y = pl.col("ym") // 100
    m = pl.col("ym") % 100
    return (
        df.with_columns(
            pl.date(y, m, 1).alias("date")  # 月初日期，仅用于展示
        )
        .select(["ym", "date", "rate"])
        .sort("ym")
    )


def download_french_rf() -> pl.DataFrame:
    """从 Ken French 网站下载并解析月度 RF。"""
    logger.info(f"下载 Ken French RF: {FRENCH_FACTORS_URL}")
    with urlopen(FRENCH_FACTORS_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    df = _parse_french_factors_csv(text)
    logger.info(f"French RF: {df.height} 个月, {df['ym'].min()} → {df['ym'].max()}")
    return df


def load_french_rf_monthly(cache_path: Path | None = None, refresh: bool = False) -> pl.DataFrame:
    """
    加载月度无风险利率。

    Returns
    -------
    pl.DataFrame
        columns: ``ym`` (int YYYYMM), ``date``, ``rate`` (decimal, e.g. 0.0022 = 0.22%)
    """
    cache = cache_path or DEFAULT_CACHE
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)

    df = download_french_rf()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    logger.info(f"French RF 已缓存: {cache}")
    return df


def align_rf_to_return_dates(
    returns_wide: pl.DataFrame,
    rf: pl.DataFrame,
) -> pl.DataFrame:
    """
    按 YYYYMM 将 RF 对齐到收益宽表的 ``date`` 列，供 ``FeatureEngineer`` 使用。

    Returns
    -------
    pl.DataFrame
        columns: ``date``, ``rate``（与 ``returns_wide`` 每行 date 一一对应）
    """
    ret = returns_wide.select("date").with_columns(
        (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("ym")
    )
    return (
        ret.join(rf.select(["ym", "rate"]), on="ym", how="left")
        .select(["date", pl.col("rate").fill_null(0.0).alias("rate")])
    )


def apply_excess_returns(
    returns_long: pl.DataFrame,
    rf: pl.DataFrame,
    ym_col: str = "ym",
) -> pl.DataFrame:
    """
    长表收益减 RF：``excess_ret = ret - RF_t``（按 YYYYMM 对齐）。

    Parameters
    ----------
    returns_long : pl.DataFrame
        需含 ``ym``（或 date 可推导）与 ``ret`` 列。
    """
    if ym_col not in returns_long.columns:
        raise ValueError(f"returns 缺少 {ym_col} 列")

    out = (
        returns_long.join(rf.select(["ym", "rate"]), on="ym", how="left")
        .with_columns(
            (pl.col("ret") - pl.col("rate").fill_null(0.0)).alias("ret")
        )
        .drop("rate")
    )
    missing = out.filter(pl.col("ret").is_null()).height
    if missing:
        logger.warning(f"超额收益转换后仍有 {missing} 行 ret 为 null")
    return out
