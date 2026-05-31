"""论文 Section 2.1 CRSP 样本筛选（Murray et al. 2024）。

训练与回测共用同一套规则：
  1. SHRCD ∈ {10,11} 等价：普通股（USIncFlg + EQTY + ShareType/SecuritySubType）
  2. EXCHCD ∈ {1,2,3} 等价：PrimaryExch ∈ {N, A, Q, P}
  3. 月 t−12 … t−1 收益全非缺失（12 月完整回溯）
  4. 月 t−1 末市值可算（MthCap > 0）

CRSP Stock Monthlies (SF) 字段映射见 ``crsp_data/CRSP_DATA_ANALYSIS.md`` §6.3。
"""

from __future__ import annotations

from datetime import date as dt_date
from pathlib import Path

import polars as pl
from loguru import logger

from cbm.core.types import FeatureSet

PAPER_PRIMARY_EXCHANGES: tuple[str, ...] = ("N", "A", "Q", "P")
PAPER_NYSE_EXCHANGE: str = "N"

_DEFAULT_PARQUET = (
    Path(__file__).resolve().parent.parent.parent / "crsp_data" / "crsp2525_monthly.parquet"
)

_PANEL_COLUMNS = [
    "PERMNO",
    "MthCalDt",
    "MthRet",
    "MthCap",
    "PrimaryExch",
    "USIncFlg",
    "SecurityType",
    "ShareType",
    "SecuritySubType",
    "DelReasonType",
]


def paper_ordinary_share_expr() -> pl.Expr:
    """SHRCD ∈ {10, 11} 等价（不含交易所）。"""
    return (
        (pl.col("USIncFlg") == "Y")
        & (pl.col("SecurityType") == "EQTY")
        & (
            pl.col("ShareType").is_in(["NS", "AD"])
            | pl.col("SecuritySubType").is_in(["COM", "COMAD"])
        )
    )


def paper_security_filter_expr() -> pl.Expr:
    """SHRCD∈{10,11} 且 EXCHCD∈{1,2,3} 的 SF 格式等价表达式。"""
    return paper_ordinary_share_expr() & pl.col("PrimaryExch").is_in(
        list(PAPER_PRIMARY_EXCHANGES)
    )


def apply_paper_security_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(paper_security_filter_expr())


def apply_shumway_delisting(df: pl.DataFrame, ret_col: str = "ret") -> pl.DataFrame:
    """强制退市且 MthRet 缺失时填 −30%（Shumway 1997 近似）。"""
    if "del_reason" not in df.columns:
        return df
    return df.with_columns(
        pl.when(
            pl.col(ret_col).is_null()
            & pl.col("del_reason").cast(pl.Utf8).str.starts_with("5")
        )
        .then(pl.lit(-0.30))
        .otherwise(pl.col(ret_col))
        .alias(ret_col)
    )


def ym_from_date_col(col: str = "date") -> pl.Expr:
    return (pl.col(col).dt.year() * 100 + pl.col(col).dt.month()).alias("ym")


def prev_ym(ym: int, n: int = 1) -> int:
    y, m = divmod(ym, 100)
    m -= n
    while m <= 0:
        m += 12
        y -= 1
    return y * 100 + m


def panel_load_start(test_start: dt_date, lookback_months: int = 12) -> dt_date:
    """加载面板所需最早日期：测试首月 t 的 t−12 … t−1 收益 + t−1 市值。"""
    y, m = test_start.year, test_start.month
    total = y * 12 + m - (lookback_months + 1)
    py, pm = divmod(total, 12)
    if pm == 0:
        py -= 1
        pm = 12
    return dt_date(py, pm, 1)


def compute_monthly_eligibility(
    returns: pl.DataFrame,
    lookback_months: int = 12,
) -> pl.DataFrame:
    """
    对每个月 ym（组合月 t），返回 t−12 … t−1 收益全非缺失的 (ym, permno)。
    """
    valid = (
        returns.filter(pl.col("ret").is_not_null())
        .select(["permno", "ym"])
        .unique()
    )
    if valid.is_empty():
        return pl.DataFrame({"ym": [], "permno": []})

    by_ym: dict[int, set[str]] = {}
    for permno, ym in valid.iter_rows():
        by_ym.setdefault(ym, set()).add(permno)

    all_yms = sorted(returns["ym"].unique().to_list())
    rows: list[dict] = []

    for ym in all_yms:
        lag_yms = [prev_ym(ym, k) for k in range(1, lookback_months + 1)]
        sets = [by_ym.get(ly, set()) for ly in lag_yms]
        if not all(sets):
            continue
        for p in set.intersection(*sets):
            rows.append({"ym": ym, "permno": p})

    if not rows:
        return pl.DataFrame({"ym": [], "permno": []})
    return pl.DataFrame(rows)


def load_paper_crsp_panel(
    data_start: dt_date,
    data_end: dt_date,
    parquet_path: Path | None = None,
    lookback_months: int = 12,
    handle_delisting: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    加载论文筛选后的 CRSP 长表面板。

    Returns
    -------
    returns : date, permno, ret, ym
    mcap : date, permno, mcap, ym  （仅 MthCap > 0）
    eligible : ym, permno  （当月 t 满足 12 月收益完整，用于组组合前筛选）
    exchanges : date, permno, primary_exch, ym  （Table 3 NYSE 断点用）
    """
    path = parquet_path or _DEFAULT_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"未找到 CRSP Parquet: {path}")

    lf = pl.scan_parquet(path).select(_PANEL_COLUMNS)
    lf = apply_paper_security_filter(lf)
    lf = lf.filter(
        (pl.col("MthCalDt") >= pl.lit(data_start))
        & (pl.col("MthCalDt") <= pl.lit(data_end))
    )

    df = (
        lf.select(
            pl.col("MthCalDt").cast(pl.Date).alias("date"),
            pl.col("PERMNO").cast(pl.Utf8).alias("permno"),
            pl.col("MthRet").alias("ret"),
            pl.col("MthCap").alias("mcap"),
            pl.col("PrimaryExch").cast(pl.Utf8).alias("primary_exch"),
            pl.col("DelReasonType").cast(pl.Utf8).alias("del_reason"),
        )
        .collect()
    )

    if handle_delisting:
        df = apply_shumway_delisting(df)

    returns = df.select(["date", "permno", "ret"]).with_columns(ym_from_date_col())
    mcap = (
        df.filter(pl.col("mcap").is_not_null() & (pl.col("mcap") > 0))
        .select(["date", "permno", "mcap"])
        .with_columns(ym_from_date_col())
    )
    eligible = compute_monthly_eligibility(returns, lookback_months=lookback_months)
    exchanges = (
        df.select(["date", "permno", "primary_exch"])
        .with_columns(ym_from_date_col())
    )

    logger.info(
        f"论文筛选面板 {data_start}→{data_end}: "
        f"收益 {returns.height:,} 行, 市值 {mcap.height:,} 行, "
        f"合格 (ym,permno) {eligible.height:,} 对"
    )
    return returns, mcap, eligible, exchanges


def eligible_permno_set(eligible: pl.DataFrame, ym: int) -> set[str]:
    sub = eligible.filter(pl.col("ym") == ym)
    if sub.is_empty():
        return set()
    return set(sub["permno"].to_list())


def _prev_ym_expr(col: str = "ym") -> pl.Expr:
    y = pl.col(col) // 100
    m = pl.col(col) % 100
    return (
        pl.when(m == 1)
        .then((y - 1) * 100 + 12)
        .otherwise(y * 100 + (m - 1))
        .alias("ym_mcap")
    )


def compute_paper_training_eligibility(
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    lookback_months: int = 12,
) -> pl.DataFrame:
    """
    论文 Section 2.1 训练/预测样本：(ym, permno) 满足
      - t−12 … t−1 收益完整
      - t−1 末市值可算（MthCap > 0）
    """
    base = compute_monthly_eligibility(returns, lookback_months=lookback_months)
    if base.is_empty():
        return base
    mcap_keys = mcap.select(["ym", "permno"]).unique()
    return (
        base.with_columns(_prev_ym_expr("ym"))
        .join(
            mcap_keys.rename({"ym": "ym_mcap"}),
            on=["ym_mcap", "permno"],
            how="inner",
        )
        .select(["ym", "permno"])
    )


def filter_feature_set_by_eligible(
    features: FeatureSet,
    eligible: pl.DataFrame,
) -> FeatureSet:
    """按 (ym, permno) 合格表筛选 ``FeatureSet``（训练与预测共用）。"""
    import numpy as np
    import pandas as pd

    by_ym: dict[int, set[str]] = {}
    for sub in eligible.partition_by("ym", maintain_order=True):
        by_ym[sub["ym"][0]] = set(sub["permno"].to_list())

    def _to_ym(d) -> int:
        ts = pd.Timestamp(d)
        return int(ts.year * 100 + ts.month)

    mask = np.array(
        [
            features.tickers[i] in by_ym.get(_to_ym(features.dates[i]), set())
            for i in range(len(features))
        ],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("论文 Section 2.1 筛选后无有效特征样本")

    n_keep = int(mask.sum())
    logger.info(
        f"论文 Section 2.1 特征筛选: {len(features):,} → {n_keep:,} 样本 "
        f"({n_keep / len(features) * 100:.1f}%)"
    )

    def _slice(arr):
        return None if arr is None else arr[mask]

    return FeatureSet(
        features=features.features[mask],
        targets=features.targets[mask],
        dates=features.dates[mask],
        tickers=features.tickers[mask],
        market_caps=_slice(features.market_caps),
        excess_returns=_slice(features.excess_returns),
    )


def apply_paper_sample_filter_to_features(
    features,
    data_start: dt_date,
    data_end: dt_date,
    parquet_path: Path | None = None,
):
    """加载论文面板并对 ``FeatureSet`` 做 Section 2.1 筛选。"""
    returns, mcap, _, _ = load_paper_crsp_panel(
        data_start, data_end, parquet_path=parquet_path,
    )
    eligible = compute_paper_training_eligibility(returns, mcap)
    return filter_feature_set_by_eligible(features, eligible)
