"""
Wind CSI800 成分日频 CSV 目录 -> 月度 ``StockData``（与 spx_local 接口完全对称）。

数据格式说明
------------
每个交易日存放在 ``{root}/{year}/YYYYMMDD.csv``，各文件结构为：

    wind_code, sec_name, i_weight, industry, open, high, low, close, VOLUME

注意：
- 文件中**无 date 列**，日期从文件名（YYYYMMDD.csv）推断。
- 收盘价列名为小写 ``close``（SPX 为大写 ``CLOSE``）。
- Wind 代码后缀为 ``.SZ`` / ``.SH``（如 ``000001.SZ``）。

设计要点
--------
- 月末收盘价：每个 (年-月, ticker) 取**该月最后一个交易日**收盘价。
- 将月末实际交易日规范化为日历月末日（与 spx_local 完全一致）。
- 支持月度宽表 Parquet 缓存，避免每次重扫 ~4500 个 CSV。
- ``wind_code_to_ticker`` 默认去掉 ``.SZ`` / ``.SH`` 等后缀，也可保留原代码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterator, List, Literal, Optional, Sequence, TYPE_CHECKING, Union

import polars as pl
from loguru import logger

from cbm.core.types import StockData

if TYPE_CHECKING:
    from cbm.core.engine import PortfolioEngine

# Wind 代码后缀正则：匹配 ".SZ" / ".SH" / ".BJ" / ".O" / ".N" 等
_WIND_SUFFIX_RE = re.compile(r"\.[A-Za-z]{1,3}$")


def default_wind_code_to_ticker(wind_code: str) -> str:
    """
    CSI800 Wind 代码 -> ticker（去掉交易所后缀）。

    例：``000001.SZ`` -> ``000001``；``600519.SH`` -> ``600519``。
    若无后缀则原样返回。
    """
    s = wind_code.strip()
    if not s:
        raise ValueError("wind_code 为空")
    m = _WIND_SUFFIX_RE.search(s)
    if m:
        return s[: m.start()]
    return s


def iter_csi800_daily_csv_paths(
    dataset_root: Union[str, Path],
    start_date: Union[str, date],
    end_date: Union[str, date],
) -> Iterator[Path]:
    """
    遍历 ``{root}/{year}/YYYYMMDD.csv``，返回在 [start_date, end_date] 内的路径。
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root.resolve()}")

    start = _parse_date(start_date)
    end   = _parse_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start} 晚于 end_date {end}")

    for year in range(start.year, end.year + 1):
        ydir = root / str(year)
        if not ydir.is_dir():
            continue
        for p in sorted(ydir.glob("*.csv")):
            try:
                d = datetime.strptime(p.stem, "%Y%m%d").date()
            except ValueError:
                logger.debug(f"跳过非 YYYYMMDD 命名的文件: {p}")
                continue
            if start <= d <= end:
                yield p


def _parse_date(d: Union[str, date]) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d.strip()[:10], "%Y-%m-%d").date()


def _wind_to_ticker_expr(map_ticker: Callable[[str], str]) -> pl.Expr:
    """若使用默认映射则用正则表达式（快）；否则逐元素调用 Python 函数。"""
    if map_ticker is default_wind_code_to_ticker:
        return pl.col("wind_code").str.replace(r"\.[A-Za-z]{1,3}$", "")
    return pl.col("wind_code").map_elements(map_ticker, return_dtype=pl.Utf8)


def _read_csi800_day_file(p: Path) -> Optional[pl.DataFrame]:
    """
    读取单日 CSI800 CSV，返回三列：date（从文件名推断）、wind_code、close。

    兼容 GBK/GB18030/UTF-8 编码，``sec_name``、``industry`` 等含中文列只读
    ``wind_code`` 和 ``close``，避免编码问题导致整行失败。
    """
    # 从文件名推断日期
    try:
        file_date = datetime.strptime(p.stem, "%Y%m%d").date()
    except ValueError:
        logger.warning(f"无法从文件名解析日期，跳过: {p}")
        return None

    cols = ["wind_code", "close"]
    overrides = {"close": pl.Float64}
    base_kw: dict = {
        "columns": cols,
        "schema_overrides": overrides,
        "infer_schema_length": 5000,
    }

    df: Optional[pl.DataFrame] = None
    for enc in ("gb18030", "gbk", "utf8"):
        try:
            kw = dict(base_kw)
            if enc != "utf8":
                kw["encoding"] = enc
            df = pl.read_csv(p, **kw)
            break
        except Exception:
            continue

    if df is None:
        try:
            df = pl.read_csv(p, **base_kw, encoding="utf8-lossy")
        except Exception:
            pass

    if df is None:
        import pandas as pd
        for enc in ("gb18030", "gbk", "utf-8", "latin1"):
            try:
                dfp = pd.read_csv(p, usecols=cols, encoding=enc)
                df = pl.from_pandas(dfp)
                break
            except Exception:
                continue

    if df is None or df.height == 0:
        logger.warning(f"读取失败或为空，跳过: {p}")
        return None

    return df.with_columns(
        pl.lit(file_date).cast(pl.Date).alias("date")
    ).select(["date", "wind_code", "close"])


def _read_daily_long(
    paths: Sequence[Path],
    map_ticker: Callable[[str], str],
) -> pl.DataFrame:
    """读取多个日文件为长表：date, ticker, close。"""
    if not paths:
        raise ValueError("未找到任何符合日期范围的 CSV 文件；请检查路径与起止日期")

    ticker_expr = _wind_to_ticker_expr(map_ticker)
    frames: List[pl.DataFrame] = []
    for p in paths:
        df = _read_csi800_day_file(p)
        if df is None:
            continue
        df = df.with_columns(ticker_expr.alias("ticker")).select(["date", "ticker", "close"])
        df = df.filter(pl.col("ticker").is_not_null() & pl.col("close").is_not_null())
        frames.append(df)

    if not frames:
        raise ValueError("所有 CSV 读取失败或为空，无法构建面板")

    out = pl.concat(frames, how="vertical")
    out = out.sort("date").unique(subset=["date", "ticker"], keep="last")
    return out


def _daily_to_month_end_close(daily: pl.DataFrame) -> pl.DataFrame:
    """
    按 (年-月, ticker) 聚合为**月末最后交易日**收盘价。
    先按 (ticker, date) 排序再在组内取 last()，确保是真实月末价。
    """
    ym = pl.col("date").dt.strftime("%Y-%m")
    return (
        daily.sort(["ticker", "date"])
        .with_columns(ym.alias("ym"))
        .group_by(["ym", "ticker"])
        .agg(
            pl.col("date").last().alias("date"),
            pl.col("close").last().alias("close"),
        )
        .drop("ym")
        .sort("date")
    )


def _normalize_to_month_end_date(monthly_long: pl.DataFrame) -> pl.DataFrame:
    """
    将月末实际交易日规范化为日历月末日（如 2020-12-30 → 2020-12-31）。
    确保宽表每个自然月精确对应一行，12 期回溯与论文设置对齐。
    """
    normalized = monthly_long.with_columns(
        (
            pl.col("date")
            .dt.truncate("1mo")
            .dt.offset_by("1mo")
            .dt.offset_by("-1d")
        ).alias("date")
    )
    return normalized.unique(subset=["date", "ticker"], keep="last").sort("date")


def _wide_prices_from_long(monthly_long: pl.DataFrame) -> pl.DataFrame:
    """长表 -> 宽表 prices（date + 各 ticker 列）。"""
    return monthly_long.pivot(on="ticker", index="date", values="close").sort("date")


def _forward_fill_prices_wide(prices: pl.DataFrame) -> pl.DataFrame:
    """对每列价格按时间前向填充（处理成分股进出导致的稀疏 NaN）。"""
    tickers = [c for c in prices.columns if c != "date"]
    if not tickers:
        return prices
    out = prices.sort("date")
    return out.with_columns([pl.col(c).forward_fill().alias(c) for c in tickers])


def _monthly_returns_from_wide_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """宽表价格 -> 月度简单收益率；移除第一期（无上一月价格），保留行内 NaN。"""
    ticker_cols = [c for c in prices.columns if c != "date"]
    if not ticker_cols:
        raise ValueError("宽表价格中没有任何 ticker 列")
    rets = prices.sort("date").with_columns(
        [pl.col(c).pct_change().alias(c) for c in ticker_cols]
    )
    return rets.slice(1)


def _filter_tickers(prices: pl.DataFrame, tickers: Optional[Sequence[str]]) -> pl.DataFrame:
    if not tickers:
        return prices
    want = {t.strip() for t in tickers if t and str(t).strip()}
    cols = ["date"] + [c for c in prices.columns if c != "date" and c in want]
    missing = want - set(cols)
    if missing:
        logger.warning(
            f"下列 ticker 在价格表中不存在，已忽略: "
            f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
        )
    if len(cols) <= 1:
        raise ValueError("按 tickers 过滤后无任何可用标的列")
    return prices.select(cols)


@dataclass
class _BuildConfig:
    dataset_root: Path
    start_date: date
    end_date: date
    tickers: Optional[Sequence[str]]
    map_ticker: Callable[[str], str]


def build_stock_data_from_csi800_folder(
    dataset_root: Union[str, Path],
    *,
    start_date: Union[str, date],
    end_date: Union[str, date],
    tickers: Optional[Sequence[str]] = None,
    wind_code_to_ticker: Optional[Callable[[str], str]] = None,
    monthly_wide_parquet_cache: Optional[Union[str, Path]] = None,
    refresh_monthly_cache: bool = False,
    fill_missing_prices: Literal[False, "forward"] = "forward",
) -> StockData:
    """
    从 CSI800_volume_price 目录构建月度 ``StockData``，接口与 spx_local 完全对称。

    Parameters
    ----------
    dataset_root
        数据根目录（其下为 ``2007``…``2025`` 子文件夹，内为 ``YYYYMMDD.csv``）。
    start_date, end_date
        闭区间；支持 ``YYYY-MM-DD`` 字符串或 ``datetime.date``。
    tickers
        若给定，仅保留这些列（映射后的 ticker）；``None`` 保留全部标的。
    wind_code_to_ticker
        自定义 Wind 代码映射函数；默认去掉 ``.SZ`` / ``.SH`` 等后缀。
    monthly_wide_parquet_cache
        月度宽表 Parquet 缓存路径；存在时直接读取，跳过 CSV 扫描。
    refresh_monthly_cache
        True 时忽略已有缓存，从 CSV 重建。
    fill_missing_prices
        ``"forward"`` 时对稀疏 NaN 做时间前向填充（默认开启，与 spx_local 一致）。

    Returns
    -------
    StockData
        ``prices`` / ``returns`` 与其他适配器语义一致；``market_cap`` 为 ``None``。
    """
    cfg = _BuildConfig(
        dataset_root=Path(dataset_root),
        start_date=_parse_date(start_date),
        end_date=_parse_date(end_date),
        tickers=tickers,
        map_ticker=wind_code_to_ticker or default_wind_code_to_ticker,
    )

    cache_path = Path(monthly_wide_parquet_cache) if monthly_wide_parquet_cache else None

    if cache_path and cache_path.is_file() and not refresh_monthly_cache:
        logger.info(f"从 Parquet 缓存加载月度宽表: {cache_path}")
        prices = pl.read_parquet(cache_path)
        if prices.get_column("date").dtype != pl.Date:
            prices = prices.with_columns(pl.col("date").cast(pl.Date))
    else:
        paths = list(iter_csi800_daily_csv_paths(cfg.dataset_root, cfg.start_date, cfg.end_date))
        logger.info(f"共 {len(paths)} 个交易日文件，读取并聚合为月度收盘价…")
        daily = _read_daily_long(paths, cfg.map_ticker)
        daily = daily.filter(
            (pl.col("date") >= pl.lit(cfg.start_date))
            & (pl.col("date") <= pl.lit(cfg.end_date))
        )
        monthly_long = _daily_to_month_end_close(daily)
        monthly_long = _normalize_to_month_end_date(monthly_long)
        monthly_long = monthly_long.filter(
            (pl.col("date") >= pl.lit(cfg.start_date))
            & (pl.col("date") <= pl.lit(cfg.end_date))
        )
        prices = _wide_prices_from_long(monthly_long)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prices.write_parquet(cache_path)
            logger.info(f"月度宽表已写入缓存: {cache_path}")

    prices = _filter_tickers(prices, cfg.tickers)

    # 去掉全为空的列
    ticker_cols = [c for c in prices.columns if c != "date"]
    non_null_counts = prices.select([pl.col(c).is_not_null().sum() for c in ticker_cols])
    counts_row = non_null_counts.row(0, named=False)
    keep = ["date"] + [ticker_cols[i] for i, n in enumerate(counts_row) if n > 0]
    prices = prices.select(keep)

    price_fill_note = "未对缺失价做填充（严格成分内月末价）。"
    if fill_missing_prices == "forward":
        prices = _forward_fill_prices_wide(prices)
        price_fill_note = (
            "价格已按时间前向填充（缺失月继承上一月末价，对应月收益可为 0）。"
        )

    returns = _monthly_returns_from_wide_prices(prices)

    meta = {
        "source": "csi800_local_wind_csv",
        "dataset_root": str(cfg.dataset_root.resolve()),
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "n_tickers": len(prices.columns) - 1,
        "interval": "1mo",
        "price_fill": fill_missing_prices if fill_missing_prices else "none",
        "note": "月末为当月最后一个交易日；收益率为月度简单收益率。" + price_fill_note,
    }

    logger.info(
        f"CSI800 本地月度面板: 价格行 {prices.height}, "
        f"收益行 {returns.height}, 标的数 {len(prices.columns) - 1}"
    )

    return StockData(
        prices=prices,
        returns=returns,
        market_cap=None,
        metadata=meta,
    )


def save_monthly_wide_prices_parquet(prices: pl.DataFrame, path: Union[str, Path]) -> None:
    """将月度宽表价格写入 Parquet，供 ``monthly_wide_parquet_cache`` 参数复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(path)


def attach_stock_data(engine: "PortfolioEngine", data: StockData) -> None:
    """
    将 ``StockData`` 挂到 ``PortfolioEngine`` 上（等价于 ``load_data``）。
    调用后执行 ``engine.prepare_features()`` 即可继续管道。
    """
    engine._data = data  # noqa: SLF001


def load_stock_data_from_monthly_wide_parquet(path: Union[str, Path]) -> StockData:
    """仅从已保存的月度宽表 Parquet 构建 ``StockData``（无原始 CSV 时可用）。"""
    prices = pl.read_parquet(path)
    if "date" not in prices.columns:
        raise ValueError("parquet 必须包含 date 列")
    if prices.get_column("date").dtype != pl.Date:
        prices = prices.with_columns(pl.col("date").cast(pl.Date))
    returns = _monthly_returns_from_wide_prices(prices)
    return StockData(
        prices=prices,
        returns=returns,
        market_cap=None,
        metadata={"source": "csi800_local_parquet_only", "path": str(Path(path).resolve())},
    )
