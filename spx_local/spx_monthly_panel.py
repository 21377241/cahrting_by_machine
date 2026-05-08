"""
Wind SPX 成分日频 CSV 目录 -> 月度 ``StockData``（与 Yahoo 适配器结构对齐）。

设计要点
--------
- 仅读取 ``date``, ``wind_code``, ``CLOSE``，避免 ``sec_name`` 中文编码问题。
- 月度收盘价：每个 (年-月, ticker) 取**该月最后一个交易日**的收盘价。
- 收益率：按列 ``pct_change``，**只去掉首行**（全截面无上一期收益），
  保留行内 NaN（成分进出指数），与稀疏面板一致；**不做全行 dropna**。
- ``wind_code``（如 ``AAPL.O``）经 ``default_wind_code_to_ticker`` 映射为
  ``AAPL`` 等列名，便于与常见 ticker 习惯一致。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Sequence, TYPE_CHECKING, Union

import polars as pl
from loguru import logger

from cbm.core.types import StockData

if TYPE_CHECKING:
    from cbm.core.engine import PortfolioEngine


_WIND_SUFFIX_RE = re.compile(r"\.([A-Za-z]{1,2})$")


def default_wind_code_to_ticker(wind_code: str) -> str:
    """
    Wind 证券代码 -> 常用 ticker 形式（去掉交易所后缀 ``.O`` / ``.N`` 等）。

    规则：若末尾为 ``.<1~2 位字母>`` 则去掉该后缀；否则原样返回并 strip。
    例：``AAPL.O`` -> ``AAPL``；``BRK-B.N`` -> ``BRK-B``；``A.N`` -> ``A``。
    """
    s = wind_code.strip()
    if not s:
        raise ValueError("wind_code 为空")
    m = _WIND_SUFFIX_RE.search(s)
    if m:
        return s[: m.start()]
    return s


def iter_spx_daily_csv_paths(
    dataset_root: Union[str, Path],
    start_date: Union[str, date],
    end_date: Union[str, date],
) -> Iterator[Path]:
    """
    按 ``readme`` 约定遍历 ``{root}/{year}/YYYYMMDD.csv``，并过滤在
    ``[start_date, end_date]`` 内的文件路径（按文件名日期，且目录存在）。
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root.resolve()}")

    start = _parse_date(start_date)
    end = _parse_date(end_date)
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
    """默认映射用正则去掉 ``.O`` / ``.N`` 等后缀；自定义映射退回逐元素 Python 调用。"""
    if map_ticker is default_wind_code_to_ticker:
        return pl.col("wind_code").str.replace(r"\.[A-Za-z]{1,2}$", "")
    return pl.col("wind_code").map_elements(map_ticker, return_dtype=pl.Utf8)


def _read_spx_day_file(p: Path) -> Optional[pl.DataFrame]:
    """
    读取单日 Wind 导出 CSV（仅三列），兼容 GBK/GB18030 与含非法 UTF-8 字节文件。

    ``sec_name`` 等列可能含中文或损坏字节；Polars 默认 UTF-8 整行解码会失败，
    故依次尝试中文编码与 ``utf8-lossy``，最后用 pandas ``encoding_errors`` 兜底。
    """
    cols = ["date", "wind_code", "CLOSE"]
    overrides = {"CLOSE": pl.Float64}
    base_kw: dict = {
        "columns": cols,
        "schema_overrides": overrides,
        "infer_schema_length": 5000,
    }

    for enc in ("gb18030", "gbk", "utf8"):
        try:
            kw = dict(base_kw)
            if enc != "utf8":
                kw["encoding"] = enc
            return pl.read_csv(p, **kw)
        except Exception:
            continue

    try:
        return pl.read_csv(p, **base_kw, encoding="utf8-lossy")
    except TypeError:
        # 旧版 Polars 无 utf8-lossy
        pass
    except Exception:
        pass

    import pandas as pd

    for enc in ("gb18030", "gbk", "utf-8"):
        try:
            dfp = pd.read_csv(p, usecols=cols, encoding=enc)
            return pl.from_pandas(dfp)
        except Exception:
            continue
    try:
        dfp = pd.read_csv(
            p,
            usecols=cols,
            encoding="utf-8",
            encoding_errors="replace",
        )
        return pl.from_pandas(dfp)
    except TypeError:
        # 旧版 pandas 无 encoding_errors
        try:
            dfp = pd.read_csv(p, usecols=cols, encoding="latin1")
            return pl.from_pandas(dfp)
        except Exception as e:
            logger.warning(f"读取失败，跳过 {p}: {e}")
            return None
    except Exception:
        try:
            dfp = pd.read_csv(p, usecols=cols, encoding="latin1")
            return pl.from_pandas(dfp)
        except Exception as e:
            logger.warning(f"读取失败，跳过 {p}: {e}")
            return None


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
        df = _read_spx_day_file(p)
        if df is None:
            continue
        if df.height == 0:
            continue
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
            ticker_expr.alias("ticker"),
            pl.col("CLOSE").alias("close"),
        ).select(["date", "ticker", "close"])
        df = df.filter(pl.col("date").is_not_null() & pl.col("ticker").is_not_null())
        frames.append(df)

    if not frames:
        raise ValueError("所有 CSV 读取失败或为空，无法构建面板")

    out = pl.concat(frames, how="vertical")
    # 同一日同一 ticker 多条（极少）：保留最后一条
    out = out.sort("date").unique(subset=["date", "ticker"], keep="last")
    return out


def _daily_to_month_end_close(daily: pl.DataFrame) -> pl.DataFrame:
    """
    按 (年-月, ticker) 聚合为月末**最后交易日**收盘价。

    必须先按 ``(ticker, date)`` 排序再在组内取 ``last()``；若仅 ``sort("date")``
    后 ``group_by``，组内行序不保证按日递增，``last()`` 可能不是月末收盘价。
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
    将月末实际交易日规范化为**日历月末日**（如 2011-04-29 → 2011-04-30）。

    问题根源
    --------
    不同 ticker 在同一自然月的最后交易日可能差 1–2 个交易日（如某月最后一
    个交易日对 A 是 4 月 28 日、对 B 是 4 月 29 日），``pivot`` 后同一自然月
    出现多行（multi-row-per-month）：

    - 全表 ~375 行 ÷ ~175 个自然月 ≈ 每月 2.1 行；
    - ``FeatureEngineer`` 的「12 期回溯」实际只覆盖 ~5–6 个自然月；
    - 前向填充产生大量 0 收益行，特征截面方差极小；
    - 神经网络退化为预测常数，所有股票落入 p1，p2–p5 全 NaN。

    修复逻辑
    --------
    将每行 ``date``（实际交易日）映射为所在自然月的最后一个日历日：

        下月第一日 - 1 天  （Polars 日期算术，自动处理闰年与大小月）

    归一化后宽表每个自然月精确对应一行，12 期回溯 = 12 个自然月，与
    Murray 等（2024）论文设置严格对齐。
    """
    normalized = monthly_long.with_columns(
        (
            pl.col("date")
            .dt.truncate("1mo")       # → 本月 1 日
            .dt.offset_by("1mo")      # → 下月 1 日
            .dt.offset_by("-1d")      # → 本月末日（自动适配闰年/大小月）
        ).alias("date")
    )
    # _daily_to_month_end_close 已保证每 (ym, ticker) 唯一；此处保险去重
    return normalized.unique(subset=["date", "ticker"], keep="last").sort("date")


def _wide_prices_from_long(monthly_long: pl.DataFrame) -> pl.DataFrame:
    """长表 -> 宽表 prices（date + 各 ticker 列）。"""
    wide = monthly_long.pivot(on="ticker", index="date", values="close").sort("date")
    return wide


def _forward_fill_prices_wide(prices: pl.DataFrame) -> pl.DataFrame:
    """
    按时间对每列价格做前向填充，使未在指数内的月份继承上一有效月末价。

    用于配合 ``cbm`` 特征工程（12 期滚动积）：原始稀疏 NaN 会导致几乎所有
    (月×股) 的累积收益为 NaN，从而 **0 条训练样本**。填充后近似「未调整仓位则
    价格不变、月收益为 0」，与常见面板 ML 处理一致；**经济含义与严格成分进出
    不同**，见 ``metadata["price_fill"]``。
    """
    tickers = [c for c in prices.columns if c != "date"]
    if not tickers:
        return prices
    out = prices.sort("date")
    return out.with_columns([pl.col(c).forward_fill().alias(c) for c in tickers])


def _monthly_returns_from_wide_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """
    与 Yahoo 路径一致：对每列做简单收益率 pct_change；
    仅移除第一期（截面无上一月价格），保留后续行中的 NaN。
    """
    ticker_cols = [c for c in prices.columns if c != "date"]
    if not ticker_cols:
        raise ValueError("宽表价格中没有任何 ticker 列")

    rets = prices.sort("date").with_columns(
        [pl.col(c).pct_change().alias(c) for c in ticker_cols]
    )
    # 仅去掉第一期（无上一月价格）；保留行内 NaN 以反映当月无成交/未在指数内等情况
    rets = rets.slice(1)
    return rets


def _filter_tickers(prices: pl.DataFrame, tickers: Optional[Sequence[str]]) -> pl.DataFrame:
    if not tickers:
        return prices
    want = {t.strip() for t in tickers if t and str(t).strip()}
    cols = ["date"] + [c for c in prices.columns if c != "date" and c in want]
    missing = want - set(cols)
    if missing:
        logger.warning(f"下列 ticker 在价格表中不存在，已忽略: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")
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


def build_stock_data_from_spx_folder(
    dataset_root: Union[str, Path],
    *,
    start_date: Union[str, date],
    end_date: Union[str, date],
    tickers: Optional[Sequence[str]] = None,
    wind_code_to_ticker: Optional[Callable[[str], str]] = None,
    monthly_wide_parquet_cache: Optional[Union[str, Path]] = None,
    refresh_monthly_cache: bool = False,
    fill_missing_prices: Literal[False, "forward"] = False,
) -> StockData:
    """
    从 SPX_volume_price 类目录构建月度 ``StockData``。

    Parameters
    ----------
    dataset_root
        数据根目录（其下为 ``2011``、``2012``… 子文件夹，内为 ``YYYYMMDD.csv``）。
    start_date, end_date
        闭区间；支持 ``YYYY-MM-DD`` 字符串或 ``datetime.date``。
    tickers
        若给定，仅保留这些列（映射后的 ticker）；``None`` 表示保留数据中出现的全部标的。
    wind_code_to_ticker
        自定义 Wind 代码映射；默认 ``default_wind_code_to_ticker``。
    monthly_wide_parquet_cache
        若提供路径且文件存在且 ``refresh_monthly_cache=False``，则直接从该 parquet
        读取**已聚合好的月度宽表价格**（列：date + tickers），跳过 CSV 扫描。
        写入可使用 ``save_monthly_wide_prices_parquet``。
    refresh_monthly_cache
        为 True 时忽略已有 parquet 缓存，从 CSV 重建。
    fill_missing_prices
        若为 ``"forward"``，在生成收益前对宽表价格按时间 **前向填充**（见
        ``_forward_fill_prices_wide``），以便 ``FeatureEngineer`` 在成分进出频繁的
        SPX 面板上能得到非零训练样本。默认 ``False`` 保持严格缺失。

    Returns
    -------
    StockData
        ``prices`` / ``returns`` 与 Yahoo 适配器语义一致；``market_cap`` 为 ``None``。
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
        logger.info(f"从 parquet 缓存加载月度宽表: {cache_path}")
        prices = pl.read_parquet(cache_path)
        if prices.get_column("date").dtype != pl.Date:
            prices = prices.with_columns(pl.col("date").cast(pl.Date))
    else:
        paths = list(iter_spx_daily_csv_paths(cfg.dataset_root, cfg.start_date, cfg.end_date))
        logger.info(f"共 {len(paths)} 个交易日文件，读取并聚合为月度收盘价…")
        daily = _read_daily_long(paths, cfg.map_ticker)
        daily = daily.filter(
            (pl.col("date") >= pl.lit(cfg.start_date)) & (pl.col("date") <= pl.lit(cfg.end_date))
        )
        monthly_long = _daily_to_month_end_close(daily)
        # 将实际末交易日统一映射到日历月末日，确保宽表每自然月仅一行
        monthly_long = _normalize_to_month_end_date(monthly_long)
        monthly_long = monthly_long.filter(
            (pl.col("date") >= pl.lit(cfg.start_date)) & (pl.col("date") <= pl.lit(cfg.end_date))
        )
        prices = _wide_prices_from_long(monthly_long)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prices.write_parquet(cache_path)
            logger.info(f"月度宽表已写入缓存: {cache_path}")

    prices = _filter_tickers(prices, cfg.tickers)
    # 去掉全为空的列，减少无效维度
    ticker_cols = [c for c in prices.columns if c != "date"]
    non_null_counts = prices.select([pl.col(c).is_not_null().sum() for c in ticker_cols])
    counts_row = non_null_counts.row(0, named=False)
    keep = ["date"] + [ticker_cols[i] for i, n in enumerate(counts_row) if n > 0]
    prices = prices.select(keep)

    price_fill_note = "未对缺失价做填充（严格成分内月末价）。"
    if fill_missing_prices == "forward":
        prices = _forward_fill_prices_wide(prices)
        price_fill_note = (
            "价格已按时间前向填充（用于 ML 管道；缺失月继承上一月末价，对应月收益可为 0）。"
        )

    returns = _monthly_returns_from_wide_prices(prices)

    meta = {
        "source": "spx_local_wind_csv",
        "dataset_root": str(cfg.dataset_root.resolve()),
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "n_tickers": len(prices.columns) - 1,
        "interval": "1mo",
        "price_fill": fill_missing_prices if fill_missing_prices else "none",
        "note": (
            "月末为当月最后一个交易日；收益率为月度简单收益率。"
            + price_fill_note
        ),
    }

    logger.info(
        f"SPX 本地月度面板: 价格行 {prices.height}, 收益行 {returns.height}, 标的数 {len(prices.columns) - 1}"
    )

    return StockData(
        prices=prices,
        returns=returns,
        market_cap=None,
        metadata=meta,
    )


def save_monthly_wide_prices_parquet(prices: pl.DataFrame, path: Union[str, Path]) -> None:
    """将月度宽表价格写入 parquet，供 ``monthly_wide_parquet_cache`` 复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(path)


def attach_stock_data(engine: "PortfolioEngine", data: StockData) -> None:
    """
    将 ``StockData`` 挂到 ``PortfolioEngine`` 上，等价于完成一次数据加载。

    说明：使用引擎实例属性 ``_data``（包内未提供公共 setter），
    本函数为官方扩展点的轻量封装，调用后请执行 ``engine.prepare_features()``。
    """
    engine._data = data  # noqa: SLF001 — 有意注入，避免改动 cbm 源码


def load_stock_data_from_monthly_wide_parquet(path: Union[str, Path]) -> StockData:
    """
    仅从已保存的月度宽表 parquet 构建 ``StockData``（无 CSV 根目录时可用）。
    """
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
        metadata={"source": "spx_local_parquet_only", "path": str(Path(path).resolve())},
    )
