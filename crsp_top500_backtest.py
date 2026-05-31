"""CRSP Table 6 回测：Top500/Top500 十分位组合（论文 Murray et al. 2024）

设定（Section 4.4 / Table 6）：
  - 样本筛选：论文 Section 2.1（普通股 + 三交易所 + 12月收益完整 + t-1 市值）
  - Breakpoints 池 = 合格样本中市值最大的 500 只（t-1 末）
  - Holdings 池   = 同上 Top 500
  - 在 Top 500 内按 MLER 升序分 10 组
  - 组合收益 = 市值加权（权重为 t-1 末市值）
  - 测试期默认 1963-07 → 2022-12

用法::

    python crsp_top500_backtest.py --run-dir result/run_20260511_035019_cnn_lstm_...
    python crsp_top500_backtest.py --run-dir ... --regenerate-forecasts
    python crsp_top500_backtest.py --run-dir ... --test-start 2011-01 --test-end 2022-12
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from datetime import date as dt_date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

# ── Windows DLL 修复 ──────────────────────────────────────────────────────────
for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

from cbm import CBMConfig, PortfolioEngine
from cbm.data.crsp_sample_filter import (
    PAPER_NYSE_EXCHANGE,
    load_paper_crsp_panel,
    panel_load_start,
)
from cbm.data.french_rf import apply_excess_returns, load_french_rf_monthly
from cbm.core.config import BacktestConfig, ModelConfig
from cbm.ml import ModelRegistry

ROOT        = Path(__file__).resolve().parent
CRSP_PARQ   = ROOT / "crsp_data" / "crsp2525_monthly.parquet"
RESULT_ROOT = ROOT / "result"

TOP_N            = 500
N_PORTFOLIOS     = 10
TEST_PERIOD      = ("1963-07", "2022-12")   # 论文 Table 6
DATA_START       = "1927-01-01"             # 重新生成预测时需要足够历史
DATA_END         = "2022-12-31"
NEWEY_WEST_LAGS  = 12


def _find_latest_crsp_run() -> Path:
    candidates = sorted(RESULT_ROOT.glob("run_*_cnn_lstm_*"))
    if not candidates:
        raise FileNotFoundError(f"未找到 run 目录: {RESULT_ROOT}")
    return candidates[-1]


def _period_bounds(period: tuple[str, str]) -> tuple[dt_date, dt_date]:
    y0, m0 = map(int, period[0].split("-"))
    y1, m1 = map(int, period[1].split("-"))
    last = calendar.monthrange(y1, m1)[1]
    return dt_date(y0, m0, 1), dt_date(y1, m1, last)


def _ym_expr(col: str = "date") -> pl.Expr:
    return (pl.col(col).dt.year() * 100 + pl.col(col).dt.month()).alias("ym")


def _add_ym(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_ym_expr())


def _prev_ym(ym: int) -> int:
    y, m = divmod(ym, 100)
    if m == 1:
        return (y - 1) * 100 + 12
    return y * 100 + (m - 1)

def _newey_west_tstat(returns: np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n == 0:
        return float("nan")
    mu = float(np.mean(r))
    if n < lags + 2:
        se = float(np.std(r)) / np.sqrt(n)
        return mu / se if se > 0 else float("nan")
    d = r - mu
    var = float(np.sum(d ** 2) / n)
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * float(np.sum(d[lag:] * d[:-lag]) / n)
    se = np.sqrt(var / n)
    return mu / se if se > 0 else float("nan")


def _performance_metrics(returns: np.ndarray) -> dict:
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return {
            "mean_return": np.nan, "std_dev": np.nan, "sharpe_ratio": np.nan,
            "t_statistic": np.nan, "annualized_return": np.nan,
            "max_drawdown": np.nan, "n_months": 0,
        }
    mean_r = float(np.mean(r))
    std_r = float(np.std(r))
    sharpe = (mean_r / std_r * np.sqrt(12)) if std_r > 0 else 0.0
    tstat = _newey_west_tstat(r)
    cagr = float(np.prod(1 + r) ** (12 / len(r)) - 1)
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.min((cum - peak) / peak))
    return {
        "mean_return": mean_r, "std_dev": std_r, "sharpe_ratio": sharpe,
        "t_statistic": float(tstat), "annualized_return": cagr,
        "max_drawdown": mdd, "n_months": len(r),
    }


def load_forecasts_long(run_dir: Path) -> pl.DataFrame:
    parq = run_dir / "forecasts.parquet"
    if not parq.exists():
        raise FileNotFoundError(f"未找到 forecasts.parquet: {parq}")
    wide = pl.read_parquet(parq)
    tickers = [c for c in wide.columns if c != "date"]
    long = (
        wide.unpivot(on=tickers, index="date", variable_name="permno", value_name="score")
        .drop_nulls("score")
        .filter(pl.col("score").is_not_nan())
        .with_columns(pl.col("date").cast(pl.Date))
    )
    logger.info(f"预测宽表: {wide.height} 月 × {len(tickers)} PERMNO")
    return long


def regenerate_forecasts(run_dir: Path, test_period: tuple[str, str]) -> pl.DataFrame:
    """加载 CRSP 数据 + 已保存模型，重新生成测试期预测。"""
    model_base = run_dir / "model"
    subdirs = [p for p in model_base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"model/ 子目录为空: {model_base}")

    config = CBMConfig(
        model=ModelConfig(device="cpu", batch_size=512),
        backtest=BacktestConfig(test_period=test_period),
    )
    engine = PortfolioEngine(config)

    registry = ModelRegistry(path=str(model_base))
    model_id, model_data = registry.load(str(subdirs[0]))
    engine._models[model_id] = model_data  # noqa: SLF001

    diag_path = run_dir / "diagnostics.json"
    universe = "crsp_all"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        universe = diag.get("universe", universe)

    logger.info("加载 CRSP 数据并生成特征（用于重新预测）…")
    engine.load_data(
        source="crsp_local",
        universe=universe,
        start_date=DATA_START,
        end_date=DATA_END,
    )
    engine.prepare_features()
    forecast = engine.forecast(model_id=model_id, test_period=test_period)
    return load_forecasts_long_from_wide(forecast.values)


def load_forecasts_long_from_wide(wide: pl.DataFrame) -> pl.DataFrame:
    tickers = [c for c in wide.columns if c != "date"]
    return (
        wide.unpivot(on=tickers, index="date", variable_name="permno", value_name="score")
        .drop_nulls("score")
        .filter(pl.col("score").is_not_nan())
        .with_columns(pl.col("date").cast(pl.Date))
    )


def load_crsp_panel(
    data_start: dt_date,
    data_end: dt_date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """加载论文筛选后的 CRSP 长表：returns / mcap / eligible / exchanges。"""
    return load_paper_crsp_panel(data_start, data_end, parquet_path=CRSP_PARQ)


def _calc_breakpoints(values: list[float], n_portfolios: int) -> np.ndarray:
    pct = np.linspace(0, 100, n_portfolios + 1)[1:-1]
    return np.percentile(values, pct)


def _assign_portfolios(scores: dict[str, float], breakpoints: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for permno, val in scores.items():
        p = 1
        for bp in breakpoints:
            if val > bp:
                p += 1
        out[permno] = p
    return out


def _value_weighted_return(
    permnos: list[str],
    rets: dict[str, float],
    caps: dict[str, float],
) -> float:
    weights: dict[str, float] = {}
    for p in permnos:
        if p in rets and p in caps and caps[p] > 0:
            weights[p] = caps[p]
    if not weights:
        return float("nan")
    total = sum(weights.values())
    return sum(rets[p] * weights[p] / total for p in weights)


def _ym_in_period(ym: int, period: tuple[str, str]) -> bool:
    lo, hi = _period_bounds(period)
    ym_lo = lo.year * 100 + lo.month
    ym_hi = hi.year * 100 + hi.month
    return ym_lo <= ym <= ym_hi


def compute_decile_portfolios(
    forecasts: pl.DataFrame,
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    test_period: tuple[str, str],
    top_n: int | None = TOP_N,
    n_portfolios: int = N_PORTFOLIOS,
    eligible: pl.DataFrame | None = None,
    use_nyse_breakpoints: bool = False,
    exchanges: pl.DataFrame | None = None,
    exclude_periods: list[tuple[str, str]] | None = None,
) -> dict:
    """
    长表十分位组合（市值加权，按月 partition，避免宽表扫描）。

    每月 t（按 YYYYMM 对齐，论文 Section 2.1）：
      1. 在合格样本内，用 t-1 末市值确定断点池（``top_n=None`` 为全合格样本）
      2. 在断点池内用 MLER（date=t）算断点并分组
         （``use_nyse_breakpoints=True`` 时仅用 NYSE 股票算断点，Table 3）
      3. 用 t 月收益 + t-1 末市值做价值加权

    ``top_n=500`` 时对应 Table 6 Top500/Top500。
    """
    t_start, t_end = _period_bounds(test_period)
    ym_start = t_start.year * 100 + t_start.month
    ym_end = t_end.year * 100 + t_end.month

    fc = _add_ym(forecasts)
    ret = _add_ym(returns)
    mc = _add_ym(mcap)

    mcap_by_ym = {sub["ym"][0]: sub for sub in mc.partition_by("ym", maintain_order=True)}
    fc_by_ym = {sub["ym"][0]: sub for sub in fc.partition_by("ym", maintain_order=True)}
    ret_by_ym = {sub["ym"][0]: sub for sub in ret.partition_by("ym", maintain_order=True)}

    eligible_by_ym: dict[int, set[str]] = {}
    if eligible is not None and not eligible.is_empty():
        for sub in eligible.partition_by("ym", maintain_order=True):
            eligible_by_ym[sub["ym"][0]] = set(sub["permno"].to_list())

    nyse_by_ym: dict[int, set[str]] = {}
    if use_nyse_breakpoints:
        if exchanges is None:
            raise ValueError("use_nyse_breakpoints=True 需要 exchanges 面板")
        nyse = exchanges.filter(pl.col("primary_exch") == PAPER_NYSE_EXCHANGE)
        for sub in nyse.partition_by("ym", maintain_order=True):
            nyse_by_ym[sub["ym"][0]] = set(sub["permno"].to_list())

    exclude = exclude_periods or []
    test_yms = sorted(
        ym for ym in fc_by_ym
        if ym_start <= ym <= ym_end
        and not any(_ym_in_period(ym, p) for p in exclude)
    )

    port_returns: dict[int, list[float]] = {i: [] for i in range(1, n_portfolios + 1)}
    all_dates: list[dt_date] = []
    diag_rows: list[dict] = []

    for ym in test_yms:
        ym_prev = _prev_ym(ym)
        if ym_prev not in mcap_by_ym:
            continue

        mcap_prev = mcap_by_ym[ym_prev]
        if eligible_by_ym:
            elig = eligible_by_ym.get(ym, set())
            if not elig:
                continue
            mcap_prev = mcap_prev.filter(pl.col("permno").is_in(list(elig)))

        fc_t = fc_by_ym.get(ym)
        ret_t = ret_by_ym.get(ym)
        if fc_t is None or ret_t is None:
            continue

        d = fc_t["date"][0]
        if not isinstance(d, dt_date):
            d = dt_date.fromisoformat(str(d)[:10])

        if top_n is not None:
            universe = mcap_prev.sort("mcap", descending=True).head(top_n).select("permno")
        else:
            universe = mcap_prev.select("permno")
        if universe.height < n_portfolios:
            continue

        pool = (
            universe.join(fc_t.select(["permno", "score"]), on="permno", how="inner")
            .join(mcap_prev.select(["permno", "mcap"]), on="permno", how="inner")
            .join(ret_t.select(["permno", "ret"]), on="permno", how="inner")
        )
        if pool.height < n_portfolios:
            for i in range(1, n_portfolios + 1):
                port_returns[i].append(float("nan"))
            all_dates.append(d)
            diag_rows.append({
                "date": d, "ym": ym,
                "n_eligible": len(eligible_by_ym.get(ym, [])) if eligible_by_ym else None,
                "n_universe": universe.height, "n_pool": pool.height, "empty": True,
            })
            continue

        scores = dict(zip(pool["permno"].to_list(), pool["score"].to_list()))
        caps = dict(zip(pool["permno"].to_list(), pool["mcap"].to_list()))
        rets = dict(zip(pool["permno"].to_list(), pool["ret"].to_list()))

        if use_nyse_breakpoints:
            nyse_set = nyse_by_ym.get(ym, set())
            bp_scores = [s for p, s in scores.items() if p in nyse_set]
            n_nyse = len(bp_scores)
            if n_nyse < n_portfolios:
                for i in range(1, n_portfolios + 1):
                    port_returns[i].append(float("nan"))
                all_dates.append(d)
                diag_rows.append({
                    "date": d, "ym": ym,
                    "n_eligible": len(eligible_by_ym.get(ym, [])) if eligible_by_ym else None,
                    "n_universe": universe.height, "n_pool": pool.height,
                    "n_nyse_breakpoint": n_nyse, "empty": True,
                })
                continue
            bps = _calc_breakpoints(bp_scores, n_portfolios)
        else:
            n_nyse = None
            bps = _calc_breakpoints(list(scores.values()), n_portfolios)
        assigns = _assign_portfolios(scores, bps)

        counts = {i: 0 for i in range(1, n_portfolios + 1)}
        for i in range(1, n_portfolios + 1):
            members = [p for p, g in assigns.items() if g == i]
            counts[i] = len(members)
            port_returns[i].append(_value_weighted_return(members, rets, caps))

        all_dates.append(d)
        diag_rows.append({
            "date": d,
            "ym": ym,
            "n_eligible": len(eligible_by_ym.get(ym, [])) if eligible_by_ym else None,
            "n_universe": universe.height,
            "n_pool": pool.height,
            "n_nyse_breakpoint": n_nyse,
            "n_p1": counts[1],
            "n_p10": counts[n_portfolios],
            "bp_unique": len(set(float(x) for x in bps)),
        })

    ls = [
        (port_returns[n_portfolios][i] - port_returns[1][i])
        if not (np.isnan(port_returns[n_portfolios][i]) or np.isnan(port_returns[1][i]))
        else float("nan")
        for i in range(len(all_dates))
    ]

    metrics: dict[str, dict] = {}
    for i in range(1, n_portfolios + 1):
        metrics[str(i)] = _performance_metrics(np.array(port_returns[i]))
    metrics["long_short"] = _performance_metrics(np.array(ls))

    return {
        "dates": all_dates,
        "returns": port_returns,
        "long_short": ls,
        "metrics": metrics,
        "diagnostics": diag_rows,
    }


def compute_top500_decile_portfolios(
    forecasts: pl.DataFrame,
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    test_period: tuple[str, str],
    top_n: int = TOP_N,
    n_portfolios: int = N_PORTFOLIOS,
    eligible: pl.DataFrame | None = None,
) -> dict:
    """Table 6 Top500/Top500 十分位（``compute_decile_portfolios`` 的便捷封装）。"""
    return compute_decile_portfolios(
        forecasts, returns, mcap, test_period,
        top_n=top_n, n_portfolios=n_portfolios, eligible=eligible,
    )


def print_results(results: dict, test_period: tuple[str, str]) -> None:
    print("\n" + "=" * 78)
    print("  Table 6 — Top500/Top500 十分位组合（市值加权）")
    print(f"  测试期: {test_period[0]} → {test_period[1]}")
    print("=" * 78)
    print(f"  {'组合':<12}  {'月数':>4}  {'月均收益':>9}  {'年化收益':>9}  "
          f"{'Sharpe':>7}  {'NW-t':>7}  {'最大回撤':>9}")
    print("-" * 78)

    for key in [str(i) for i in range(1, N_PORTFOLIOS + 1)] + ["long_short"]:
        m = results["metrics"][key]
        label = f"P{key}" if key != "long_short" else "L/S (P10-P1)"
        print(
            f"  {label:<12}  {m['n_months']:>4}  "
            f"{m['mean_return']*100:>8.3f}%  {m['annualized_return']*100:>8.2f}%  "
            f"{m['sharpe_ratio']:>7.3f}  {m['t_statistic']:>7.2f}  "
            f"{m['max_drawdown']*100:>8.2f}%"
        )
    print("=" * 78)

    # 论文 Table 6 Top500 对照
    print("\n  论文 Table 6 Top500/Top500 参考值（1963-07→2022-12）：")
    print("    P1=0.10%, P10=0.82%, L/S=0.72% (t=4.37)")


def save_results(
    run_dir: Path,
    results: dict,
    test_period: tuple[str, str],
    top_n: int,
    return_type: str,
) -> Path:
    out = run_dir / "top500_decile_portfolios"
    if return_type == "excess":
        out = run_dir / "top500_decile_portfolios_excess"
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "table6_top500_top500",
        "return_type": return_type,
        "sample_filter": "paper_section_2_1",
        "breakpoints_universe": f"eligible_top_{top_n}_by_mcap",
        "holdings_universe": f"eligible_top_{top_n}_by_mcap",
        "weighting": "value",
        "n_portfolios": N_PORTFOLIOS,
        "test_period": list(test_period),
        "newey_west_lags": NEWEY_WEST_LAGS,
        "performance": results["metrics"],
    }
    (out / "performance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    dates = results["dates"]
    for i in range(1, N_PORTFOLIOS + 1):
        pl.DataFrame({
            "date": [str(d) for d in dates],
            "return": results["returns"][i],
        }).write_csv(out / f"portfolio_{i}_returns.csv")

    pl.DataFrame({
        "date": [str(d) for d in dates],
        "return": results["long_short"],
    }).write_csv(out / "portfolio_long_short_returns.csv")

    if results["diagnostics"]:
        pl.DataFrame(results["diagnostics"]).write_csv(out / "monthly_diagnostics.csv")

    logger.info(f"结果已保存: {out.resolve()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSP Table 6 Top500/Top500 十分位回测")
    parser.add_argument("--run-dir", type=str, default=None, help="含 model/ 与 forecasts.parquet 的 run 目录")
    parser.add_argument("--test-start", type=str, default=TEST_PERIOD[0])
    parser.add_argument("--test-end", type=str, default=TEST_PERIOD[1])
    parser.add_argument("--top-n", type=int, default=TOP_N, help="市值 Top N（默认 500）")
    parser.add_argument(
        "--regenerate-forecasts", action="store_true",
        help="重新加载 CRSP 数据并生成预测（测试期较长时必需）",
    )
    parser.add_argument(
        "--excess-returns", action="store_true", default=True,
        help="使用超额收益（MthRet − Ken French 月度 RF），默认开启，对齐论文",
    )
    parser.add_argument(
        "--total-returns", action="store_true", default=False,
        help="使用 CRSP 总收益 MthRet（不减 RF）",
    )
    parser.add_argument(
        "--refresh-rf", action="store_true",
        help="重新下载 Ken French RF 并刷新本地缓存",
    )
    args = parser.parse_args()

    use_excess = args.excess_returns and not args.total_returns
    return_type = "excess" if use_excess else "total"

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_crsp_run()
    test_period = (args.test_start, args.test_end)
    t_start, t_end = _period_bounds(test_period)

    print(f"\nRun 目录 : {run_dir.resolve()}")
    print(f"测试期   : {test_period[0]} → {test_period[1]}")
    print(f"设定     : Top{args.top_n}/Top{args.top_n} 断点+持仓, 市值加权, {N_PORTFOLIOS} 分位")
    print(f"收益口径 : {'超额收益 (MthRet − RF)' if use_excess else '总收益 (MthRet)'}")

    if args.regenerate_forecasts:
        print("\n[1/3] 重新生成 MLER 预测…")
        forecasts = regenerate_forecasts(run_dir, test_period)
    else:
        print("\n[1/3] 加载已保存预测…")
        forecasts = load_forecasts_long(run_dir)
        fc_dates = forecasts["date"].unique().sort().to_list()
        if fc_dates and (fc_dates[0] > t_start or fc_dates[-1] < t_end):
            logger.warning(
                f"forecasts.parquet 覆盖 {fc_dates[0]}→{fc_dates[-1]}，"
                f"未完全覆盖测试期；请加 --regenerate-forecasts"
            )

    data_start = panel_load_start(t_start)
    print(f"\n[2/3] 加载 CRSP 收益与市值（论文筛选，自 {data_start} 起）…")
    returns, mcap, eligible, _ = load_crsp_panel(data_start, t_end)

    if use_excess:
        print("       加载 Ken French 月度 RF 并转为超额收益…")
        rf = load_french_rf_monthly(refresh=args.refresh_rf)
        n_before = returns.height
        returns = apply_excess_returns(returns, rf)
        matched = returns.join(rf.select("ym"), on="ym", how="inner").height
        logger.info(f"超额收益: {n_before:,} 行, RF 匹配 ym 覆盖 {rf.height} 个月")

    print(f"\n[3/3] 构建 Top{args.top_n}/Top{args.top_n} 十分位组合（先合格样本 → Top{args.top_n}）…")
    results = compute_top500_decile_portfolios(
        forecasts, returns, mcap, test_period, top_n=args.top_n, eligible=eligible,
    )

    print_results(results, test_period)
    out_dir = save_results(run_dir, results, test_period, args.top_n, return_type)
    print(f"\n输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
