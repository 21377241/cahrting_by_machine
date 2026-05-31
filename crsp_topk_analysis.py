"""Top-K 股票组合回测分析

基于已保存的 forecasts.parquet 和 CRSP 收益数据，构建 Top-K 等权组合
（K = 5, 10, 20），并计算与分位组合相同的绩效指标。

用法::

    python crsp_topk_analysis.py
    python crsp_topk_analysis.py --run-dir result/run_<timestamp>_<tag>
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ── Windows DLL 修复 ──────────────────────────────────────────────────────────
for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

# ── 路径常量 ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
CRSP_PARQ   = ROOT / "crsp_data" / "crsp2525_monthly.parquet"
RESULT_ROOT = ROOT / "result"

TOP_K_LIST  = [5, 10, 20, 50, 100, 500]   # 要评估的 K 值
TEST_PERIOD = ("2011-01", "2022-12")   # 测试期（与 crsp_run.py 一致）
NEWEY_WEST_LAGS = 12            # NW 标准误滞后阶数

# ═══════════════════════════════════════════════════════════════════════════════


def _find_latest_run_dir() -> Path:
    """找最新的 crsp_all cnn_lstm run 目录（按名称排序取最后一个）。"""
    candidates = sorted(RESULT_ROOT.glob("run_*_cnn_lstm_*"))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {RESULT_ROOT} 中找到任何 run 目录。\n"
            "请先运行 crsp_run.py 生成结果。"
        )
    return candidates[-1]


def _newey_west_tstat(returns: np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
    """Newey-West 调整 t 统计量（零均值原假设）。"""
    n = len(returns)
    if n == 0:
        return float("nan")
    mean = np.mean(returns)
    if n < lags + 2:
        se = np.std(returns) / np.sqrt(n)
        return mean / se if se > 0 else float("nan")
    demeaned = returns - mean
    var = np.sum(demeaned ** 2) / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        autocovar = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        var += 2 * w * autocovar
    se = np.sqrt(var / n)
    return mean / se if se > 0 else float("nan")


def _performance_metrics(returns: np.ndarray) -> dict:
    """计算与 crsp_run.py 相同口径的绩效指标。"""
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return dict(mean_return=np.nan, std_dev=np.nan, sharpe_ratio=np.nan,
                    t_statistic=np.nan, annualized_return=np.nan,
                    max_drawdown=np.nan, n_months=0)
    mean_r  = float(np.mean(r))
    std_r   = float(np.std(r))
    sharpe  = (mean_r / std_r * np.sqrt(12)) if std_r > 0 else 0.0
    tstat   = _newey_west_tstat(r)
    n_years = len(r) / 12
    cagr    = float(np.prod(1 + r) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    cum     = np.cumprod(1 + r)
    rolling_max = np.maximum.accumulate(cum)
    max_dd  = float(np.min((cum - rolling_max) / rolling_max))
    return dict(mean_return=mean_r, std_dev=std_r, sharpe_ratio=sharpe,
                t_statistic=float(tstat), annualized_return=cagr,
                max_drawdown=max_dd, n_months=len(r))


def _period_to_date_bounds(period: tuple[str, str]) -> tuple[pl.Date, pl.Date]:
    """将 ('YYYY-MM', 'YYYY-MM') 转为 pl.Date 月初/月末边界。"""
    y0, m0 = map(int, period[0].split("-"))
    y1, m1 = map(int, period[1].split("-"))
    last_day = calendar.monthrange(y1, m1)[1]
    import datetime
    return (
        pl.lit(datetime.date(y0, m0, 1)).cast(pl.Date),
        pl.lit(datetime.date(y1, m1, last_day)).cast(pl.Date),
    )


def load_forecasts(run_dir: Path) -> pl.DataFrame:
    """
    加载并返回 long 格式预测表。

    输入 forecasts.parquet 为宽表：date | PERMNO_A | PERMNO_B | …
    输出 long 表：date | permno | score
    """
    parq = run_dir / "forecasts.parquet"
    if not parq.exists():
        raise FileNotFoundError(f"未找到 forecasts.parquet: {parq}")
    wide = pl.read_parquet(parq)
    print(f"  [forecasts] 宽表: {wide.height} 个月 × {wide.width - 1} 个 PERMNO")

    ticker_cols = [c for c in wide.columns if c != "date"]
    long = wide.unpivot(
        on=ticker_cols,
        index="date",
        variable_name="permno",
        value_name="score",
    ).drop_nulls("score").filter(pl.col("score").is_not_nan())
    return long


def load_crsp_returns(test_start: str, test_end: str) -> pl.DataFrame:
    """
    从 CRSP parquet 加载测试期月度收益，返回 long 表：date | permno | ret
    """
    if not CRSP_PARQ.exists():
        raise FileNotFoundError(f"未找到 CRSP Parquet: {CRSP_PARQ}")

    import datetime
    y0, m0 = map(int, test_start.split("-"))
    y1, m1 = map(int, test_end.split("-"))
    last_day = calendar.monthrange(y1, m1)[1]
    d_start = datetime.date(y0, m0, 1)
    d_end   = datetime.date(y1, m1, last_day)

    raw = pl.read_parquet(CRSP_PARQ, columns=["PERMNO", "MthCalDt", "MthRet"])
    filtered = (
        raw
        .filter(
            (pl.col("MthCalDt") >= pl.lit(d_start)) &
            (pl.col("MthCalDt") <= pl.lit(d_end))
        )
        .rename({"PERMNO": "permno_int", "MthCalDt": "date", "MthRet": "ret"})
        .with_columns(pl.col("permno_int").cast(pl.Utf8).alias("permno"))
        .select(["date", "permno", "ret"])
        .drop_nulls("ret")
    )
    print(f"  [CRSP]      测试期 {test_start}→{test_end}: "
          f"{filtered.height:,} 行, {filtered['date'].n_unique()} 个月份")
    return filtered


def compute_topk_portfolios(
    forecasts_long: pl.DataFrame,
    returns_long: pl.DataFrame,
    k_list: list[int],
) -> dict[int, dict]:
    """
    对每个 K，构建 Top-K 等权组合并返回月度收益序列。

    时序对齐：在预测日期 t，取预测得分最高的 K 只股票，
    使用这些股票在同一日期 t 的实际月收益作为组合当月收益。
    （与原始分位组合构建方式完全一致：forecast date == return date）
    """
    # 合并预测与实际收益（内连接：只保留两边都有数据的 date×permno）
    joined = forecasts_long.join(returns_long, on=["date", "permno"], how="inner")
    print(f"  [join]      匹配样本: {joined.height:,} 行")

    results: dict[int, dict] = {}

    for k in k_list:
        # 按月份：按预测得分降序取 top k，计算等权收益
        monthly_ret = (
            joined
            .sort("score", descending=True)
            .group_by("date")
            .agg(
                pl.col("score").len().alias("n_stocks_with_score"),
                # top-k：取 score 最高的 k 只（已排降序，取前 k 个 ret 的均值）
                pl.col("ret").sort_by(pl.col("score"), descending=True)
                             .head(k)
                             .mean()
                             .alias("port_ret"),
            )
            .sort("date")
            .filter(pl.col("n_stocks_with_score") >= k)  # 确保当月有足够股票
        )

        ret_arr = monthly_ret["port_ret"].to_numpy()
        dates   = monthly_ret["date"].to_list()
        metrics = _performance_metrics(ret_arr)
        results[k] = {"dates": dates, "returns": ret_arr, "metrics": metrics}

    return results


def print_results_table(topk_results: dict[int, dict]) -> None:
    """打印 Top-K 绩效汇总表。"""
    print("\n" + "=" * 66)
    print("  Top-K 等权组合绩效（测试期：2011-01 → 2022-12）")
    print("=" * 66)
    print(f"  {'组合':<12}  {'月数':>4}  {'月均收益':>9}  {'年化收益':>9}  "
          f"{'年化Sharpe':>10}  {'t统计量':>8}  {'最大回撤':>9}")
    print("-" * 66)
    for k, res in sorted(topk_results.items()):
        m = res["metrics"]
        print(
            f"  Top-{k:<7}  "
            f"{m['n_months']:>4}  "
            f"{m['mean_return']*100:>8.3f}%  "
            f"{m['annualized_return']*100:>8.2f}%  "
            f"{m['sharpe_ratio']:>10.3f}  "
            f"{m['t_statistic']:>8.2f}  "
            f"{m['max_drawdown']*100:>8.2f}%"
        )
    print("=" * 66)


def save_results(
    run_dir: Path,
    topk_results: dict[int, dict],
) -> None:
    """将 top-k 绩效与月度收益序列保存到 run 目录中。"""
    out_dir = run_dir / "topk_portfolios"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 汇总 JSON
    summary = {}
    for k, res in topk_results.items():
        m = res["metrics"].copy()
        m.pop("n_months", None)
        summary[f"top{k}"] = m
    (out_dir / "topk_performance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 各 K 的月度收益 CSV
    for k, res in topk_results.items():
        df = pl.DataFrame({
            "date":    [str(d) for d in res["dates"]],
            "return":  res["returns"].tolist(),
        })
        df.write_csv(out_dir / f"top{k}_monthly_returns.csv")

    print(f"\n结果已保存至: {out_dir.resolve()}")
    print("  topk_performance.json   — 汇总绩效指标")
    for k in topk_results:
        print(f"  top{k}_monthly_returns.csv — 月度收益序列")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSP Top-K 组合分析")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="指定 run 目录路径；默认自动选取最新的 cnn_lstm run 目录",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir()
    print(f"\n使用 run 目录: {run_dir.resolve()}")

    # ── 加载诊断参数（获取 test_period） ─────────────────────────────────────
    diag_path = run_dir / "diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        test_period = tuple(diag.get("test_period", list(TEST_PERIOD)))
    else:
        test_period = TEST_PERIOD
    print(f"测试期: {test_period[0]} → {test_period[1]}")

    print("\n[1/3] 加载模型预测…")
    forecasts_long = load_forecasts(run_dir)

    print("\n[2/3] 加载 CRSP 测试期收益…")
    returns_long = load_crsp_returns(test_period[0], test_period[1])

    print(f"\n[3/3] 构建 Top-K 组合（K = {TOP_K_LIST}）…")
    topk_results = compute_topk_portfolios(forecasts_long, returns_long, TOP_K_LIST)

    print_results_table(topk_results)
    save_results(run_dir, topk_results)


if __name__ == "__main__":
    main()
