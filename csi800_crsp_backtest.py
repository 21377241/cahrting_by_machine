"""使用预训练的 CRSP CNN-LSTM 模型对 CSI800 数据进行样本外回测

步骤：
  1. 加载本地 CSI800 月度面板数据（从 Parquet 缓存，< 1 秒）
  2. 从已保存的 run 目录中加载 CRSP 训练好的 CNN-LSTM 集成模型
  3. 在 CSI800 全样本上生成预测
  4. 构建十分位（Decile）组合并分析绩效
  5. 构建 Top-K 等权组合（K = 5, 10, 20, 50）并分析绩效
  6. 将结果保存到 result/ 目录

用法::

    python csi800_crsp_backtest.py
    python csi800_crsp_backtest.py --run-dir result/run_20260511_035019_cnn_lstm_...
    python csi800_crsp_backtest.py --test-start 2015-01 --test-end 2025-12
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

import numpy as np
import polars as pl
from loguru import logger

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import BacktestConfig, ModelConfig, PortfolioConfig, TrainingConfig
from cbm.core.types import ReturnVariable
from cbm.ml import ModelRegistry
from cbm.utils.results_io import save_pipeline_results
from csi800_local import attach_stock_data, build_stock_data_from_csi800_folder

# ═══════════════════════════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════════════════════════

ROOT        = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"

DEFAULT_RUN_DIR = RESULT_ROOT / "run_20260511_035019_cnn_lstm_20260511_024025_fb0c7e72"

CSI800_ROOT  = Path(r"E:\phd\LLM_trading\CNN_trading\CSI800_volume_price")
CSI800_CACHE = ROOT / "data" / "csi800_monthly_wide.parquet"
CSI800_START = "2007-01-01"
CSI800_END   = "2025-12-31"

# 测试期（2015 之后数据质量更好；可通过命令行覆盖）
TEST_PERIOD = ("2015-01", "2025-12")

N_PORTFOLIOS        = 10
PORTFOLIO_WEIGHTING = "equal"
TOP_K_LIST          = [5, 10, 20, 50]
NEWEY_WEST_LAGS     = 12

# ═══════════════════════════════════════════════════════════════════════════════


def _find_latest_crsp_run() -> Path:
    candidates = sorted(RESULT_ROOT.glob("run_*_cnn_lstm_*"))
    if not candidates:
        raise FileNotFoundError("未找到任何 cnn_lstm run 目录，请先运行 crsp_run.py")
    return candidates[-1]


def load_csi800_data(engine: PortfolioEngine, fill_prices: str | bool = False) -> None:
    sd = build_stock_data_from_csi800_folder(
        CSI800_ROOT,
        start_date=CSI800_START,
        end_date=CSI800_END,
        monthly_wide_parquet_cache=CSI800_CACHE,
        refresh_monthly_cache=False,
        fill_missing_prices=fill_prices,  # type: ignore[arg-type]
    )
    attach_stock_data(engine, sd)
    logger.info(
        f"CSI800 月度数据: {len(sd.tickers)} 标的, "
        f"收益表 {sd.returns.height} 行, "
        f"区间 {sd.date_range[0]} ~ {sd.date_range[1]}"
    )


def load_crsp_model(engine: PortfolioEngine, run_dir: Path) -> str:
    model_base = run_dir / "model"
    if not model_base.exists():
        raise FileNotFoundError(f"model/ 子目录不存在: {model_base}")
    model_subdirs = [p for p in model_base.iterdir() if p.is_dir()]
    if not model_subdirs:
        raise FileNotFoundError(f"model/ 目录内无模型子目录: {model_base}")
    model_path = model_subdirs[0]

    registry = ModelRegistry(path=str(model_base))
    loaded_id, model_data = registry.load(str(model_path))
    engine._models[loaded_id] = model_data  # noqa: SLF001
    logger.info(f"已加载 CRSP 模型: {loaded_id}")
    return loaded_id


# ── Newey-West t 统计量 & 绩效指标 ──────────────────────────────────────────

def _newey_west_tstat(returns: np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
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
        var += 2 * w * np.sum(demeaned[lag:] * demeaned[:-lag]) / n
    se = np.sqrt(var / n)
    return mean / se if se > 0 else float("nan")


def _performance_metrics(returns: np.ndarray) -> dict:
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return dict(mean_return=np.nan, std_dev=np.nan, sharpe_ratio=np.nan,
                    t_statistic=np.nan, annualized_return=np.nan,
                    max_drawdown=np.nan, n_months=0)
    mean_r = float(np.mean(r))
    std_r  = float(np.std(r))
    sharpe = (mean_r / std_r * np.sqrt(12)) if std_r > 0 else 0.0
    tstat  = _newey_west_tstat(r)
    n_years = len(r) / 12
    cagr   = float(np.prod(1 + r) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    cum    = np.cumprod(1 + r)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
    return dict(mean_return=mean_r, std_dev=std_r, sharpe_ratio=sharpe,
                t_statistic=float(tstat), annualized_return=cagr,
                max_drawdown=max_dd, n_months=len(r))


def compute_topk_portfolios(
    forecasts_wide: pl.DataFrame,
    returns_wide: pl.DataFrame,
    k_list: list[int],
) -> dict[int, dict]:
    ticker_cols = [c for c in forecasts_wide.columns if c != "date"]
    fc_long = (
        forecasts_wide
        .unpivot(on=ticker_cols, index="date", variable_name="ticker", value_name="score")
        .drop_nulls("score")
        .filter(pl.col("score").is_not_nan())
    )
    ret_cols = [c for c in returns_wide.columns if c != "date"]
    ret_long = (
        returns_wide
        .unpivot(on=ret_cols, index="date", variable_name="ticker", value_name="ret")
        .drop_nulls("ret")
    )
    joined = fc_long.join(ret_long, on=["date", "ticker"], how="inner")
    logger.info(f"[Top-K] 预测 × 收益 匹配样本: {joined.height:,} 行")

    results: dict[int, dict] = {}
    for k in k_list:
        monthly_ret = (
            joined
            .sort("score", descending=True)
            .group_by("date")
            .agg(
                pl.col("score").len().alias("n_with_score"),
                pl.col("ret")
                  .sort_by(pl.col("score"), descending=True)
                  .head(k)
                  .mean()
                  .alias("port_ret"),
            )
            .sort("date")
            .filter(pl.col("n_with_score") >= k)
        )
        ret_arr = monthly_ret["port_ret"].to_numpy()
        dates   = monthly_ret["date"].to_list()
        metrics = _performance_metrics(ret_arr)
        results[k] = {"dates": dates, "returns": ret_arr, "metrics": metrics}
        logger.info(
            f"  Top-{k:>2}: {metrics['n_months']} 月, "
            f"月均 {metrics['mean_return']*100:.3f}%, "
            f"Sharpe {metrics['sharpe_ratio']:.3f}, "
            f"t={metrics['t_statistic']:.2f}"
        )
    return results


def print_decile_table(performance: dict) -> None:
    print("\n" + "=" * 72)
    print("  十分位（Decile）组合绩效")
    print("=" * 72)
    print(f"  {'组合':<14}  {'月均收益':>9}  {'年化Sharpe':>10}  {'t统计量':>8}  {'年化收益':>9}  {'最大回撤':>9}")
    print("-" * 72)
    for name, m in sorted(performance.items(), key=lambda x: x[0]):
        print(
            f"  {name:<14}  {m.mean_return*100:>8.3f}%  "
            f"{m.sharpe_ratio:>10.3f}  "
            f"{m.t_statistic:>8.2f}  "
            f"{m.annualized_return*100:>8.2f}%  "
            f"{m.max_drawdown*100:>8.2f}%"
        )
    ls = performance.get("long_short")
    if ls:
        print("─" * 72)
        print(
            f"\n  多空组合 (P10-P1):\n"
            f"    月均超额收益: {ls.mean_return*100:.3f}%\n"
            f"    年化 Sharpe : {ls.sharpe_ratio:.3f}\n"
            f"    t 统计量    : {ls.t_statistic:.2f}\n"
            f"    年化收益    : {ls.annualized_return*100:.2f}%\n"
            f"    最大回撤    : {ls.max_drawdown*100:.2f}%"
        )
    print("=" * 72)


def print_topk_table(topk_results: dict[int, dict], test_start: str, test_end: str) -> None:
    print("\n" + "=" * 72)
    print(f"  Top-K 等权组合绩效（测试期：{test_start} → {test_end}）")
    print("=" * 72)
    print(
        f"  {'组合':<12}  {'月数':>4}  {'月均收益':>9}  "
        f"{'年化收益':>9}  {'年化Sharpe':>10}  {'t统计量':>8}  {'最大回撤':>9}"
    )
    print("-" * 72)
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
    print("=" * 72)


def save_all_results(
    engine: PortfolioEngine,
    model_id: str,
    performance: dict,
    forecasts,
    portfolios,
    topk_results: dict[int, dict],
    diagnostics: dict,
) -> Path:
    out_dir = save_pipeline_results(
        engine=engine,
        model_id=model_id,
        performance=performance,
        forecasts=forecasts,
        portfolios=portfolios,
        result_dir=RESULT_ROOT,
        diagnostics=diagnostics,
        experiment_tag="csi800_crsp_transfer_cnn_lstm",
    )

    topk_dir = out_dir / "topk_portfolios"
    topk_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for k, res in topk_results.items():
        m = res["metrics"].copy()
        m.pop("n_months", None)
        summary[f"top{k}"] = m
    (topk_dir / "topk_performance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    for k, res in topk_results.items():
        df = pl.DataFrame({
            "date":   [str(d) for d in res["dates"]],
            "return": res["returns"].tolist(),
        })
        df.write_csv(topk_dir / f"top{k}_monthly_returns.csv")

    logger.info(f"Top-K 结果已保存至: {topk_dir.resolve()}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI800 × CRSP 预训练模型回测")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="CRSP 训练 run 目录；默认使用 DEFAULT_RUN_DIR")
    parser.add_argument("--test-start", type=str, default=TEST_PERIOD[0],
                        help=f"测试期开始月（YYYY-MM），默认 {TEST_PERIOD[0]}")
    parser.add_argument("--test-end", type=str, default=TEST_PERIOD[1],
                        help=f"测试期结束月（YYYY-MM），默认 {TEST_PERIOD[1]}")
    parser.add_argument("--fill-prices", action="store_true", default=False,
                        help="对缺失价格做前向填充（默认关闭；开启会引入僵尸股）")
    args = parser.parse_args()

    test_period = (args.test_start, args.test_end)

    if args.run_dir:
        crsp_run_dir = Path(args.run_dir)
    elif DEFAULT_RUN_DIR.exists():
        crsp_run_dir = DEFAULT_RUN_DIR
    else:
        crsp_run_dir = _find_latest_crsp_run()

    print("\n" + "=" * 70)
    print("  CSI800 数据 × CRSP 预训练 CNN-LSTM 回测")
    print("=" * 70)
    print(f"  CRSP run 目录 : {crsp_run_dir.resolve()}")
    print(f"  CSI800 数据   : {CSI800_ROOT}")
    print(f"  测试期        : {test_period[0]} → {test_period[1]}")
    print(f"  十分位组合    : {N_PORTFOLIOS}")
    print(f"  Top-K         : {TOP_K_LIST}")
    print("=" * 70 + "\n")

    diag_path = crsp_run_dir / "diagnostics.json"
    crsp_diag: dict = {}
    if diag_path.exists():
        crsp_diag = json.loads(diag_path.read_text(encoding="utf-8"))
        print(f"CRSP 模型信息: arch={crsp_diag.get('architecture')}, "
              f"优化期={crsp_diag.get('optimization_period')}")

    config = CBMConfig(
        model=ModelConfig(device="cpu", batch_size=512),
        training=TrainingConfig(
            optimization_period=("2007-01", "2014-12"),
            return_variable=ReturnVariable.RET_RANK_NORM,
        ),
        portfolio=PortfolioConfig(
            n_portfolios=N_PORTFOLIOS,
            weighting=PORTFOLIO_WEIGHTING,
        ),
        backtest=BacktestConfig(test_period=test_period),
    )
    engine = PortfolioEngine(config)

    fill_mode = "forward" if args.fill_prices else False
    print(f"  价格填充模式  : {'前向填充（含僵尸股）' if fill_mode else '不填充（仅活跃成分股）'}")

    # ── 步骤 1：加载 CSI800 数据 ──────────────────────────────────────────────
    print("步骤 1/5  加载 CSI800 月度数据…")
    load_csi800_data(engine, fill_prices=fill_mode)

    # ── 步骤 2：特征工程 ──────────────────────────────────────────────────────
    print("\n步骤 2/5  特征工程（CR1…CR12）…")
    engine.prepare_features()
    fs = engine.features
    print(
        f"  CSI800 特征集: {len(fs):,} 样本, "
        f"特征维度 {fs.features.shape[1]}, "
        f"月份数 {len(set(fs.dates.tolist()))}"
    )

    # ── 步骤 3：加载 CRSP 预训练模型 ─────────────────────────────────────────
    print("\n步骤 3/5  加载 CRSP 预训练模型…")
    model_id = load_crsp_model(engine, crsp_run_dir)
    ensemble = engine._models[model_id]["model"]  # noqa: SLF001
    n_members = len(ensemble.models) if hasattr(ensemble, "models") else "?"
    print(f"  模型 ID : {model_id}")
    print(f"  集成成员: {n_members} 个")

    # ── 步骤 4：样本外预测 ───────────────────────────────────────────────────
    print(f"\n步骤 4/5  样本外预测 ({test_period[0]} → {test_period[1]})…")
    forecasts = engine.forecast(model_id=model_id, test_period=test_period)
    print(f"  预测覆盖月数 : {forecasts.values.height}")
    print(f"  预测标的数   : {forecasts.values.width - 1}")

    # ── 步骤 5a：十分位组合 ───────────────────────────────────────────────────
    print(f"\n步骤 5/5  构建十分位组合（{N_PORTFOLIOS} 分位）并分析绩效…")
    portfolios = engine.construct_portfolios(
        forecasts=forecasts,
        n_portfolios=N_PORTFOLIOS,
        weighting=PORTFOLIO_WEIGHTING,
    )
    performance = engine.analyze_performance(portfolios)
    print_decile_table(performance)

    # ── 步骤 5b：Top-K 组合 ───────────────────────────────────────────────────
    print(f"\n构建 Top-K 等权组合（K = {TOP_K_LIST}）…")
    topk_results = compute_topk_portfolios(
        forecasts_wide=forecasts.values,
        returns_wide=engine.data.returns,
        k_list=TOP_K_LIST,
    )
    print_topk_table(topk_results, test_period[0], test_period[1])

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    diagnostics = {
        "data_source": "csi800_local_wind_csv",
        "fill_missing_prices": str(fill_mode),
        "csi800_root": str(CSI800_ROOT.resolve()),
        "csi800_start_date": CSI800_START,
        "csi800_end_date": CSI800_END,
        "test_period": list(test_period),
        "n_portfolios": N_PORTFOLIOS,
        "portfolio_weighting": PORTFOLIO_WEIGHTING,
        "top_k_list": TOP_K_LIST,
        "pretrained_model_id": model_id,
        "pretrained_run_dir": str(crsp_run_dir.resolve()),
        "crsp_training_info": {
            "architecture": crsp_diag.get("architecture"),
            "optimization_period": crsp_diag.get("optimization_period"),
            "return_variable": crsp_diag.get("return_variable"),
            "n_ensemble": crsp_diag.get("n_ensemble"),
        },
    }

    out_dir = save_all_results(
        engine=engine,
        model_id=model_id,
        performance=performance,
        forecasts=forecasts,
        portfolios=portfolios,
        topk_results=topk_results,
        diagnostics=diagnostics,
    )

    print(f"\n所有结果已保存至: {out_dir.resolve()}")
    print("  performance.json            — 十分位绩效指标")
    print("  forecasts.parquet           — CSI800 预测宽表")
    print("  portfolio_*_returns.parquet — 各组合月度收益")
    print("  topk_portfolios/            — Top-K 绩效与月度收益 CSV")
    print("  diagnostics.json            — 完整参数记录")


if __name__ == "__main__":
    main()
