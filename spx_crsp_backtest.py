"""使用预训练的 CRSP CNN-LSTM 模型对 SPX 数据进行样本外回测

步骤：
  1. 加载本地 SPX 月度面板数据
  2. 从已保存的 run 目录中加载 CRSP 训练好的 CNN-LSTM 集成模型
  3. 在 SPX 全样本上生成预测（测试期：2021-01 → 2025-12）
  4. 构建十分位（Decile）组合并分析绩效
  5. 构建 Top-K 等权组合（K = 5, 10, 20, 50）并分析绩效
  6. 将结果保存到 result/ 目录

用法::

    python spx_crsp_backtest.py
    python spx_crsp_backtest.py --run-dir result/run_20260511_035019_cnn_lstm_...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Windows DLL 修复 ──────────────────────────────────────────────────────────
_dll_dirs: list = []
for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        _dll_dirs.append(os.add_dll_directory(_lib_path))

import numpy as np
import polars as pl
from loguru import logger

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import BacktestConfig, ModelConfig, PortfolioConfig, TrainingConfig
from cbm.core.types import Architecture, LossFunction, ReturnVariable, WeightingScheme
from cbm.ml import ModelRegistry
from cbm.utils.results_io import save_pipeline_results
from spx_local import attach_stock_data, build_stock_data_from_spx_folder

# ═══════════════════════════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════════════════════════

ROOT          = Path(__file__).resolve().parent
RESULT_ROOT   = ROOT / "result"

# CRSP 训练 run 目录（默认使用指定目录，亦可通过命令行 --run-dir 覆盖）
DEFAULT_RUN_DIR = RESULT_ROOT / "run_20260511_035019_cnn_lstm_20260511_024025_fb0c7e72"

# SPX 数据源
SPX_DATA_ROOT      = Path(r"E:\phd\LLM_trading\CNN_trading\SPX_volume_price")
SPX_START_DATE     = "2011-04-01"
SPX_END_DATE       = "2025-12-31"
MONTHLY_WIDE_CACHE = ROOT / "data" / "spx_monthly_wide_v2.parquet"

# 测试期（SPX 全样本覆盖 2011-04 ~ 2025-12；测试期取后段）
TEST_PERIOD = ("2021-01", "2025-12")

# 组合参数
N_PORTFOLIOS        = 10      # 十分位
PORTFOLIO_WEIGHTING = "equal"
TOP_K_LIST          = [5, 10, 20, 50]
NEWEY_WEST_LAGS     = 12

# ═══════════════════════════════════════════════════════════════════════════════


def _find_latest_crsp_run() -> Path:
    """自动找最新的 CRSP CNN-LSTM run 目录。"""
    candidates = sorted(RESULT_ROOT.glob("run_*_cnn_lstm_*"))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {RESULT_ROOT} 找到任何 cnn_lstm run 目录，"
            "请先运行 crsp_run.py 或通过 --run-dir 手动指定。"
        )
    return candidates[-1]


def load_spx_data(engine: PortfolioEngine) -> None:
    """加载本地 SPX 数据并挂到 engine 上。"""
    if not SPX_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"SPX 数据目录不存在: {SPX_DATA_ROOT}")

    stock_data = build_stock_data_from_spx_folder(
        SPX_DATA_ROOT,
        start_date=SPX_START_DATE,
        end_date=SPX_END_DATE,
        tickers=None,
        monthly_wide_parquet_cache=MONTHLY_WIDE_CACHE,
        refresh_monthly_cache=False,
        fill_missing_prices="forward",
    )
    attach_stock_data(engine, stock_data)
    logger.info(
        f"SPX 月度数据: {len(stock_data.tickers)} 标的, "
        f"收益表 {stock_data.returns.height} 行, "
        f"区间 {stock_data.date_range[0]} ~ {stock_data.date_range[1]}"
    )


def load_crsp_model(engine: PortfolioEngine, run_dir: Path) -> str:
    """从 run 目录加载已训练的 CRSP 模型，注入到 engine 并返回 model_id。"""
    model_base = run_dir / "model"
    if not model_base.exists():
        raise FileNotFoundError(f"model/ 子目录不存在: {model_base}")

    # model/ 下只有一个子目录，即 model_id 目录
    model_subdirs = [p for p in model_base.iterdir() if p.is_dir()]
    if not model_subdirs:
        raise FileNotFoundError(f"model/ 目录内无模型子目录: {model_base}")
    model_path = model_subdirs[0]
    model_id   = model_path.name

    registry = ModelRegistry(path=str(model_base))
    loaded_id, model_data = registry.load(str(model_path))

    engine._models[loaded_id] = model_data  # noqa: SLF001
    logger.info(f"已加载 CRSP 模型: {loaded_id}（来自 {model_path}）")
    return loaded_id


# ─────────────────────────────────────────────────────────────
# Top-K 绩效计算（与 crsp_topk_analysis.py 口径完全一致）
# ─────────────────────────────────────────────────────────────

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
        autocovar = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        var += 2 * w * autocovar
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
    cagr = float(np.prod(1 + r) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    cum  = np.cumprod(1 + r)
    rolling_max = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - rolling_max) / rolling_max))
    return dict(mean_return=mean_r, std_dev=std_r, sharpe_ratio=sharpe,
                t_statistic=float(tstat), annualized_return=cagr,
                max_drawdown=max_dd, n_months=len(r))


def compute_topk_portfolios(
    forecasts_wide: pl.DataFrame,
    returns_wide: pl.DataFrame,
    k_list: list[int],
) -> dict[int, dict]:
    """
    基于预测宽表和收益宽表构建 Top-K 等权组合。

    对齐方式：预测日期 t 得到的排名，直接使用同一 t 月实际收益
    （与 PortfolioConstructor 分位排序口径一致）。
    """
    # 转为 long 格式
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
        mean_r = m.mean_return * 100
        print(
            f"  {name:<14}  {mean_r:>8.3f}%  "
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
    run_dir: Path,
    engine: PortfolioEngine,
    model_id: str,
    performance: dict,
    forecasts,
    portfolios,
    topk_results: dict[int, dict],
    diagnostics: dict,
) -> Path:
    """保存十分位结果（通过 save_pipeline_results）和 Top-K 结果。"""
    out_run_dir = save_pipeline_results(
        engine=engine,
        model_id=model_id,
        performance=performance,
        forecasts=forecasts,
        portfolios=portfolios,
        result_dir=RESULT_ROOT,
        diagnostics=diagnostics,
        experiment_tag="spx_crsp_transfer_cnn_lstm",
    )

    # ── Top-K JSON + CSV ──────────────────────────────────────────────────────
    topk_dir = out_run_dir / "topk_portfolios"
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
    return out_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="SPX × CRSP 预训练模型回测")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="CRSP 训练 run 目录；默认使用 DEFAULT_RUN_DIR 或最新目录",
    )
    args = parser.parse_args()

    if args.run_dir:
        crsp_run_dir = Path(args.run_dir)
    elif DEFAULT_RUN_DIR.exists():
        crsp_run_dir = DEFAULT_RUN_DIR
    else:
        crsp_run_dir = _find_latest_crsp_run()

    print("\n" + "=" * 70)
    print("  SPX 数据 × CRSP 预训练 CNN-LSTM 回测")
    print("=" * 70)
    print(f"  CRSP run 目录 : {crsp_run_dir.resolve()}")
    print(f"  SPX 数据目录  : {SPX_DATA_ROOT}")
    print(f"  测试期        : {TEST_PERIOD[0]} → {TEST_PERIOD[1]}")
    print(f"  十分位组合    : {N_PORTFOLIOS}")
    print(f"  Top-K         : {TOP_K_LIST}")
    print("=" * 70 + "\n")

    # ── 读取 CRSP 诊断参数 ────────────────────────────────────────────────────
    diag_path = crsp_run_dir / "diagnostics.json"
    crsp_diag: dict = {}
    if diag_path.exists():
        crsp_diag = json.loads(diag_path.read_text(encoding="utf-8"))
        print(f"CRSP 模型信息: arch={crsp_diag.get('architecture')}, "
              f"优化期={crsp_diag.get('optimization_period')}")

    # ── 初始化 engine（注意：预测时不再训练，只用于特征提取和组合构建）───────
    config = CBMConfig(
        model=ModelConfig(device="cpu", batch_size=512),
        training=TrainingConfig(
            optimization_period=("2012-01", "2020-12"),   # 不实际训练，占位
            return_variable=ReturnVariable.RET_RANK_NORM,  # 特征目标与 CRSP 一致
        ),
        portfolio=PortfolioConfig(
            n_portfolios=N_PORTFOLIOS,
            weighting=PORTFOLIO_WEIGHTING,
        ),
        backtest=BacktestConfig(test_period=TEST_PERIOD),
    )
    engine = PortfolioEngine(config)

    # ── 步骤 1：加载 SPX 数据 ─────────────────────────────────────────────────
    print("步骤 1/5  加载 SPX 月度数据…")
    load_spx_data(engine)

    # ── 步骤 2：特征工程（CR1…CR12 累积收益）────────────────────────────────
    print("\n步骤 2/5  特征工程（CR1…CR12）…")
    engine.prepare_features()
    fs = engine.features
    print(
        f"  SPX 特征集: {len(fs):,} 样本, "
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

    # ── 步骤 4：样本外预测（SPX 测试期）──────────────────────────────────────
    print(f"\n步骤 4/5  样本外预测 ({TEST_PERIOD[0]} → {TEST_PERIOD[1]})…")
    forecasts = engine.forecast(model_id=model_id, test_period=TEST_PERIOD)
    print(f"  预测覆盖月数 : {forecasts.values.height}")
    print(f"  预测标的数   : {forecasts.values.width - 1}")

    # ── 步骤 5a：十分位组合构建与绩效分析 ───────────────────────────────────
    print(f"\n步骤 5/5  构建十分位组合（{N_PORTFOLIOS} 分位）并分析绩效…")
    portfolios = engine.construct_portfolios(
        forecasts=forecasts,
        n_portfolios=N_PORTFOLIOS,
        weighting=PORTFOLIO_WEIGHTING,
    )
    performance = engine.analyze_performance(portfolios)
    print_decile_table(performance)

    # ── 步骤 5b：Top-K 组合分析 ───────────────────────────────────────────────
    print(f"\n构建 Top-K 等权组合（K = {TOP_K_LIST}）…")
    topk_results = compute_topk_portfolios(
        forecasts_wide=forecasts.values,
        returns_wide=engine.data.returns,
        k_list=TOP_K_LIST,
    )
    print_topk_table(topk_results, TEST_PERIOD[0], TEST_PERIOD[1])

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    diagnostics = {
        "data_source": "spx_local_wind_csv",
        "spx_root": str(SPX_DATA_ROOT.resolve()),
        "spx_start_date": SPX_START_DATE,
        "spx_end_date": SPX_END_DATE,
        "test_period": list(TEST_PERIOD),
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
        run_dir=crsp_run_dir,
        engine=engine,
        model_id=model_id,
        performance=performance,
        forecasts=forecasts,
        portfolios=portfolios,
        topk_results=topk_results,
        diagnostics=diagnostics,
    )

    print(f"\n所有结果已保存至: {out_dir.resolve()}")
    print("  performance.json          — 十分位绩效指标")
    print("  forecasts.parquet         — SPX 预测宽表")
    print("  portfolio_*_returns.parquet — 各组合月度收益")
    print("  topk_portfolios/           — Top-K 绩效与月度收益 CSV")
    print("  diagnostics.json          — 完整参数记录")


if __name__ == "__main__":
    main()
