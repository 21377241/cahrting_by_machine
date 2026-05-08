"""CRSP 数据训练入口脚本

目标变量 : 逆正态秩变换  ret_rank_norm  (Φ⁻¹[rank/(N+1)])
训练/验证分割 : 随机分割 val_split_random=True
架构       : CNN-LSTM（论文最优）
数据源     : crsp_local（本地 crsp2525(monthly).csv 或其 Parquet 版本）

快速使用::

    cd E:\\phd\\LLM_trading\\charting_by_machine
    python crsp_run.py

如需全量论文复现（更长时间，更多集成）可修改下方 CONFIG 区域的参数。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Windows DLL 修复（torch 在跨盘目录下需要手动注册 DLL 搜索路径） ──────────
_dll_dirs: list = []
for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        _dll_dirs.append(os.add_dll_directory(_lib_path))

import polars as pl
from loguru import logger

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import BacktestConfig, ModelConfig, PortfolioConfig, TrainingConfig
from cbm.core.types import Architecture, LossFunction, ReturnVariable, WeightingScheme
from cbm.utils.results_io import save_pipeline_results

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — 按需修改
# ═══════════════════════════════════════════════════════════════════════════════

# ── 数据范围 ──────────────────────────────────────────────────────────────────
# 「快速验证」默认：1990-2022（33年），NYSE+AMEX+NASDAQ 全部普通股
# 「论文对齐」可改为 start_date="1927-01-01" / end_date="2022-12-31"
START_DATE = "1990-01-01"
END_DATE   = "2022-12-31"

# CRSP 宇宙：crsp_all / crsp_nyse / crsp_amex / crsp_nasdaq
UNIVERSE = "crsp_all"

# ── 时期划分（月，含两端） ─────────────────────────────────────────────────────
# 优化期（训练）
OPTIMIZATION_PERIOD = ("1990-01", "2010-12")
# 测试期（样本外预测）
TEST_PERIOD = ("2011-01", "2022-12")

# ── 目标变量（固定为逆正态秩变换，即论文最优设定） ─────────────────────────────
RETURN_VARIABLE = ReturnVariable.RET_RANK_NORM   # Φ⁻¹[rank/(N+1)]

# ── 训练参数 ──────────────────────────────────────────────────────────────────
ARCHITECTURE      = Architecture.CNN_LSTM          # 论文最优架构
LOSS_FUNCTION     = LossFunction.MSE
SAMPLE_WEIGHTING  = WeightingScheme.EWPM           # 每月内等权（论文最优）
VAL_SPLIT_RANDOM  = True                           # 随机分割（论文方法）
VALIDATION_SIZE   = 0.30                           # 30% 验证集
N_ENSEMBLE        = 5                              # 集成次数（论文=30；此处5次供快速测试）
RANDOM_SEED       = 42

# ── 设备与批次 ────────────────────────────────────────────────────────────────
# CRSP 全量样本约 130 万（训练集），LSTM 一次性前向传播会超出 GPU 显存。
# 强制使用 CPU，mini-batch 训练不受显存限制。
DEVICE     = "cpu"
BATCH_SIZE = 512   # CPU 下可适当加大

# ── 组合构建 ──────────────────────────────────────────────────────────────────
N_PORTFOLIOS        = 10        # 十分位组合
PORTFOLIO_WEIGHTING = "equal"   # 首次运行用等权（CRSP 市值已有，改 "value" 即可价值加权）

# ── 结果输出 ──────────────────────────────────────────────────────────────────
RESULT_DIR = Path(__file__).resolve().parent / "result"

# ═══════════════════════════════════════════════════════════════════════════════


def _check_parquet() -> None:
    """提示用户先运行 convert_to_parquet.py 以加快后续加载。"""
    parquet = Path(__file__).resolve().parent / "crsp_data" / "crsp2525_monthly.parquet"
    csv     = Path(__file__).resolve().parent / "crsp_data" / "crsp2525(monthly).csv"

    if not parquet.exists():
        if csv.exists():
            logger.warning(
                "Parquet 版本不存在，将使用原始 CSV（首次加载较慢，约 3-10 分钟）。\n"
                "建议提前运行：\n"
                "    python crsp_data/convert_to_parquet.py\n"
                "以后每次仅需 10-30 秒加载数据。"
            )
        else:
            raise FileNotFoundError(
                "未找到 CRSP 数据文件！\n"
                f"  期望 CSV    : {csv}\n"
                f"  期望 Parquet: {parquet}"
            )
    else:
        logger.info(f"检测到 Parquet 文件: {parquet}")


def _print_data_summary(engine: PortfolioEngine) -> None:
    r = engine.data.returns
    tickers = engine.data.tickers
    logger.info(
        f"数据概览: {len(tickers):,} 个证券 (PERMNO), "
        f"{r.height} 个月份, "
        f"区间 {r.get_column('date').min()} ~ {r.get_column('date').max()}"
    )
    # 打印前5行×前6列样例
    sample_cols = ["date"] + tickers[:6]
    print("\n收益宽表样例（前5月 × 前6只PERMNO）：")
    print(r.select(sample_cols).head(5))


def _print_feature_summary(engine: PortfolioEngine) -> None:
    fs = engine.features
    if fs is None:
        return
    print(
        f"\n特征集概览:\n"
        f"  样本总数   : {len(fs):,}\n"
        f"  特征维度   : {fs.features.shape[1]}  (CR1…CR12)\n"
        f"  目标变量   : {RETURN_VARIABLE.value}\n"
        f"  唯一月份数 : {len(set(fs.dates.tolist())):,}\n"
        f"  唯一证券数 : {len(set(fs.tickers.tolist())):,}"
    )


def main() -> None:
    _check_parquet()

    # ── 1. 构建配置 ────────────────────────────────────────────────────────────
    config = CBMConfig(
        model=ModelConfig(
            device=DEVICE,
            batch_size=BATCH_SIZE,
        ),
        training=TrainingConfig(
            optimization_period=OPTIMIZATION_PERIOD,
            loss_function=LOSS_FUNCTION,
            weighting=SAMPLE_WEIGHTING,
            return_variable=RETURN_VARIABLE,       # 逆正态秩变换
            val_split_random=VAL_SPLIT_RANDOM,     # 随机分割
            validation_size=VALIDATION_SIZE,
            n_ensemble=N_ENSEMBLE,
            random_seed=RANDOM_SEED,
        ),
        portfolio=PortfolioConfig(
            n_portfolios=N_PORTFOLIOS,
            weighting=PORTFOLIO_WEIGHTING,
        ),
        backtest=BacktestConfig(test_period=TEST_PERIOD),
    )

    print("\n" + "=" * 60)
    print("  CRSP 数据训练  —  逆正态秩变换 + 随机分割")
    print("=" * 60)
    print(f"  数据范围     : {START_DATE} → {END_DATE}")
    print(f"  宇宙         : {UNIVERSE}")
    print(f"  优化期       : {OPTIMIZATION_PERIOD[0]} → {OPTIMIZATION_PERIOD[1]}")
    print(f"  测试期       : {TEST_PERIOD[0]} → {TEST_PERIOD[1]}")
    print(f"  目标变量     : {RETURN_VARIABLE.value}  (逆正态秩变换)")
    print(f"  验证分割     : 随机 {int(VALIDATION_SIZE*100)}%")
    print(f"  集成次数     : {N_ENSEMBLE}")
    print(f"  架构         : {ARCHITECTURE.value}")
    print(f"  计算设备     : {DEVICE}  (batch={BATCH_SIZE})")
    print("=" * 60 + "\n")

    engine = PortfolioEngine(config)

    # ── 2. 加载 CRSP 数据 ──────────────────────────────────────────────────────
    print("步骤 1/5  加载 CRSP 数据（首次较慢，后续命中缓存）…")
    engine.load_data(
        source="crsp_local",
        universe=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    _print_data_summary(engine)

    # ── 3. 特征工程 ────────────────────────────────────────────────────────────
    print("\n步骤 2/5  特征工程（CR1…CR12 累积收益 + 逆正态秩变换目标）…")
    engine.prepare_features()
    _print_feature_summary(engine)

    # ── 4. 训练模型 ────────────────────────────────────────────────────────────
    print(f"\n步骤 3/5  训练 {ARCHITECTURE.value.upper()} 集成（{N_ENSEMBLE} 个成员）…")
    t_start = datetime.now()
    model_id = engine.train_model(
        architecture=ARCHITECTURE.value,
        optimization_period=OPTIMIZATION_PERIOD,
    )
    elapsed = (datetime.now() - t_start).total_seconds()

    train_metrics = engine._models[model_id]["metrics"]  # noqa: SLF001
    print(f"\n  模型 ID       : {model_id}")
    print(f"  训练耗时      : {elapsed:.1f}s")
    print(f"  训练 Spearman : {train_metrics['train_spearman_corr']:.4f}")
    print(f"  验证 Spearman : {train_metrics['val_spearman_corr']:.4f}")
    print(f"  验证 MSE      : {train_metrics['val_mse']:.6f}")
    print(f"  验证分割方式  : {train_metrics['val_split']}")

    # ── 5. 样本外预测 ──────────────────────────────────────────────────────────
    print(f"\n步骤 4/5  样本外预测 ({TEST_PERIOD[0]} → {TEST_PERIOD[1]})…")
    forecasts = engine.forecast(model_id=model_id, test_period=TEST_PERIOD)
    print(f"  预测覆盖月数  : {forecasts.values.height}")

    # ── 6. 组合构建与绩效分析 ──────────────────────────────────────────────────
    print(f"\n步骤 5/5  构建 {N_PORTFOLIOS} 个分位组合并分析绩效…")
    portfolios = engine.construct_portfolios(
        forecasts=forecasts,
        n_portfolios=N_PORTFOLIOS,
        weighting=PORTFOLIO_WEIGHTING,
    )
    performance = engine.analyze_performance(portfolios)

    print("\n" + "─" * 50)
    print(f"  {'组合':<12}  {'月均超额收益':>10}  {'年化Sharpe':>10}  {'t统计量':>8}")
    print("─" * 50)
    for name, m in sorted(performance.items(), key=lambda x: x[0]):
        mean_r  = m.mean_return * 100   # 转为 %
        sharpe  = m.sharpe_ratio
        tstat   = m.t_statistic
        print(f"  {name:<12}  {mean_r:>9.2f}%  {sharpe:>10.3f}  {tstat:>8.2f}")

    ls = performance.get("long_short")
    if ls:
        print("─" * 50)
        print(
            f"\n  多空组合 (P10–P1):\n"
            f"    月均超额收益: {ls.mean_return*100:.3f}%\n"
            f"    年化 Sharpe : {ls.sharpe_ratio:.3f}\n"
            f"    t 统计量    : {ls.t_statistic:.2f}\n"
            f"    年化收益    : {ls.annualized_return*100:.2f}%\n"
            f"    最大回撤    : {ls.max_drawdown*100:.2f}%"
        )

    # ── 7. 保存结果 ────────────────────────────────────────────────────────────
    diagnostics = {
        "data_source": "crsp_local",
        "universe": UNIVERSE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "optimization_period": list(OPTIMIZATION_PERIOD),
        "test_period": list(TEST_PERIOD),
        "return_variable": RETURN_VARIABLE.value,
        "val_split_random": VAL_SPLIT_RANDOM,
        "validation_size": VALIDATION_SIZE,
        "n_ensemble": N_ENSEMBLE,
        "architecture": ARCHITECTURE.value,
        "loss_function": LOSS_FUNCTION.value,
        "sample_weighting": SAMPLE_WEIGHTING.value,
        "n_portfolios": N_PORTFOLIOS,
        "portfolio_weighting": PORTFOLIO_WEIGHTING,
        "training_metrics": {
            k: float(v) if hasattr(v, "item") else v
            for k, v in train_metrics.items()
        },
        "crsp_n_securities": len(engine.data.tickers),
        "crsp_n_months": engine.data.returns.height,
    }

    run_dir = save_pipeline_results(
        engine=engine,
        model_id=model_id,
        performance=performance,
        forecasts=forecasts,
        portfolios=portfolios,
        result_dir=RESULT_DIR,
        diagnostics=diagnostics,
        experiment_tag=f"crsp_{UNIVERSE}_{ARCHITECTURE.value}_{RETURN_VARIABLE.value}",
    )

    print(f"\n结果已保存至: {run_dir.resolve()}")
    print("  performance.json  — 绩效指标")
    print("  forecasts.parquet — 预测宽表")
    print("  portfolios/       — 各组合收益序列")
    print("  diagnostics.json  — 完整参数记录")


if __name__ == "__main__":
    main()
