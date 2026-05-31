"""CRSP 论文设定训练：超额收益目标 + 拟合月训练 + 6 段扩展窗口，保存各窗口模型权重。

与 ``crsp_run.py`` 的区别：
  - 训练目标基于超额收益（Ken French RF）
  - 仅在拟合月上训练（偶年偶月 + 奇年奇月）
  - 6 段扩展窗口各训一套 ensemble，合并样本外预测
  - 每窗口模型保存至 ``result/run_*/models/window_YYYYMM/``，供迁移验证加载

用法::

    python crsp_paper_train.py
    python crsp_paper_train.py --n-ensemble 30
    python crsp_paper_train.py --device cpu          # 显存不足时强制 CPU
    python crsp_paper_train.py --windows 0,1
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date as dt_date, datetime
from pathlib import Path

for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

from loguru import logger

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import ModelConfig, TrainingConfig
from cbm.core.types import Architecture, LossFunction, ReturnVariable, WeightingScheme
from cbm.data import FeatureEngineer
from cbm.data.crsp_sample_filter import (
    apply_paper_sample_filter_to_features,
    panel_load_start,
)
from cbm.data.french_rf import align_rf_to_return_dates, load_french_rf_monthly
from cbm.ml import Forecaster, ModelRegistry
from cbm.ml.forecaster import merge_forecast_wides
from cbm.ml.models.pytorch_impl import get_device
from cbm.ml.paper_training import (
    EXPANDING_WINDOWS,
    PaperModelTrainer,
    window_model_dir_name,
)

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result"

START_DATE = "1927-01-01"
END_DATE = "2022-12-31"
UNIVERSE = "crsp_all"
ARCHITECTURE = Architecture.CNN_LSTM
LOSS_FUNCTION = LossFunction.MSE
SAMPLE_WEIGHTING = WeightingScheme.EWPM
RETURN_VARIABLE = ReturnVariable.RET_RANK_NORM
VAL_SPLIT_RANDOM = True
VALIDATION_SIZE = 0.30
N_ENSEMBLE_DEFAULT = 5
RANDOM_SEED = 42
DEVICE = "auto"   # 有 CUDA 则用 GPU；框架已 mini-batch 训练/分块预测
BATCH_SIZE = 256


def _check_parquet() -> None:
    parquet = ROOT / "crsp_data" / "crsp2525_monthly.parquet"
    if not parquet.exists():
        csv = ROOT / "crsp_data" / "crsp2525(monthly).csv"
        if not csv.exists():
            raise FileNotFoundError(f"未找到 CRSP 数据: {parquet}")


def _prepare_excess_features(engine: PortfolioEngine) -> None:
    rf = load_french_rf_monthly()
    rf_aligned = align_rf_to_return_dates(engine.data.returns, rf)
    engineer = FeatureEngineer(return_variable=RETURN_VARIABLE)
    raw = engineer.create_features(
        engine.data,
        risk_free_rate=rf_aligned,
    )
    data_start = panel_load_start(dt_date(1927, 1, 1))
    data_end = dt_date(2022, 12, 31)
    engine._features = apply_paper_sample_filter_to_features(  # noqa: SLF001
        raw, data_start, data_end,
    )
    logger.info(
        f"超额收益特征（Section 2.1 筛选后）: {len(engine._features):,} 样本"  # noqa: SLF001
    )


def _parse_window_indices(spec: str) -> list[int]:
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSP 论文设定扩展窗口训练")
    parser.add_argument("--n-ensemble", type=int, default=N_ENSEMBLE_DEFAULT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--windows", default=None, help="窗口索引逗号分隔，默认 0-5")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    window_indices = (
        list(range(len(EXPANDING_WINDOWS)))
        if args.windows is None
        else _parse_window_indices(args.windows)
    )

    _check_parquet()

    config = CBMConfig(
        model=ModelConfig(device=args.device, batch_size=args.batch_size),
        training=TrainingConfig(
            loss_function=LOSS_FUNCTION,
            weighting=SAMPLE_WEIGHTING,
            return_variable=RETURN_VARIABLE,
            val_split_random=VAL_SPLIT_RANDOM,
            validation_size=VALIDATION_SIZE,
            n_ensemble=args.n_ensemble,
            random_seed=RANDOM_SEED,
        ),
    )

    run_tag = f"crsp_paper_{UNIVERSE}_{ARCHITECTURE.value}_excess"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.result_dir / f"run_{stamp}_{run_tag}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  CRSP 论文设定训练  —  超额收益 + 拟合月 + 扩展窗口")
    print("=" * 64)
    print(f"  数据范围     : {START_DATE} → {END_DATE}")
    print(f"  目标变量     : {RETURN_VARIABLE.value}（基于超额收益）")
    print(f"  架构/损失/权重: {ARCHITECTURE.value} / {LOSS_FUNCTION.value} / {SAMPLE_WEIGHTING.value}")
    print(f"  CR 预处理   : 无（论文：不对 CR1–12 标准化）")
    print(f"  训练样本    : Section 2.1（12月收益 + t−1 市值）")
    print(f"  batch_size  : {args.batch_size}（论文 256）")
    print(f"  拟合月训练   : 是（偶年偶月 + 奇年奇月）")
    print(f"  集成次数     : {args.n_ensemble}")
    print(f"  计算设备     : {get_device(args.device)}  (配置: {args.device})")
    print(f"  扩展窗口     : {len(window_indices)} 个")
    print(f"  输出目录     : {run_dir}")
    print("=" * 64 + "\n")

    engine = PortfolioEngine(config)

    print("步骤 1/3  加载 CRSP 数据…")
    engine.load_data(
        source="crsp_local",
        universe=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("步骤 2/3  特征工程（Ken French RF 超额收益）…")
    _prepare_excess_features(engine)
    features = engine.features

    trainer = PaperModelTrainer(
        architecture=ARCHITECTURE,
        model_config=config.model,
        training_config=config.training,
    )
    registry = ModelRegistry(path=str(models_dir))

    window_records: list[dict] = []
    forecast_wides = []
    t_all = datetime.now()

    print(f"\n步骤 3/3  扩展窗口训练（{len(window_indices)} 段）…")
    for idx in window_indices:
        win = EXPANDING_WINDOWS[idx]
        train_end = win["train_end"]
        forecast_start = win["forecast_start"]
        forecast_end = win["forecast_end"]
        opt_period = ("1927-01", train_end)

        print(f"\n── 窗口 {idx}: 训练 1927-01 → {train_end}，预测 {forecast_start} → {forecast_end}")
        t0 = datetime.now()

        ensemble, metrics = trainer.train(
            features=features,
            optimization_period=opt_period,
            n_ensemble=args.n_ensemble,
            fitting_months_only=True,
        )

        dir_name = window_model_dir_name(train_end)
        model_id = dir_name
        model_data = {
            "model": ensemble,
            "metrics": metrics,
            "config": {
                "architecture": ARCHITECTURE.value,
                "optimization_period": opt_period,
                "train_end": train_end,
                "forecast_period": [forecast_start, forecast_end],
                "loss_function": LOSS_FUNCTION.value,
                "weighting": SAMPLE_WEIGHTING.value,
                "return_variable": RETURN_VARIABLE.value,
                "excess_returns": True,
                "fitting_months_only": True,
                "paper_sample_filter": True,
                "feature_preprocessing": "none",
                "batch_size": args.batch_size,
                "n_ensemble": args.n_ensemble,
            },
        }
        save_path = registry.save(model_id, model_data, metrics=metrics)

        fc = Forecaster(model=ensemble).predict(
            features=features,
            test_period=(forecast_start, forecast_end),
        )
        forecast_wides.append(fc.values)

        elapsed = (datetime.now() - t0).total_seconds()
        nf_corr = metrics.get("non_fitting_spearman_corr")
        nf_str = f"{nf_corr:.4f}" if nf_corr is not None else "n/a"
        print(
            f"  完成 ({elapsed:.0f}s) | val Spearman={metrics['val_spearman_corr']:.4f} | "
            f"non-fit Spearman={nf_str} | 预测月数={fc.values.height}"
        )

        window_records.append({
            "index": idx,
            "train_start": "1927-01",
            "train_end": train_end,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "model_id": model_id,
            "model_dir": dir_name,
            "model_path": save_path,
            "metrics": metrics,
            "n_forecast_months": fc.values.height,
        })

    combined_wide = merge_forecast_wides(forecast_wides)
    combined_wide.write_parquet(run_dir / "forecasts.parquet")

    total_elapsed = (datetime.now() - t_all).total_seconds()
    meta = {
        "experiment": "crsp_paper_expanding_window",
        "created_at": datetime.now().isoformat(),
        "data_source": "crsp_local",
        "universe": UNIVERSE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "excess_returns": True,
        "rf_source": "ken_french_monthly",
        "fitting_months_only": True,
        "paper_sample_filter": True,
        "feature_preprocessing": "none",
        "batch_size": args.batch_size,
        "architecture": ARCHITECTURE.value,
        "loss_function": LOSS_FUNCTION.value,
        "weighting": SAMPLE_WEIGHTING.value,
        "return_variable": RETURN_VARIABLE.value,
        "n_ensemble": args.n_ensemble,
        "validation_size": VALIDATION_SIZE,
        "val_split_random": VAL_SPLIT_RANDOM,
        "total_elapsed_seconds": total_elapsed,
        "n_forecast_months": combined_wide.height,
        "forecast_range": [
            str(combined_wide["date"].min()),
            str(combined_wide["date"].max()),
        ],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "windows.json").write_text(
        json.dumps({"windows": window_records}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print(f"  训练完成，总耗时 {total_elapsed/3600:.2f} h")
    print(f"  合并预测: {combined_wide.height} 月")
    print(f"  模型目录: {models_dir}")
    print(f"  预测文件: {run_dir / 'forecasts.parquet'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
