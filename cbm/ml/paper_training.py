"""论文设定下的 ML 训练：拟合月划分、非拟合月评估、扩展窗口常量。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

from cbm.core.config import ModelConfig, TrainingConfig
from cbm.core.types import Architecture, FeatureSet
from cbm.ml.preprocessing import DataPreprocessor, compute_sample_weights
from cbm.ml.trainer import MODEL_REGISTRY, EnsembleModel, _period_str_to_end_date


# 论文 Section 3.2：6 段扩展窗口（起点均为 1927-01）
EXPANDING_WINDOWS: List[Dict[str, str]] = [
    {"train_end": "1963-06", "forecast_start": "1963-07", "forecast_end": "1974-12"},
    {"train_end": "1974-12", "forecast_start": "1975-01", "forecast_end": "1984-12"},
    {"train_end": "1984-12", "forecast_start": "1985-01", "forecast_end": "1994-12"},
    {"train_end": "1994-12", "forecast_start": "1995-01", "forecast_end": "2004-12"},
    {"train_end": "2004-12", "forecast_start": "2005-01", "forecast_end": "2014-12"},
    {"train_end": "2014-12", "forecast_start": "2015-01", "forecast_end": "2022-12"},
]


def is_fitting_month(year: int, month: int) -> bool:
    """偶年偶月 + 奇年奇月 为拟合月（论文 Section 3.1）。"""
    return (year % 2 == 0 and month % 2 == 0) or (year % 2 == 1 and month % 2 == 1)


def fitting_month_mask(dates: np.ndarray) -> np.ndarray:
    """对 numpy 日期数组返回拟合月布尔掩码。"""
    mask = np.empty(len(dates), dtype=bool)
    for i, d in enumerate(dates):
        ts = pd.Timestamp(d)
        mask[i] = is_fitting_month(ts.year, ts.month)
    return mask


def monthly_mean_spearman(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    min_stocks: int = 10,
) -> float:
    """时间序列平均的月度截面 Spearman 相关（论文评估指标）。"""
    corrs: list[float] = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < min_stocks:
            continue
        rho, _ = spearmanr(predictions[m], targets[m])
        if rho is not None and not np.isnan(rho):
            corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else float("nan")


class PaperModelTrainer:
    """
    论文训练流程：仅在拟合月上训练，随机 70/30 验证，ensemble 平均。

    与 ``ModelTrainer`` 的区别：
    - 训练样本限制为优化期内的**拟合月**
    - CR1–CR12 **不做** StandardScaler（论文保留幅度信息）
    - 额外报告优化期内**非拟合月**上 forecast vs **超额收益** 的 Spearman
    """

    def __init__(
        self,
        architecture: Architecture,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        self.architecture = architecture
        self.model_config = model_config
        self.training_config = training_config
        if architecture not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.model_class = MODEL_REGISTRY[architecture]

    def train(
        self,
        features: FeatureSet,
        optimization_period: Tuple[str, str],
        n_ensemble: int = 30,
        fitting_months_only: bool = True,
    ) -> Tuple[EnsembleModel, Dict[str, Any]]:
        start_date = np.datetime64(optimization_period[0] + "-01", "D")
        end_date = _period_str_to_end_date(optimization_period[1])

        period_mask = (features.dates >= start_date) & (features.dates <= end_date)
        if fitting_months_only:
            fit_mask = fitting_month_mask(features.dates)
            train_pool_mask = period_mask & fit_mask
            non_fit_mask = period_mask & ~fit_mask
        else:
            train_pool_mask = period_mask
            non_fit_mask = np.zeros(len(features), dtype=bool)

        X = features.features[train_pool_mask]
        y = features.targets[train_pool_mask]
        dates = features.dates[train_pool_mask]
        market_caps = (
            features.market_caps[train_pool_mask]
            if features.market_caps is not None
            else None
        )

        if len(X) == 0:
            raise ValueError(
                f"拟合月训练样本为空：period={optimization_period}, "
                f"fitting_months_only={fitting_months_only}"
            )

        logger.info(
            f"Paper training {self.architecture.value}: "
            f"{len(X):,} fitting-month samples in {optimization_period}"
        )

        weights = compute_sample_weights(
            dates=dates,
            market_caps=market_caps,
            scheme=self.training_config.weighting.value,
        )

        import torch

        preprocessor = DataPreprocessor(method="none")
        X_scaled = preprocessor.fit_transform(X)

        n_total = len(X_scaled)
        n_val = int(n_total * self.training_config.validation_size)
        n_train = n_total - n_val
        use_random_split = self.training_config.val_split_random
        split_label = "random" if use_random_split else "time-series"

        rng_report = np.random.default_rng(self.training_config.random_seed)
        report_idx = (
            rng_report.permutation(n_total)
            if use_random_split
            else np.arange(n_total)
        )
        report_train_idx = report_idx[n_val:]
        report_val_idx = report_idx[:n_val]

        models = []
        for i in tqdm(range(n_ensemble), desc="Training ensemble"):
            seed_i = self.training_config.random_seed + i
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            model = self.model_class(device=self.model_config.device)
            model.build(
                input_size=self.model_config.input_size,
                cnn_filters=self.model_config.cnn_filters,
                cnn_kernel_size=self.model_config.cnn_kernel_size,
                lstm_hidden_size=self.model_config.lstm_hidden_size,
                lstm_num_layers=self.model_config.lstm_num_layers,
                dropout=self.model_config.dropout,
            )

            if use_random_split:
                idx_i = np.random.permutation(n_total)
                val_idx = idx_i[:n_val]
                train_idx = idx_i[n_val:]
            else:
                train_idx = np.arange(n_train)
                val_idx = np.arange(n_train, n_total)

            w_train_i = weights[train_idx]
            w_train_i = w_train_i * (len(w_train_i) / w_train_i.sum())

            model.fit(
                X=X_scaled[train_idx],
                y=y[train_idx],
                weights=w_train_i,
                validation_data=(X_scaled[val_idx], y[val_idx]),
                epochs=self.model_config.epochs,
                batch_size=self.model_config.batch_size,
                learning_rate=self.model_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                early_stopping_patience=self.model_config.early_stopping_patience,
                loss_function=self.training_config.loss_function.value,
                grad_clip_norm=self.model_config.grad_clip_norm,
                verbose=False,
            )
            models.append(model)

        ensemble = EnsembleModel(models=models, preprocessor=preprocessor)

        train_pred = ensemble.predict(X[report_train_idx])
        val_pred = ensemble.predict(X[report_val_idx])
        train_corr, _ = spearmanr(train_pred, y[report_train_idx])
        val_corr, _ = spearmanr(val_pred, y[report_val_idx])

        metrics: Dict[str, Any] = {
            "train_spearman_corr": float(train_corr),
            "val_spearman_corr": float(val_corr),
            "train_mse": float(np.mean((train_pred - y[report_train_idx]) ** 2)),
            "val_mse": float(np.mean((val_pred - y[report_val_idx]) ** 2)),
            "n_samples": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "val_split": split_label,
            "fitting_months_only": fitting_months_only,
            "feature_preprocessing": "none",
            "optimization_period": list(optimization_period),
        }

        if non_fit_mask.any():
            X_nf = features.features[non_fit_mask]
            y_nf = features.targets[non_fit_mask]
            d_nf = features.dates[non_fit_mask]
            nf_pred = ensemble.predict(X_nf)
            excess_nf = (
                features.excess_returns[non_fit_mask]
                if features.excess_returns is not None
                else y_nf
            )
            metrics["non_fitting_spearman_corr"] = monthly_mean_spearman(
                nf_pred, excess_nf, d_nf
            )
            metrics["non_fitting_spearman_target"] = monthly_mean_spearman(
                nf_pred, y_nf, d_nf
            )
            metrics["n_non_fitting_samples"] = int(non_fit_mask.sum())
            logger.info(
                f"Non-fitting month Spearman (vs excess): "
                f"{metrics['non_fitting_spearman_corr']:.4f}"
            )

        logger.info(f"Val Spearman (fitting months): {val_corr:.4f}")
        return ensemble, metrics


def window_model_dir_name(train_end: str) -> str:
    """将 '1963-06' 转为模型目录名 'window_196306'。"""
    y, m = train_end.split("-")
    return f"window_{y}{m}"
