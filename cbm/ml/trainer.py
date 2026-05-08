"""Model training orchestration."""

import calendar
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from loguru import logger
from tqdm import tqdm

from cbm.core.config import ModelConfig, TrainingConfig
from cbm.core.types import Architecture, FeatureSet, LossFunction, WeightingScheme
from cbm.ml.models.base import BaseNeuralNetworkModel
from cbm.ml.models.pytorch_impl import (
    PyTorchCNN,
    PyTorchCNNLSTM,
    PyTorchFNN,
    PyTorchLSTM,
)
from cbm.ml.preprocessing import DataPreprocessor, compute_sample_weights


def _period_str_to_end_date(period: str) -> np.datetime64:
    """
    将 'YYYY-MM' 格式的月份字符串转为该月最后一天的 numpy datetime64[D]。

    直接用 ``np.datetime64("2020-12")`` 比较日精度日期时等价于 2020-12-01，
    会错误排除 12-02 至 12-31 的所有行。此函数转为 2020-12-31 后比较正确。
    """
    y, m = map(int, period.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    return np.datetime64(f"{y:04d}-{m:02d}-{last_day:02d}", "D")


# Model registry mapping architecture to implementation
MODEL_REGISTRY: Dict[Architecture, Type[BaseNeuralNetworkModel]] = {
    Architecture.FNN: PyTorchFNN,
    Architecture.CNN: PyTorchCNN,
    Architecture.LSTM: PyTorchLSTM,
    Architecture.CNN_LSTM: PyTorchCNNLSTM,
}


class EnsembleModel:
    """
    Ensemble of neural network models.
    
    Following the paper, forecasts are averaged across multiple model fits
    (default: 30) to reduce variance and improve robustness.
    
    Parameters
    ----------
    models : list
        List of fitted BaseNeuralNetworkModel instances.
    preprocessor : DataPreprocessor
        Fitted data preprocessor.
    """
    
    def __init__(
        self,
        models: List[BaseNeuralNetworkModel],
        preprocessor: DataPreprocessor,
    ):
        self.models = models
        self.preprocessor = preprocessor
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions by averaging.
        
        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Averaged predictions with shape (n_samples,).
        """
        X_scaled = self.preprocessor.transform(X)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Average across ensemble
        return np.mean(predictions, axis=0)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Returns
        -------
        tuple
            (mean predictions, standard deviation across ensemble)
        """
        X_scaled = self.preprocessor.transform(X)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


class ModelTrainer:
    """
    Orchestrates ML model training following paper methodology.
    
    Handles:
    - Time-series aware train/validation splitting
    - Sample weighting (EW, EWPM, EWPMVW)
    - Ensemble training
    - Hyperparameter configuration
    
    Parameters
    ----------
    architecture : Architecture
        Neural network architecture to use.
    model_config : ModelConfig
        Model hyperparameters.
    training_config : TrainingConfig
        Training parameters.
        
    Example
    -------
    >>> trainer = ModelTrainer(
    ...     architecture=Architecture.CNN_LSTM,
    ...     model_config=model_config,
    ...     training_config=training_config,
    ... )
    >>> model, metrics = trainer.train(features, optimization_period)
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
        
        # Get model class
        if architecture not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.model_class = MODEL_REGISTRY[architecture]
    
    def train(
        self,
        features: FeatureSet,
        optimization_period: Tuple[str, str],
        n_ensemble: int = 30,
    ) -> Tuple[EnsembleModel, Dict[str, Any]]:
        """
        Train an ensemble of models.
        
        Parameters
        ----------
        features : FeatureSet
            Feature set with features, targets, dates, and tickers.
        optimization_period : tuple
            Training period as (start_month, end_month).
        n_ensemble : int
            Number of models in ensemble.
            
        Returns
        -------
        tuple
            (EnsembleModel, training_metrics)
        """
        logger.info(f"Training {self.architecture.value} ensemble with {n_ensemble} models")
        
        # 将 "YYYY-MM" 转为该月末日（含整个末月），避免月精度 datetime64 比较截断
        start_date = np.datetime64(optimization_period[0] + "-01", "D")
        end_date = _period_str_to_end_date(optimization_period[1])

        mask = (features.dates >= start_date) & (features.dates <= end_date)
        
        X = features.features[mask]
        y = features.targets[mask]
        dates = features.dates[mask]
        market_caps = features.market_caps[mask] if features.market_caps is not None else None
        
        logger.info(f"Training on {len(X)} samples from {optimization_period}")
        
        # Compute sample weights
        weights = compute_sample_weights(
            dates=dates,
            market_caps=market_caps,
            scheme=self.training_config.weighting.value,
        )
        
        import torch

        # Create preprocessor — fit on full optimization-period data to keep scale consistent
        preprocessor = DataPreprocessor(method="standard")
        X_scaled = preprocessor.fit_transform(X)

        n_total = len(X_scaled)
        n_val   = int(n_total * self.training_config.validation_size)
        n_train = n_total - n_val

        use_random_split = self.training_config.val_split_random
        split_label = "random" if use_random_split else "time-series"
        logger.info(
            f"Val split: {split_label}, train={n_train}, val={n_val} "
            f"({self.training_config.validation_size:.0%})"
        )

        # Fixed reporting split (seed=random_seed) — used only to compute final metrics
        rng_report = np.random.default_rng(self.training_config.random_seed)
        if use_random_split:
            report_idx = rng_report.permutation(n_total)
        else:
            report_idx = np.arange(n_total)
        report_train_idx = report_idx[n_val:]
        report_val_idx   = report_idx[:n_val]

        # Train ensemble
        models = []
        all_histories = []

        for i in tqdm(range(n_ensemble), desc="Training ensemble"):
            # Set seed BEFORE build: controls weight init, making each member distinct
            seed_i = self.training_config.random_seed + i
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            # Build model (weight initialisation now seeded)
            model = self.model_class(device=self.model_config.device)
            model.build(
                input_size=self.model_config.input_size,
                cnn_filters=self.model_config.cnn_filters,
                cnn_kernel_size=self.model_config.cnn_kernel_size,
                lstm_hidden_size=self.model_config.lstm_hidden_size,
                lstm_num_layers=self.model_config.lstm_num_layers,
                dropout=self.model_config.dropout,
            )

            # Per-member train/val split
            if use_random_split:
                # Each member draws its own random 70/30 partition (paper method)
                idx_i     = np.random.permutation(n_total)
                val_idx   = idx_i[:n_val]
                train_idx = idx_i[n_val:]
            else:
                # Time-series split: last validation_size fraction is validation
                train_idx = np.arange(n_train)
                val_idx   = np.arange(n_train, n_total)

            X_train_i = X_scaled[train_idx]
            y_train_i = y[train_idx]
            X_val_i   = X_scaled[val_idx]
            y_val_i   = y[val_idx]

            w_train_i = weights[train_idx]
            # Normalize to mean=1 (not sum=1) so that the per-sample gradient
            # magnitude stays comparable to unweighted MSE.  With sum=1 over
            # ~920 k samples, each weight ≈ 1e-6; combined with .mean() in the
            # batch loss this pushes gradient norms below Adam's epsilon (1e-8),
            # causing complete training collapse.  mean=1 keeps relative EWPM
            # ratios intact while restoring a gradient scale of O(1/batch_size).
            w_train_i = w_train_i * (len(w_train_i) / w_train_i.sum())

            history = model.fit(
                X=X_train_i,
                y=y_train_i,
                weights=w_train_i,
                validation_data=(X_val_i, y_val_i),
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
            all_histories.append(history)

        # Create ensemble model
        ensemble = EnsembleModel(models=models, preprocessor=preprocessor)

        # Compute reporting metrics on the fixed split
        from scipy.stats import spearmanr

        train_pred = ensemble.predict(X[report_train_idx])
        val_pred   = ensemble.predict(X[report_val_idx])

        train_corr, _ = spearmanr(train_pred, y[report_train_idx])
        val_corr,   _ = spearmanr(val_pred,   y[report_val_idx])

        train_mse = np.mean((train_pred - y[report_train_idx]) ** 2)
        val_mse   = np.mean((val_pred   - y[report_val_idx])   ** 2)

        metrics = {
            "train_spearman_corr": train_corr,
            "val_spearman_corr":   val_corr,
            "train_mse": train_mse,
            "val_mse":   val_mse,
            "n_samples": n_total,
            "n_train":   n_train,
            "n_val":     n_val,
            "val_split": split_label,
        }

        logger.info(f"Training complete. Val Spearman correlation: {val_corr:.4f}")
        
        return ensemble, metrics
    
    def cross_validate(
        self,
        features: FeatureSet,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Uses expanding window approach suitable for financial time series.
        
        Parameters
        ----------
        features : FeatureSet
            Feature set.
        n_folds : int
            Number of folds.
            
        Returns
        -------
        dict
            Cross-validation results.
        """
        unique_dates = np.unique(features.dates)
        n_dates = len(unique_dates)
        fold_size = n_dates // n_folds
        
        results = []
        
        for fold in range(n_folds - 1):
            # Expanding window: train on all data up to fold
            train_end_idx = (fold + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + fold_size, n_dates)
            
            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Get training and validation masks
            train_mask = np.isin(features.dates, train_dates)
            val_mask = np.isin(features.dates, val_dates)
            
            # Train single model for CV
            model = self.model_class(device=self.model_config.device)
            model.build(input_size=self.model_config.input_size)
            
            preprocessor = DataPreprocessor()
            X_train = preprocessor.fit_transform(features.features[train_mask])
            X_val = preprocessor.transform(features.features[val_mask])
            
            model.fit(
                X=X_train,
                y=features.targets[train_mask],
                epochs=self.model_config.epochs,
                verbose=False,
            )
            
            val_pred = model.predict(X_val)
            
            from scipy.stats import spearmanr
            corr, _ = spearmanr(val_pred, features.targets[val_mask])
            
            results.append({
                "fold": fold,
                "train_size": train_mask.sum(),
                "val_size": val_mask.sum(),
                "spearman_corr": corr,
            })
        
        return {
            "folds": results,
            "mean_corr": np.mean([r["spearman_corr"] for r in results]),
            "std_corr": np.std([r["spearman_corr"] for r in results]),
        }
