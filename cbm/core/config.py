"""Configuration classes for the charting-by-machines package."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from cbm.core.types import (
    Architecture,
    DataSource,
    LossFunction,
    PortfolioMethod,
    ReturnVariable,
    WeightingScheme,
)


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    
    source: DataSource = DataSource.YAHOO
    tickers: Optional[List[str]] = None
    universe: Optional[str] = None  # e.g., "sp500", "russell1000"
    start_date: str = "2010-01-01"
    end_date: str = "2023-12-31"
    cache_dir: str = "./data/cache"
    risk_free_rate: str = "^IRX"  # 13-week Treasury Bill
    
    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        import re
        if not re.match(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"Date must be in YYYY-MM-DD format: {v}")
        return v


class ModelConfig(BaseModel):
    """Configuration for ML model architecture and hyperparameters."""
    
    architecture: Architecture = Architecture.CNN_LSTM
    input_size: int = 12  # 12 cumulative returns
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    # CNN specific
    cnn_filters: int = 32
    cnn_kernel_size: int = 3
    
    # LSTM specific
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0  # max gradient norm; set 0 to disable clipping
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    optimization_period: Tuple[str, str] = ("2010-01", "2018-12")
    loss_function: LossFunction = LossFunction.MSE
    weighting: WeightingScheme = WeightingScheme.EWPM
    return_variable: ReturnVariable = ReturnVariable.RET_NORM
    
    # Cross-validation
    n_folds: int = 5
    validation_size: float = 0.2
    val_split_random: bool = True  # True: random split (paper); False: time-series split
    
    # Ensemble
    n_ensemble: int = 30  # Number of model fits to average
    random_seed: int = 42
    
    # Regularization
    weight_decay: float = 1e-5


class PortfolioConfig(BaseModel):
    """Configuration for portfolio construction."""
    
    method: PortfolioMethod = PortfolioMethod.UNIVARIATE
    n_portfolios: int = 10  # Number of quantile portfolios
    weighting: str = "value"  # "value" or "equal"
    rebalance_freq: str = "monthly"
    
    # Control variables for bivariate/trivariate sorts
    control_variables: List[str] = Field(default_factory=list)
    
    # Transaction costs
    transaction_costs: float = 0.001  # 10 bps
    
    # NYSE breakpoints
    use_nyse_breakpoints: bool = True


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    
    test_period: Tuple[str, str] = ("2019-01", "2023-12")
    benchmark: str = "market_cap_weighted"
    
    # Factor models for risk adjustment
    factor_models: List[str] = Field(
        default_factory=lambda: ["CAPM", "FF3", "FFC4", "FF5"]
    )
    
    # Newey-West lags for t-statistics
    newey_west_lags: int = 12


class TrackingConfig(BaseModel):
    """Configuration for experiment tracking."""
    
    mlflow_uri: Optional[str] = None
    experiment_name: str = "charting-by-machines"
    log_level: str = "INFO"
    save_models: bool = True
    model_dir: str = "./models"
    results_dir: str = Field(
        default="./result",
        description="训练与回测摘要、预测与组合收益等默认输出目录（相对当前工作目录）",
    )


class CBMConfig(BaseModel):
    """Main configuration class combining all sub-configs."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "CBMConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    @classmethod
    def paper_config(cls) -> "CBMConfig":
        """Configuration matching the original paper settings."""
        return cls(
            model=ModelConfig(
                architecture=Architecture.CNN_LSTM,
                cnn_filters=32,
                lstm_hidden_size=64,
            ),
            training=TrainingConfig(
                optimization_period=("1927-01", "1963-06"),
                loss_function=LossFunction.MSE,
                weighting=WeightingScheme.EWPM,
                return_variable=ReturnVariable.RET_RANK_NORM,  # paper uses Phi^{-1}[rank/(N+1)]
                n_ensemble=30,
            ),
            backtest=BacktestConfig(
                test_period=("1963-07", "2022-12"),
            ),
        )
