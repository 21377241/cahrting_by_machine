"""Core type definitions for the charting-by-machines package."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


class Architecture(str, Enum):
    """Supported neural network architectures."""
    FNN = "fnn"
    CNN = "cnn"
    LSTM = "lstm"
    CNN_LSTM = "cnn_lstm"


class LossFunction(str, Enum):
    """Supported loss functions."""
    MSE = "mse"
    MAE = "mae"


class WeightingScheme(str, Enum):
    """Weighting schemes for training observations."""
    EW = "ew"          # Equal-weighted
    EWPM = "ewpm"      # Equal-weighted per month
    EWPMVW = "ewpmvw"  # Equal-weighted per month, value-weighted


class ReturnVariable(str, Enum):
    """Target return variable transformations.

    RET           : raw excess return (no transformation)
    RET_STD       : cross-sectional standardization by std only (mean not removed)
    RET_NORM      : cross-sectional z-score  (default)
                    y = (r - mean) / std
    RET_RANK_NORM : cross-sectional inverse-normal rank transform  (paper method)
                    y = Phi^{-1}[ rank_i / (N + 1) ]
                    Rank-based; robust to outliers; bounded output.
    RET_PCTL      : cross-sectional percentile rank in (0, 1]
    """
    RET           = "ret"
    RET_STD       = "ret_std"
    RET_NORM      = "ret_norm"       # z-score (default)
    RET_RANK_NORM = "ret_rank_norm"  # inverse-normal rank (paper)
    RET_PCTL      = "ret_pctl"


class PortfolioMethod(str, Enum):
    """Portfolio construction methods."""
    UNIVARIATE = "univariate"
    BIVARIATE = "bivariate"
    TRIVARIATE = "trivariate"


class DataSource(str, Enum):
    """Supported data sources."""
    YAHOO = "yahoo"
    WRDS = "wrds"
    LOCAL = "local"


@dataclass
class DateRange:
    """Represents a date range for time periods."""
    start: str  # Format: YYYY-MM
    end: str    # Format: YYYY-MM
    
    def to_tuple(self) -> Tuple[str, str]:
        return (self.start, self.end)
    
    @classmethod
    def from_tuple(cls, t: Tuple[str, str]) -> "DateRange":
        return cls(start=t[0], end=t[1])


@dataclass
class StockData:
    """Container for stock market data using Polars DataFrames."""
    prices: pl.DataFrame           # OHLCV data with date column and ticker columns
    returns: pl.DataFrame          # Return data with date column and ticker columns
    market_cap: Optional[pl.DataFrame] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def tickers(self) -> List[str]:
        # Exclude the date column
        return [c for c in self.returns.columns if c != "date"]
    
    @property
    def date_range(self) -> Tuple[date, date]:
        dates = self.returns.get_column("date")
        return dates.min(), dates.max()
    
    def get_returns_for_ticker(self, ticker: str) -> pl.Series:
        """Get returns for a specific ticker."""
        return self.returns.get_column(ticker)
    
    def get_prices_for_ticker(self, ticker: str) -> pl.Series:
        """Get prices for a specific ticker."""
        return self.prices.get_column(ticker)


@dataclass
class FeatureSet:
    """Container for ML features (cumulative returns)."""
    features: np.ndarray           # Shape: (n_samples, 12) - 12 cumulative returns
    targets: np.ndarray            # Shape: (n_samples,) - next month return
    dates: np.ndarray              # Shape: (n_samples,) - date indices
    tickers: np.ndarray            # Shape: (n_samples,) - ticker indices
    market_caps: Optional[np.ndarray] = None  # Shape: (n_samples,)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def to_dict(self) -> Dict:
        return {
            "features": self.features,
            "targets": self.targets,
            "dates": self.dates,
            "tickers": self.tickers,
            "market_caps": self.market_caps,
        }


@dataclass
class Forecast:
    """Container for ML-based return forecasts using Polars."""
    values: pl.DataFrame           # Date column + ticker columns
    model_id: str
    created_at: str
    metadata: Dict = field(default_factory=dict)
    
    def get_forecast(self, date_val: str, ticker: str) -> float:
        """Get forecast for specific date and ticker."""
        row = self.values.filter(pl.col("date") == date_val)
        if row.height == 0:
            return float('nan')
        return row.get_column(ticker)[0]
    
    def get_date_forecasts(self, date_val: str) -> Dict[str, float]:
        """Get all forecasts for a specific date."""
        row = self.values.filter(pl.col("date") == date_val)
        if row.height == 0:
            return {}
        result = {}
        for col in row.columns:
            if col != "date":
                result[col] = row.get_column(col)[0]
        return result


@dataclass
class Portfolio:
    """Container for a single portfolio."""
    name: str
    holdings: pl.DataFrame         # Date column + ticker columns with weights
    returns: pl.DataFrame          # Date column + return column
    metadata: Dict = field(default_factory=dict)
    
    def get_returns_series(self) -> pl.Series:
        """Get returns as a Polars Series."""
        return self.returns.get_column("return")


@dataclass 
class PortfolioSet:
    """Container for a set of portfolios (e.g., decile portfolios)."""
    portfolios: Dict[str, Portfolio]
    long_short: Optional[Portfolio] = None
    
    def __getitem__(self, key: str) -> Portfolio:
        return self.portfolios[key]
    
    def keys(self) -> List[str]:
        return list(self.portfolios.keys())


@dataclass
class PerformanceMetrics:
    """Performance metrics for a portfolio."""
    mean_return: float             # Average monthly return
    std_dev: float                 # Standard deviation
    sharpe_ratio: float            # Annualized Sharpe ratio
    t_statistic: float             # Newey-West t-stat
    annualized_return: float       # CAGR
    max_drawdown: float            # Maximum drawdown
    alpha: Optional[Dict[str, float]] = None  # Factor model alphas
    factor_loadings: Optional[Dict[str, Dict[str, float]]] = None
    
    def summary(self) -> str:
        lines = [
            f"Mean Monthly Return: {self.mean_return:.2%}",
            f"Standard Deviation: {self.std_dev:.2%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"t-statistic: {self.t_statistic:.2f}",
            f"Annualized Return: {self.annualized_return:.2%}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
        ]
        if self.alpha:
            lines.append("Alphas:")
            for model, alpha in self.alpha.items():
                lines.append(f"  {model}: {alpha:.2%}")
        return "\n".join(lines)


@dataclass
class BacktestResult:
    """Results from a backtest."""
    portfolio_set: PortfolioSet
    performance: Dict[str, PerformanceMetrics]
    cumulative_returns: pl.DataFrame
    turnover: Optional[pl.DataFrame] = None
    transaction_costs: Optional[pl.DataFrame] = None
    
    def summary(self) -> pl.DataFrame:
        """Create summary DataFrame of all portfolio metrics."""
        data = []
        for name, metrics in self.performance.items():
            data.append({
                "portfolio": name,
                "mean_return": metrics.mean_return,
                "std_dev": metrics.std_dev,
                "sharpe": metrics.sharpe_ratio,
                "t_stat": metrics.t_statistic,
            })
        return pl.DataFrame(data)
