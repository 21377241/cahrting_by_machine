"""Risk analysis and factor model utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class FactorModel:
    """
    Container for factor model specification.
    
    Attributes
    ----------
    name : str
        Model name (e.g., "CAPM", "FF3", "FF5").
    factors : list of str
        Factor names.
    description : str
        Model description.
    """
    name: str
    factors: List[str]
    description: str


# Standard factor models
FACTOR_MODELS = {
    "CAPM": FactorModel(
        name="CAPM",
        factors=["MKT-RF"],
        description="Capital Asset Pricing Model",
    ),
    "FF3": FactorModel(
        name="FF3",
        factors=["MKT-RF", "SMB", "HML"],
        description="Fama-French 3-Factor Model",
    ),
    "FFC4": FactorModel(
        name="FFC4",
        factors=["MKT-RF", "SMB", "HML", "UMD"],
        description="Carhart 4-Factor Model (with momentum)",
    ),
    "FFCREV": FactorModel(
        name="FFCREV",
        factors=["MKT-RF", "SMB", "HML", "UMD", "STR"],
        description="Carhart model with short-term reversal",
    ),
    "FF5": FactorModel(
        name="FF5",
        factors=["MKT-RF", "SMB", "HML", "RMW", "CMA"],
        description="Fama-French 5-Factor Model",
    ),
    "Q": FactorModel(
        name="Q",
        factors=["MKT-RF", "ME", "IA", "ROE"],
        description="Hou-Xue-Zhang Q-Factor Model",
    ),
}


class RiskAnalyzer:
    """
    Risk analysis utilities for portfolios.
    
    Computes various risk metrics including:
    - Value at Risk (VaR)
    - Expected Shortfall (CVaR)
    - Beta and correlation analysis
    - Rolling statistics
    
    Parameters
    ----------
    confidence_level : float
        Confidence level for VaR/CVaR (default: 0.95).
    rolling_window : int
        Window size for rolling calculations (default: 36 months).
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        rolling_window: int = 36,
    ):
        self.confidence_level = confidence_level
        self.rolling_window = rolling_window
    
    def compute_var(
        self,
        returns: np.ndarray,
        method: str = "historical",
    ) -> float:
        """
        Compute Value at Risk.
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
        method : str
            VaR method ("historical", "parametric").
            
        Returns
        -------
        float
            VaR at the specified confidence level.
        """
        returns = returns[~np.isnan(returns)]
        
        if method == "historical":
            return np.percentile(returns, (1 - self.confidence_level) * 100)
        
        elif method == "parametric":
            from scipy.stats import norm
            mean = np.mean(returns)
            std = np.std(returns)
            return mean + std * norm.ppf(1 - self.confidence_level)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def compute_cvar(self, returns: np.ndarray) -> float:
        """
        Compute Conditional Value at Risk (Expected Shortfall).
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
            
        Returns
        -------
        float
            CVaR at the specified confidence level.
        """
        returns = returns[~np.isnan(returns)]
        var = self.compute_var(returns)
        return np.mean(returns[returns <= var])
    
    def compute_beta(
        self,
        returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> float:
        """
        Compute portfolio beta.
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
        market_returns : np.ndarray
            Market returns.
            
        Returns
        -------
        float
            Portfolio beta.
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(returns) | np.isnan(market_returns))
        port_ret = returns[valid_mask]
        mkt_ret = market_returns[valid_mask]
        
        if len(port_ret) == 0:
            return 0
        
        covariance = np.cov(port_ret, mkt_ret)[0, 1]
        market_variance = np.var(mkt_ret)
        
        return covariance / market_variance if market_variance > 0 else 0
    
    def compute_rolling_stats(
        self,
        returns: np.ndarray,
        dates: List,
    ) -> pl.DataFrame:
        """
        Compute rolling statistics.
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
        dates : List
            Dates corresponding to returns.
            
        Returns
        -------
        pl.DataFrame
            Rolling mean, std, and Sharpe ratio.
        """
        n = len(returns)
        rolling_mean = []
        rolling_std = []
        rolling_sharpe = []
        
        for i in range(n):
            if i < self.rolling_window - 1:
                rolling_mean.append(np.nan)
                rolling_std.append(np.nan)
                rolling_sharpe.append(np.nan)
            else:
                window = returns[i - self.rolling_window + 1:i + 1]
                window = window[~np.isnan(window)]
                if len(window) > 0:
                    mean = np.mean(window)
                    std = np.std(window)
                    rolling_mean.append(mean)
                    rolling_std.append(std)
                    rolling_sharpe.append((mean / std) * np.sqrt(12) if std > 0 else 0)
                else:
                    rolling_mean.append(np.nan)
                    rolling_std.append(np.nan)
                    rolling_sharpe.append(np.nan)
        
        return pl.DataFrame({
            "date": dates,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "rolling_sharpe": rolling_sharpe,
        })
    
    def compute_drawdown_series(self, returns: np.ndarray, dates: List) -> pl.DataFrame:
        """
        Compute drawdown time series.
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
        dates : List
            Dates corresponding to returns.
            
        Returns
        -------
        pl.DataFrame
            Drawdown at each point.
        """
        cumulative = np.cumprod(1 + np.nan_to_num(returns, nan=0))
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return pl.DataFrame({
            "date": dates,
            "drawdown": drawdown.tolist(),
        })
    
    def compute_correlation_matrix(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> pl.DataFrame:
        """
        Compute correlation matrix across portfolios.
        
        Parameters
        ----------
        returns_dict : dict
            Dictionary mapping portfolio names to return arrays.
            
        Returns
        -------
        pl.DataFrame
            Correlation matrix.
        """
        names = list(returns_dict.keys())
        n = len(names)
        
        # Build correlation matrix
        corr_matrix = np.zeros((n, n))
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                ret_i = returns_dict[name_i]
                ret_j = returns_dict[name_j]
                valid_mask = ~(np.isnan(ret_i) | np.isnan(ret_j))
                if valid_mask.sum() > 0:
                    corr_matrix[i, j] = np.corrcoef(ret_i[valid_mask], ret_j[valid_mask])[0, 1]
                else:
                    corr_matrix[i, j] = np.nan
        
        # Create DataFrame
        data = {"portfolio": names}
        for i, name in enumerate(names):
            data[name] = corr_matrix[:, i].tolist()
        
        return pl.DataFrame(data)
    
    def compute_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """
        Compute Information Ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Portfolio returns.
        benchmark_returns : np.ndarray
            Benchmark returns.
            
        Returns
        -------
        float
            Information ratio (annualized).
        """
        valid_mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
        active_returns = returns[valid_mask] - benchmark_returns[valid_mask]
        
        if len(active_returns) == 0 or np.std(active_returns) == 0:
            return 0
        
        return (np.mean(active_returns) / np.std(active_returns)) * np.sqrt(12)


class FactorDataLoader:
    """
    Load factor data from Kenneth French's data library.
    
    Note: In production, this would download from the actual data source.
    This is a placeholder that generates synthetic factor data.
    """
    
    def __init__(self, cache_dir: str = "./data/factors"):
        self.cache_dir = cache_dir
    
    def load_factors(
        self,
        model_name: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """
        Load factor returns for specified model and date range.
        
        Parameters
        ----------
        model_name : str
            Factor model name.
        start_date : str
            Start date.
        end_date : str
            End date.
            
        Returns
        -------
        pl.DataFrame
            Factor returns with date column and factor columns.
        """
        if model_name not in FACTOR_MODELS:
            raise ValueError(f"Unknown factor model: {model_name}")
        
        model = FACTOR_MODELS[model_name]
        
        # In production, load from Kenneth French's website or cache
        # For now, return placeholder
        logger.warning(
            f"Factor data loading not implemented. "
            f"Need to integrate with pandas_datareader or direct download."
        )
        
        return pl.DataFrame({"date": []})
    
    def download_ff_factors(self) -> None:
        """
        Download Fama-French factor data from Kenneth French's website.
        
        This would use pandas_datareader or direct HTTP download.
        """
        try:
            import pandas_datareader.data as web
            
            # Download FF 3-factor
            ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench")
            
            # Download momentum
            mom = web.DataReader("F-F_Momentum_Factor", "famafrench")
            
            logger.info("Factor data downloaded successfully")
            
        except ImportError:
            logger.warning("pandas_datareader not installed")
        except Exception as e:
            logger.error(f"Failed to download factor data: {e}")
