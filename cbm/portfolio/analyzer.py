"""Performance analysis for portfolios."""

from typing import Dict, List, Optional

import numpy as np
import polars as pl
from loguru import logger
from scipy import stats

from cbm.core.types import PerformanceMetrics, Portfolio, PortfolioSet


class PerformanceAnalyzer:
    """
    Analyze portfolio performance with risk adjustment.
    
    Computes standard performance metrics and factor model alphas
    following the methodology in Murray, Xia, Xiao (2024).
    
    Parameters
    ----------
    factor_models : list of str
        Factor models to use for alpha calculation.
    newey_west_lags : int
        Number of lags for Newey-West standard errors.
        
    Example
    -------
    >>> analyzer = PerformanceAnalyzer(factor_models=["CAPM", "FF3", "FF5"])
    >>> performance = analyzer.analyze(portfolios)
    """
    
    def __init__(
        self,
        factor_models: List[str] = ["CAPM", "FF3", "FFC4", "FF5"],
        newey_west_lags: int = 12,
    ):
        self.factor_models = factor_models
        self.newey_west_lags = newey_west_lags
        self._factor_data: Optional[pl.DataFrame] = None
    
    def analyze(
        self,
        portfolio_set: PortfolioSet,
        risk_free_rate: Optional[pl.DataFrame] = None,
    ) -> Dict[str, PerformanceMetrics]:
        """
        Analyze performance of all portfolios in the set.
        
        Parameters
        ----------
        portfolio_set : PortfolioSet
            Set of portfolios to analyze.
        risk_free_rate : pl.DataFrame, optional
            Risk-free rate for excess return calculation.
            
        Returns
        -------
        dict
            Performance metrics for each portfolio.
        """
        logger.info("Analyzing portfolio performance")
        
        results = {}
        
        # Analyze each portfolio
        for name, portfolio in portfolio_set.portfolios.items():
            results[name] = self._analyze_portfolio(portfolio, risk_free_rate)
        
        # Analyze long-short portfolio
        if portfolio_set.long_short is not None:
            results["long_short"] = self._analyze_portfolio(
                portfolio_set.long_short, risk_free_rate
            )
        
        return results
    
    def _analyze_portfolio(
        self,
        portfolio: Portfolio,
        risk_free_rate: Optional[pl.DataFrame] = None,
    ) -> PerformanceMetrics:
        """Analyze a single portfolio."""
        # Get returns as numpy array
        returns_data = portfolio.returns
        
        # Handle both numpy arrays and DataFrames
        if isinstance(returns_data, np.ndarray):
            returns = returns_data
        elif isinstance(returns_data, pl.DataFrame):
            if "return" not in returns_data.columns:
                return PerformanceMetrics(
                    mean_return=np.nan,
                    std_dev=np.nan,
                    sharpe_ratio=np.nan,
                    t_statistic=np.nan,
                    annualized_return=np.nan,
                    max_drawdown=np.nan,
                )
            returns = returns_data.get_column("return").to_numpy()
        else:
            returns = np.array(returns_data)
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return PerformanceMetrics(
                mean_return=np.nan,
                std_dev=np.nan,
                sharpe_ratio=np.nan,
                t_statistic=np.nan,
                annualized_return=np.nan,
                max_drawdown=np.nan,
            )
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        
        # Annualized Sharpe ratio (assuming monthly data)
        sharpe_ratio = (mean_return / std_dev) * np.sqrt(12) if std_dev > 0 else 0
        
        # Newey-West t-statistic
        t_stat = self._newey_west_tstat(returns)
        
        # Annualized return (CAGR)
        cumulative = np.prod(1 + returns)
        n_years = len(returns) / 12
        annualized_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Factor model alphas (placeholder)
        alphas = {}
        factor_loadings = {}
        
        return PerformanceMetrics(
            mean_return=mean_return,
            std_dev=std_dev,
            sharpe_ratio=sharpe_ratio,
            t_statistic=t_stat,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            alpha=alphas if alphas else None,
            factor_loadings=factor_loadings if factor_loadings else None,
        )
    
    def _newey_west_tstat(self, returns: np.ndarray) -> float:
        """
        Calculate Newey-West adjusted t-statistic.
        
        Tests null hypothesis that mean return equals zero.
        """
        n = len(returns)
        mean = np.mean(returns)
        
        if n < self.newey_west_lags + 2:
            # Fall back to simple t-stat
            return mean / (np.std(returns) / np.sqrt(n))
        
        # Calculate Newey-West standard error
        demeaned = returns - mean
        
        # Base variance
        var = np.sum(demeaned ** 2) / n
        
        # Add autocovariance terms
        for lag in range(1, self.newey_west_lags + 1):
            weight = 1 - lag / (self.newey_west_lags + 1)
            autocovar = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
            var += 2 * weight * autocovar
        
        se = np.sqrt(var / n)
        
        return mean / se if se > 0 else 0
    
    def create_summary_table(
        self,
        performance: Dict[str, PerformanceMetrics],
    ) -> pl.DataFrame:
        """
        Create summary table matching paper format.
        
        Parameters
        ----------
        performance : dict
            Performance metrics for each portfolio.
            
        Returns
        -------
        pl.DataFrame
            Summary table with mean, std, Sharpe, t-stat.
        """
        data = []
        
        for name, metrics in performance.items():
            data.append({
                "portfolio": name,
                "mean_pct": metrics.mean_return * 100,
                "std_pct": metrics.std_dev * 100,
                "sharpe": metrics.sharpe_ratio,
                "t_stat": metrics.t_statistic,
            })
        
        return pl.DataFrame(data)
