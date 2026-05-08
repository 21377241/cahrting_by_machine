"""Statistical metrics utilities."""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from scipy import stats


def calculate_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 12,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns.
    risk_free_rate : float
        Monthly risk-free rate.
    annualization_factor : int
        Factor for annualization (12 for monthly data).
        
    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return {
            "mean_return": np.nan,
            "std_dev": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "positive_months_pct": np.nan,
            "best_month": np.nan,
            "worst_month": np.nan,
            "annualized_return": np.nan,
            "annualized_std": np.nan,
        }
    
    excess_returns = returns - risk_free_rate
    
    metrics = {
        "mean_return": np.mean(returns),
        "std_dev": np.std(returns),
        "skewness": stats.skew(returns),
        "kurtosis": stats.kurtosis(returns),
        "sharpe_ratio": (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(annualization_factor) if np.std(excess_returns) > 0 else 0,
        "sortino_ratio": _sortino_ratio(excess_returns, annualization_factor),
        "max_drawdown": _max_drawdown(returns),
        "calmar_ratio": _calmar_ratio(returns, annualization_factor),
        "positive_months_pct": np.mean(returns > 0),
        "best_month": np.max(returns),
        "worst_month": np.min(returns),
    }
    
    # Annualized metrics
    metrics["annualized_return"] = (1 + metrics["mean_return"]) ** annualization_factor - 1
    metrics["annualized_std"] = metrics["std_dev"] * np.sqrt(annualization_factor)
    
    return metrics


def _sortino_ratio(
    returns: np.ndarray,
    annualization_factor: int = 12,
) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
    
    return (np.mean(returns) / downside_std) * np.sqrt(annualization_factor)


def _max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    return np.min(drawdowns)


def _calmar_ratio(
    returns: np.ndarray,
    annualization_factor: int = 12,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    max_dd = abs(_max_drawdown(returns))
    
    if max_dd == 0:
        return np.inf
    
    annualized_return = (1 + np.mean(returns)) ** annualization_factor - 1
    return annualized_return / max_dd


def newey_west_se(
    residuals: np.ndarray,
    lags: int = 12,
) -> float:
    """
    Calculate Newey-West standard error.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residual series.
    lags : int
        Number of lags for HAC estimation.
        
    Returns
    -------
    float
        Newey-West standard error.
    """
    n = len(residuals)
    
    # Base variance
    var = np.sum(residuals ** 2) / n
    
    # Add autocovariance terms with Bartlett weights
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovar = np.sum(residuals[lag:] * residuals[:-lag]) / n
        var += 2 * weight * autocovar
    
    return np.sqrt(var / n)


def spearman_correlation_monthly(
    forecasts: pl.DataFrame,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate monthly cross-sectional Spearman correlations.
    
    Parameters
    ----------
    forecasts : pl.DataFrame
        Return forecasts with date column and ticker columns.
    actuals : pl.DataFrame
        Actual returns with date column and ticker columns.
        
    Returns
    -------
    pl.DataFrame
        Monthly Spearman correlations with date and correlation columns.
    """
    correlations = []
    
    forecast_dates = forecasts.get_column("date").to_list()
    actual_dates = actuals.get_column("date").to_list()
    
    common_dates = [d for d in forecast_dates if d in actual_dates]
    
    tickers = [c for c in forecasts.columns if c != "date"]
    
    for date in common_dates:
        fcst_row = forecasts.filter(pl.col("date") == date)
        actual_row = actuals.filter(pl.col("date") == date)
        
        fcst_vals = []
        actual_vals = []
        
        for ticker in tickers:
            if ticker not in actual_row.columns:
                continue
            
            fcst_val = fcst_row.get_column(ticker)[0]
            actual_val = actual_row.get_column(ticker)[0]
            
            if fcst_val is not None and actual_val is not None:
                if not np.isnan(fcst_val) and not np.isnan(actual_val):
                    fcst_vals.append(fcst_val)
                    actual_vals.append(actual_val)
        
        if len(fcst_vals) >= 10:
            corr, _ = stats.spearmanr(fcst_vals, actual_vals)
            correlations.append({"date": date, "correlation": corr})
    
    return pl.DataFrame(correlations)


def calculate_ic(
    forecasts: pl.DataFrame,
    actuals: pl.DataFrame,
) -> Dict[str, float]:
    """
    Calculate Information Coefficient (IC) statistics.
    
    Parameters
    ----------
    forecasts : pl.DataFrame
        Return forecasts with date column and ticker columns.
    actuals : pl.DataFrame
        Actual returns with date column and ticker columns.
        
    Returns
    -------
    dict
        IC statistics including mean, std, IR.
    """
    monthly_corr = spearman_correlation_monthly(forecasts, actuals)
    
    if monthly_corr.height == 0:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ir": np.nan,
            "positive_ic_pct": np.nan,
            "n_months": 0,
        }
    
    corr_vals = monthly_corr.get_column("correlation").to_numpy()
    mean_ic = np.mean(corr_vals)
    std_ic = np.std(corr_vals)
    
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": mean_ic / std_ic if std_ic > 0 else 0,
        "positive_ic_pct": np.mean(corr_vals > 0),
        "n_months": len(corr_vals),
    }
