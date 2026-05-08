"""Visualization utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def prepare_equity_curve_data(
    returns: np.ndarray,
    dates: Optional[List] = None,
    initial_value: float = 1.0,
) -> pl.DataFrame:
    """
    Prepare equity curve data for visualization.
    
    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns.
    dates : list, optional
        Date labels for x-axis.
    initial_value : float
        Starting portfolio value.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with date and value columns for plotting.
    """
    cumulative = initial_value * np.cumprod(1 + returns)
    
    # Prepend initial value
    values = np.concatenate([[initial_value], cumulative])
    
    if dates is not None:
        # Add initial date placeholder
        if len(dates) == len(returns):
            date_labels = ["Start"] + list(dates)
        else:
            date_labels = list(dates)
    else:
        date_labels = list(range(len(values)))
    
    return pl.DataFrame({
        "period": date_labels,
        "value": values,
    })


def prepare_drawdown_data(
    returns: np.ndarray,
    dates: Optional[List] = None,
) -> pl.DataFrame:
    """
    Prepare drawdown series for visualization.
    
    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns.
    dates : list, optional
        Date labels for x-axis.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with date and drawdown columns.
    """
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    
    if dates is not None:
        date_labels = list(dates)
    else:
        date_labels = list(range(len(drawdowns)))
    
    return pl.DataFrame({
        "period": date_labels,
        "drawdown": drawdowns,
    })


def prepare_monthly_returns_data(
    returns: np.ndarray,
    dates: List,
) -> pl.DataFrame:
    """
    Prepare monthly returns heatmap data.
    
    Parameters
    ----------
    returns : np.ndarray
        Monthly returns.
    dates : list
        Date objects with year and month.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with year, month, and return columns.
    """
    records = []
    
    for i, date in enumerate(dates):
        if i < len(returns):
            year = date.year if hasattr(date, "year") else 0
            month = date.month if hasattr(date, "month") else 0
            records.append({
                "year": year,
                "month": month,
                "return": returns[i],
            })
    
    return pl.DataFrame(records)


def prepare_portfolio_weights_data(
    weights: Dict[str, np.ndarray],
    tickers: List[str],
    dates: List,
) -> pl.DataFrame:
    """
    Prepare portfolio weights time series for visualization.
    
    Parameters
    ----------
    weights : dict
        Dictionary mapping portfolio names to weight arrays.
    tickers : list
        List of ticker symbols.
    dates : list
        List of dates.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with date, portfolio, ticker, and weight columns.
    """
    records = []
    
    for portfolio_name, weight_matrix in weights.items():
        for i, date in enumerate(dates):
            if i < weight_matrix.shape[0]:
                for j, ticker in enumerate(tickers):
                    if j < weight_matrix.shape[1]:
                        records.append({
                            "date": date,
                            "portfolio": portfolio_name,
                            "ticker": ticker,
                            "weight": weight_matrix[i, j],
                        })
    
    return pl.DataFrame(records)


def prepare_factor_exposure_data(
    factor_loadings: np.ndarray,
    factor_names: List[str],
    portfolios: List[str],
) -> pl.DataFrame:
    """
    Prepare factor exposure data for bar charts.
    
    Parameters
    ----------
    factor_loadings : np.ndarray
        Matrix of factor loadings (portfolios x factors).
    factor_names : list
        Names of factors.
    portfolios : list
        Names of portfolios.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with portfolio, factor, and loading columns.
    """
    records = []
    
    for i, portfolio in enumerate(portfolios):
        if i < factor_loadings.shape[0]:
            for j, factor in enumerate(factor_names):
                if j < factor_loadings.shape[1]:
                    records.append({
                        "portfolio": portfolio,
                        "factor": factor,
                        "loading": factor_loadings[i, j],
                    })
    
    return pl.DataFrame(records)


def prepare_rolling_stats_data(
    returns: np.ndarray,
    dates: List,
    window: int = 12,
) -> pl.DataFrame:
    """
    Prepare rolling statistics for time series plot.
    
    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns.
    dates : list
        Date labels.
    window : int
        Rolling window size.
        
    Returns
    -------
    pl.DataFrame
        DataFrame with date, rolling_mean, rolling_std, rolling_sharpe.
    """
    n = len(returns)
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    rolling_sharpe = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_returns = returns[i - window + 1:i + 1]
        rolling_mean[i] = np.mean(window_returns)
        rolling_std[i] = np.std(window_returns)
        if rolling_std[i] > 0:
            rolling_sharpe[i] = rolling_mean[i] / rolling_std[i] * np.sqrt(12)
    
    return pl.DataFrame({
        "date": list(dates),
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_sharpe": rolling_sharpe,
    })


def format_performance_table(
    metrics: Dict[str, Dict[str, float]],
) -> pl.DataFrame:
    """
    Format performance metrics as a display table.
    
    Parameters
    ----------
    metrics : dict
        Nested dictionary of {portfolio: {metric: value}}.
        
    Returns
    -------
    pl.DataFrame
        Formatted table with portfolio names as columns.
    """
    if not metrics:
        return pl.DataFrame()
    
    # Get all unique metric names
    all_metrics = set()
    for portfolio_metrics in metrics.values():
        all_metrics.update(portfolio_metrics.keys())
    
    # Build table
    records = []
    for metric_name in sorted(all_metrics):
        record = {"metric": metric_name}
        for portfolio_name, portfolio_metrics in metrics.items():
            record[portfolio_name] = portfolio_metrics.get(metric_name, np.nan)
        records.append(record)
    
    return pl.DataFrame(records)
