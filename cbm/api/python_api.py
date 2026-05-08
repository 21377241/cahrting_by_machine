"""High-level Python API for common use cases."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr

from cbm.core.config import CBMConfig
from cbm.core.engine import PortfolioEngine
from cbm.core.types import BacktestResult, Forecast, PerformanceMetrics, PortfolioSet


def run_pipeline(
    tickers: Optional[List[str]] = None,
    universe: Optional[str] = "sp500",
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    optimization_period: Tuple[str, str] = ("2010-01", "2015-12"),
    test_period: Tuple[str, str] = ("2016-01", "2023-12"),
    architecture: str = "cnn_lstm",
    n_portfolios: int = 10,
    config: Optional[CBMConfig] = None,
) -> BacktestResult:
    """
    Run the complete ML portfolio selection pipeline.
    
    This is a convenience function that runs all steps:
    1. Load data
    2. Prepare features
    3. Train model
    4. Generate forecasts
    5. Construct portfolios
    6. Analyze performance
    
    Parameters
    ----------
    tickers : list of str, optional
        Specific tickers to include.
    universe : str, optional
        Stock universe ("sp500", "dow30", etc.).
    start_date : str
        Data start date (YYYY-MM-DD).
    end_date : str
        Data end date (YYYY-MM-DD).
    optimization_period : tuple
        Model training period as (start, end) in YYYY-MM format.
    test_period : tuple
        Out-of-sample test period.
    architecture : str
        Neural network architecture.
    n_portfolios : int
        Number of quantile portfolios.
    config : CBMConfig, optional
        Full configuration object (overrides other parameters).
        
    Returns
    -------
    BacktestResult
        Complete backtest results.
        
    Example
    -------
    >>> result = run_pipeline(
    ...     universe="sp500",
    ...     start_date="2015-01-01",
    ...     end_date="2023-12-31",
    ...     optimization_period=("2015-01", "2019-12"),
    ...     test_period=("2020-01", "2023-12"),
    ... )
    >>> print(result.summary())
    """
    # Create or update config
    if config is None:
        config = CBMConfig()
    
    # Update config with provided parameters
    config.data.start_date = start_date
    config.data.end_date = end_date
    config.data.tickers = tickers
    config.data.universe = universe
    config.training.optimization_period = optimization_period
    config.backtest.test_period = test_period
    config.portfolio.n_portfolios = n_portfolios
    
    # Set architecture
    from cbm.core.types import Architecture
    config.model.architecture = Architecture(architecture)
    
    # Initialize engine and run
    engine = PortfolioEngine(config=config)
    
    return engine.run_backtest(tickers=tickers, universe=universe)


def quick_backtest(
    returns: pl.DataFrame,
    forecasts: pl.DataFrame,
    n_portfolios: int = 10,
    weighting: str = "equal",
) -> Dict[str, PerformanceMetrics]:
    """
    Quick backtest using pre-computed forecasts and returns.
    
    This is useful when you already have forecasts and want to
    quickly evaluate their performance.
    
    Parameters
    ----------
    returns : pl.DataFrame
        Stock returns with date column and tickers as other columns.
    forecasts : pl.DataFrame
        Return forecasts with same structure as returns.
    n_portfolios : int
        Number of quantile portfolios.
    weighting : str
        Portfolio weighting ("equal" or "value").
        
    Returns
    -------
    dict
        Performance metrics for each portfolio.
        
    Example
    -------
    >>> performance = quick_backtest(
    ...     returns=stock_returns,
    ...     forecasts=my_forecasts,
    ...     n_portfolios=10,
    ... )
    >>> print(performance["long_short"].sharpe_ratio)
    """
    from cbm.core.types import Forecast
    from cbm.portfolio import PortfolioConstructor, PerformanceAnalyzer
    
    # Create Forecast object
    forecast_obj = Forecast(
        values=forecasts,
        model_id="external",
        created_at="",
    )
    
    # Construct portfolios
    constructor = PortfolioConstructor(
        n_portfolios=n_portfolios,
        weighting=weighting,
    )
    
    portfolios = constructor.construct(
        forecasts=forecast_obj,
        returns=returns,
    )
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    performance = analyzer.analyze(portfolios)
    
    return performance


def compare_architectures(
    tickers: Optional[List[str]] = None,
    universe: str = "sp500",
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    architectures: List[str] = ["fnn", "cnn", "lstm", "cnn_lstm"],
) -> pl.DataFrame:
    """
    Compare performance across different model architectures.
    
    Parameters
    ----------
    tickers : list of str, optional
        Specific tickers.
    universe : str
        Stock universe.
    start_date : str
        Start date.
    end_date : str
        End date.
    architectures : list of str
        Architectures to compare.
        
    Returns
    -------
    pl.DataFrame
        Comparison table with performance metrics.
    """
    results = []
    
    config = CBMConfig()
    config.data.start_date = start_date
    config.data.end_date = end_date
    config.data.universe = universe
    
    engine = PortfolioEngine(config=config)
    engine.load_data(tickers=tickers, universe=universe)
    engine.prepare_features()
    
    for arch in architectures:
        # Train model
        model_id = engine.train_model(architecture=arch)
        
        # Generate forecasts
        forecasts = engine.forecast(model_id=model_id)
        
        # Construct portfolios
        portfolios = engine.construct_portfolios(forecasts=forecasts)
        
        # Analyze
        performance = engine.analyze_performance(portfolios=portfolios)
        
        # Store long-short performance
        ls_perf = performance.get("long_short")
        if ls_perf:
            results.append({
                "architecture": arch,
                "mean_return": ls_perf.mean_return,
                "std_dev": ls_perf.std_dev,
                "sharpe_ratio": ls_perf.sharpe_ratio,
                "t_statistic": ls_perf.t_statistic,
            })
    
    return pl.DataFrame(results)


def evaluate_forecasts(
    forecasts: pl.DataFrame,
    actual_returns: pl.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate forecast quality using standard metrics.
    
    Parameters
    ----------
    forecasts : pl.DataFrame
        Return forecasts with date column and ticker columns.
    actual_returns : pl.DataFrame
        Actual realized returns with same structure.
        
    Returns
    -------
    dict
        Evaluation metrics.
    """
    # Get common dates
    forecast_dates = set(forecasts.get_column("date").to_list())
    actual_dates = set(actual_returns.get_column("date").to_list())
    common_dates = list(forecast_dates.intersection(actual_dates))
    
    # Get common tickers (excluding date column)
    forecast_tickers = set(c for c in forecasts.columns if c != "date")
    actual_tickers = set(c for c in actual_returns.columns if c != "date")
    common_tickers = list(forecast_tickers.intersection(actual_tickers))
    
    if not common_dates or not common_tickers:
        return {
            "spearman_correlation": np.nan,
            "pearson_correlation": np.nan,
            "mse": np.nan,
            "mae": np.nan,
            "n_observations": 0,
        }
    
    # Flatten data for comparison
    fcst_vals = []
    actual_vals = []
    
    for date in common_dates:
        fcst_row = forecasts.filter(pl.col("date") == date)
        actual_row = actual_returns.filter(pl.col("date") == date)
        
        for ticker in common_tickers:
            fcst_val = fcst_row.get_column(ticker)[0]
            actual_val = actual_row.get_column(ticker)[0]
            
            if fcst_val is not None and actual_val is not None:
                if not np.isnan(fcst_val) and not np.isnan(actual_val):
                    fcst_vals.append(fcst_val)
                    actual_vals.append(actual_val)
    
    fcst = np.array(fcst_vals)
    actual = np.array(actual_vals)
    
    if len(fcst) == 0:
        return {
            "spearman_correlation": np.nan,
            "pearson_correlation": np.nan,
            "mse": np.nan,
            "mae": np.nan,
            "n_observations": 0,
        }
    
    # Calculate metrics
    spearman_corr, _ = spearmanr(fcst, actual)
    pearson_corr, _ = pearsonr(fcst, actual)
    mse = np.mean((fcst - actual) ** 2)
    mae = np.mean(np.abs(fcst - actual))
    
    return {
        "spearman_correlation": spearman_corr,
        "pearson_correlation": pearson_corr,
        "mse": mse,
        "mae": mae,
        "n_observations": len(fcst),
    }
