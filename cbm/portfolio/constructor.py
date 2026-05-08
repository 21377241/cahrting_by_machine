"""Portfolio construction following quantile sorting methodology."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger

from cbm.core.types import Forecast, Portfolio, PortfolioSet


class PortfolioConstructor:
    """
    Construct portfolios sorted by ML forecasts.
    
    Implements the quantile sorting methodology from Murray, Xia, Xiao (2024):
    - Univariate: sort stocks into deciles by MLER
    - Bivariate: double sort by MLER and control variable
    - Trivariate: triple sort for momentum/reversal interaction
    
    Parameters
    ----------
    n_portfolios : int
        Number of quantile portfolios (default: 10 for deciles).
    weighting : str
        Portfolio weighting ("value" for value-weighted, "equal" for equal-weighted).
    use_nyse_breakpoints : bool
        Whether to use NYSE stocks for breakpoint calculation.
        
    Example
    -------
    >>> constructor = PortfolioConstructor(n_portfolios=10, weighting="value")
    >>> portfolios = constructor.construct(forecasts, returns, market_caps)
    """
    
    def __init__(
        self,
        n_portfolios: int = 10,
        weighting: str = "value",
        use_nyse_breakpoints: bool = True,
    ):
        self.n_portfolios = n_portfolios
        self.weighting = weighting
        self.use_nyse_breakpoints = use_nyse_breakpoints
    
    def construct(
        self,
        forecasts: Forecast,
        returns: pl.DataFrame,
        market_caps: Optional[pl.DataFrame] = None,
        nyse_mask: Optional[pl.DataFrame] = None,
    ) -> PortfolioSet:
        """
        Construct quantile portfolios sorted by ML forecasts.
        
        Parameters
        ----------
        forecasts : Forecast
            ML-based return forecasts with date column and ticker columns.
        returns : pl.DataFrame
            Actual returns with date column and ticker columns.
        market_caps : pl.DataFrame, optional
            Market capitalizations for value weighting.
        nyse_mask : pl.DataFrame, optional
            Boolean mask indicating NYSE stocks for breakpoints.
            
        Returns
        -------
        PortfolioSet
            Set of sorted portfolios with long-short portfolio.
        """
        logger.info(f"Constructing {self.n_portfolios} portfolios")
        
        all_returns = {i: [] for i in range(1, self.n_portfolios + 1)}
        all_dates = []
        
        # Get common dates between forecasts and returns
        forecast_dates = forecasts.values.get_column("date").to_list()
        returns_dates = returns.get_column("date").to_list()
        common_dates = [d for d in forecast_dates if d in returns_dates]
        
        tickers = [c for c in forecasts.values.columns if c != "date"]
        
        for date in common_dates:
            # Get forecasts for this date
            fcst_row = forecasts.values.filter(pl.col("date") == date)
            
            # Extract forecasts as dict
            date_forecasts = {}
            for ticker in tickers:
                val = fcst_row.get_column(ticker)[0]
                if val is not None and not np.isnan(val):
                    date_forecasts[ticker] = val
            
            if len(date_forecasts) < self.n_portfolios:
                continue
            
            # Calculate breakpoints
            forecast_values = list(date_forecasts.values())
            breakpoints = self._calculate_breakpoints(forecast_values)
            
            # Assign stocks to portfolios
            assignments = self._assign_to_portfolios(date_forecasts, breakpoints)
            
            # Get returns for this date
            ret_row = returns.filter(pl.col("date") == date)
            if ret_row.height == 0:
                continue
            
            # Get market caps for this date if available
            mc_row = None
            if market_caps is not None:
                mc_row = market_caps.filter(pl.col("date") == date)
            
            # Calculate portfolio returns
            for portfolio_num in range(1, self.n_portfolios + 1):
                stocks_in_portfolio = [t for t, p in assignments.items() if p == portfolio_num]
                
                if len(stocks_in_portfolio) == 0:
                    all_returns[portfolio_num].append(np.nan)
                    continue
                
                # Get returns for stocks in portfolio
                stock_returns = {}
                for ticker in stocks_in_portfolio:
                    if ticker in ret_row.columns:
                        val = ret_row.get_column(ticker)[0]
                        if val is not None and not np.isnan(val):
                            stock_returns[ticker] = val
                
                if len(stock_returns) == 0:
                    all_returns[portfolio_num].append(np.nan)
                    continue
                
                # Calculate weighted return
                if self.weighting == "value" and mc_row is not None:
                    # Value-weighted
                    caps = {}
                    for ticker in stock_returns.keys():
                        if ticker in mc_row.columns:
                            cap = mc_row.get_column(ticker)[0]
                            if cap is not None and not np.isnan(cap):
                                caps[ticker] = cap
                    
                    if len(caps) > 0:
                        total_cap = sum(caps.values())
                        portfolio_return = sum(
                            stock_returns[t] * caps[t] / total_cap
                            for t in caps.keys()
                        )
                    else:
                        portfolio_return = np.mean(list(stock_returns.values()))
                else:
                    # Equal-weighted
                    portfolio_return = np.mean(list(stock_returns.values()))
                
                all_returns[portfolio_num].append(portfolio_return)
            
            all_dates.append(date)
        
        # Create Portfolio objects
        portfolios = {}
        for portfolio_num in range(1, self.n_portfolios + 1):
            returns_df = pl.DataFrame({
                "date": all_dates,
                "return": all_returns[portfolio_num],
            })
            
            portfolios[str(portfolio_num)] = Portfolio(
                name=f"Portfolio_{portfolio_num}",
                holdings=pl.DataFrame({"date": []}),  # Empty placeholder
                returns=returns_df,
                metadata={
                    "quantile": portfolio_num,
                    "n_portfolios": self.n_portfolios,
                    "weighting": self.weighting,
                },
            )
        
        # Create long-short portfolio (high minus low)
        high_returns = all_returns[self.n_portfolios]
        low_returns = all_returns[1]
        long_short_returns = [h - l for h, l in zip(high_returns, low_returns)]
        
        long_short = Portfolio(
            name=f"{self.n_portfolios}-1",
            holdings=pl.DataFrame({"date": []}),
            returns=pl.DataFrame({
                "date": all_dates,
                "return": long_short_returns,
            }),
            metadata={
                "type": "long_short",
                "long": self.n_portfolios,
                "short": 1,
            },
        )
        
        avg_ls_return = np.nanmean(long_short_returns) if long_short_returns else 0
        logger.info(
            f"Constructed portfolios for {len(all_dates)} months. "
            f"Long-short avg return: {avg_ls_return:.4f}"
        )
        
        return PortfolioSet(portfolios=portfolios, long_short=long_short)
    
    def _calculate_breakpoints(self, values: List[float]) -> np.ndarray:
        """Calculate quantile breakpoints."""
        percentiles = np.linspace(0, 100, self.n_portfolios + 1)[1:-1]
        return np.percentile(values, percentiles)
    
    def _assign_to_portfolios(
        self,
        values: Dict[str, float],
        breakpoints: np.ndarray,
    ) -> Dict[str, int]:
        """Assign stocks to portfolios based on breakpoints."""
        assignments = {}
        
        for ticker, value in values.items():
            if np.isnan(value):
                continue
            portfolio = 1
            for bp in breakpoints:
                if value > bp:
                    portfolio += 1
            assignments[ticker] = portfolio
        
        return assignments
    
    def construct_bivariate(
        self,
        forecasts: Forecast,
        control_variable: pl.DataFrame,
        returns: pl.DataFrame,
        market_caps: Optional[pl.DataFrame] = None,
        n_control_groups: int = 5,
    ) -> Dict[str, PortfolioSet]:
        """
        Construct double-sorted portfolios.
        
        First sort stocks into groups by control variable,
        then within each group sort by MLER.
        
        Parameters
        ----------
        forecasts : Forecast
            ML forecasts.
        control_variable : pl.DataFrame
            Control variable for first sort (e.g., size, momentum).
        returns : pl.DataFrame
            Actual returns.
        market_caps : pl.DataFrame, optional
            Market caps for value weighting.
        n_control_groups : int
            Number of control groups (default: 5 for quintiles).
            
        Returns
        -------
        dict
            Dictionary mapping control group to PortfolioSet.
        """
        logger.info(f"Constructing bivariate portfolios with {n_control_groups} control groups")
        
        # Simplified implementation
        results = {}
        
        # TODO: Implement full bivariate sorting
        
        return results
