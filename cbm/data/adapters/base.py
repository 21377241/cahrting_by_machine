"""Abstract base class for data adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import polars as pl

from cbm.core.types import StockData


class DataAdapter(ABC):
    """
    Abstract base class for market data providers.
    
    This class defines the interface that all data adapters must implement.
    Concrete implementations include Yahoo Finance, WRDS, and local file adapters.
    
    Example
    -------
    >>> adapter = YahooFinanceAdapter()
    >>> data = adapter.fetch_data(
    ...     tickers=["AAPL", "MSFT"],
    ...     start_date="2020-01-01",
    ...     end_date="2023-12-31"
    ... )
    """
    
    @abstractmethod
    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> StockData:
        """
        Fetch stock data for specified tickers and date range.
        
        Parameters
        ----------
        tickers : list of str
            List of ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        **kwargs
            Additional adapter-specific parameters.
            
        Returns
        -------
        StockData
            Container with prices, returns, and metadata.
        """
        pass
    
    @abstractmethod
    def get_universe(self, universe_name: str) -> List[str]:
        """
        Get list of tickers for a predefined universe.
        
        Parameters
        ----------
        universe_name : str
            Name of the universe (e.g., "sp500", "russell1000").
            
        Returns
        -------
        list of str
            List of ticker symbols in the universe.
        """
        pass
    
    @abstractmethod
    def fetch_risk_free_rate(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """
        Fetch risk-free rate data.
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
            
        Returns
        -------
        pl.DataFrame
            Risk-free rate with date and rate columns.
        """
        pass
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Validate and filter tickers.
        
        Parameters
        ----------
        tickers : list of str
            List of ticker symbols to validate.
            
        Returns
        -------
        list of str
            List of valid ticker symbols.
        """
        # Default implementation returns all tickers
        return tickers
    
    def adjust_for_corporate_actions(
        self,
        data: StockData,
    ) -> StockData:
        """
        Adjust prices for splits and dividends.
        
        Parameters
        ----------
        data : StockData
            Raw stock data.
            
        Returns
        -------
        StockData
            Adjusted stock data.
        """
        # Default implementation returns data unchanged
        return data
