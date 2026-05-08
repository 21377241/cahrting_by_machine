"""Yahoo Finance data adapter implementation."""

from typing import Dict, List, Optional
import warnings
from datetime import datetime

import polars as pl
import numpy as np
from loguru import logger

from cbm.core.types import StockData
from cbm.data.adapters.base import DataAdapter


# Predefined stock universes
UNIVERSES = {
    "sp500": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "ACN",
        "DHR", "NEE", "DIS", "VZ", "ADBE", "WFC", "PM", "TXN", "CMCSA", "NKE",
        "RTX", "UPS", "ORCL", "CRM", "HON", "MS", "QCOM", "INTC", "IBM", "BA",
    ],  # Subset of S&P 500 for demonstration
    "dow30": [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    ],
    "tech_large": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
        "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT",
    ],
}


class YahooFinanceAdapter(DataAdapter):
    """
    Data adapter for Yahoo Finance.
    
    This adapter fetches stock data from Yahoo Finance using the yfinance library.
    It handles data cleaning, adjustment for corporate actions, and provides
    access to predefined stock universes.
    
    Parameters
    ----------
    timeout : int, optional
        Request timeout in seconds (default: 30).
    retries : int, optional
        Number of retries for failed requests (default: 3).
        
    Example
    -------
    >>> adapter = YahooFinanceAdapter()
    >>> data = adapter.fetch_data(
    ...     tickers=["AAPL", "MSFT"],
    ...     start_date="2020-01-01",
    ...     end_date="2023-12-31"
    ... )
    """
    
    def __init__(self, timeout: int = 30, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
    
    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1mo",
        include_market_cap: bool = True,
        **kwargs,
    ) -> StockData:
        """
        Fetch stock data from Yahoo Finance.
        
        Parameters
        ----------
        tickers : list of str
            List of ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        interval : str, optional
            Data frequency ("1d", "1wk", "1mo"). Default: "1mo".
        include_market_cap : bool, optional
            Whether to fetch market cap data. Default: True.
            
        Returns
        -------
        StockData
            Container with prices, returns, and metadata.
        """
        import time
        import pandas as pd
        import yfinance as yf
        
        logger.info(f"Fetching data for {len(tickers)} tickers from Yahoo Finance")
        
        # Fetch each ticker individually with retry + delay to avoid rate limiting
        frames = {}
        for ticker in tickers:
            for attempt in range(self.retries):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df = yf.download(
                            tickers=ticker,
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            auto_adjust=True,
                            progress=False,
                            timeout=self.timeout,
                        )
                    if not df.empty:
                        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                        frames[ticker] = close
                        logger.debug(f"Downloaded {ticker} ({len(df)} rows)")
                        break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"{ticker} attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                logger.warning(f"Skipping {ticker}: all {self.retries} attempts failed")
            time.sleep(1)  # polite delay between tickers
        
        if not frames:
            raise ValueError("No data retrieved from Yahoo Finance")
        
        data = pd.DataFrame(frames)
        
        # data is already a DataFrame with ticker columns
        prices_pd = data.copy()
        
        # Drop tickers with all NaN values
        valid_cols = prices_pd.columns[prices_pd.notna().any()]
        prices_pd = prices_pd[valid_cols]
        
        # Forward-fill missing values (handle holidays, etc.)
        prices_pd = prices_pd.ffill()
        
        # Calculate returns
        returns_pd = prices_pd.pct_change().dropna()
        
        # Convert to Polars - reset index to get date as column
        prices_pd_reset = prices_pd.reset_index()
        prices_pd_reset.columns = ["date"] + list(prices_pd.columns)
        prices = pl.from_pandas(prices_pd_reset)
        
        returns_pd_reset = returns_pd.reset_index()
        returns_pd_reset.columns = ["date"] + list(returns_pd.columns)
        returns = pl.from_pandas(returns_pd_reset)
        
        # Fetch market cap if requested
        market_cap = None
        if include_market_cap:
            market_cap = self._fetch_market_caps(list(prices_pd.columns), prices)
        
        # Create StockData container
        stock_data = StockData(
            prices=prices,
            returns=returns,
            market_cap=market_cap,
            metadata={
                "source": "yahoo_finance",
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "n_tickers": len(prices.columns) - 1,  # Exclude date column
            },
        )
        
        logger.info(f"Retrieved data for {len(stock_data.tickers)} tickers")
        
        return stock_data
    
    def _fetch_market_caps(
        self,
        tickers: List[str],
        prices: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Fetch market capitalization data.
        
        For Yahoo Finance, we approximate monthly market cap using
        the most recent available data.
        """
        import yfinance as yf
        
        logger.debug(f"Fetching market caps for {len(tickers)} tickers")
        
        market_caps = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get("marketCap", np.nan)
                market_caps[ticker] = market_cap
            except Exception:
                market_caps[ticker] = np.nan
        
        # Create DataFrame with constant market cap (simplified)
        # In production, you'd want historical market caps
        dates = prices.get_column("date").to_list()
        
        data = {"date": dates}
        for ticker in tickers:
            data[ticker] = [market_caps.get(ticker, np.nan)] * len(dates)
        
        return pl.DataFrame(data)
    
    def get_universe(self, universe_name: str) -> List[str]:
        """
        Get list of tickers for a predefined universe.
        
        Parameters
        ----------
        universe_name : str
            Name of the universe (e.g., "sp500", "dow30", "tech_large").
            
        Returns
        -------
        list of str
            List of ticker symbols in the universe.
        """
        universe_lower = universe_name.lower()
        
        if universe_lower not in UNIVERSES:
            available = ", ".join(UNIVERSES.keys())
            raise ValueError(
                f"Unknown universe '{universe_name}'. Available: {available}"
            )
        
        return UNIVERSES[universe_lower]
    
    def fetch_risk_free_rate(
        self,
        start_date: str,
        end_date: str,
        ticker: str = "^IRX",
    ) -> pl.DataFrame:
        """
        Fetch risk-free rate from Yahoo Finance.
        
        Uses the 13-week Treasury Bill rate (^IRX) by default.
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        ticker : str, optional
            Risk-free rate ticker. Default: "^IRX" (13-week T-bill).
            
        Returns
        -------
        pl.DataFrame
            Monthly risk-free rate with date and rate columns.
        """
        import yfinance as yf
        
        logger.debug(f"Fetching risk-free rate from {ticker}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1mo",
                progress=False,
            )
        
        if rf_data.empty:
            logger.warning("Could not fetch risk-free rate, using 0")
            import pandas as pd
            dates = pd.date_range(start_date, end_date, freq="MS")
            return pl.DataFrame({
                "date": dates.tolist(),
                "rate": [0.0] * len(dates),
            })
        
        # Convert from percentage to decimal and annualize to monthly
        rf_data = rf_data.reset_index()
        rf_data = rf_data[["Date", "Close"]]
        rf_data.columns = ["date", "rate"]
        rf_data["rate"] = rf_data["rate"] / 100 / 12
        
        return pl.from_pandas(rf_data)
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Validate tickers by checking if they exist on Yahoo Finance.
        
        Parameters
        ----------
        tickers : list of str
            List of ticker symbols to validate.
            
        Returns
        -------
        list of str
            List of valid ticker symbols.
        """
        import yfinance as yf
        
        valid_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                if stock.info.get("regularMarketPrice") is not None:
                    valid_tickers.append(ticker)
            except Exception:
                logger.debug(f"Invalid ticker: {ticker}")
        
        return valid_tickers
