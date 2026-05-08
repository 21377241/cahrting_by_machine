"""Synthetic data adapter for offline testing."""

from typing import List, Optional
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from cbm.core.types import StockData
from cbm.data.adapters.base import DataAdapter


class SyntheticAdapter(DataAdapter):
    """
    Data adapter that generates realistic synthetic stock data for offline testing.

    Simulates monthly prices using geometric Brownian motion with
    cross-sectional correlation, producing the same StockData structure
    as the Yahoo Finance adapter.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default: 42).
    annual_mu : float, optional
        Annualised drift (default: 0.08).
    annual_sigma : float, optional
        Annualised volatility (default: 0.20).
    """

    UNIVERSES = {
        "sp500": [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
            "TMO", "ABT", "ACN", "DHR", "NEE", "DIS", "VZ", "ADBE", "WFC",
            "PM", "TXN", "CMCSA", "NKE", "RTX", "UPS", "ORCL", "CRM", "HON",
            "MS", "QCOM", "INTC", "IBM", "BA",
        ],
        "dow30": [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
            "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
            "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V",
            "VZ", "WBA", "WMT",
        ],
        "tech_large": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
            "ORCL", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN",
            "IBM", "NOW", "INTU", "AMAT",
        ],
    }

    def __init__(self, seed: int = 42, annual_mu: float = 0.08, annual_sigma: float = 0.20):
        self.seed = seed
        self.annual_mu = annual_mu
        self.annual_sigma = annual_sigma

    # ------------------------------------------------------------------
    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1mo",
        include_market_cap: bool = True,
        **kwargs,
    ) -> StockData:
        """Generate synthetic monthly OHLCV data via geometric Brownian motion."""
        rng = np.random.default_rng(self.seed)

        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n_periods = len(dates)
        n_tickers = len(tickers)

        # Monthly params
        mu = self.annual_mu / 12
        sigma = self.annual_sigma / np.sqrt(12)

        # Correlated returns via one common factor
        factor = rng.standard_normal(n_periods)
        idio = rng.standard_normal((n_periods, n_tickers))
        log_returns = mu - 0.5 * sigma ** 2 + sigma * (
            0.5 * factor[:, None] + np.sqrt(1 - 0.25) * idio
        )

        # Random starting prices between 20 and 300
        start_prices = rng.uniform(20, 300, n_tickers)
        prices_np = start_prices * np.exp(np.cumsum(log_returns, axis=0))

        prices_pd = pd.DataFrame(prices_np, index=dates, columns=tickers)

        # --- build Polars DataFrames ------------------------------------------------
        prices_reset = prices_pd.reset_index().rename(columns={"index": "date"})
        prices = pl.from_pandas(prices_reset)

        returns_pd = prices_pd.pct_change().dropna()
        returns_reset = returns_pd.reset_index().rename(columns={"index": "date"})
        returns = pl.from_pandas(returns_reset)

        # --- synthetic market cap ---------------------------------------------------
        market_cap = None
        if include_market_cap:
            shares = rng.integers(500_000_000, 10_000_000_000, n_tickers)
            mc_data = {"date": prices.get_column("date").to_list()}
            for i, t in enumerate(tickers):
                mc_data[t] = (prices_pd[t] * shares[i]).tolist()
            market_cap = pl.DataFrame(mc_data)

        logger.info(
            f"Generated synthetic data for {n_tickers} tickers "
            f"over {n_periods} months"
        )

        return StockData(
            prices=prices,
            returns=returns,
            market_cap=market_cap,
            metadata={
                "source": "synthetic",
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "n_tickers": n_tickers,
            },
        )

    # ------------------------------------------------------------------
    def get_universe(self, universe_name: str) -> List[str]:
        key = universe_name.lower()
        if key not in self.UNIVERSES:
            available = ", ".join(self.UNIVERSES.keys())
            raise ValueError(f"Unknown universe '{universe_name}'. Available: {available}")
        return self.UNIVERSES[key]

    # ------------------------------------------------------------------
    def fetch_risk_free_rate(self, start_date: str, end_date: str, **kwargs) -> pl.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        return pl.DataFrame({"date": dates.tolist(), "rate": [0.004] * len(dates)})

    def validate_tickers(self, tickers: List[str]) -> List[str]:
        return tickers
