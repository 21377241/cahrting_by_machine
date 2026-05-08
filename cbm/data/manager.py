"""Data manager for orchestrating data loading from multiple sources."""

from pathlib import Path
from typing import Dict, List, Optional, Type
import hashlib
import json

import polars as pl
from loguru import logger

from cbm.core.types import StockData
from cbm.data.adapters.base import DataAdapter
from cbm.data.adapters.yahoo_finance import YahooFinanceAdapter
from cbm.data.adapters.synthetic import SyntheticAdapter
from cbm.data.adapters.crsp_local import CRSPLocalAdapter


# Registry of available adapters
ADAPTER_REGISTRY: Dict[str, Type[DataAdapter]] = {
    "yahoo": YahooFinanceAdapter,
    "synthetic": SyntheticAdapter,
    "crsp_local": CRSPLocalAdapter,
}


class DataManager:
    """
    Manages data loading from multiple sources with caching.
    
    This class provides a unified interface for loading stock data from
    different sources (Yahoo Finance, WRDS, local files) with automatic
    caching to parquet files for efficiency.
    
    Parameters
    ----------
    source : str
        Data source to use ("yahoo", "wrds", "local").
    cache_dir : str, optional
        Directory for caching data. Default: "./data/cache".
    use_cache : bool, optional
        Whether to use cached data if available. Default: True.
        
    Example
    -------
    >>> manager = DataManager(source="yahoo")
    >>> data = manager.load_data(
    ...     universe="sp500",
    ...     start_date="2010-01-01",
    ...     end_date="2023-12-31"
    ... )
    """
    
    def __init__(
        self,
        source: str = "yahoo",
        cache_dir: str = "./data/cache",
        use_cache: bool = True,
    ):
        self.source = source.lower()
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        
        # Initialize adapter
        if self.source not in ADAPTER_REGISTRY:
            available = ", ".join(ADAPTER_REGISTRY.keys())
            raise ValueError(f"Unknown source '{source}'. Available: {available}")
        
        self.adapter = ADAPTER_REGISTRY[self.source]()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(
        self,
        tickers: Optional[List[str]] = None,
        universe: Optional[str] = None,
        start_date: str = "2010-01-01",
        end_date: str = "2023-12-31",
        **kwargs,
    ) -> StockData:
        """
        Load stock data from the configured source.
        
        Parameters
        ----------
        tickers : list of str, optional
            List of ticker symbols. Either tickers or universe required.
        universe : str, optional
            Predefined universe name. Either tickers or universe required.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        **kwargs
            Additional parameters passed to the adapter.
            
        Returns
        -------
        StockData
            Container with prices, returns, and metadata.
        """
        # Resolve tickers from universe if provided
        if universe is not None:
            tickers = self.adapter.get_universe(universe)
            logger.info(f"Resolved universe '{universe}' to {len(tickers)} tickers")
        
        if tickers is None or len(tickers) == 0:
            raise ValueError("Must specify either tickers or universe")
        
        # Check cache
        cache_key = self._get_cache_key(tickers, start_date, end_date)
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        
        if self.use_cache and cache_path.exists():
            logger.info(f"Loading data from cache: {cache_path}")
            return self._load_from_cache(cache_path)
        
        # Fetch from source
        logger.info(f"Fetching data from {self.source}")
        data = self.adapter.fetch_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(data, cache_path)
            logger.info(f"Data cached to: {cache_path}")
        
        return data
    
    def _get_cache_key(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        """Generate a unique cache key for the data request."""
        content = f"{self.source}_{sorted(tickers)}_{start_date}_{end_date}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _save_to_cache(self, data: StockData, path: Path) -> None:
        """Save StockData to parquet files."""
        # Save prices
        prices_path = path.with_suffix(".prices.parquet")
        data.prices.write_parquet(prices_path)
        
        # Save returns
        returns_path = path.with_suffix(".returns.parquet")
        data.returns.write_parquet(returns_path)
        
        # Save market cap if available
        if data.market_cap is not None:
            mc_path = path.with_suffix(".market_cap.parquet")
            data.market_cap.write_parquet(mc_path)
        
        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(data.metadata, f)
    
    def _load_from_cache(self, path: Path) -> StockData:
        """Load StockData from parquet files."""
        # Load prices
        prices_path = path.with_suffix(".prices.parquet")
        prices = pl.read_parquet(prices_path)
        
        # Load returns
        returns_path = path.with_suffix(".returns.parquet")
        returns = pl.read_parquet(returns_path)
        
        # Load market cap if available
        mc_path = path.with_suffix(".market_cap.parquet")
        market_cap = None
        if mc_path.exists():
            market_cap = pl.read_parquet(mc_path)
        
        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        
        return StockData(
            prices=prices,
            returns=returns,
            market_cap=market_cap,
            metadata=metadata,
        )
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
            logger.info("Cache cleared")
