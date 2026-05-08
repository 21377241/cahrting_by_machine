"""CRSP Stock Monthlies local file adapter.

Reads the ``crsp2525(monthly).csv`` (or its Parquet equivalent) and converts
the CRSP long-format panel into the wide-format ``StockData`` expected by the
``cbm`` pipeline—**zero changes to any existing source file are required**.

Supported universes (pass to ``DataManager.load_data(universe=...)``):
    ``crsp_all``      – NYSE + AMEX + NASDAQ ordinary shares
    ``crsp_nyse``     – NYSE ordinary shares only
    ``crsp_amex``     – AMEX ordinary shares only
    ``crsp_nasdaq``   – NASDAQ ordinary shares only

Typical usage::

    from cbm.data.manager import DataManager

    mgr = DataManager(source="crsp_local", cache_dir="./data/cache")
    data = mgr.load_data(
        universe="crsp_all",
        start_date="1963-07-01",
        end_date="2022-12-31",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from loguru import logger

from cbm.core.types import StockData
from cbm.data.adapters.base import DataAdapter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root
_DEFAULT_CSV = _PKG_ROOT / "crsp_data" / "crsp2525(monthly).csv"
_DEFAULT_PARQUET = _PKG_ROOT / "crsp_data" / "crsp2525_monthly.parquet"

# ---------------------------------------------------------------------------
# Constants: universe → PrimaryExch codes
# ---------------------------------------------------------------------------

#: Map from universe sentinel string (returned by ``get_universe``) to the list
#: of ``PrimaryExch`` values that should be included in the sample.
_SENTINEL_EXCHANGES: Dict[str, List[str]] = {
    "__CRSP_ALL__":     ["N", "A", "Q", "P"],
    "__CRSP_NYSE__":    ["N"],
    "__CRSP_AMEX__":    ["A"],
    "__CRSP_NASDAQ__":  ["Q", "P"],
}

_UNIVERSE_SENTINELS: Dict[str, str] = {
    "crsp_all":     "__CRSP_ALL__",
    "crsp_nyse":    "__CRSP_NYSE__",
    "crsp_amex":    "__CRSP_AMEX__",
    "crsp_nasdaq":  "__CRSP_NASDAQ__",
}

# CRSP PrimaryExch string → traditional EXCHCD integer (kept in metadata)
_EXCHCD_MAP: Dict[str, int] = {"N": 1, "A": 2, "Q": 3, "P": 3, "Z": 3, "C": 3}

# DelReasonType codes that correspond to forced (involuntary) delistings.
# Codes 500–591 in the CRSP tradition map to these prefix strings.
_FORCED_DELIST_PREFIXES = ("5",)


class CRSPLocalAdapter(DataAdapter):
    """Data adapter for a local CRSP Stock Monthlies CSV (or Parquet) file.

    Parameters
    ----------
    data_path : Path, optional
        Explicit path to the ``.csv`` or ``.parquet`` file.  When omitted the
        adapter first looks for the pre-converted Parquet file next to the CSV
        (much faster), then falls back to the raw CSV.
    identifier : str
        Column used as the ticker/column identifier in the resulting wide
        tables.  ``"PERMNO"`` (default) is stable across name changes;
        ``"Ticker"`` is human-readable but may vary over time.
    filter_ordinary_shares : bool
        When ``True`` (default) retain only rows where
        ``USIncFlg == "Y"`` **and** ``SecurityType == "EQTY"``, which
        approximates the paper's ``SHRCD in {10, 11}`` filter.
    infer_schema_length : int
        Passed to ``pl.scan_csv`` for type inference (default: 50 000).
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        identifier: str = "PERMNO",
        filter_ordinary_shares: bool = True,
        infer_schema_length: int = 50_000,
    ) -> None:
        if data_path is not None:
            self.data_path = Path(data_path)
        elif _DEFAULT_PARQUET.exists():
            self.data_path = _DEFAULT_PARQUET
            logger.info(f"Using pre-converted Parquet: {self.data_path}")
        elif _DEFAULT_CSV.exists():
            self.data_path = _DEFAULT_CSV
            logger.info(f"Using raw CSV (slow first load): {self.data_path}")
        else:
            raise FileNotFoundError(
                f"CRSP data file not found. Expected:\n"
                f"  Parquet: {_DEFAULT_PARQUET}\n"
                f"  CSV    : {_DEFAULT_CSV}\n"
                "Run  crsp_data/convert_to_parquet.py  to create the Parquet file."
            )

        self.identifier = identifier
        self.filter_ordinary_shares = filter_ordinary_shares
        self.infer_schema_length = infer_schema_length
        self._is_parquet = self.data_path.suffix.lower() == ".parquet"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan(self) -> pl.LazyFrame:
        """Return a lazy frame over the data file."""
        if self._is_parquet:
            return pl.scan_parquet(self.data_path)
        return pl.scan_csv(
            self.data_path,
            infer_schema_length=self.infer_schema_length,
            null_values=["", ".", "NA", "NaN", "C", "B"],
        )

    @staticmethod
    def _to_yyyymm(date_str: str) -> int:
        """Convert ``"YYYY-MM-DD"`` or ``"YYYY-MM"`` to integer ``YYYYMM``."""
        parts = date_str.split("-")
        return int(parts[0]) * 100 + int(parts[1])

    def _base_filter(
        self,
        lf: pl.LazyFrame,
        start_ym: int,
        end_ym: int,
        exchanges: List[str],
    ) -> pl.LazyFrame:
        """Apply time, exchange, and (optionally) share-type filters."""
        lf = lf.filter(
            (pl.col("YYYYMM") >= start_ym) & (pl.col("YYYYMM") <= end_ym)
        )
        lf = lf.filter(pl.col("PrimaryExch").is_in(exchanges))
        if self.filter_ordinary_shares:
            lf = lf.filter(
                (pl.col("USIncFlg") == "Y") & (pl.col("SecurityType") == "EQTY")
            )
        return lf

    @staticmethod
    def _apply_delisting_adjustment(df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing ``MthRet`` with ``-0.30`` for forced delistings.

        This approximates the Shumway (1997) adjustment used in the paper.
        Forced delistings are identified by ``DelReasonType`` starting with
        ``"5"`` (CRSP codes 500–591) when ``MthRet`` is null.
        """
        return df.with_columns(
            pl.when(
                pl.col("ret").is_null()
                & pl.col("del_reason").cast(pl.Utf8).str.starts_with("5")
            )
            .then(pl.lit(-0.30))
            .otherwise(pl.col("ret"))
            .alias("ret")
        )

    # ------------------------------------------------------------------
    # DataAdapter interface
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        handle_delisting: bool = True,
        **kwargs,
    ) -> StockData:
        """Load CRSP data and return wide-format ``StockData``.

        Parameters
        ----------
        tickers
            Either a sentinel returned by :meth:`get_universe` (e.g.
            ``["__CRSP_ALL__"]``) **or** a list of explicit PERMNO strings
            (e.g. ``["10001", "14593"]``).
        start_date, end_date
            Date range.  Format ``"YYYY-MM-DD"`` or ``"YYYY-MM"``.
        handle_delisting
            When ``True`` (default), fill missing ``MthRet`` with ``-0.30``
            for forced delistings (Shumway 1997 approximation).

        Returns
        -------
        StockData
            Wide-format container:

            * ``prices``    – ``date × PERMNO`` monthly closing prices
            * ``returns``   – ``date × PERMNO`` monthly total returns (``MthRet``)
            * ``market_cap`` – ``date × PERMNO`` month-end market cap (``MthCap``)
        """
        start_ym = self._to_yyyymm(start_date)
        end_ym = self._to_yyyymm(end_date)

        # Resolve exchange filter from sentinel or default to all
        if len(tickers) == 1 and tickers[0] in _SENTINEL_EXCHANGES:
            exchanges = _SENTINEL_EXCHANGES[tickers[0]]
            permno_filter: Optional[List[int]] = None
        else:
            exchanges = ["N", "A", "Q", "P"]
            permno_filter = []
            for t in tickers:
                try:
                    permno_filter.append(int(t))
                except ValueError:
                    logger.warning(f"Ignoring non-integer PERMNO: {t!r}")

        logger.info(
            f"Loading CRSP: {start_date} → {end_date}, "
            f"exchanges={exchanges}, "
            f"permno_filter={'all' if permno_filter is None else len(permno_filter)}"
        )

        lf = self._scan()
        lf = self._base_filter(lf, start_ym, end_ym, exchanges)

        if permno_filter:
            lf = lf.filter(pl.col("PERMNO").is_in(permno_filter))

        # Select and rename to internal names (minimise memory before collect)
        lf = lf.select(
            [
                pl.col("MthCalDt").cast(pl.Date).alias("date"),
                pl.col("PERMNO").cast(pl.Utf8).alias("_id"),
                pl.col("MthPrc").abs().alias("price"),
                pl.col("MthRet").alias("ret"),
                pl.col("MthCap").alias("market_cap"),
                pl.col("MthDelFlg").alias("del_flg"),
                # DelReasonType may be null for non-delisted rows
                pl.col("DelReasonType").cast(pl.Utf8).alias("del_reason"),
            ]
        )

        df = lf.collect()

        if df.is_empty():
            raise ValueError(
                "No rows returned from CRSP file for the given filters.\n"
                f"  start_ym={start_ym}, end_ym={end_ym}, exchanges={exchanges}"
            )

        logger.info(f"Collected {df.height:,} long-format rows; pivoting…")

        # Delisting adjustment before pivot (operates on long format)
        if handle_delisting:
            df = self._apply_delisting_adjustment(df)

        # Keep only the columns needed for pivot (drop helper cols)
        df_pivot = df.select(["date", "_id", "price", "ret", "market_cap"])

        # Long → Wide pivot for each metric
        prices_wide = (
            df_pivot.pivot(values="price", index="date", on="_id", aggregate_function="first")
            .sort("date")
        )
        returns_wide = (
            df_pivot.pivot(values="ret", index="date", on="_id", aggregate_function="first")
            .sort("date")
        )
        mcap_wide = (
            df_pivot.pivot(values="market_cap", index="date", on="_id", aggregate_function="first")
            .sort("date")
        )

        n_tickers = len([c for c in returns_wide.columns if c != "date"])
        n_months = returns_wide.height

        logger.info(
            f"CRSP StockData ready: {n_tickers:,} securities × {n_months} months"
        )

        return StockData(
            prices=prices_wide,
            returns=returns_wide,
            market_cap=mcap_wide,
            metadata={
                "source": "crsp_local",
                "file": str(self.data_path),
                "identifier": self.identifier,
                "start_date": start_date,
                "end_date": end_date,
                "n_tickers": n_tickers,
                "n_months": n_months,
                "exchanges": exchanges,
                "handle_delisting": handle_delisting,
                "filter_ordinary_shares": self.filter_ordinary_shares,
            },
        )

    def get_universe(self, universe_name: str) -> List[str]:
        """Return a sentinel that encodes the requested CRSP universe.

        The sentinel is interpreted by :meth:`fetch_data` to apply the
        correct exchange filter without additional round-trips to the file.

        Supported names
        ---------------
        ``crsp_all``, ``crsp_nyse``, ``crsp_amex``, ``crsp_nasdaq``
        """
        name_lower = universe_name.lower()
        if name_lower not in _UNIVERSE_SENTINELS:
            available = ", ".join(_UNIVERSE_SENTINELS.keys())
            raise ValueError(
                f"Unknown CRSP universe '{universe_name}'. "
                f"Available: {available}"
            )
        return [_UNIVERSE_SENTINELS[name_lower]]

    def fetch_risk_free_rate(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Return a zero risk-free rate series covering the requested period.

        The CRSP file does not include Tbill rates.  For proper excess-return
        calculation supply Ken French's monthly RF data separately and pass it
        to ``FeatureEngineer.create_features(risk_free_rate=rf_df)``.

        The returned DataFrame has columns ``date`` (``pl.Date``) and
        ``rate`` (``pl.Float64``), matching the interface expected by
        ``FeatureEngineer``.
        """
        start_ym = self._to_yyyymm(start_date)
        end_ym = self._to_yyyymm(end_date)

        dates_df = (
            self._scan()
            .filter(
                (pl.col("YYYYMM") >= start_ym) & (pl.col("YYYYMM") <= end_ym)
            )
            .select(pl.col("MthCalDt").cast(pl.Date).alias("date"))
            .unique("date")
            .sort("date")
            .collect()
        )

        return dates_df.with_columns(pl.lit(0.0).alias("rate"))

    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """For CRSP, all PERMNO strings in the file are considered valid."""
        # Sentinels are always valid
        if len(tickers) == 1 and tickers[0] in _SENTINEL_EXCHANGES:
            return tickers
        # For explicit PERMNOs we trust the caller; validation would require
        # a full file scan which is expensive.
        return tickers
