"""One-time preprocessing: convert crsp2525(monthly).csv → Parquet.

Run this script **once** before using CRSPLocalAdapter.  The resulting
``crsp2525_monthly.parquet`` is ~80-90% smaller than the CSV and loads
10-30× faster with Polars.

Usage::

    cd E:\\phd\\LLM_trading\\charting_by_machine
    python crsp_data/convert_to_parquet.py

Expected runtime: 3-8 minutes (depends on disk speed).
Expected output size: 200-400 MB (vs 2.46 GB CSV).
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
CSV_PATH = _HERE / "crsp2525(monthly).csv"
PARQUET_PATH = _HERE / "crsp2525_monthly.parquet"

# Columns that should remain as strings (to preserve leading zeros, codes, etc.)
_STRING_COLS = {
    "HdrCUSIP", "HdrCUSIP9", "CUSIP", "CUSIP9",
    "PrimaryExch", "ConditionalType", "ExchangeTier", "TradingStatusFlg",
    "SecurityNm", "ShareClass", "USIncFlg", "IssuerType", "SecurityType",
    "SecuritySubType", "ShareType", "SecurityActiveFlg",
    "DelActionType", "DelStatusType", "DelReasonType", "DelPaymentType",
    "Ticker", "TradingSymbol", "IssuerNm",
    "MthCompFlg", "MthCompSubFlg", "MthPrcFlg", "MthDtFlg", "MthDelFlg",
    "MthVolFlg", "MthFacShrFlg", "ShrSource", "ShrFacType", "ShrAdrFlg",
    "DisOrdinaryFlg", "DisType", "DisFreqType", "DisPaymentType",
    "DisDetailType", "DisTaxType", "DisOrigCurType",
}

# Columns to parse as Date
_DATE_COLS = {
    "SecInfoStartDt", "SecInfoEndDt", "SecurityBegDt", "SecurityEndDt",
    "MthCalDt", "MthPrcDt", "MthPrevDt",
    "ShrStartDt", "ShrEndDt",
    "DisExDt", "DisDeclareDt", "DisRecordDt", "DisPayDt",
}


def _build_schema_overrides() -> dict:
    """Return dtype overrides for scan_csv."""
    overrides: dict = {}
    for col in _STRING_COLS:
        overrides[col] = pl.Utf8
    # Date columns: keep as Utf8 during scan, cast after (scan_csv date parsing
    # can be fragile with mixed CRSP formats)
    for col in _DATE_COLS:
        overrides[col] = pl.Utf8
    return overrides


def convert(
    csv_path: Path = CSV_PATH,
    parquet_path: Path = PARQUET_PATH,
    compression: str = "zstd",
    row_group_size: int = 200_000,
) -> None:
    """Convert CSV to Parquet with proper types.

    Parameters
    ----------
    csv_path
        Source CSV file.
    parquet_path
        Destination Parquet file.
    compression
        Parquet compression codec.  ``"zstd"`` gives the best size/speed
        trade-off; alternatives: ``"snappy"``, ``"lz4"``, ``"uncompressed"``.
    row_group_size
        Polars row-group size hint for the Parquet writer.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Source  : {csv_path}  ({csv_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Target  : {parquet_path}")
    print(f"Codec   : {compression}")
    print()

    t0 = time.perf_counter()

    print("Step 1/3  Scanning CSV and casting date columns…")
    schema_overrides = _build_schema_overrides()

    lf = pl.scan_csv(
        csv_path,
        infer_schema_length=100_000,
        null_values=["", ".", "NA", "NaN"],
        schema_overrides=schema_overrides,
    )

    # Cast Utf8 date columns to pl.Date where possible
    date_casts = []
    for col in _DATE_COLS:
        date_casts.append(
            pl.col(col)
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias(col)
        )

    lf = lf.with_columns(date_casts)

    print("Step 2/3  Writing Parquet (streaming)…")
    lf.sink_parquet(
        parquet_path,
        compression=compression,
        row_group_size=row_group_size,
    )

    elapsed = time.perf_counter() - t0
    size_mb = parquet_path.stat().st_size / 1e6

    print(f"Step 3/3  Done in {elapsed:.1f}s")
    print()
    print(f"Output size : {size_mb:.1f} MB")
    print(
        f"Compression : {csv_path.stat().st_size / parquet_path.stat().st_size:.1f}× "
        f"vs original CSV"
    )
    print()
    print("You can now use  DataManager(source='crsp_local')  with full speed.")


def verify(parquet_path: Path = PARQUET_PATH, n_sample: int = 5) -> None:
    """Quick sanity check: print schema + first rows of key columns."""
    print(f"\nVerifying {parquet_path} …")
    df = pl.scan_parquet(parquet_path).head(n_sample).collect()

    key_cols = [
        c for c in [
            "PERMNO", "YYYYMM", "MthCalDt", "MthPrc", "MthRet",
            "MthCap", "ShrOut", "PrimaryExch", "USIncFlg", "SecurityType",
            "MthDelFlg", "DelReasonType", "vwretd",
        ]
        if c in df.columns
    ]

    print(f"\nSchema (selected columns):")
    for col in key_cols:
        dtype = df.schema[col]
        print(f"  {col:<22} {dtype}")

    print(f"\nFirst {n_sample} rows (selected columns):")
    print(df.select(key_cols))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CRSP monthly CSV to Parquet"
    )
    parser.add_argument(
        "--csv", type=Path, default=CSV_PATH,
        help="Path to source CSV file",
    )
    parser.add_argument(
        "--parquet", type=Path, default=PARQUET_PATH,
        help="Path for output Parquet file",
    )
    parser.add_argument(
        "--compression", default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After conversion, print a schema+sample verification",
    )
    args = parser.parse_args()

    convert(args.csv, args.parquet, compression=args.compression)

    if args.verify:
        verify(args.parquet)
