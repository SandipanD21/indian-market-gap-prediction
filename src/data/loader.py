import polars as pl
from pathlib import Path
from typing import Dict

from src.config.settings import DataConfig


# ----------------------------
# Column alias normalization
# ----------------------------

COLUMN_ALIASES: Dict[str, str] = {
    "Date": "date",
    "date": "date",

    "Open": "open",
    "open": "open",

    "High": "high",
    "high": "high",

    "Low": "low",
    "low": "low",

    "Close": "close",
    "close": "close",
    "Adj Close": "close",

    "Volume": "volume",
    "volume": "volume",
}


CANONICAL_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


# ----------------------------
# Loader
# ----------------------------

def load_ohlcv(config: DataConfig) -> pl.DataFrame:
    """
    Load and normalize OHLCV data.

    Guarantees:
    - canonical column names
    - required columns exist
    - datetime parsed
    - no null values
    - logically valid OHLC
    - sorted by date
    - returns only canonical columns
    """

    # ----------------------------
    # File existence
    # ----------------------------
    data_path = Path(config.data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {config.data_path}")

    # ----------------------------
    # Read CSV
    # ----------------------------
    df = pl.read_csv(data_path)

    if df.is_empty():
        raise ValueError("CSV file is empty.")

    # ----------------------------
    # Apply alias mapping
    # ----------------------------
    rename_map = {
        col: COLUMN_ALIASES[col]
        for col in df.columns
        if col in COLUMN_ALIASES
    }

    df = df.rename(rename_map)

    # ----------------------------
    # Ensure required columns
    # ----------------------------
    missing = set(config.required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns after alias mapping: {missing}"
        )

    # ----------------------------
    # Restrict to canonical columns
    # ----------------------------
    df = df.select(list(CANONICAL_COLUMNS & set(df.columns)))

    # ----------------------------
    # Parse date column
    # ----------------------------
    try:
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Datetime, strict=True)
        )
    except Exception as e:
        raise ValueError(f"Date parsing failed: {e}")

    # ----------------------------
    # Ensure numeric types
    # ----------------------------
    df = df.with_columns([
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ])

    # ----------------------------
    # Null checks
    # ----------------------------
    null_counts = df.select(
        [pl.col(c).null_count().alias(c) for c in df.columns]
    ).row(0)

    if any(v > 0 for v in null_counts):
        raise ValueError(
            f"Null values detected in OHLCV columns: {null_counts}"
        )

    # ----------------------------
    # Logical OHLC validation
    # ----------------------------
    invalid_rows = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low")) |
        (pl.col("low") > pl.min_horizontal("open", "close", "high"))
    )

    if invalid_rows.height > 0:
        raise ValueError(
            f"Invalid OHLC values detected in {invalid_rows.height} rows."
        )

    # ----------------------------
    # Sort
    # ----------------------------
    df = df.sort("date", descending=not config.sort_ascending)

    return df