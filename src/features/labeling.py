import polars as pl


REQUIRED_COLUMNS = {"date", "open", "close"}


def validate_market_dataframe(df: pl.DataFrame) -> None:
    """
    Validate dataframe structure and values before labeling.
    """

    if df.is_empty():
        raise ValueError("Input dataframe is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["open"] <= 0).any():
        raise ValueError("Column 'open' contains non-positive values.")

    if (df["close"] <= 0).any():
        raise ValueError("Column 'close' contains non-positive values.")

    # Ensure chronological order
    if not df["date"].is_sorted():
        raise ValueError("Data must be sorted by date before labeling.")


def generate_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate labels for gap prediction.

    Label definition:
        1 -> Gap Up
        0 -> Gap Down

    Gap %:
        (Open_t - Close_{t-1}) / Close_{t-1}
    """

    validate_market_dataframe(df)

    df = df.with_columns(
        pl.col("close").shift(1).alias("prev_close")
    )

    df = df.with_columns(
        (
            (pl.col("open") - pl.col("prev_close"))
            / pl.col("prev_close")
        ).alias("gap")
    )

    # First row has no previous close
    df = df.drop_nulls(subset=["gap"])

    df = df.with_columns(
        (pl.col("gap") > 0)
        .cast(pl.Int8)
        .alias("label")
    )

    return df