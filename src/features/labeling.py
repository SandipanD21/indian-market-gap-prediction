import polars as pl


REQUIRED_COLUMNS = {"trade_date", "open_price", "close_price"}


def validate_market_dataframe(df: pl.DataFrame) -> None:
    """
    Validates required columns and basic constraints.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.height == 0:
        raise ValueError("Input dataframe is empty.")

    if (df["open_price"] <= 0).any():
        raise ValueError("open_price contains non-positive values.")

    if (df["close_price"] <= 0).any():
        raise ValueError("close_price contains non-positive values.")


def generate_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates binary labels:
        1 -> GAP_UP
        0 -> GAP_DOWN
    """

    validate_market_dataframe(df)

    df = df.sort("trade_date")

    df = df.with_columns(
        (
            (pl.col("open_price") - pl.col("close_price").shift(1))
            / pl.col("close_price").shift(1)
            * 100
        ).alias("gap_percent")
    )

    # Remove first row where shift created null
    df = df.drop_nulls(subset=["gap_percent"])

    df = df.with_columns(
        (pl.col("gap_percent") > 0)
        .cast(pl.Int8)
        .alias("label")
    )

    return df
