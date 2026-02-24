import polars as pl
from typing import Iterable


# -------------------------------------------------------
# helpers
# -------------------------------------------------------

def _validate_windows(df: pl.DataFrame, windows: Iterable[int]) -> None:
    """
    Ensure rolling windows are usable for the dataset.
    Fail hard if they are not.
    """
    n = df.height

    if n < 10:
        raise ValueError("Dataset too small to compute rolling features.")

    for w in windows:
        if w <= 1:
            raise ValueError(f"Rolling window must be >1. Got {w}")

        if w >= n:
            raise ValueError(
                f"Rolling window {w} is larger than dataset ({n})."
            )


# -------------------------------------------------------
# feature builder
# -------------------------------------------------------

def build_rolling_features(
    df: pl.DataFrame,
    windows: Iterable[int] = (5, 10, 20)
) -> pl.DataFrame:
    """
    Create rolling features using ONLY past data.

    Important:
    Features for time T use data only up to T-1.
    """

    if df.is_empty():
        raise ValueError("Input dataframe is empty.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    windows = sorted(set(windows))
    _validate_windows(df, windows)

    # ---------------------------------------------------
    # lagged base columns (prevents leakage)
    # ---------------------------------------------------

    df = df.with_columns(
        [
            pl.col("close").shift(1).alias("prev_close"),
            pl.col("volume").shift(1).alias("prev_volume"),
            pl.col("high").shift(1).alias("prev_high"),
            pl.col("low").shift(1).alias("prev_low"),
        ]
    )

    # ---------------------------------------------------
    # overnight gap
    # ---------------------------------------------------

    df = df.with_columns(
        (
            (pl.col("open") - pl.col("prev_close"))
            / pl.col("prev_close")
        ).alias("overnight_gap")
    )

    # ---------------------------------------------------
    # daily return (lagged)
    # ---------------------------------------------------

    df = df.with_columns(
        (
            (pl.col("prev_close") - pl.col("prev_close").shift(1))
            / pl.col("prev_close").shift(1)
        ).alias("prev_return")
    )

    # ---------------------------------------------------
    # rolling features
    # ---------------------------------------------------

    feature_exprs = []

    for w in windows:

        feature_exprs.extend(
            [
                pl.col("prev_return")
                .rolling_mean(w)
                .alias(f"return_mean_{w}"),

                pl.col("prev_return")
                .rolling_std(w)
                .alias(f"return_vol_{w}"),

                pl.col("prev_volume")
                .rolling_mean(w)
                .alias(f"volume_mean_{w}"),

                pl.col("prev_volume")
                .rolling_std(w)
                .alias(f"volume_vol_{w}"),
            ]
        )

    df = df.with_columns(feature_exprs)

    # ---------------------------------------------------
    # ATR style volatility
    # ---------------------------------------------------

    df = df.with_columns(
        (
            (pl.col("prev_high") - pl.col("prev_low"))
            / pl.col("prev_close")
        ).alias("range_pct")
    )

    for w in windows:
        df = df.with_columns(
            pl.col("range_pct")
            .rolling_mean(w)
            .alias(f"range_mean_{w}")
        )

    return df