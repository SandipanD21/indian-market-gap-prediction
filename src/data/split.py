import math
import polars as pl
from typing import Tuple

from src.config.settings import SplitConfig


def train_test_split(
    df: pl.DataFrame, config: SplitConfig
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Chronologically split a time-series dataframe into train and test sets.

    Assumptions
    ----------
    - Data is already cleaned and sorted by date by the loader.
    - 'date' column exists and is datetime.
    """

    # --------------------------------------------------
    # basic validation
    # --------------------------------------------------
    if df.is_empty():
        raise ValueError("Input dataframe is empty.")

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    n = df.height

    # --------------------------------------------------
    # ratio based split
    # --------------------------------------------------
    if config.split_ratio is not None:

        idx = math.floor(n * config.split_ratio)

        if idx <= 0 or idx >= n:
            raise ValueError(
                f"Split ratio {config.split_ratio} produces empty train/test (n={n})"
            )

        train = df.slice(0, idx)
        test = df.slice(idx)

    # --------------------------------------------------
    # date based split
    # --------------------------------------------------
    else:
        try:
            split_dt = (
                pl.Series([config.split_date])
                .str.strptime(pl.Datetime, strict=True)
                .item()
            )
        except Exception as exc:
            raise ValueError(f"Invalid split_date: {exc}")

        train = df.filter(pl.col("date") <= split_dt)
        test = df.filter(pl.col("date") > split_dt)

        if train.is_empty() or test.is_empty():
            raise ValueError(
                f"Split date {config.split_date} results in empty dataset "
                f"(train={train.height}, test={test.height})"
            )

    return train, test