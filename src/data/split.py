import polars as pl
from typing import Tuple

from src.config.settings import SplitConfig


# ---------------------------------------------------------------------------
# Train/Test split utilities
# ---------------------------------------------------------------------------


def train_test_split(
    df: pl.DataFrame, config: SplitConfig
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split a time‑series dataframe into training and testing portions.

    The helper performs a series of sanity checks before actually slicing the
    data:

    * input ``df`` must not be empty
    * ``date`` column must exist and be of a datetime-like type
    * values in ``date`` must be strictly increasing (no duplicates or
      disorder)
    * either ``config.split_ratio`` **or** ``config.split_date`` must be
      supplied (validation happens in :class:`SplitConfig`)
    * resulting train/test sets must both contain at least one row

    The split behaves as follows:

    * when ``split_ratio`` is provided the earliest ``ratio`` portion of the
      data is returned as the training set; the remainder becomes the test set
    * when ``split_date`` is provided all rows with ``date <= split_date`` go
      into the training set and the remainder into the test set

    The function returns ``(train_df, test_df)``.
    """

    # ------------------------------------------------------------------
    # basic sanity
    # ------------------------------------------------------------------
    if df.is_empty():
        raise ValueError("Input dataframe is empty; nothing to split.")

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column for splitting.")

    # make sure we have a comparable dtype (Polars will raise if not)
    try:
        ser = df.select(pl.col("date")).to_series()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to access 'date' column: %s" % exc)

    # ------------------------------------------------------------------
    # ordering / duplicates
    # ------------------------------------------------------------------
    # require strictly ascending timestamps
    if not ser.is_sorted():
        raise ValueError("'date' column must be sorted in ascending order.")

    if ser.n_unique() != ser.len():
        raise ValueError("Duplicate timestamps detected in 'date' column.")

    # ------------------------------------------------------------------
    # perform split
    # ------------------------------------------------------------------
    n = df.height

    if config.split_ratio is not None:
        # ratio case
        idx = int(n * config.split_ratio)
        # guard against empty slices
        if idx <= 0 or idx >= n:
            raise ValueError(
                "Split ratio %s produces empty train or test set (n=%s)"
                % (config.split_ratio, n)
            )

        train = df.slice(0, idx)
        test = df.slice(idx, n - idx)
    else:
        # date case
        try:
            # convert provided string to datetime using polars parser
            split_val = (
                pl.Series([config.split_date])
                .str.strptime(pl.Datetime, strict=True)
                .item()
            )
        except Exception as exc:
            raise ValueError(f"split_date could not be parsed: {exc}")

        train = df.filter(pl.col("date") <= split_val)
        test = df.filter(pl.col("date") > split_val)

    # ------------------------------------------------------------------
    # metadata checks
    # ------------------------------------------------------------------
    if train.is_empty() or test.is_empty():
        raise ValueError(
            "Split produced empty dataset: train size %s, test size %s"
            % (train.height, test.height)
        )

    return train, test
