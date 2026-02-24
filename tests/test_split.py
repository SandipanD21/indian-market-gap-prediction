import polars as pl
import pytest

from src.data.split import train_test_split
from src.config.settings import SplitConfig


def make_df(dates, values=None):
    df = pl.DataFrame({
        "date": dates,
        "val": values or list(range(len(dates)))
    }).with_columns(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    return df


def test_split_by_ratio():
    df = make_df(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    cfg = SplitConfig(split_ratio=0.5, split_date=None, train_label="train", test_label="test")
    train, test = train_test_split(df, cfg)
    assert train.height == 2
    assert test.height == 2
    assert train[0, "date"] < test[0, "date"]


def test_split_by_date():
    df = make_df(["2024-01-01", "2024-01-05", "2024-01-10"])
    cfg = SplitConfig(split_ratio=None, split_date="2024-01-05", train_label="t", test_label="s")
    train, test = train_test_split(df, cfg)
    # inclusive
    assert train.height == 2
    assert test.height == 1


def test_unsorted_raises():
    df = make_df(["2024-01-03", "2024-01-01", "2024-01-02"])
    cfg = SplitConfig(split_ratio=0.5, split_date=None, train_label="t", test_label="s")
    with pytest.raises(ValueError):
        train_test_split(df, cfg)


def test_duplicates_raises():
    df = make_df(["2024-01-01", "2024-01-01", "2024-01-02"])
    cfg = SplitConfig(split_ratio=0.5, split_date=None, train_label="t", test_label="s")
    with pytest.raises(ValueError):
        train_test_split(df, cfg)


def test_empty_input():
    cfg = SplitConfig(split_ratio=0.5, split_date=None, train_label="t", test_label="s")
    with pytest.raises(ValueError):
        train_test_split(pl.DataFrame(), cfg)


def test_ratio_edge_case():
    df = make_df(["2024-01-01", "2024-01-02"])
    cfg = SplitConfig(split_ratio=0.1, split_date=None, train_label="t", test_label="s")
    with pytest.raises(ValueError):
        train_test_split(df, cfg)


def test_date_out_of_range():
    df = make_df(["2024-01-01", "2024-01-02"])
    cfg = SplitConfig(split_ratio=None, split_date="2030-01-01", train_label="t", test_label="s")
    with pytest.raises(ValueError):
        train_test_split(df, cfg)
