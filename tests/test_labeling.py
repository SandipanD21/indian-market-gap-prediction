import polars as pl
from src.features.labeling import generate_labels


def test_generate_labels_basic_case():
    df = pl.DataFrame(
        {
            "trade_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "open_price": [100, 105, 102],
            "close_price": [100, 100, 103],
        }
    ).with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    labeled = generate_labels(df)

    assert labeled.height == 2

    # Day 2: 105 vs 100 -> GAP_UP
    assert labeled[0, "label"] == 1

    # Day 3: 102 vs 100 -> GAP_UP
    assert labeled[1, "label"] == 1
