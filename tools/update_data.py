from __future__ import annotations

from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import get_prices


UNIVERSE = ["SPY", "TLT", "GLD", "QQQ", "EFA", "VNQ", "DBC", "BIL", "BTC-USD"]
START_DATE = "2005-01-01"
CACHE_PATH = Path("data/prices.parquet")


def main() -> None:
    prices = get_prices(
        tickers=UNIVERSE,
        start=START_DATE,
        end=None,
        cache_path=str(CACHE_PATH),
        refresh=False,
        incremental=True,
        strict_inception=False,
    )

    print(f"Updated cache: {CACHE_PATH}")
    print(f"Rows: {len(prices)}")
    print(f"Start: {prices.index.min()}")
    print(f"End: {prices.index.max()}")
    print(f"Columns: {', '.join(prices.columns)}")
    print(prices.tail().to_string())


if __name__ == "__main__":
    main()
