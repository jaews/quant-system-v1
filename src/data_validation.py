from __future__ import annotations

from typing import List

import pandas as pd


def validate_price_frame(prices: pd.DataFrame) -> None:
    """Raise if the price frame violates core invariants."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices.index must be a DatetimeIndex")
    if prices.index.tz is not None:
        raise ValueError("prices.index must be timezone-naive")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index must be strictly increasing")
    if prices.index.has_duplicates:
        raise ValueError("prices.index must not contain duplicate dates")
    if prices.columns.has_duplicates:
        raise ValueError("prices.columns must not contain duplicates")


def report_missing_days(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per-ticker missing-value counts and ratios."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    total_rows = max(len(prices.index), 1)
    missing = prices.isna().sum().astype(int)
    report = pd.DataFrame({
        "missing_days": missing,
        "missing_ratio": missing.astype(float) / float(total_rows),
    })
    return report


def report_inception_dates(prices: pd.DataFrame) -> pd.Series:
    """Return the first valid date for each ticker."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")

    data = {}
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        data[str(col)] = pd.Timestamp(first_valid) if first_valid is not None else pd.NaT
    return pd.Series(data, dtype="datetime64[ns]")


def detect_large_gaps(prices: pd.DataFrame, min_gap_days: int = 5) -> pd.DataFrame:
    """Return missing-data gaps of at least `min_gap_days` within valid history."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    if min_gap_days < 1:
        raise ValueError("min_gap_days must be >= 1")

    records: List[dict] = []
    for col in prices.columns:
        series = prices[col]
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        if first_valid is None or last_valid is None:
            continue

        window = series.loc[first_valid:last_valid]
        is_missing = window.isna()
        if not is_missing.any():
            continue

        group_ids = is_missing.ne(is_missing.shift(fill_value=False)).cumsum()
        for _, mask in is_missing.groupby(group_ids):
            if not bool(mask.iloc[0]):
                continue
            gap_len = int(mask.sum())
            if gap_len < min_gap_days:
                continue
            records.append({
                "ticker": str(col),
                "start": pd.Timestamp(mask.index[0]),
                "end": pd.Timestamp(mask.index[-1]),
                "length": gap_len,
            })

    if not records:
        return pd.DataFrame(columns=["ticker", "start", "end", "length"])

    return pd.DataFrame.from_records(records, columns=["ticker", "start", "end", "length"])
