from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


def _download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download Adjusted Close prices from yfinance for the given tickers and date range.

    Returns a DataFrame indexed by trading days with columns for each ticker containing
    the Adjusted Close prices.

    References SYSTEM_SPEC.md: data source must be yfinance and use Adjusted Close only.
    This function performs a raw download and does not perform cleaning beyond basic
    construction of the price DataFrame.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is required to download price data. Install via pip.") from e

    if not tickers:
        return pd.DataFrame()

    # yfinance can accept a list; request full OHLCV then attempt to extract 'Adj Close'
    data = yf.download(tickers, start=start, end=end, progress=False, threads=True)

    if data is None or data.empty:
        return pd.DataFrame()

    adj = None

    # Case 1: top-level 'Adj Close' column exists (single-level columns)
    if isinstance(data, pd.DataFrame) and 'Adj Close' in data.columns:
        adj = data['Adj Close']

    # Case 2: MultiIndex columns where level 0 contains 'Adj Close'
    elif isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            try:
                adj = data.xs('Adj Close', axis=1, level=0, drop_level=True)
            except Exception:
                adj = None
        elif 'Adj Close' in data.columns.levels[1]:
            try:
                adj = data.xs('Adj Close', axis=1, level=1, drop_level=True)
            except Exception:
                adj = None

    # Case 3: If the DataFrame has same number of columns as tickers and numeric values,
    # assume it's already Adjusted Close (fallback)
    if adj is None:
        if isinstance(data, pd.DataFrame) and data.shape[1] == len(tickers):
            numeric = True
            for c in data.columns:
                if not pd.api.types.is_numeric_dtype(data[c].dtype):
                    numeric = False
                    break
            if numeric:
                adj = data.copy()

    # Final fallback: try selecting columns that match tickers
    if adj is None:
        cols = [c for c in data.columns if str(c) in tickers or (isinstance(c, tuple) and c[-1] in tickers)]
        if cols:
            adj = data.loc[:, cols]

    if adj is None:
        return pd.DataFrame()

    # Ensure DataFrame
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    # Normalize column names to ticker symbols where possible
    new_cols = []
    for c in adj.columns:
        if isinstance(c, tuple):
            # prefer the element that matches a ticker
            matched = None
            for part in c:
                if str(part) in tickers:
                    matched = str(part)
                    break
            new_cols.append(matched or str(c))
        else:
            new_cols.append(str(c))
    adj.columns = new_cols

    # Keep only requested tickers (preserve order)
    present = [t for t in tickers if t in adj.columns]
    adj = adj.loc[:, present]

    return adj


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price DataFrame to conform with SYSTEM_SPEC.md rules.

    Cleaning rules implemented:
    - Align all tickers to a common trading-day index
    - Forward-fill ONLY after first valid observation (ffill is per-column and will not fill
      leading NaNs)
    - Do NOT forward-fill leading NaN values
    - Drop rows where all tickers are NaN
    - Ensure dtype is float and index is timezone-naive
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a copy and ensure datetime index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows that could not be parsed as dates
    df = df[~df.index.isna()]

    # Sort ascending
    df = df.sort_index()

    # Ensure timezone-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None).tz_localize(None)

    # Forward-fill after first valid observation per column (ffill does not fill leading NaNs)
    df = df.ffill()

    # Drop rows where all tickers are NaN
    df = df.dropna(how='all')

    # Ensure dtype float
    try:
        df = df.astype(float)
    except Exception:
        # Coerce on failure
        df = df.apply(pd.to_numeric, errors='coerce')

    return df


def load_prices(tickers: List[str], start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Load Adjusted Close price series for `tickers` between `start` and `end`.

    - Returns a DataFrame of Adjusted Close prices with trading-day datetime index
      (timezone-naive) and columns equal to the requested tickers.
    - Caching: a `data_cache` folder is created in the repository root. Cache file name
      format: `prices_{start}_{end}.csv`.
    - If `use_cache` is True and the cache file exists, loads from cache. Missing tickers
      (not present in cache) are downloaded and merged, and the cache is updated.

    This function strictly implements the data rules in SYSTEM_SPEC.md. It does not
    perform any signal computation, shifting, or other strategy logic (no look-ahead).

    Raises:
        ValueError: if no data is returned for any requested ticker.
    """
    # Validate inputs
    tickers = [t for t in tickers]  # ensure list-like of strings
    if not tickers:
        raise ValueError("`tickers` must be a non-empty list of ticker symbols.")

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"prices_{start}_{end}.csv"

    df_cache = pd.DataFrame()
    if use_cache and cache_file.exists():
        try:
            df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except Exception:
            df_cache = pd.DataFrame()

    # Determine which tickers are missing from cache
    needed = [t for t in tickers if t not in df_cache.columns]

    downloaded = pd.DataFrame()
    if needed:
        downloaded = _download_prices(needed, start, end)

    # Merge cache and newly downloaded. Avoid duplicate columns after concat.
    if not df_cache.empty and not downloaded.empty:
        df = pd.concat([df_cache, downloaded], axis=1, join='outer')
        # Drop duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
    elif not df_cache.empty:
        df = df_cache
    else:
        df = downloaded

    # If cache existed but did not include desired tickers and we didn't download any data
    # (e.g., network issue), ensure df still contains something
    if df is None or df.empty:
        raise ValueError("No price data available for the requested date range.")

    # Clean
    df = _clean_prices(df)

    # If some tickers originally requested are missing entirely from df, warn.
    # Use robust checks that handle duplicate column names or unexpected structures.
    def _col_all_na(frame: pd.DataFrame, col) -> bool:
        series_or_df = frame[col]
        if isinstance(series_or_df, pd.DataFrame):
            return bool(series_or_df.isna().all().all())
        return bool(series_or_df.isna().all())

    missing_all = [t for t in tickers if (t not in df.columns) or _col_all_na(df, t)]
    for t in missing_all:
        print(f"Warning: ticker '{t}' has no data in the requested range ({start} to {end}).")

    # Ensure at least one ticker has data
    cols_with_data = [c for c in df.columns if not _col_all_na(df, c)]
    if not cols_with_data:
        raise ValueError("No price data returned for any requested ticker.")

    # Subset to requested tickers preserving column order
    available = [t for t in tickers if t in df.columns]
    result = df.reindex(columns=available)

    # Update cache: save the full merged DataFrame (not just subset) so future calls
    # can reuse existing data. Do not overwrite cache if it did not previously exist and
    # there was a download failure (df may be empty handled above).
    try:
        # When writing, ensure index has no timezone and is sorted
        to_save = df.sort_index()
        if to_save.index.tz is not None:
            to_save.index = to_save.index.tz_convert(None).tz_localize(None)
        to_save.to_csv(cache_file)
    except Exception:
        # Cache write failure should not prevent return of results
        pass

    return result


if __name__ == "__main__":
    # Smoke test per requirement
    test_tickers = ["SPY", "TLT", "GLD"]
    df = load_prices(test_tickers, "2015-01-01", "2024-01-01")
    print(df.head())
    print(df.tail())
