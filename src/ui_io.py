from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Dict, Iterable, Tuple

import pandas as pd


def compute_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def load_prices_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Load prices from CSV or parquet bytes into a DataFrame with DatetimeIndex.

    Does not mutate input beyond ensuring index is DatetimeIndex and sorted.
    """
    bio = BytesIO(data)
    if filename.lower().endswith(('.parquet', '.pq')):
        df = pd.read_parquet(bio)
    else:
        df = pd.read_csv(bio, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # sort and ensure unique index for downstream determinism
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def validate_prices(df: pd.DataFrame, required_tickers: Iterable[str] = ("BIL",)) -> Dict[str, object]:
    """Return a dict summarizing validation checks for the prices DataFrame."""
    out = {}
    if df is None:
        out['loaded'] = False
        out['error'] = 'no dataframe provided'
        return out

    out['loaded'] = True
    out['is_datetime_index'] = isinstance(df.index, pd.DatetimeIndex)
    out['monotonic_increasing'] = bool(df.index.is_monotonic_increasing)
    out['has_duplicates'] = bool(df.index.duplicated().any())
    out['n_rows'] = int(len(df))
    out['n_columns'] = int(df.shape[1])
    out['missing_values'] = int(df.isna().sum().sum())
    # check required tickers
    req = list(required_tickers)
    present = [t for t in req if t in df.columns]
    missing = [t for t in req if t not in df.columns]
    out['required_present'] = present
    out['required_missing'] = missing
    return out


def config_to_tuple(cfg: Dict) -> Tuple[Tuple[str, object], ...]:
    """Convert config dict to a sorted, hashable tuple for caching keys."""
    return tuple(sorted(cfg.items()))
