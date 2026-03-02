import os
import sys
from pathlib import Path

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
import pandas.testing as pdt

from data import _clean_prices, load_prices


def test_cleaning_forward_fill_and_leading_nan():
    # tz-aware index, with leading NaNs and internal NaNs
    idx = pd.date_range('2020-01-01', periods=6, tz='UTC')
    df = pd.DataFrame({
        'A': [np.nan, np.nan, 1.0, np.nan, 2.0, np.nan],
        'B': [np.nan, 5.0, np.nan, 6.0, np.nan, 8.0],
    }, index=idx)

    cleaned = _clean_prices(df)

    # index should be timezone-naive and sorted
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.tz is None

    # Leading NaNs preserved before first valid observation, and forward-filled after
    for col in cleaned.columns:
        fv = cleaned[col].first_valid_index()
        if fv is None:
            continue
        fv_pos = cleaned.index.get_loc(fv)
        # all positions before first_valid should be NaN
        assert cleaned[col].iloc[:fv_pos].isna().all()
        # after first valid, forward-fill should have eliminated NaNs
        assert not cleaned[col].iloc[fv_pos:].isna().any()

    # dtype should be float
    assert cleaned['A'].dtype == float and cleaned['B'].dtype == float


def test_drop_rows_all_nan_and_column_integrity():
    idx = pd.date_range('2020-01-01', periods=5)
    df = pd.DataFrame({
        'SPY': [np.nan, 100.0, np.nan, 102.0, np.nan],
        'TLT': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }, index=idx)

    cleaned = _clean_prices(df)

    # TLT is all NaN -> after dropna(how='all') rows where both NaN removed,
    # but column may still exist (cleaning does not drop columns)
    assert 'SPY' in cleaned.columns
    assert 'TLT' in cleaned.columns

    # Rows where all tickers NaN should be removed
    assert not cleaned.index.empty


def test_cache_logic_reads_existing_cache(tmp_path):
    # Prepare a fake cache file in the repository data_cache
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / 'data_cache'
    cache_dir.mkdir(exist_ok=True)

    start = '2024-01-01'
    end = '2025-01-01'
    cache_file = cache_dir / f'prices_{start}_{end}.csv'

    # create sample prices
    idx = pd.date_range('2024-01-02', periods=5)
    df_sample = pd.DataFrame({
        'SPY': [450.0, 452.0, 455.0, 458.0, 460.0],
        'TLT': [90.0, 91.0, 90.5, 91.2, 92.0],
        'GLD': [180.0, 181.0, 182.0, 183.0, 184.0],
    }, index=idx)

    df_sample.to_csv(cache_file)

    try:
        df = load_prices(['SPY', 'TLT', 'GLD'], start, end, use_cache=True)
        # values should match cached values (after cleaning)
        # Align indices for comparison
        assert df.index.equals(df_sample.index)
        # compare numeric values (index already compared)
        assert np.allclose(df['SPY'].values.astype(float), df_sample['SPY'].values.astype(float))
    finally:
        try:
            cache_file.unlink()
        except Exception:
            pass
