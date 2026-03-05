import io
import hashlib
import json

import pytest
import pandas as pd

from src import ui_io


def test_compute_sha1():
    b = b'hello world'
    h = ui_io.compute_sha1(b)
    assert h == hashlib.sha1(b).hexdigest()


def test_load_prices_bytes_csv_and_duplicates_removed():
    # create sample dataframe with duplicate dates and unordered
    idx = pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02', '2020-01-02'])
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    buf = io.BytesIO()
    df.to_csv(buf)
    data = buf.getvalue()

    out = ui_io.load_prices_bytes(data, 'sample.csv')
    # index should be DatetimeIndex, sorted, and duplicates removed
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert len(out.index) == 3


def test_validate_prices_present_and_missing_tickers():
    idx = pd.date_range('2020-01-01', periods=3, freq='D')
    df = pd.DataFrame({'SPY': [1, 2, 3], 'BIL': [0.99, 0.98, 0.97]}, index=idx)
    v = ui_io.validate_prices(df, required_tickers=('BIL', 'FOO'))
    assert v['loaded'] is True
    assert v['is_datetime_index'] is True
    assert v['required_present'] == ['BIL']
    assert 'FOO' in v['required_missing']


def test_config_to_tuple_is_deterministic():
    a = {'b': 2, 'a': 1}
    t1 = ui_io.config_to_tuple(a)
    t2 = ui_io.config_to_tuple({'a': 1, 'b': 2})
    assert t1 == t2


def test_load_prices_parquet_if_supported():
    # attempt to write parquet; skip if engine not available
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pytest.skip('pyarrow not available for parquet test')

    idx = pd.date_range('2020-01-01', periods=2, freq='D')
    df = pd.DataFrame({'X': [1.0, 2.0]}, index=idx)
    buf = io.BytesIO()
    df.to_parquet(buf)
    data = buf.getvalue()
    out = ui_io.load_prices_bytes(data, 'file.parquet')
    assert list(out.columns) == ['X']
    assert len(out) == 2


def test_slice_prices_strict_inception():
    idx = pd.date_range('2020-01-01', periods=4, freq='D')
    df = pd.DataFrame({
        'SPY': [1.0, 2.0, 3.0, 4.0],
        'TLT': [float('nan'), 5.0, 6.0, 7.0],
    }, index=idx)

    out = ui_io.slice_prices(df, tickers=('SPY', 'TLT'), start='2020-01-01', end=None, strict_inception=True)
    assert out.index.min() == pd.Timestamp('2020-01-02')


def test_parse_json_config_from_string():
    cfg = {'target_vol': 0.1, 'band': 0.05}
    out = ui_io.parse_json_config(json.dumps(cfg))
    assert out == cfg
