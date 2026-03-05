import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import pandas.testing as pdt

from data import _clean_prices, _extract_adj_close, get_prices


def test_get_prices_loads_cache_only_without_network(tmp_path, monkeypatch):
    cache_path = tmp_path / "prices.parquet"
    idx = pd.bdate_range("2020-01-01", periods=3)
    cached = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "TLT": [50.0, 50.5, 51.0],
        },
        index=idx,
    )
    cached.to_parquet(cache_path)

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr("data._download_adj_close", _unexpected_download)

    out = get_prices(
        ["SPY", "TLT"],
        start="2020-01-01",
        end="2020-01-31",
        cache_path=str(cache_path),
        refresh=False,
        incremental=False,
        strict_inception=False,
    )

    pdt.assert_frame_equal(out, cached, check_freq=False)


def test_get_prices_incremental_update_appends_only_new_rows(tmp_path, monkeypatch):
    cache_path = tmp_path / "prices.parquet"
    idx = pd.bdate_range("2020-01-01", periods=2)
    cached = pd.DataFrame({"SPY": [100.0, 101.0]}, index=idx)
    cached.to_parquet(cache_path)

    calls = []

    def _fake_download(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        update_idx = pd.DatetimeIndex([pd.Timestamp("2020-01-03")])
        return pd.DataFrame({"SPY": [102.0]}, index=update_idx)

    monkeypatch.setattr("data._download_adj_close", _fake_download)

    out = get_prices(
        ["SPY"],
        start="2020-01-01",
        end="2020-01-10",
        cache_path=str(cache_path),
        refresh=False,
        incremental=True,
        strict_inception=False,
    )

    assert calls == [(("SPY",), "2020-01-03", "2020-01-10")]
    assert list(out.index) == list(pd.bdate_range("2020-01-01", periods=3))
    assert out.loc[pd.Timestamp("2020-01-03"), "SPY"] == 102.0

    persisted = pd.read_parquet(cache_path)
    assert list(persisted.index) == list(pd.bdate_range("2020-01-01", periods=3))
    assert not persisted.index.has_duplicates


def test_get_prices_returns_monotonic_unique_index(tmp_path):
    cache_path = tmp_path / "prices.parquet"
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-02"),
        ]
    )
    cached = pd.DataFrame({"SPY": [102.0, 101.0, 101.5]}, index=idx)
    cached.to_parquet(cache_path)

    out = get_prices(
        ["SPY"],
        start="2020-01-01",
        end="2020-01-10",
        cache_path=str(cache_path),
        refresh=False,
        incremental=False,
        strict_inception=False,
    )

    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert list(out.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert out.loc[pd.Timestamp("2020-01-02"), "SPY"] == 101.5


def test_extract_adj_close_handles_multiindex_yfinance_shape():
    idx = pd.bdate_range("2020-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["SPY", "TLT"]])
    raw = pd.DataFrame(
        [
            [100.0, 200.0, 101.0, 201.0],
            [102.0, 202.0, 103.0, 203.0],
        ],
        index=idx,
        columns=columns,
    )

    out = _extract_adj_close(raw, ["SPY", "TLT"])

    expected = pd.DataFrame(
        {
            "SPY": [100.0, 102.0],
            "TLT": [200.0, 202.0],
        },
        index=idx,
    )
    pdt.assert_frame_equal(out, expected)


def test_get_prices_strict_inception_trims_to_latest_inception(tmp_path):
    cache_path = tmp_path / "prices.parquet"
    idx = pd.bdate_range("2020-01-01", periods=5)
    cached = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "TLT": [np.nan, np.nan, 50.0, 51.0, 52.0],
        },
        index=idx,
    )
    cached.to_parquet(cache_path)

    out = get_prices(
        ["SPY", "TLT"],
        start="2020-01-01",
        end="2020-01-31",
        cache_path=str(cache_path),
        refresh=False,
        incremental=False,
        strict_inception=True,
    )

    assert out.index.min() == pd.Timestamp("2020-01-03")
    assert out.notna().all().all()


def test_clean_prices_does_not_forward_fill_across_inception():
    idx = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.DataFrame(
        {
            "SPY": [np.nan, 100.0, np.nan, 102.0, np.nan],
            "BIL": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    out = _clean_prices(raw, ["SPY", "BIL"])

    assert pd.isna(out.loc[pd.Timestamp("2020-01-01"), "SPY"])
    assert out.loc[pd.Timestamp("2020-01-03"), "SPY"] == 100.0
    assert out.loc[pd.Timestamp("2020-01-07"), "SPY"] == 102.0
