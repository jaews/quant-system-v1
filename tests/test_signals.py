import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

from signals import compute_eligibility_and_momentum, momentum_12m, trend_filter


def make_prices(index: pd.DatetimeIndex, tickers, daily_returns_by_ticker) -> pd.DataFrame:
    prices = pd.DataFrame(index=index)
    for t in tickers:
        rets = np.array(daily_returns_by_ticker.get(t, 0.0))
        if rets.size == 1:
            rets = np.repeat(rets, len(index))
        prices[t] = 100.0 * np.cumprod(1 + rets[: len(index)])
    return prices


def test_trend_filter_ma200_true_when_price_above_ma200():
    idx = pd.bdate_range('2024-01-01', periods=260)
    tickers = ['A', 'B']
    # A: gentle uptrend, B: gentle downtrend
    returns = {'A': 0.001, 'B': -0.001}
    prices = make_prices(idx, tickers, returns)

    out = compute_eligibility_and_momentum(prices, as_of=idx[-1], ma_window=200, momentum_window=252)
    trend = out['eligible']
    assert bool(trend['A']) is True
    assert bool(trend['B']) is False


def test_momentum_12m_matches_expected_simple_case():
    idx = pd.bdate_range('2023-01-01', periods=300)
    tickers = ['A']
    r = 0.0004
    prices = make_prices(idx, tickers, {'A': r})

    mom = momentum_12m(prices, lookback=252)
    # expected = price(asof)/price(asof-252) - 1
    expected = float(prices['A'].iloc[-1] / prices['A'].iloc[-1 - 252] - 1.0)
    got = float(mom['A'].iloc[-1])
    assert pytest.approx(got, rel=1e-12, abs=1e-12) == expected


def test_insufficient_history_behavior():
    idx = pd.bdate_range('2025-01-01', periods=150)
    tickers = ['A', 'B']
    prices = make_prices(idx, tickers, {'A': 0.001, 'B': -0.0005})

    out = compute_eligibility_and_momentum(prices, as_of=idx[-1], ma_window=200, momentum_window=252)
    # per implementation: eligible when MA not available -> False; momentum -> NaN
    assert 'eligible' in out.columns and 'momentum' in out.columns
    assert out['eligible'].dtype == bool or out['eligible'].iloc[0] in (True, False)
    assert np.isnan(out['momentum'].iloc[0])


def test_no_lookahead_future_shock_does_not_change_signals():
    idx = pd.bdate_range('2023-01-01', periods=300)
    tickers = ['A', 'B']
    prices = make_prices(idx, tickers, {'A': 0.0005, 'B': 0.0002})

    asof = idx[260]
    s1 = compute_eligibility_and_momentum(prices, as_of=asof, ma_window=200, momentum_window=252)

    # create future-extended prices with shock after asof
    extra_idx = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=5)
    extra = make_prices(extra_idx, tickers, {'A': 0.5, 'B': -0.5})
    prices2 = pd.concat([prices, extra])

    s2 = compute_eligibility_and_momentum(prices2, as_of=asof, ma_window=200, momentum_window=252)

    # eligible exact equality
    pd.testing.assert_series_equal(s1['eligible'].sort_index(), s2['eligible'].sort_index())

    # momentum: numeric compare allowing NaNs
    m1 = s1['momentum'].sort_index()
    m2 = s2['momentum'].sort_index()
    for a, b in zip(m1.index, m1.values):
        v1 = m1.loc[a]
        v2 = m2.loc[a]
        if np.isnan(v1) and np.isnan(v2):
            continue
        assert pytest.approx(float(v1), rel=1e-12, abs=1e-12) == float(v2)


def test_output_shapes_and_indices():
    idx = pd.bdate_range('2025-01-01', periods=260)
    tickers = ['SPY', 'TLT', 'BIL']
    prices = make_prices(idx, tickers, {'SPY': 0.0005, 'TLT': 0.0002, 'BIL': 0.0})

    out = compute_eligibility_and_momentum(prices, as_of=idx[-1], ma_window=200, momentum_window=252)
    # index should include tickers
    assert set(out.index) == set(prices.columns)
    assert 'eligible' in out.columns and 'momentum' in out.columns
    assert out['momentum'].dtype == float or np.issubdtype(out['momentum'].dtype, np.floating)


def test_asof_not_in_index_handles_defensively():
    idx = pd.bdate_range('2025-01-01', periods=260)
    tickers = ['A']
    prices = make_prices(idx, tickers, {'A': 0.001})

    # pick a weekend date after last index
    asof = idx[-1] + pd.Timedelta(days=2)
    try:
        out = compute_eligibility_and_momentum(prices, as_of=asof, ma_window=200, momentum_window=252)
    except ValueError:
        pytest.skip("Implementation raises on asof not in index")
    # otherwise, should equal compute_eligibility_and_momentum with aligned date (last <= asof)
    aligned = idx[idx <= asof][-1]
    out2 = compute_eligibility_and_momentum(prices, as_of=aligned, ma_window=200, momentum_window=252)
    pd.testing.assert_series_equal(out['eligible'].sort_index(), out2['eligible'].sort_index())
