import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

from monitor import get_last_monday, compute_current_state


def make_prices(index: pd.DatetimeIndex, tickers, daily_returns_by_ticker) -> pd.DataFrame:
    prices = pd.DataFrame(index=index)
    for t in tickers:
        rets = np.array(daily_returns_by_ticker.get(t, 0.0))
        if rets.size == 1:
            rets = np.repeat(rets, len(index))
        prices[t] = 100.0 * np.cumprod(1 + rets[: len(index)])
    return prices


def make_equity_curve_from_prices(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    rets = prices.pct_change().fillna(0.0)
    w = weights.reindex(prices.columns).fillna(0.0).astype(float)
    port_rets = rets.dot(w.values)
    eq = (1.0 + port_rets).cumprod()
    eq.index = prices.index
    return eq


def test_get_last_monday_returns_most_recent_monday_in_index():
    idx = pd.bdate_range('2025-02-03', periods=15)  # spans 3 weeks
    # ensure there are Mondays in the index
    last_monday = get_last_monday(idx)
    assert isinstance(last_monday, pd.Timestamp)
    assert last_monday.weekday() == 0
    assert last_monday in idx
    assert last_monday <= idx.max()


def test_get_last_monday_raises_if_no_monday():
    # construct index with Tue-Fri only for two weeks
    full = pd.bdate_range('2025-02-03', periods=10)
    no_mondays = full[full.weekday != 0]
    # sanity check no Monday
    assert all(d.weekday() != 0 for d in no_mondays)
    with pytest.raises(ValueError):
        get_last_monday(no_mondays)


def _base_universe(days=270):
    idx = pd.bdate_range('2023-01-01', periods=days)
    tickers = ['SPY', 'TLT', 'BIL', 'BTC-USD']
    # gentle positive returns for SPY/TLT, flat BIL, zero vol BTC
    daily = {
        'SPY': 0.0005,
        'TLT': 0.0003,
        'BIL': 0.0,
        'BTC-USD': 0.0001,
    }
    prices = make_prices(idx, tickers, daily)
    return prices


def test_compute_current_state_returns_required_keys_and_types():
    prices = _base_universe(days=270)
    idx = prices.index
    weights = pd.Series({'SPY': 0.5, 'TLT': 0.3, 'BIL': 0.2, 'BTC-USD': 0.0})
    equity = make_equity_curve_from_prices(prices, weights)

    state = compute_current_state(prices, equity)
    expected_keys = {"asof", "target_weights", "realized_vol", "vol_scale", "drawdown", "dd_bucket", "next_rebalance", "alerts"}
    assert expected_keys.issubset(set(state.keys()))
    assert state['asof'] == prices.index.max()
    assert isinstance(state['target_weights'], pd.Series)
    assert isinstance(state['alerts'], list)
    assert isinstance(state['dd_bucket'], str)


def test_compute_current_state_target_weights_validity():
    prices = _base_universe(days=270)
    weights = pd.Series({'SPY': 0.5, 'TLT': 0.3, 'BIL': 0.2, 'BTC-USD': 0.0})
    equity = make_equity_curve_from_prices(prices, weights)

    state = compute_current_state(prices, equity)
    w = state['target_weights']
    assert (w >= -1e-12).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert 'BIL' in w.index


def test_compute_current_state_no_lookahead_future_shock_does_not_change_prior_asof():
    base = _base_universe(days=120)
    weights = pd.Series({'SPY': 0.6, 'TLT': 0.3, 'BIL': 0.1, 'BTC-USD': 0.0})
    equity = make_equity_curve_from_prices(base, weights)

    state1 = compute_current_state(base, equity)

    # create future-extended prices with a shock after base.asof
    future = base.copy()
    extra_idx = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=5)
    extra = make_prices(extra_idx, base.columns, {'SPY': -0.5, 'TLT': -0.5, 'BIL': 0.0, 'BTC-USD': 0.0})
    prices2 = pd.concat([future, extra])

    # compute state at the same asof by slicing the extended series
    state2 = compute_current_state(prices2.loc[: base.index[-1]], equity)

    # target weights and numeric diagnostics should match
    w1 = state1['target_weights'].reindex(sorted(state1['target_weights'].index)).fillna(0.0)
    w2 = state2['target_weights'].reindex(sorted(state1['target_weights'].index)).fillna(0.0)
    assert np.allclose(w1.values, w2.values, atol=1e-12)
    # alerts and dd_bucket should match
    assert state1['dd_bucket'] == state2['dd_bucket']
    assert state1['alerts'] == state2['alerts']


def test_dd_bucket_transitions_and_alerts():
    idx = pd.bdate_range('2025-01-01', periods=4)
    tickers = ['SPY', 'TLT', 'BIL']
    # prices constant (not relevant for dd tests)
    prices = make_prices(idx, tickers, {'SPY': 0.0, 'TLT': 0.0, 'BIL': 0.0})

    cases = [(-0.16, 'dd15'), (-0.23, 'dd22'), (-0.31, 'dd30'), (-0.09, 'recovery')]
    for dd_val, expected_bucket in cases:
        # build equity with a prior peak then drop to desired drawdown
        peak = 1.2
        last = peak * (1.0 + dd_val)
        eq = pd.Series([1.0, peak, peak, last], index=idx)
        state = compute_current_state(prices, eq)
        assert state['dd_bucket'] == expected_bucket
        if expected_bucket == 'dd22':
            assert state['target_weights'].get('BIL', 0.0) >= 0.5
        if expected_bucket == 'dd30':
            assert pytest.approx(float(state['target_weights'].get('BIL', 0.0)), rel=1e-9) == 1.0


def test_vol_scale_alert_when_below_one():
    # build sufficient history (>252 business days) and include high vol recent returns
    idx = pd.bdate_range('2023-01-01', periods=270)
    tickers = ['SPY', 'TLT', 'BIL']
    # alternating +/-5% returns for SPY (high vol), small for TLT
    # small positive drift plus alternating large moves to keep realized vol high
    # low activity first 200 days to form MA, then high-vol recent window
    spy_rets = []
    for i in range(len(idx)):
        if i < 200:
            spy_rets.append(0.0)
        else:
            spy_rets.append(0.001 + (0.05 if i % 2 == 0 else -0.05))
    returns = {
        'SPY': np.array(spy_rets),
        'TLT': 0.0002,
        'BIL': 0.0,
    }
    prices = make_prices(idx, tickers, returns)
    weights = pd.Series({'SPY': 0.7, 'TLT': 0.2, 'BIL': 0.1})
    equity = make_equity_curve_from_prices(prices, weights)

    state = compute_current_state(prices, equity)
    assert state['vol_scale'] < 1.0
    assert 'vol_scale_below_1' in state['alerts']


def test_determinism_same_inputs_same_outputs():
    prices = _base_universe(days=270)
    weights = pd.Series({'SPY': 0.5, 'TLT': 0.3, 'BIL': 0.2, 'BTC-USD': 0.0})
    equity = make_equity_curve_from_prices(prices, weights)

    s1 = compute_current_state(prices, equity)
    s2 = compute_current_state(prices.copy(), equity.copy())

    # compare target weights
    w1 = s1['target_weights'].reindex(sorted(s1['target_weights'].index)).fillna(0.0)
    w2 = s2['target_weights'].reindex(sorted(s1['target_weights'].index)).fillna(0.0)
    assert np.allclose(w1.values, w2.values, atol=1e-12)

    # compare numeric diagnostics
    for k in ['realized_vol', 'vol_scale', 'drawdown']:
        v1 = float(s1.get(k, float('nan')))
        v2 = float(s2.get(k, float('nan')))
        if np.isnan(v1) and np.isnan(v2):
            continue
        assert pytest.approx(v1, rel=1e-9, abs=1e-12) == v2

    # alerts identical
    assert s1['alerts'] == s2['alerts']
