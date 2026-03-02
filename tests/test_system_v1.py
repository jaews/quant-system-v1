import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

from backtest import get_rebalance_schedule, run_backtest
from monitor import compute_current_state


def make_prices(index, tickers, returns_by_ticker):
    prices = pd.DataFrame(index=index)
    for t in tickers:
        rets = returns_by_ticker.get(t, 0.0) if isinstance(returns_by_ticker, dict) else returns_by_ticker
        rets = np.asarray(rets)
        if rets.size == 1:
            rets = np.repeat(rets, len(index))
        prices[t] = 100.0 * np.cumprod(1 + rets)
    # ensure BIL exists
    if 'BIL' not in prices.columns:
        prices['BIL'] = 100.0
    return prices


def test_run_backtest_completes():
    idx = pd.bdate_range('2025-01-01', periods=90)
    tickers = ['A', 'B', 'BTC-USD', 'ETH-USD']
    rets = {'A': 0.001, 'B': 0.0003, 'BTC-USD': 0.002, 'ETH-USD': 0.0015}
    prices = make_prices(idx, tickers, rets)

    res = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 2})
    assert 'equity' in res and 'weights' in res and 'trades' in res
    assert isinstance(res['equity'], pd.Series)


def test_no_lookahead_integration():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B']
    rets = {'A': 0.001, 'B': 0.0005}
    prices_clean = make_prices(idx, tickers, rets)

    sched = get_rebalance_schedule(idx)
    assert len(sched) >= 1
    R = sched[0]
    E = None

    # create shock on decision day R (should not affect target)
    prices_shock = prices_clean.copy()
    if R in prices_shock.index:
        prices_shock.loc[R] = prices_shock.loc[R] * 0.3

    res_clean = run_backtest(prices_clean, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})
    res_shock = run_backtest(prices_shock, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})

    # find first exec date
    E = res_clean['trades'].iloc[0]['exec_date']
    w_clean = res_clean['weights'].loc[E]
    w_shock = res_shock['weights'].loc[E]
    assert all(np.isclose(w_clean.values, w_shock.values, atol=1e-12))


def test_long_only_and_sum_invariants():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B', 'BTC-USD', 'ETH-USD']
    rets = {'A': 0.0005, 'B': -0.0001, 'BTC-USD': 0.001, 'ETH-USD': 0.0008}
    prices = make_prices(idx, tickers, rets)
    res = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 2})
    w = res['weights']
    sums = w.sum(axis=1)
    assert (np.isclose(sums.values, 1.0, atol=1e-6)).all()
    assert (w >= -1e-12).all().all()


def test_crypto_cap_invariant():
    idx = pd.bdate_range('2025-01-01', periods=90)
    tickers = ['A', 'BTC-USD', 'ETH-USD']
    rets = {'A': 0.0005, 'BTC-USD': 0.002, 'ETH-USD': 0.0018}
    prices = make_prices(idx, tickers, rets)
    res = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 3})
    w = res['weights']
    cap = 0.25
    # check daily crypto sleeve <= cap + tiny tol
    crypto = w.get('BTC-USD', pd.Series(0, index=w.index)) + w.get('ETH-USD', pd.Series(0, index=w.index))
    assert (crypto <= cap + 1e-9).all()


def test_rebalance_schedule_last_friday():
    idx = pd.bdate_range('2025-01-01', periods=70)
    sched = get_rebalance_schedule(idx)
    for R in sched:
        assert R.weekday() == 4
        assert R in idx


def test_cash_fallback_when_no_eligibility():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B']
    prices = make_prices(idx, tickers, -0.01)
    res = run_backtest(prices, config={'ma_window': 20, 'mom_lookback': 20, 'top_n': 2})
    for rec in res['trades'].itertuples():
        E = rec.exec_date
        w = res['weights'].loc[E]
        assert 'BIL' in w.index
        assert pytest.approx(w['BIL'], rel=1e-9) == 1.0


def test_determinism_of_run_backtest():
    idx = pd.bdate_range('2025-01-01', periods=90)
    tickers = ['A', 'B']
    rets = {'A': 0.0008, 'B': 0.0002}
    prices = make_prices(idx, tickers, rets)
    r1 = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})
    r2 = run_backtest(prices.copy(), config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})
    assert r1['equity'].equals(r2['equity'])
    # compare trades key columns
    t1 = r1['trades'][['decision_date', 'exec_date', 'traded', 'turnover', 'cost']].reset_index(drop=True)
    t2 = r2['trades'][['decision_date', 'exec_date', 'traded', 'turnover', 'cost']].reset_index(drop=True)
    pd.testing.assert_frame_equal(t1, t2)
