import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
import pytest

from backtest import (
    get_rebalance_schedule,
    next_trading_day,
    run_backtest,
)


def make_prices(index, tickers, returns_by_ticker):
    prices = pd.DataFrame(index=index)
    for t in tickers:
        rets = returns_by_ticker.get(t, 0.0) if isinstance(returns_by_ticker, dict) else returns_by_ticker
        rets = np.asarray(rets)
        if rets.size == 1:
            rets = np.repeat(rets, len(index))
        prices[t] = 100.0 * np.cumprod(1 + rets)
    return prices


def test_rebalance_schedule_last_trading_friday():
    idx = pd.bdate_range('2025-01-01', periods=60)
    sched = get_rebalance_schedule(idx)
    assert len(sched) >= 2
    for R in sched:
        assert R in idx
        assert R.weekday() == 4  # Friday
        # ensure it's the last friday in that month's trading days
        month_idx = idx[idx.to_period('M') == R.to_period('M')]
        fridays = month_idx[month_idx.weekday == 4]
        assert fridays[-1] == R


def test_execution_is_next_trading_day():
    idx = pd.bdate_range('2025-01-01', periods=30)
    sched = get_rebalance_schedule(idx)
    for R in sched:
        E = next_trading_day(idx, R)
        assert E > R
        assert E in idx
    # test ValueError when no next day
    last = idx[-1]
    with pytest.raises(ValueError):
        next_trading_day(idx, last)


def test_transaction_costs_applied_results_differ():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B']
    # modest drift so portfolio moves and rebalances
    rets = {'A': 0.001, 'B': 0.0005}
    prices = make_prices(idx, tickers, rets)

    res0 = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1, 'tx_cost': 0.0, 'band': 0.01})
    res1 = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1, 'tx_cost': 0.0015, 'band': 0.01})

    eq0 = res0['equity']
    eq1 = res1['equity']
    assert not eq0.equals(eq1)
    assert float(eq0.iloc[-1]) != float(eq1.iloc[-1])


def test_band_rebalance_skips_small_changes():
    idx = pd.bdate_range('2025-01-01', periods=40)
    tickers = ['A']
    # flat prices -> no momentum change -> target stays BIL or unchanged
    prices = make_prices(idx, tickers, 0.0)
    res = run_backtest(prices, config={'ma_window': 10, 'mom_lookback': 10, 'top_n': 1, 'tx_cost': 0.0015, 'band': 0.05})
    trades = res['trades']
    # every trade entry should have traded==False when no selection
    assert all(trades['traded'] == False)
    assert all(trades['turnover'] == 0.0)
    assert all(trades['cost'] == 0.0)


def test_no_lookahead_end_to_end_shock_on_decision_or_exec_day():
    idx = pd.bdate_range('2025-01-01', periods=60)
    tickers = ['A', 'B']
    rets = {'A': 0.001, 'B': 0.0005}
    prices_clean = make_prices(idx, tickers, rets)

    # choose a rebalance R (use schedule)
    sched = get_rebalance_schedule(idx)
    assert len(sched) >= 1
    R = sched[0]
    E = next_trading_day(idx, R)

    # dataset with shock on decision day R (should not affect decision since asof = R-1)
    prices_shock_R = prices_clean.copy()
    # introduce large move at R (if R exists in index)
    if R in prices_shock_R.index:
        prices_shock_R.loc[R] = prices_shock_R.loc[R] * 0.5

    # dataset with shock on execution day E
    prices_shock_E = prices_clean.copy()
    if E in prices_shock_E.index:
        prices_shock_E.loc[E] = prices_shock_E.loc[E] * 0.5

    res_clean = run_backtest(prices_clean, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})
    res_shock_R = run_backtest(prices_shock_R, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})
    res_shock_E = run_backtest(prices_shock_E, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 1})

    # compare target holdings at exec date E
    w_clean = res_clean['weights'].loc[E]
    w_shockR = res_shock_R['weights'].loc[E]
    w_shockE = res_shock_E['weights'].loc[E]

    # decision shouldn't change because of R shock (as-of uses R-1)
    assert all(np.isclose(w_clean.values, w_shockR.values, atol=1e-12))
    # but exec-day shock may change post-exec equity but not target weights
    assert all(np.isclose(w_clean.values, w_shockE.values, atol=1e-12))


def test_weights_validity_every_day():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B']
    prices = make_prices(idx, tickers, {'A': 0.001, 'B': -0.0002})
    res = run_backtest(prices, config={'ma_window': 5, 'mom_lookback': 5, 'top_n': 2})
    w = res['weights']
    # sum to 1 each day
    sums = w.sum(axis=1)
    assert all(np.isclose(sums.values, 1.0, atol=1e-6))
    # non-negative
    assert (w >= -1e-12).all().all()


def test_cash_fallback_all_in_bil_when_no_assets_eligible():
    idx = pd.bdate_range('2025-01-01', periods=80)
    tickers = ['A', 'B']
    # design prices so momentum undefined/negative relative to ma -> no eligibility
    prices = make_prices(idx, tickers, -0.01)
    res = run_backtest(prices, config={'ma_window': 20, 'mom_lookback': 20, 'top_n': 2})
    trades = res['trades']
    # at each exec date, weights should be 100% BIL
    for rec in res['trades'].itertuples():
        E = rec.exec_date
        w = res['weights'].loc[E]
        # BIL column exists
        assert 'BIL' in w.index
        assert pytest.approx(w['BIL'], rel=1e-9) == 1.0
        assert np.isclose(w.values.sum(), 1.0, atol=1e-6)
        assert (w >= -1e-12).all()
