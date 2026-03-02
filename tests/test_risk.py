import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

from risk import (
    realized_vol,
    current_drawdown,
    apply_vol_target,
    apply_drawdown_governor,
    apply_risk_controls,
)
from backtest import run_backtest


def make_prices(n_days=40, vol_type='low', tickers=('A', 'B')):
    idx = pd.bdate_range('2025-01-01', periods=n_days)
    prices = pd.DataFrame(index=idx)
    for t in tickers:
        if vol_type == 'high':
            # deterministic alternating +/- 5% moves
            rets = np.array([0.05 if i % 2 == 0 else -0.05 for i in range(n_days)])
        elif vol_type == 'med':
            rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n_days)])
        else:
            # low vol tiny moves
            rets = np.array([0.0001 for _ in range(n_days)])
        prices[t] = 100.0 * np.cumprod(1 + rets)
    # BIL is flat cash
    prices['BIL'] = 100.0
    return prices


def make_equity_from_weights(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    # compute daily portfolio value series from prices and static weights
    rets = prices.pct_change().fillna(0.0)
    # align weights to columns
    w = weights.reindex(prices.columns).fillna(0.0).astype(float)
    # portfolio returns
    port_rets = rets.dot(w.values)
    eq = (1 + port_rets).cumprod()
    eq.index = prices.index
    return eq


def assert_weights_valid(w: pd.Series):
    assert all(v >= -1e-12 for v in w.values)
    assert pytest.approx(float(w.sum()), rel=1e-9) == 1.0


def test_realized_vol_uses_exact_lookback_returns():
    idx = pd.bdate_range('2025-01-01', periods=12)
    rets = np.array([0.50, -0.40, 0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.04, -0.01])
    prices = pd.DataFrame(index=idx)
    prices['A'] = 100.0 * np.cumprod(1 + rets)
    prices['BIL'] = 100.0
    w = pd.Series({'A': 1.0, 'BIL': 0.0})

    vol = realized_vol(prices, w, asof=idx[-1], lookback=5)
    expected = float(np.std(rets[-5:], ddof=0) * np.sqrt(252))
    assert pytest.approx(vol, rel=1e-12) == expected


def test_realized_vol_nan_when_lookback_history_insufficient():
    prices = make_prices(n_days=6, vol_type='med')
    w = pd.Series({'A': 0.7, 'B': 0.3, 'BIL': 0.0})
    vol = realized_vol(prices, w, asof=prices.index[-1], lookback=10)
    assert np.isnan(vol)


def test_current_drawdown_matches_peak_relative():
    idx = pd.bdate_range('2025-01-01', periods=4)
    eq = pd.Series([1.0, 1.2, 1.1, 1.08], index=idx)
    dd = current_drawdown(eq, asof=idx[-1])
    assert pytest.approx(dd, rel=1e-12) == (1.08 / 1.2 - 1.0)


def test_high_vol_risky_reduced():
    prices = make_prices(n_days=40, vol_type='high')
    base = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    asof = prices.index[-1]
    # use small lookback to make test fast and deterministic
    w_new, diag = apply_vol_target(base, prices, asof, target_vol=0.12, lookback=10)
    risky_before = base.drop('BIL').sum()
    risky_after = w_new.drop('BIL').sum()
    assert risky_after < risky_before
    assert_weights_valid(w_new)


def test_low_vol_unchanged():
    prices = make_prices(n_days=40, vol_type='low')
    base = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    asof = prices.index[-1]
    w_new, diag = apply_vol_target(base, prices, asof, target_vol=0.12, lookback=10)
    # low vol -> scale should be 1.0 -> risky unchanged (within tolerance)
    assert pytest.approx(float(w_new.drop('BIL').sum()), rel=1e-9) == pytest.approx(float(base.drop('BIL').sum()), rel=1e-9)
    assert_weights_valid(w_new)


def test_dd_minus_15_reduces_risky_by_30_percent():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.15)
    # risky scaled by 0.7
    assert pytest.approx(new_w['A'], rel=1e-9) == pytest.approx(0.4 * 0.7, rel=1e-9)
    assert pytest.approx(new_w['B'], rel=1e-9) == pytest.approx(0.4 * 0.7, rel=1e-9)
    # BIL increased by freed amount
    assert new_w['BIL'] > w['BIL']
    assert_weights_valid(new_w)


def test_dd_minus_22_sets_bil_at_least_half():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.22)
    assert new_w['BIL'] >= 0.5
    # risky sum should be remaining (<= 0.5)
    assert pytest.approx(new_w.drop('BIL').sum(), rel=1e-9) == pytest.approx(1.0 - new_w['BIL'], rel=1e-9)
    assert_weights_valid(new_w)


def test_dd_minus_30_full_bil():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.30)
    assert pytest.approx(new_w['BIL'], rel=1e-9) == 1.0
    for k in ['A', 'B']:
        assert pytest.approx(new_w.get(k, 0.0), rel=1e-9) == 0.0
    assert_weights_valid(new_w)


def test_recovery_no_change():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.09)
    # unchanged
    assert pytest.approx(new_w['A'], rel=1e-9) == pytest.approx(w['A'], rel=1e-9)
    assert pytest.approx(new_w['B'], rel=1e-9) == pytest.approx(w['B'], rel=1e-9)
    assert pytest.approx(new_w['BIL'], rel=1e-9) == pytest.approx(w['BIL'], rel=1e-9)
    assert_weights_valid(new_w)


def test_drawdown_between_minus_22_and_minus_15_uses_dd15_rule():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.2199)
    assert diag['dd_rule'] == 'dd15'
    assert pytest.approx(float(new_w.drop('BIL').sum()), rel=1e-9) == pytest.approx(0.8 * 0.7, rel=1e-9)
    assert_weights_valid(new_w)


def test_drawdown_between_minus_30_and_minus_22_uses_dd22_rule():
    w = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    new_w, diag = apply_drawdown_governor(w, -0.2999)
    assert diag['dd_rule'] == 'dd22'
    assert new_w['BIL'] >= 0.5
    assert_weights_valid(new_w)


def test_no_lookahead_rebalance_shock():
    # build prices up to asof
    prices = make_prices(n_days=40, vol_type='med')
    asof = prices.index[-1]
    rebalance_date = asof + pd.tseries.offsets.BDay(1)

    base = pd.Series({'A': 0.4, 'B': 0.4, 'BIL': 0.2})
    equity = make_equity_from_weights(prices, base)

    # prices_with_shock includes an extra row at rebalance_date with a large shock
    prices_with_shock = prices.copy()
    # compute shock deterministically: A down 50%, B down 50% on rebalance_date
    shock_row = prices.iloc[-1] * 0.5
    shock_row.name = rebalance_date
    prices_with_shock = pd.concat([prices_with_shock, shock_row.to_frame().T])

    w1, d1 = apply_risk_controls(base, prices, equity, rebalance_date, config={'target_vol': 0.12, 'lookback': 10})
    w2, d2 = apply_risk_controls(base, prices_with_shock, equity, rebalance_date, config={'target_vol': 0.12, 'lookback': 10})

    # results must be identical (no look-ahead)
    assert all(abs(w1 - w2) <= 1e-12)
    assert d1['asof'] == d2['asof']
    assert_weights_valid(w1)


def test_future_data_does_not_change_past_results():
    # prepare a full price history and a truncated subset
    full = make_prices(n_days=80, vol_type='med', tickers=('A', 'B'))
    truncated = full.iloc[:40]

    # run backtest on truncated dataset
    res1 = run_backtest(truncated)
    eq1 = res1['equity']
    w1 = res1['weights'].loc[truncated.index]

    # run backtest on extended dataset (future data appended)
    res2 = run_backtest(full)
    eq2 = res2['equity'].reindex(truncated.index)
    w2 = res2['weights'].reindex(truncated.index).ffill().fillna(0.0)

    # equity for the original period must be identical (no look-ahead)
    # use a tight tolerance for floating point equality
    assert eq1.index.equals(eq2.index)
    assert np.allclose(eq1.values, eq2.values, atol=1e-12, rtol=0)

    # weights during the original period must also be identical
    # align columns and compare
    w1a = w1.reindex(sorted(w1.columns), axis=1).fillna(0.0)
    w2a = w2.reindex(sorted(w1.columns), axis=1).fillna(0.0)
    diff = (w1a - w2a).abs().values.max()
    assert diff <= 1e-12
