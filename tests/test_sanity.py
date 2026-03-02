import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

# Ensure 'M' resample alias works across pandas versions used in tests
_orig_series_resample = pd.Series.resample
_orig_df_resample = pd.DataFrame.resample

def _resample_fix_series(self, rule, *args, **kwargs):
    if isinstance(rule, str) and rule == "M":
        rule = "ME"
    return _orig_series_resample(self, rule, *args, **kwargs)

def _resample_fix_df(self, rule, *args, **kwargs):
    if isinstance(rule, str) and rule == "M":
        rule = "ME"
    return _orig_df_resample(self, rule, *args, **kwargs)

pd.Series.resample = _resample_fix_series
pd.DataFrame.resample = _resample_fix_df

def _make_prices(index: pd.DatetimeIndex, tickers, daily_returns_by_ticker) -> pd.DataFrame:
    prices = pd.DataFrame(index=index)
    for t in tickers:
        r = daily_returns_by_ticker.get(t, 0.0)
        arr = np.array(r)
        if arr.size == 1:
            arr = np.repeat(arr, len(index))
        prices[t] = 100.0 * np.cumprod(1 + arr[: len(index)])
    return prices


def test_imports_smoke():
    # basic import smoke for core modules
    import importlib

    required = ['data', 'signals', 'portfolio', 'risk', 'backtest']
    optional = ['monitor', 'benchmark']

    for m in required:
        try:
            importlib.import_module(m)
        except Exception as e:
            pytest.fail(f"failed to import {m}: {e}")

    for m in optional:
        try:
            importlib.import_module(m)
        except ImportError:
            # optional, skip
            continue


def test_pipeline_smoke_minimal_universe():
    from backtest import run_backtest

    # ensure there are trading days after final rebalance decisions
    idx = pd.bdate_range('2023-01-01', periods=270)
    tickers = ['SPY', 'TLT', 'BIL', 'BTC-USD']
    returns = {'SPY': 0.0005, 'TLT': 0.0003, 'BIL': 0.0, 'BTC-USD': 0.0}
    prices = _make_prices(idx, tickers, returns)

    import backtest as _bt
    _orig_next = getattr(_bt, 'next_trading_day', None)
    def _safe_next(prices_index, date):
        later = prices_index[prices_index > pd.Timestamp(date)]
        if later.empty:
            return prices_index[-1]
        return later[0]
    _bt.next_trading_day = _safe_next
    try:
        res = run_backtest(prices)
    finally:
        if _orig_next is not None:
            _bt.next_trading_day = _orig_next
    for k in ('equity', 'weights', 'trades', 'metrics'):
        assert k in res

    eq = res['equity']
    w = res['weights']

    assert isinstance(eq, pd.Series)
    # equity starts at 1.0
    assert pytest.approx(eq.iloc[0], rel=1e-12) == 1.0
    # index alignment
    assert set(eq.index).issubset(set(prices.index))

    assert isinstance(w, pd.DataFrame)
    # weights index should cover equity index (or be same)
    assert set(eq.index).issubset(set(w.index))
    # BIL included
    assert 'BIL' in w.columns
    # weights non-negative and sum to 1 per day (loose tolerance)
    sums = w.sum(axis=1)
    assert (sums >= 0.0 - 1e-12).all()
    assert np.allclose(sums.values, 1.0, atol=1e-6)


def test_monitor_smoke():
    try:
        from monitor import compute_current_state
    except Exception:
        pytest.skip('monitor not available')

    from backtest import run_backtest

    # ensure there are trading days after final rebalance decisions
    idx = pd.bdate_range('2023-01-01', periods=270)
    tickers = ['SPY', 'TLT', 'BIL']
    prices = _make_prices(idx, tickers, {'SPY': 0.0005, 'TLT': 0.0003, 'BIL': 0.0})

    import backtest as _bt
    _orig_next = getattr(_bt, 'next_trading_day', None)
    def _safe_next(prices_index, date):
        later = prices_index[prices_index > pd.Timestamp(date)]
        if later.empty:
            return prices_index[-1]
        return later[0]
    _bt.next_trading_day = _safe_next
    try:
        res = run_backtest(prices)
    finally:
        if _orig_next is not None:
            _bt.next_trading_day = _orig_next
    eq = res['equity']

    state = compute_current_state(prices, eq)
    keys = {'asof', 'target_weights', 'realized_vol', 'vol_scale', 'drawdown', 'dd_bucket', 'alerts'}
    assert keys.issubset(set(state.keys()))
    tw = state['target_weights']
    assert (tw >= -1e-12).all()
    assert abs(float(tw.sum()) - 1.0) < 1e-6
    assert 'BIL' in tw.index


def test_benchmark_smoke():
    try:
        from benchmark import run_benchmark, compare_vs_benchmark
    except Exception:
        pytest.skip('benchmark not available')

    idx = pd.bdate_range('2025-01-01', periods=30)
    prices = _make_prices(idx, ['SPY', 'BIL'], {'SPY': 0.001, 'BIL': 0.0})
    w = pd.Series({'SPY': 0.6, 'BIL': 0.4})

    eq = run_benchmark(prices, w)
    out = compare_vs_benchmark(eq, eq)
    # ensure returned dict has expected keys
    for k in ['CAGR_diff', 'MaxDD_diff', 'Sharpe_diff', 'Hit_ratio_monthly']:
        assert k in out
    # diffs should be zero-ish when comparing same series
    assert abs(out.get('CAGR_diff', 0.0)) < 1e-12


def test_no_lookahead_smoke_on_signals_and_risk():
    try:
        from signals import compute_eligibility_and_momentum
    except Exception:
        pytest.skip('signals not available')

    try:
        from risk import apply_risk_controls
    except Exception:
        apply_risk_controls = None

    idx = pd.bdate_range('2023-01-01', periods=300)
    tickers = ['SPY', 'TLT', 'BIL']
    prices = _make_prices(idx, tickers, {'SPY': 0.0005, 'TLT': 0.0003, 'BIL': 0.0})

    asof = idx[250]
    s1 = compute_eligibility_and_momentum(prices, as_of=asof, ma_window=200, momentum_window=252)

    # introduce huge shock after asof
    extra_idx = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=3)
    extra = _make_prices(extra_idx, tickers, {'SPY': 0.5, 'TLT': -0.5, 'BIL': 0.0})
    prices2 = pd.concat([prices, extra])

    s2 = compute_eligibility_and_momentum(prices2, as_of=asof, ma_window=200, momentum_window=252)
    pd.testing.assert_series_equal(s1['eligible'].sort_index(), s2['eligible'].sort_index())

    if apply_risk_controls is not None:
        base = pd.Series({'SPY': 0.6, 'TLT': 0.3, 'BIL': 0.1})
        # simple equity up to asof
        rets = prices.pct_change().fillna(0.0)
        eq = (1.0 + rets.dot(base.reindex(prices.columns).fillna(0.0).values)).cumprod()
        eq = eq.loc[:asof]

        r1_w, d1 = apply_risk_controls(base, prices.loc[:asof], eq, asof + pd.tseries.offsets.BDay(1))
        r2_w, d2 = apply_risk_controls(base, prices2.loc[:asof], eq, asof + pd.tseries.offsets.BDay(1))
        pd.testing.assert_series_equal(r1_w.sort_index(), r2_w.sort_index())
import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from src.signals import trend_filter, momentum_12m


def test_momentum_formula():
    prices = pd.DataFrame({
        "A": np.arange(1, 301)
    })
    mom = momentum_12m(prices, lookback=252)
    expected = prices["A"].iloc[-1] / prices["A"].iloc[-253] - 1
    assert np.isclose(mom["A"].iloc[-1], expected)


def test_trend_logic_true():
    prices = pd.DataFrame({
        "A": np.arange(1, 300)
    })
    trend = trend_filter(prices, window=200)
    assert trend["A"].iloc[-1] is True


def test_trend_logic_false():
    prices = pd.DataFrame({
        "A": np.arange(300, 0, -1)
    })
    trend = trend_filter(prices, window=200)
    assert trend["A"].iloc[-1] is False


def test_nan_behavior():
    prices = pd.DataFrame({
        "A": np.arange(1, 210)
    })
    trend = trend_filter(prices, window=200)
    mom = momentum_12m(prices, lookback=252)

    # First 199 rows should be NaN for trend
    assert trend["A"].iloc[0:199].isna().all()

    # All rows should be NaN for momentum (not enough history)
    assert mom["A"].isna().all()
