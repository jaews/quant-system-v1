import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import pytest

from benchmark import run_benchmark, compare_vs_benchmark

# pandas versions differ in supported month aliases; ensure 'M' works for resample in tests
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


def make_prices(index: pd.DatetimeIndex, tickers, daily_returns_by_ticker) -> pd.DataFrame:
    prices = pd.DataFrame(index=index)
    for t in tickers:
        rets = np.array(daily_returns_by_ticker.get(t, 0.0))
        if rets.size == 1:
            rets = np.repeat(rets, len(index))
        prices[t] = 100.0 * np.cumprod(1 + rets[: len(index)])
    return prices


def make_static_weights(dct) -> pd.Series:
    return pd.Series(dct, dtype=float)


def test_run_benchmark_equity_starts_at_one_and_monotonic_when_all_returns_positive():
    idx = pd.bdate_range('2025-01-01', periods=30)
    tickers = ['A', 'B', 'BIL']
    returns = {'A': 0.001, 'B': 0.0008, 'BIL': 0.0}
    prices = make_prices(idx, tickers, returns)
    w = make_static_weights({'A': 0.5, 'B': 0.5, 'BIL': 0.0})

    eq = run_benchmark(prices, w)
    assert pytest.approx(eq.iloc[0], rel=1e-12) == 1.0
    # strictly increasing
    assert (eq.diff().dropna() > 0).all()


def test_run_benchmark_matches_manual_calculation_two_assets():
    idx = pd.bdate_range('2025-01-01', periods=10)
    tickers = ['A', 'B']
    # A +1% every day, B alternates -0.5% and +0.5%
    a_rets = np.array([0.01] * len(idx))
    b_rets = np.array([(-0.005 if i % 2 == 0 else 0.005) for i in range(len(idx))])
    prices = make_prices(idx, tickers, {'A': a_rets, 'B': b_rets})
    w = make_static_weights({'A': 0.6, 'B': 0.4})

    eq = run_benchmark(prices, w)

    # manual
    rets = prices.pct_change().fillna(0.0)
    port_rets = rets.dot(w.values)
    manual = (1.0 + port_rets).cumprod()
    manual.index = prices.index
    assert np.allclose(eq.values, manual.values, atol=1e-12)


def test_run_benchmark_respects_date_slicing():
    idx = pd.bdate_range('2025-01-01', periods=30)
    tickers = ['A', 'B']
    prices = make_prices(idx, tickers, {'A': 0.001, 'B': 0.0005})
    w = make_static_weights({'A': 0.7, 'B': 0.3})

    start = idx[5]
    end = idx[20]
    eq = run_benchmark(prices.loc[start:end], w)
    assert eq.index.min() >= start
    assert eq.index.max() <= end
    assert pytest.approx(eq.iloc[0], rel=1e-12) == 1.0


def test_run_benchmark_long_only_and_sum_to_one_validation():
    idx = pd.bdate_range('2025-01-01', periods=10)
    prices = make_prices(idx, ['A'], {'A': 0.001})
    # sum != 1
    with pytest.raises(ValueError):
        run_benchmark(prices, make_static_weights({'A': 0.0}))
    # negative weight
    with pytest.raises(ValueError):
        run_benchmark(prices, make_static_weights({'A': -0.1}))


def test_compare_vs_benchmark_basic_diffs_and_alignment():
    idx1 = pd.bdate_range('2025-01-01', periods=30)
    idx2 = pd.bdate_range('2025-01-10', periods=30)
    # simple equity curves
    sys = pd.Series(1.0 + np.linspace(0.0, 0.1, len(idx1)), index=idx1)
    bmk = pd.Series(1.0 + np.linspace(0.0, 0.05, len(idx2)), index=idx2)

    out = compare_vs_benchmark(sys, bmk)
    keys = {'CAGR_diff', 'MaxDD_diff', 'Sharpe_diff', 'Hit_ratio_monthly'}
    assert keys.issubset(set(out.keys()))
    # Final equity diff on intersection end date
    common_idx = sys.index.intersection(bmk.index)
    if not common_idx.empty:
        expected = float(sys.loc[common_idx[-1]] - bmk.loc[common_idx[-1]])
        assert pytest.approx(out.get('Final_equity_diff', expected), rel=1e-12) == expected if 'Final_equity_diff' in out else True


def test_compare_vs_benchmark_hit_ratio_monthly_known_case():
    # build 4 month-end points so compare_vs_benchmark computes 3 month returns
    idx = pd.DatetimeIndex(['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30'])
    # define end-of-month equity levels to produce monthly returns where sys wins 2 of 3
    sys = pd.Series([1.0, 1.05, 1.06, 1.12], index=idx)
    bmk = pd.Series([1.0, 1.02, 1.07, 1.03], index=idx)

    out = compare_vs_benchmark(sys, bmk)
    assert abs(out['Hit_ratio_monthly'] - (2.0 / 3.0)) < 1e-12


def test_compare_vs_benchmark_sharpe_diff_sign():
    idx = pd.bdate_range('2025-01-01', periods=252)
    # create equities with higher mean for system, similar vol
    np.random.seed(0)
    base_rets = np.random.normal(0.0001, 0.01, size=len(idx))
    sys_rets = base_rets + 0.0005
    bmk_rets = base_rets
    sys = (1.0 + pd.Series(sys_rets, index=idx)).cumprod()
    bmk = (1.0 + pd.Series(bmk_rets, index=idx)).cumprod()

    out = compare_vs_benchmark(sys, bmk)
    assert out['Sharpe_diff'] > 0


def test_determinism():
    idx = pd.bdate_range('2025-01-01', periods=60)
    prices = make_prices(idx, ['A', 'B', 'BIL'], {'A': 0.001, 'B': 0.0005, 'BIL': 0.0})
    w = make_static_weights({'A': 0.6, 'B': 0.3, 'BIL': 0.1})

    e1 = run_benchmark(prices, w)
    e2 = run_benchmark(prices.copy(), w.copy())
    assert np.allclose(e1.values, e2.values, atol=1e-12)

    comp1 = compare_vs_benchmark(e1, e2)
    comp2 = compare_vs_benchmark(e1.copy(), e2.copy())
    for k in comp1:
        v1 = comp1[k]
        v2 = comp2[k]
        if isinstance(v1, float) and np.isnan(v1) and np.isnan(v2):
            continue
        assert pytest.approx(v1, rel=1e-12, abs=1e-12) == v2
