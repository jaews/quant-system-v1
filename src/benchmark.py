from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtest import compute_equity_curve, compute_metrics


def run_benchmark(prices: pd.DataFrame, weights_static) -> pd.Series:
    """Run a static-weight benchmark and return equity series starting at 1.0.

    - `weights_static` may be a pd.Series, dict or similar mapping of ticker->weight.
    - Missing tickers in `prices` are treated as zero weight.
    - We normalize weights to sum to 1 (if non-zero) and require non-negative values.
    """
    if prices is None or prices.empty:
        raise ValueError("prices must be provided and non-empty")
    if weights_static is None:
        raise ValueError("weights_static must be provided")

    # construct weight series aligned to price columns
    w = pd.Series(weights_static).reindex(prices.columns).fillna(0.0).astype(float)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / total

    # build weights_by_day DataFrame and compute equity
    w_by_day = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    # fill each day with static weights
    w_by_day.loc[:, :] = w.values

    equity = compute_equity_curve(prices, w_by_day)
    return equity


def compare_vs_benchmark(system_equity: pd.Series, benchmark_equity: pd.Series) -> Dict[str, float]:
    """Compare system vs benchmark equity series and return summary diffs.

    Returns dict with keys:
      - "CAGR_diff": system_CAGR - benchmark_CAGR
      - "MaxDD_diff": system_MaxDD - benchmark_MaxDD
      - "Sharpe_diff": system_Sharpe - benchmark_Sharpe
      - "Hit_ratio_monthly": fraction of months system_outperforms benchmark
    """
    if system_equity is None or system_equity.empty:
        raise ValueError("system_equity must be provided and non-empty")
    if benchmark_equity is None or benchmark_equity.empty:
        raise ValueError("benchmark_equity must be provided and non-empty")

    m_sys = compute_metrics(system_equity)
    m_bmk = compute_metrics(benchmark_equity)

    cagr_diff = float(m_sys.get("CAGR", np.nan) - m_bmk.get("CAGR", np.nan))
    maxdd_diff = float(m_sys.get("MaxDD", np.nan) - m_bmk.get("MaxDD", np.nan))
    sharpe_diff = float(m_sys.get("Sharpe", np.nan) - m_bmk.get("Sharpe", np.nan))

    # monthly hit ratio
    sys_month = system_equity.resample("M").last().pct_change().dropna()
    bmk_month = benchmark_equity.resample("M").last().pct_change().dropna()
    common_idx = sys_month.index.intersection(bmk_month.index)
    if common_idx.empty:
        hit_ratio = float("nan")
    else:
        sys_m = sys_month.reindex(common_idx)
        bmk_m = bmk_month.reindex(common_idx)
        wins = (sys_m > bmk_m).sum()
        hit_ratio = float(wins) / float(len(common_idx))

    return {
        "CAGR_diff": cagr_diff,
        "MaxDD_diff": maxdd_diff,
        "Sharpe_diff": sharpe_diff,
        "Hit_ratio_monthly": hit_ratio,
    }
