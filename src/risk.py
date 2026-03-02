from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def realized_vol(prices: pd.DataFrame, weights: pd.Series, asof: pd.Timestamp, lookback: int = 63) -> float:
    """Annualized realized vol of the risky sleeve using data <= `asof`.

    - `weights` is a pd.Series containing tickers and weights (may include 'BIL').
    - Uses only risky sleeve (all tickers except 'BIL').
    - Uses last `lookback` daily returns ending at `asof` (inclusive if asof in index).
    - Annualizes by sqrt(252).
    - Returns np.nan when insufficient data or risky sleeve has zero weight.
    """
    if prices is None or prices.empty:
        return float('nan')
    if weights is None or weights.empty:
        return float('nan')

    # ensure asof is not after available dates
    prior = prices.index[prices.index <= asof]
    if prior.empty:
        return float('nan')
    asof_loc = prior[-1]

    if lookback <= 0:
        return float('nan')

    # determine price window that yields `lookback` returns ending at asof_loc
    end_pos = prices.index.get_loc(asof_loc)
    start_pos = max(0, end_pos - lookback)
    window = prices.iloc[start_pos:end_pos + 1]
    if window.shape[0] < lookback + 1:
        return float('nan')

    # risky tickers
    risky = [t for t in weights.index if t != 'BIL']
    risky_weights = weights.reindex(risky).fillna(0.0).astype(float)
    risky_sum = float(risky_weights.sum())
    if risky_sum <= 0:
        return float('nan')

    # normalized risky weights
    norm_w = risky_weights / risky_sum

    # daily returns
    rets = window.pct_change().dropna(how='all')
    if rets.empty:
        return float('nan')

    # align columns
    common = [c for c in rets.columns if c in norm_w.index]
    if not common:
        return float('nan')

    port_rets = rets[common].dot(norm_w.reindex(common).values).dropna()
    if port_rets.size < lookback:
        return float('nan')

    vol = float(port_rets.std(ddof=0) * np.sqrt(252))
    if vol == 0:
        return float('nan')
    return vol


def current_drawdown(equity_curve: pd.Series, asof: pd.Timestamp) -> float:
    """Compute current drawdown using equity data <= `asof`.

    Returns equity/peak - 1 for the last point <= asof. If insufficient data returns 0.0.
    """
    if equity_curve is None or equity_curve.empty:
        return 0.0
    series = equity_curve[equity_curve.index <= asof]
    if series.empty or series.shape[0] < 1:
        return 0.0
    peak = series.cummax().iloc[-1]
    if peak == 0:
        return 0.0
    return float(series.iloc[-1] / peak - 1.0)


def apply_vol_target(weights: pd.Series, prices: pd.DataFrame, asof: pd.Timestamp, target_vol: float = 0.12, lookback: int = 63) -> Tuple[pd.Series, Dict]:
    """Scale risky sleeve to meet `target_vol` using data <= `asof`.

    Returns (new_weights, diagnostics).
    Diagnostics: realized_vol, vol_scale, risky_weight_before, risky_weight_after.
    """
    if weights is None or weights.empty:
        raise ValueError("weights must be provided")
    w = weights.copy().astype(float)
    if 'BIL' not in w.index:
        w.loc['BIL'] = 0.0

    risky_idx = [k for k in w.index if k != 'BIL']
    risky_before = float(w.loc[risky_idx].sum())

    rv = realized_vol(prices, w, asof, lookback=lookback)
    if rv is None or np.isnan(rv) or rv <= 0:
        scale = 1.0
    else:
        scale = float(min(1.0, target_vol / rv))

    for k in risky_idx:
        w.loc[k] = max(0.0, w.get(k, 0.0) * scale)

    risky_after = float(w.loc[risky_idx].sum())
    # move remainder to BIL
    remainder = risky_before - risky_after
    w.loc['BIL'] = max(0.0, w.get('BIL', 0.0) + remainder)

    # small drift correction only
    total = float(w.sum())
    if total <= 0:
        # fallback to 100% BIL
        out = pd.Series({k: 0.0 for k in w.index})
        out.loc['BIL'] = 1.0
        diag = {'realized_vol': rv, 'vol_scale': scale, 'risky_weight_before': risky_before, 'risky_weight_after': float(out.loc[risky_idx].sum())}
        return out, diag

    if abs(total - 1.0) > 1e-8:
        w = w / total

    diag = {'realized_vol': rv, 'vol_scale': scale, 'risky_weight_before': risky_before, 'risky_weight_after': float(w.loc[risky_idx].sum())}
    return w, diag


def apply_drawdown_governor(weights: pd.Series, drawdown: float) -> Tuple[pd.Series, Dict]:
    """Apply drawdown governor rules and preserve risky proportions.

    Returns (new_weights, diagnostics) where diagnostics contains 'dd_rule'.
    """
    if weights is None or weights.empty:
        raise ValueError("weights must be provided")
    w = weights.copy().astype(float)
    if 'BIL' not in w.index:
        w.loc['BIL'] = 0.0

    risky_idx = [k for k in w.index if k != 'BIL']
    risky_sum = float(w.loc[risky_idx].sum())
    diag = {'drawdown': float(drawdown), 'dd_rule': 'none', 'risky_weight_before': risky_sum, 'risky_weight_after': risky_sum}

    # Recovery: no governor
    if drawdown > -0.10:
        diag['dd_rule'] = 'recovery'
        return w, diag

    # DD <= -30% -> 100% BIL
    if drawdown <= -0.30:
        out = pd.Series({k: 0.0 for k in w.index})
        out.loc['BIL'] = 1.0
        diag.update({'dd_rule': 'dd30', 'risky_weight_after': 0.0})
        return out, diag

    # DD <= -22% -> ensure BIL >= 50%
    if drawdown <= -0.22:
        desired_bil = max(w.get('BIL', 0.0), 0.5)
        remain = max(0.0, 1.0 - desired_bil)
        if risky_sum > 0:
            for k in risky_idx:
                w.loc[k] = (w.loc[k] / risky_sum) * remain
        else:
            for k in risky_idx:
                w.loc[k] = 0.0
        w.loc['BIL'] = desired_bil
        diag.update({'dd_rule': 'dd22', 'risky_weight_after': float(w.loc[risky_idx].sum())})
        return w, diag

    # DD <= -15% -> reduce risky by 30%
    if drawdown <= -0.15:
        if risky_sum > 0:
            for k in risky_idx:
                w.loc[k] = w.loc[k] * 0.7
        reduction = risky_sum - float(w.loc[risky_idx].sum())
        w.loc['BIL'] = w.get('BIL', 0.0) + reduction
        diag.update({'dd_rule': 'dd15', 'risky_weight_after': float(w.loc[risky_idx].sum())})
        return w, diag

    return w, diag


def apply_risk_controls(base_weights: pd.Series, prices: pd.DataFrame, equity_curve: pd.Series, rebalance_date: pd.Timestamp, config: Dict | None = None) -> Tuple[pd.Series, Dict]:
    """Orchestrate vol targeting and drawdown governor.

    - asof = last trading date < rebalance_date (raise ValueError if none)
    - validate base_weights sums to ~1
    - apply vol target then drawdown governor
    - final validation: weights >=0 and sum to 1 (tiny drift correction only)
    """
    if base_weights is None or base_weights.empty:
        raise ValueError("base_weights must be provided")
    if prices is None or prices.empty:
        raise ValueError("prices must be provided")
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve must be provided")

    # determine asof: last trading date strictly before rebalance_date
    prior = prices.index[prices.index < rebalance_date]
    if prior.empty:
        raise ValueError("no available trading date before rebalance_date")
    asof = prior[-1]

    w0 = base_weights.copy().astype(float)
    if 'BIL' not in w0.index:
        w0.loc['BIL'] = 0.0

    total0 = float(w0.sum())
    if not (abs(total0 - 1.0) <= 1e-6):
        raise ValueError("base_weights must sum to 1")

    cfg = {'target_vol': 0.12, 'lookback': 63}
    if config:
        cfg.update(config)

    # vol targeting
    w_vol, diag_vol = apply_vol_target(w0, prices, asof, target_vol=cfg['target_vol'], lookback=cfg['lookback'])

    # drawdown
    dd = current_drawdown(equity_curve, asof)
    w_dd, diag_dd = apply_drawdown_governor(w_vol, dd)

    # final validation and tiny drift correction
    w_final = w_dd.copy().astype(float)
    w_final = w_final.reindex(w0.index).fillna(0.0)
    w_final = w_final.clip(lower=0.0)
    total = float(w_final.sum())
    if total <= 0:
        raise ValueError("final weights sum to zero")
    if abs(total - 1.0) > 1e-8:
        w_final = w_final / total

    diagnostics = {
        'asof': asof,
        'realized_vol': diag_vol.get('realized_vol'),
        'vol_scale': diag_vol.get('vol_scale'),
        'drawdown': float(dd),
        'dd_rule': diag_dd.get('dd_rule'),
        'risky_weight_before': diag_vol.get('risky_weight_before'),
        'risky_weight_after': diag_dd.get('risky_weight_after') if 'risky_weight_after' in diag_dd else float(w_final.drop(labels=['BIL'], errors='ignore').sum())
    }
    return w_final, diagnostics


