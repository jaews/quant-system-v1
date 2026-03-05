from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signals import compute_eligibility_and_momentum
from portfolio import build_target_weights
from risk import apply_risk_controls, current_drawdown
from backtest import get_rebalance_schedule


def get_last_monday(prices_index: pd.DatetimeIndex) -> pd.Timestamp:
    if prices_index is None or len(prices_index) == 0:
        raise ValueError("prices_index is empty")
    idx = pd.DatetimeIndex(sorted(set(prices_index)))
    mondays = idx[idx.weekday == 0]
    if mondays.empty:
        raise ValueError("no Monday in index")
    return mondays[-1]


def _aligned_equity_curve(prices_index: pd.DatetimeIndex, equity_curve: pd.Series, asof: pd.Timestamp) -> pd.Series:
    if equity_curve is None or equity_curve.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(sorted(set(prices_index)))
    eq = equity_curve.sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    aligned_idx = idx[idx <= asof].intersection(eq.index)
    if aligned_idx.empty:
        return pd.Series(dtype=float)
    return eq.reindex(aligned_idx).dropna().astype(float)


def _drawdown_state(equity_curve: pd.Series, asof: pd.Timestamp, recovery_lookback: int) -> tuple[float, float]:
    if equity_curve is None or equity_curve.empty:
        return 0.0, 0.0
    dd_series = equity_curve / equity_curve.cummax() - 1.0
    if dd_series.empty:
        return 0.0, 0.0
    current_dd = float(current_drawdown(equity_curve, asof))
    lookback = max(int(recovery_lookback), 1)
    trailing_min_dd = float(dd_series.iloc[-lookback:].min())
    return current_dd, trailing_min_dd


def _dd_bucket_from_drawdown(drawdown: float, trailing_min_dd: float) -> str:
    if trailing_min_dd <= -0.10 and drawdown > -0.10:
        return "recovery"
    if drawdown <= -0.30:
        return "dd30"
    if drawdown <= -0.22:
        return "dd22"
    if drawdown <= -0.15:
        return "dd15"
    return "normal"


def next_rebalance_date(prices_index: pd.DatetimeIndex, asof: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first valid decision date in the schedule strictly after `asof`."""
    if prices_index is None or len(prices_index) == 0:
        return None
    idx = pd.DatetimeIndex(sorted(set(prices_index)))
    asof = pd.Timestamp(asof)

    for d in get_rebalance_schedule(idx):
        if d > asof:
            return d
    return None


def compute_current_state(prices: pd.DataFrame, equity_curve: pd.Series, config: Optional[Dict] = None) -> Dict:
    """Compute monitoring state as-of last available trading date.

    Returns dict with keys described in spec. Pure function, no I/O.
    """
    if prices is None or prices.empty:
        raise ValueError("prices must be provided and non-empty")
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve must be provided and non-empty")

    cfg = {
        "ma_window": 200,
        "mom_lookback": 252,
        "top_n": 4,
        "vol_lookback": 63,
        "target_vol": 0.12,
        "crypto_cap": 0.25,
        "bil_ticker": "BIL",
        "recovery_lookback": 252,
    }
    if config:
        cfg.update(config)

    idx = prices.index.sort_values()
    asof = idx[-1]
    prices_asof = prices.loc[idx[idx <= asof]]
    equity_asof = _aligned_equity_curve(idx, equity_curve, asof)

    # compute signals as-of asof
    elig = compute_eligibility_and_momentum(prices_asof, as_of=asof, ma_window=cfg["ma_window"], momentum_window=cfg["mom_lookback"])

    base_w = build_target_weights(elig, top_n=cfg["top_n"], cash_ticker=cfg["bil_ticker"]) if not elig.empty else pd.Series({cfg["bil_ticker"]: 1.0})

    next_reb = next_rebalance_date(idx, asof)

    # Risk controls need a date strictly after the monitoring as-of so they evaluate
    # using data up to `asof`.
    rebalance_date = next_reb if next_reb is not None and next_reb > asof else (asof + pd.Timedelta(days=1))

    # apply risk controls (will validate inside)
    risk_equity = equity_asof if not equity_asof.empty else pd.Series([1.0], index=pd.DatetimeIndex([asof]))
    target_w, diag = apply_risk_controls(base_w, prices_asof, risk_equity, rebalance_date, config={"target_vol": cfg["target_vol"], "lookback": cfg["vol_lookback"]})

    # compute realized vol and drawdown explicitly
    rv = diag.get("realized_vol")
    vol_scale = diag.get("vol_scale")
    dd, trailing_min_dd = _drawdown_state(equity_asof, asof, cfg["recovery_lookback"])
    dd_bucket = _dd_bucket_from_drawdown(dd, trailing_min_dd)

    # alerts
    alerts: List[str] = []
    last_state = (config or {}).get("last_state") or {}
    last_dd_bucket = last_state.get("dd_bucket")
    if last_dd_bucket is not None and last_dd_bucket != dd_bucket:
        alerts.append(f"drawdown_bucket_changed:{last_dd_bucket}->{dd_bucket}")

    if next_reb is None:
        alerts.append("next_rebalance_unavailable")

    if vol_scale is not None and vol_scale < 1.0:
        alerts.append("vol_scale_below_1")

    if dd_bucket in {"dd15", "dd22", "dd30"}:
        alerts.append(f"drawdown_regime:{dd_bucket}")
    elif dd_bucket == "recovery":
        alerts.append("drawdown_recovery")

    # asset dropped from eligibility
    prev_elig = last_state.get("eligible") if last_state else None
    if prev_elig is not None and isinstance(prev_elig, (dict, pd.Series)):
        prev_set = set([k for k, v in dict(prev_elig).items() if bool(v)])
        curr_set = set([k for k, v in dict(elig["eligible"]).items() if bool(v)]) if not elig.empty and "eligible" in elig.columns else set()
        dropped = prev_set - curr_set
        if dropped:
            alerts.append(f"assets_dropped:{sorted(list(dropped))}")

    # BIL heavy
    bil_w = float(target_w.get(cfg["bil_ticker"], 0.0))
    if bil_w >= 0.5:
        alerts.append("bil_ge_50pct")

    # band_check vs last executed weights if provided
    band_check = None
    last_exec = (config or {}).get("last_executed_weights")
    if isinstance(last_exec, (pd.Series, dict)):
        last_exec_s = pd.Series(last_exec).reindex(target_w.index).fillna(0.0).astype(float)
        diff = (target_w - last_exec_s).abs()
        band_check = float(diff.max())

    return {
        "asof": asof,
        "target_weights": target_w.copy(),
        "realized_vol": float(rv) if rv is not None else float("nan"),
        "vol_scale": float(vol_scale) if vol_scale is not None else float("nan"),
        "drawdown": float(dd) if dd is not None else float("nan"),
        "dd_bucket": dd_bucket,
        "next_rebalance": next_reb,
        "band_check": band_check,
        "alerts": alerts,
    }
