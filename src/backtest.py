from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from signals import compute_eligibility_and_momentum
from portfolio import build_target_weights
from risk import apply_risk_controls


def get_rebalance_schedule(prices_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Last trading Friday of each executable month in `prices_index`.

    Fallback: if no Friday in that month's trading days, use last trading day of month.
    Terminal month-end dates are excluded when there is no later trading day available
    to execute the rebalance.
    """
    if prices_index is None or len(prices_index) == 0:
        return []
    idx = pd.DatetimeIndex(sorted(set(prices_index)))
    last_idx = idx[-1]
    months = idx.to_period("M").unique()
    rebalance_dates: List[pd.Timestamp] = []
    for m in months:
        month_idx = idx[idx.to_period("M") == m]
        # Fridays weekday == 4
        fridays = month_idx[month_idx.weekday == 4]
        if not fridays.empty:
            candidate = fridays[-1]
        else:
            # fallback to last trading day of month
            candidate = month_idx[-1]
        if candidate < last_idx:
            rebalance_dates.append(candidate)
    return rebalance_dates


def next_trading_day(prices_index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    """Return first trading date strictly greater than `date`.

    Raises ValueError if none found.
    """
    if prices_index is None or len(prices_index) == 0:
        raise ValueError("prices_index is empty")
    later = prices_index[prices_index > pd.Timestamp(date)]
    if later.empty:
        raise ValueError(f"no trading day after {date}")
    return later[0]


def compute_equity_curve(prices: pd.DataFrame, weights_by_day: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.Series:
    """Compute equity curve starting at 1.0.

    Portfolio return on day t uses weights held at start of day t (weights.shift(1)).
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    # align index range
    px = prices.copy()
    if start is not None:
        px = px.loc[px.index >= pd.Timestamp(start)]
    if end is not None:
        px = px.loc[px.index <= pd.Timestamp(end)]
    if px.empty:
        return pd.Series(dtype=float)

    rets = px.pct_change().fillna(0.0)
    w = weights_by_day.reindex(px.index).ffill().fillna(0.0)
    # use weights held at start of day -> shift down
    w_shift = w.shift(1).ffill().fillna(0.0)
    daily_port = (w_shift * rets).sum(axis=1)
    equity = (1.0 + daily_port).cumprod()
    equity.name = "equity"
    return equity


def _max_drawdown_from_series(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(equity: pd.Series) -> Dict:
    """Compute performance metrics; defensive for short series."""
    out: Dict = {}
    if equity is None or equity.empty:
        keys = ["CAGR", "MaxDD", "Calmar", "Sharpe", "Worst12M", "Rolling3Y_CAGR_last", "Rolling3Y_MaxDD_last"]
        return {k: float("nan") for k in keys}

    # daily returns
    rets = equity.pct_change().dropna()
    n = rets.shape[0]
    days = float(len(equity))

    # CAGR
    try:
        if days <= 0:
            cagr = float("nan")
        else:
            cagr = float(equity.iloc[-1] ** (252.0 / days) - 1.0)
    except Exception:
        cagr = float("nan")

    maxdd = _max_drawdown_from_series(equity)

    calmar = float(cagr / abs(maxdd)) if (not np.isnan(cagr) and maxdd < 0) else float("nan")

    if rets.std(ddof=0) == 0 or rets.empty:
        sharpe = float("nan")
    else:
        sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(252.0))

    # Worst 12M: worst rolling 252-day return (equity / equity.shift(252) - 1)
    if len(equity) >= 252:
        roll_252 = equity / equity.shift(252) - 1.0
        worst12 = float(roll_252.min())
    else:
        worst12 = float("nan")

    # Rolling 3Y metrics (last value) using 252*3 days
    window3y = 252 * 3
    if len(equity) >= window3y:
        # compute CAGR over rolling windows (use positional access to avoid label-based KeyError)
        roll_cagr = equity.rolling(window3y).apply(lambda x: float(x.iloc[-1] ** (252.0 / float(len(x))) - 1.0), raw=False)
        # For stability, provide last values
        roll3_cagr_last = float(roll_cagr.dropna().iloc[-1])
        # compute maxdd for last 3y window
        last_window = equity.iloc[-window3y:]
        roll3_maxdd_last = _max_drawdown_from_series(last_window)
    else:
        roll3_cagr_last = float("nan")
        roll3_maxdd_last = float("nan")

    out.update({
        "CAGR": cagr,
        "MaxDD": float(maxdd),
        "Calmar": calmar,
        "Sharpe": sharpe,
        "Worst12M": worst12,
        "Rolling3Y_CAGR_last": roll3_cagr_last,
        "Rolling3Y_MaxDD_last": roll3_maxdd_last,
    })
    return out


def run_backtest(prices: pd.DataFrame, config: Dict | None = None) -> Dict:
    """Run monthly rebalance backtest per SYSTEM_SPEC.md v1.

    Returns dict with keys: equity (Series), weights (DataFrame by day), rebalance_schedule (list), trades (DataFrame), metrics (dict)
    """
    if prices is None or prices.empty:
        raise ValueError("prices must be provided and non-empty")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index must be sorted ascending and contain unique trading days")

    cfg = {
        "ma_window": 200,
        "mom_lookback": 252,
        "top_n": 4,
        "vol_lookback": 63,
        "target_vol": 0.12,
        "band": 0.05,
        "tx_cost": 0.0015,
        "bil_ticker": "BIL",
    }
    if config:
        cfg.update(config)

    idx = prices.index
    rebalance_dates = get_rebalance_schedule(idx)

    # prepare weights_by_day starting fully in BIL
    cols = list(prices.columns)
    if cfg["bil_ticker"] not in cols:
        cols.append(cfg["bil_ticker"])  # ensure BIL exists in columns
        prices = prices.copy()
        prices[cfg["bil_ticker"]] = prices.iloc[:, 0] * 0 + 100.0

    # initialize current_weights to 100% BIL
    current_weights = pd.Series(0.0, index=cols, dtype=float)
    current_weights.loc[cfg["bil_ticker"]] = 1.0

    weights_by_day = pd.DataFrame(index=idx, columns=cols, dtype=float)
    # fill with initial weights
    weights_by_day.loc[:, :] = current_weights.values

    trades_records = []

    for R in rebalance_dates:
        # determine asof: last trading day strictly before R
        prior = idx[idx < R]
        if prior.empty:
            raise ValueError(f"no available trading date before rebalance date {R}")
        asof = prior[-1]

        # compute equity curve up to asof using current weights_by_day
        equity_so_far = compute_equity_curve(prices[cols], weights_by_day.loc[:asof])

        # signals -> eligibility/momentum table
        elig = compute_eligibility_and_momentum(prices[cols], as_of=asof, ma_window=cfg["ma_window"], momentum_window=cfg["mom_lookback"])

        # base weights
        base_w = build_target_weights(elig, top_n=cfg["top_n"], cash_ticker=cfg["bil_ticker"]) if not elig.empty else pd.Series({cfg["bil_ticker"]: 1.0})

        # apply risk controls (uses prices and equity_so_far and asof via rebalance_date)
        target_w, diag = apply_risk_controls(base_w, prices[cols], equity_so_far, R, config={"target_vol": cfg["target_vol"], "lookback": cfg["vol_lookback"]})

        # ensure BIL present and align to cols
        if cfg["bil_ticker"] not in target_w.index:
            target_w.loc[cfg["bil_ticker"]] = 0.0
        target_w = target_w.reindex(cols).fillna(0.0).astype(float)

        # execution date E
        try:
            E = next_trading_day(idx, R)
        except ValueError:
            raise ValueError(f"no execution date after rebalance decision {R}")

        # current weights at start of E are weights_by_day.loc[E]
        current_at_E = weights_by_day.loc[E].astype(float)
        diff = target_w - current_at_E
        maxdiff = float(np.max(np.abs(diff.values)))

        traded = False
        turnover = 0.0
        cost = 0.0

        if maxdiff > cfg["band"]:
            traded = True
            turnover = float(np.abs(diff).sum())
            cost = float(cfg["tx_cost"] * turnover)
            # apply target weights from E onward
            weights_by_day.loc[E:, :] = target_w.values
            current_weights = target_w.copy()
        else:
            # do not trade; weights remain as before
            traded = False
            turnover = 0.0
            cost = 0.0

        trades_records.append({
            "decision_date": R,
            "exec_date": E,
            "target_weights": target_w,
            "traded": traded,
            "turnover": turnover,
            "cost": cost,
            **{f"diag_{k}": v for k, v in diag.items()},
        })

    # compute equity from final weights_by_day
    equity = compute_equity_curve(prices[cols], weights_by_day)

    # apply costs as multiplicative hit on exec dates (affects equity from exec date onward)
    for rec in trades_records:
        c = rec.get("cost", 0.0)
        if c and c > 0:
            E = rec["exec_date"]
            if E in equity.index:
                equity.loc[E:] = equity.loc[E:] * (1.0 - float(c))

    metrics = compute_metrics(equity)

    # build trades DataFrame
    trades_df = pd.DataFrame([
        {
            "decision_date": r["decision_date"],
            "exec_date": r["exec_date"],
            "traded": r["traded"],
            "turnover": r["turnover"],
            "cost": r["cost"],
            "diag": {k: v for k, v in r.items() if k.startswith("diag_")},
        }
        for r in trades_records
    ])

    return {
        "equity": equity,
        "weights": weights_by_day,
        "rebalance_schedule": rebalance_dates,
        "trades": trades_df,
        "metrics": metrics,
    }
