from __future__ import annotations

from typing import List, Sequence

import pandas as pd


def select_top_assets(eligible_row: pd.Series, momentum_row: pd.Series, top_n: int = 4) -> List[str]:
    if eligible_row is None or momentum_row is None:
        return []
    common = eligible_row.index.intersection(momentum_row.index)
    if common.empty:
        return []
    elig = eligible_row.loc[common].astype(bool)
    mom = momentum_row.loc[common]
    mask = elig & mom.notna()
    ranked = mom[mask].sort_values(ascending=False)
    return list(ranked.index[:top_n])


def base_weights(selected: Sequence[str], top_n: int = 4, cash_ticker: str = "BIL") -> pd.Series:
    if not selected:
        return pd.Series({cash_ticker: 1.0})
    n_selected = len(selected)
    weight_per_selected = 1.0 / top_n
    weights = {t: weight_per_selected for t in selected}
    assigned = weight_per_selected * n_selected
    remainder = max(0.0, 1.0 - assigned)
    weights[cash_ticker] = weights.get(cash_ticker, 0.0) + remainder
    return pd.Series(weights)


def apply_crypto_cap(weights: pd.Series, btc_ticker: str = "BTC-USD", eth_ticker: str = "ETH-USD", cap: float = 0.25, cash_ticker: str = "BIL") -> pd.Series:
    if weights is None:
        return pd.Series()
    w = weights.copy().astype(float)
    for t in (btc_ticker, eth_ticker, cash_ticker):
        if t not in w.index:
            w.loc[t] = 0.0
    btc = float(w.loc[btc_ticker])
    eth = float(w.loc[eth_ticker])
    combined = btc + eth
    if combined <= cap or combined == 0.0:
        return normalize_weights(w)
    scale = cap / combined
    w.loc[btc_ticker] = btc * scale
    w.loc[eth_ticker] = eth * scale
    freed = combined - (w.loc[btc_ticker] + w.loc[eth_ticker])
    non_crypto = [k for k in w.index if k not in {btc_ticker, eth_ticker, cash_ticker} and w.loc[k] > 0]
    if non_crypto:
        total_non_crypto = float(w.loc[non_crypto].sum())
        if total_non_crypto > 0:
            for k in non_crypto:
                w.loc[k] = w.loc[k] + (w.loc[k] / total_non_crypto) * freed
        else:
            w.loc[cash_ticker] = w.get(cash_ticker, 0.0) + freed
    else:
        w.loc[cash_ticker] = w.get(cash_ticker, 0.0) + freed
    return normalize_weights(w)


def normalize_weights(weights: pd.Series) -> pd.Series:
    if weights is None:
        raise ValueError("weights must be provided")
    w = weights.copy()
    if not isinstance(w, pd.Series):
        w = pd.Series(w)
    w = w.clip(lower=0.0)
    total = float(w.sum())
    if total == 0.0:
        if 'BIL' in w.index:
            out = pd.Series({k: 0.0 for k in w.index})
            out.loc['BIL'] = 1.0
            return out
        raise ValueError("Sum of weights is zero and 'BIL' is not available to allocate cash")
    return w / total


def build_target_weights(eligibility: pd.DataFrame, top_n: int = 4, cash_ticker: str = "BIL", crypto_tickers: tuple[str, ...] = ("BTC-USD", "ETH-USD"), crypto_cap: float = 0.25) -> pd.Series:
    if eligibility is None or eligibility.empty:
        return pd.Series({cash_ticker: 1.0}, dtype=float)
    table = eligibility.copy()
    table["momentum"] = pd.to_numeric(table.get("momentum"), errors="coerce")
    eligible = table[table["eligible"].fillna(False)]
    eligible = eligible.drop(index=[cash_ticker], errors="ignore")
    eligible = eligible.dropna(subset=["momentum"]) if "momentum" in eligible else eligible
    selected = eligible.sort_values("momentum", ascending=False).head(top_n)
    if selected.empty:
        return pd.Series({cash_ticker: 1.0}, dtype=float)
    weights = pd.Series(0.0, index=table.index, dtype=float)
    risky_names = list(selected.index)
    equal_weight = 1.0 / top_n
    for name in risky_names:
        weights.loc[name] = equal_weight
    cash_weight = 1.0 - equal_weight * len(risky_names)
    weights.loc[cash_ticker] = weights.get(cash_ticker, 0.0) + max(cash_weight, 0.0)
    # enforce crypto cap
    crypto_names = [c for c in crypto_tickers if c in weights.index]
    crypto_weight = float(weights.loc[crypto_names].sum()) if crypto_names else 0.0
    if crypto_weight > crypto_cap:
        scale = crypto_cap / crypto_weight
        weights.loc[crypto_names] = weights.loc[crypto_names] * scale
        freed = crypto_weight - float(weights.loc[crypto_names].sum())
        non_crypto_selected = [a for a in risky_names if a not in crypto_tickers]
        if non_crypto_selected:
            bump = freed / len(non_crypto_selected)
            for asset in non_crypto_selected:
                weights.loc[asset] += bump
        else:
            weights.loc[cash_ticker] = weights.get(cash_ticker, 0.0) + freed
    weights = weights.fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series({cash_ticker: 1.0}, dtype=float)
    weights = weights / total
    if cash_ticker not in weights.index:
        weights.loc[cash_ticker] = 0.0
    weights = weights.fillna(0.0).clip(lower=0.0)
    weights = weights / float(weights.sum())
    return weights

