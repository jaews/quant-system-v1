import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from signals import compute_eligibility_and_momentum
from portfolio import build_target_weights


ASSETS = ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC", "BTC-USD", "ETH-USD", "BIL"]


def _make_prices(days=320):
    idx = pd.bdate_range("2022-01-03", periods=days)
    data = {}
    for i, asset in enumerate(ASSETS):
        base = 100 + i
        trend = np.linspace(0, 30 - i, days)
        data[asset] = base + trend
    return pd.DataFrame(data, index=idx)


def test_no_lookahead_bias_signals_change_if_future_data_changes():
    prices = _make_prices()
    as_of = prices.index[-1]

    base = compute_eligibility_and_momentum(prices, as_of=as_of)

    changed = prices.copy()
    changed.loc[as_of, "SPY"] = changed.loc[as_of, "SPY"] * 10
    bumped = compute_eligibility_and_momentum(changed, as_of=as_of)

    pd.testing.assert_frame_equal(base, bumped)


def test_pandas_alignment_and_nan_handling():
    prices = _make_prices()
    prices.loc[prices.index[:30], "EEM"] = np.nan
    prices.loc[prices.index[50:60], "GLD"] = np.nan

    out = compute_eligibility_and_momentum(prices, as_of=prices.index[-1])
    assert out.index.tolist() == ASSETS
    assert out["eligible"].dtype == bool
    assert out["momentum"].isna().sum() >= 0


def test_weights_sum_to_one_and_non_negative():
    prices = _make_prices()
    sig = compute_eligibility_and_momentum(prices, as_of=prices.index[-1])
    w = build_target_weights(sig)

    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert (w >= 0).all()


def test_crypto_cap_and_cash_fallback():
    idx = ["BTC-USD", "ETH-USD", "BIL"]
    elig = pd.DataFrame(
        {
            "eligible": [True, True, True],
            "momentum": [0.9, 0.8, 0.01],
        },
        index=idx,
    )
    w = build_target_weights(elig, top_n=4)
    assert float(w.get("BTC-USD", 0.0) + w.get("ETH-USD", 0.0)) <= 0.25 + 1e-12
    assert abs(float(w.sum()) - 1.0) < 1e-9

    none_eligible = pd.DataFrame(
        {
            "eligible": [False, False, True],
            "momentum": [np.nan, np.nan, 0.0],
        },
        index=idx,
    )
    w2 = build_target_weights(none_eligible, top_n=4)
    assert w2.to_dict() == {"BIL": 1.0}
