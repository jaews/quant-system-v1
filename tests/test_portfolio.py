import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import pytest

from portfolio import select_top_assets, base_weights, apply_crypto_cap, normalize_weights


def test_select_top_assets_basic():
    eligible = pd.Series({'A': True, 'B': False, 'C': True, 'D': True})
    momentum = pd.Series({'A': 0.10, 'B': 0.50, 'C': float('nan'), 'D': 0.20})

    # should ignore B (ineligible) and C (nan), rank A (0.10) and D (0.20) -> D then A
    selected = select_top_assets(eligible, momentum, top_n=4)
    assert selected == ['D', 'A']

    # limit to top_n
    selected2 = select_top_assets(eligible, momentum, top_n=1)
    assert selected2 == ['D']


def test_base_weights_equal_and_remainder():
    # 4 selected -> each 0.25, BIL may be present with 0
    sel4 = ['A', 'B', 'C', 'D']
    w4 = base_weights(sel4, top_n=4, cash_ticker='BIL')
    assert pytest.approx(float(w4.sum()), rel=1e-9) == 1.0
    for t in sel4:
        assert pytest.approx(w4[t], rel=1e-9) == 0.25

    # fewer than top_n -> remainder to BIL
    sel2 = ['A', 'B']
    w2 = base_weights(sel2, top_n=4, cash_ticker='BIL')
    # each selected should be 0.25, remainder 0.5 to BIL
    assert pytest.approx(w2['A'], rel=1e-9) == 0.25
    assert pytest.approx(w2['B'], rel=1e-9) == 0.25
    assert pytest.approx(w2['BIL'], rel=1e-9) == 0.5

    # zero selected -> 100% BIL
    w0 = base_weights([], top_n=4, cash_ticker='BIL')
    assert list(w0.index) == ['BIL']
    assert pytest.approx(w0['BIL'], rel=1e-9) == 1.0

    # weights non-negative and sum to 1
    assert all(v >= 0 for v in w2.values)
    assert pytest.approx(float(w2.sum()), rel=1e-9) == 1.0


def test_apply_crypto_cap_no_change_when_under_cap():
    w = pd.Series({'SPY': 0.5, 'BTC-USD': 0.10, 'ETH-USD': 0.10, 'BIL': 0.30})
    out = apply_crypto_cap(w, cap=0.25, cash_ticker='BIL')
    # BTC+ETH unchanged
    assert pytest.approx(out['BTC-USD'] + out['ETH-USD'], rel=1e-9) == pytest.approx(0.20, rel=1e-9)
    # sum to 1
    assert pytest.approx(out.sum(), rel=1e-9) == 1.0


def test_apply_crypto_cap_reduce_and_redistribute():
    # BTC+ETH = 0.4, cap=0.25, freed=0.15 -> redistributed to SPY
    w = pd.Series({'SPY': 0.6, 'BTC-USD': 0.2, 'ETH-USD': 0.2})
    out = apply_crypto_cap(w, cap=0.25, cash_ticker='BIL')
    # BTC+ETH equals cap
    assert pytest.approx(out['BTC-USD'] + out['ETH-USD'], rel=1e-9) == pytest.approx(0.25, rel=1e-9)
    # SPY should have received the freed 0.15
    assert pytest.approx(out['SPY'], rel=1e-9) == pytest.approx(0.75, rel=1e-9)
    assert pytest.approx(out.sum(), rel=1e-9) == 1.0


def test_apply_crypto_cap_redistribute_to_bil_if_no_noncrypto():
    w = pd.Series({'BTC-USD': 0.4, 'ETH-USD': 0.4, 'BIL': 0.2})
    out = apply_crypto_cap(w, cap=0.25, cash_ticker='BIL')
    assert pytest.approx(out['BTC-USD'] + out['ETH-USD'], rel=1e-9) == pytest.approx(0.25, rel=1e-9)
    # freed weight should be added to BIL
    assert out['BIL'] > 0.2
    assert pytest.approx(out.sum(), rel=1e-9) == 1.0


def test_normalize_weights_and_zero_sum_behavior():
    w = pd.Series({'A': -0.1, 'B': 0.2, 'C': 0.3})
    norm = normalize_weights(w)
    assert pytest.approx(norm.sum(), rel=1e-9) == 1.0
    assert all(v >= 0 for v in norm.values)

    # zero sum with BIL present -> 100% BIL
    wzero = pd.Series({'A': 0.0, 'B': 0.0, 'BIL': 0.0})
    norm2 = normalize_weights(wzero)
    assert pytest.approx(norm2['BIL'], rel=1e-9) == 1.0

    # zero sum without BIL -> raises
    wzero2 = pd.Series({'A': 0.0, 'B': 0.0})
    with pytest.raises(ValueError):
        normalize_weights(wzero2)
