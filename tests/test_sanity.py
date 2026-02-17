import os
import sys

# ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import data
import signals
import backtest


def test_sanity_run_backtest():
    df = data.load_sample_data(n=50, seed=1)
    sig = signals.generate_signals(df, short=3, long=7)
    res = backtest.run_backtest(df, sig)
    assert isinstance(res, dict)
    assert 'pnl' in res and 'trades' in res
    assert res['trades'] >= 0
