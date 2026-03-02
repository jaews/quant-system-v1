import pandas as pd
import traceback
import sys
import importlib

if 'backtest' in sys.modules:
    importlib.reload(sys.modules['backtest'])
import backtest

print('backtest module:', backtest)
df = pd.read_csv('data/synthetic_prices.csv', index_col=0, parse_dates=True)
print('loaded prices rows:', len(df))
try:
    res = backtest.run_backtest(df)
    print('OK, keys:', list(res.keys()))
except Exception as e:
    traceback.print_exc()
    print('EXC REPR:', repr(e))
