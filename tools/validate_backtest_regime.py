import sys
sys.path.insert(0, 'src')
import importlib
import pandas as pd

backtest = importlib.import_module('backtest')

df = pd.read_csv('data/synthetic_regime_prices.csv', index_col=0, parse_dates=True)
res = backtest.run_backtest(df)
equity = res.get('equity')
trades = res.get('trades')
metrics = res.get('metrics') or {}

print('CAGR', f"{metrics.get('CAGR'):.4f}")
print('MaxDD', f"{metrics.get('MaxDD'):.4f}")
print('Sharpe', f"{metrics.get('Sharpe'):.4f}")
print('Calmar', f"{metrics.get('Calmar'):.4f}")
print('Worst12M', f"{metrics.get('Worst12M'):.4f}")
print('Final equity', f"{float(equity.iloc[-1]):.4f}")

tot_turn = trades['turnover'].sum() if trades is not None and 'turnover' in trades.columns else 0.0
tot_cost = trades['cost'].sum() if trades is not None and 'cost' in trades.columns else 0.0
print('Total turnover', f"{tot_turn:.4f}")
print('Total cost', f"{tot_cost:.6f}")
