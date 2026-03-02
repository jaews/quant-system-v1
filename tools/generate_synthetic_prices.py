import pandas as pd
import numpy as np
from pathlib import Path

out = Path('data')
out.mkdir(exist_ok=True)
idx = pd.bdate_range('2020-01-01', periods=1250)  # ~5 years of trading days
tickers = ['SPY','TLT','BIL','BTC-USD']
prices = pd.DataFrame(index=idx)
# SPY: small positive drift with low vol
prices['SPY'] = 100 * (1 + 0.0005) ** np.arange(len(idx)) * np.cumprod(1 + np.random.normal(0, 0.001, size=len(idx)))
# TLT: small positive drift
prices['TLT'] = 100 * (1 + 0.0003) ** np.arange(len(idx)) * np.cumprod(1 + np.random.normal(0, 0.0008, size=len(idx)))
# BIL: flat cash
prices['BIL'] = 100.0
# BTC-USD: low volatility stable series
prices['BTC-USD'] = 100 * (1 + 0.0002) ** np.arange(len(idx)) * np.cumprod(1 + np.random.normal(0, 0.002, size=len(idx)))

# Save CSV
prices.to_csv(out / 'synthetic_prices.csv')
print('Wrote', str(out / 'synthetic_prices.csv'))
