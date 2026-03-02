import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

out = Path('data')
out.mkdir(exist_ok=True)

# Use end date instead of fixed periods
idx = pd.bdate_range(start='2005-01-01', end=pd.Timestamp.today())
n = len(idx)

prices = pd.DataFrame(index=idx)

def regime_returns(drift, vol, length):
    return drift + np.random.normal(0, vol, size=length)

# Split dynamically based on length
r1_len = int(n * 0.40)
r2_len = int(n * 0.08)
r3_len = int(n * 0.27)
r4_len = n - (r1_len + r2_len + r3_len)

r1 = regime_returns(0.0005, 0.01, r1_len)    # bull
r2 = regime_returns(-0.002, 0.02, r2_len)    # crash
r3 = regime_returns(0.001, 0.015, r3_len)    # recovery
r4 = regime_returns(0.0, 0.008, r4_len)      # sideways

spy_returns = np.concatenate([r1, r2, r3, r4])
prices['SPY'] = 100 * np.cumprod(1 + spy_returns)

# TLT
tlt_returns = np.concatenate([
    regime_returns(0.0003, 0.006, r1_len),
    regime_returns(0.0015, 0.01, r2_len),
    regime_returns(0.0002, 0.007, r3_len),
    regime_returns(0.0001, 0.005, r4_len)
])
prices['TLT'] = 100 * np.cumprod(1 + tlt_returns)

# BIL
prices['BIL'] = 100 * np.cumprod(
    1 + 0.00005 + np.random.normal(0, 0.0001, n)
)

# BTC
btc_returns = np.random.normal(0.0008, 0.03, n)
prices['BTC-USD'] = 100 * np.cumprod(1 + btc_returns)

prices.to_csv(out / 'synthetic_regime_prices.csv')

print("Rows:", n)
print("Wrote synthetic_regime_prices.csv")
