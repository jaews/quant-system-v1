import pandas as pd
from pathlib import Path
p = Path('data/synthetic_prices.csv')
if not p.exists():
    raise SystemExit('source file missing')

df = pd.read_csv(p, index_col=0, parse_dates=True)
last = df.iloc[-1:].copy()
nextd = df.index[-1] + pd.tseries.offsets.BDay(1)
last.index = [nextd]
df2 = pd.concat([df, last])
out = Path('data/synthetic_prices_padded.csv')
df2.to_csv(out)
print(f'wrote {out}')
