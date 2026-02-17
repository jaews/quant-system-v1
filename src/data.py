import pandas as pd
import numpy as np


def load_sample_data(n=200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
    # simple random-walk price series
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"date": dates, "close": prices}).set_index("date")
    return df
