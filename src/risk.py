def max_drawdown(returns):
    try:
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum / peak) - 1
        return float(dd.min())
    except Exception:
        return 0.0
