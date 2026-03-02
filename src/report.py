from __future__ import annotations

from typing import Dict

import pandas as pd


def simple_report(results: Dict) -> None:
    """Print a concise performance summary including CAGR and MaxDD."""
    final = results.get('final_value', 0.0)
    pnl = results.get('pnl', 0.0)
    trades = results.get('trades', 0)
    equity: pd.Series = results.get('equity')

    print(f"Trades: {trades}")
    print(f"Final value: {final:,.2f}")
    print(f"PnL: {pnl:,.2f}")
    if isinstance(equity, pd.Series) and not equity.empty:
        # compute simple CAGR
        days = (equity.index[-1] - equity.index[0]).days
        years = max(1.0, days / 365.25)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        # max drawdown
        peak = equity.cummax()
        maxdd = float((equity / peak - 1.0).min())
        print(f"CAGR: {cagr:.2%}, MaxDD: {maxdd:.2%}")

