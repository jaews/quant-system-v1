def simple_report(results):
    print(f"Trades: {results.get('trades', 0)}, PnL: {results.get('pnl', 0):.2f}")
