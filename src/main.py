from data import load_prices
from backtest import run_backtest
from report import simple_report


def main():
    # minimal runner: load prices for the universe defined in SYSTEM_SPEC.md
    universe = ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC", "BTC-USD", "ETH-USD", "BIL"]
    prices = load_prices(universe, "2015-01-01", "2024-01-01")
    res = run_backtest(prices)
    simple_report(res)
    return res


if __name__ == "__main__":
    main()
