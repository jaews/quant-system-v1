from data import load_sample_data
from signals import generate_signals
from backtest import run_backtest
from report import simple_report


def main():
    df = load_sample_data()
    sig = generate_signals(df)
    res = run_backtest(df, sig)
    simple_report(res)
    return res


if __name__ == "__main__":
    main()
