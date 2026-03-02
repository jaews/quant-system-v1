#!/usr/bin/env python3
"""Run a single deterministic summary of the quant system.

Usage (examples):
  python run_summary.py --prices-file prices.csv
  python run_summary.py --tickers SPY,TLT,BIL --start 2020-01-01 --end 2024-01-01

Notes:
- This script avoids network calls. Provide a local `--prices-file` (csv/parquet)
  or ensure the `data` module cache contains the requested range before using
  the `--tickers` flow. If neither is available the script fails fast.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

import pandas as pd


def format_f(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "nan"
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def load_prices_from_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prices file not found: {path}")
    if p.suffix.lower() in ('.parquet', '.pq'):
        return pd.read_parquet(p)
    # try csv
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    return df


def try_load_from_data_module(tickers, start, end):
    # Only use cache via src.data if available and cache file exists to avoid network.
    try:
        import data as data_mod
    except Exception:
        return None

    # inspect expected cache location used by data.load_prices
    try:
        project_root = Path(__file__).resolve().parents[0]
        cache_dir = project_root / 'data_cache'
        cache_file = cache_dir / f"prices_{start}_{end}.csv"
        if cache_file.exists():
            return data_mod.load_prices(tickers, start, end, use_cache=True)
    except Exception:
        return None
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description='Single-run system summary')
    p.add_argument('--prices-file', help='Local prices file (csv or parquet)')
    p.add_argument('--tickers', default='SPY,TLT,BIL', help='Comma list of tickers')
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2024-01-01')
    p.add_argument('--pad-mode', choices=['business', 'calendar', 'none'], default='business',
                   help='How to pad missing execution dates: business (BDay), calendar (Day), or none')
    p.add_argument('--pad-max-days', type=int, default=5,
                   help='Maximum number of days to auto-pad when pad-mode is enabled')
    p.add_argument('--persist-padded', action='store_true',
                   help='If set, save the padded prices to a new file next to the input (suffix _padded)')
    args = p.parse_args(argv)

    prices = None
    if args.prices_file:
        try:
            prices = load_prices_from_file(args.prices_file)
        except Exception as e:
            print(f"Failed to load prices from file: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
        prices = try_load_from_data_module(tickers, args.start, args.end)
        if prices is None or prices.empty:
            print("No local prices file provided and no cache available via src.data.", file=sys.stderr)
            print("Provide --prices-file or pre-populate data_cache for the requested range.")
            sys.exit(3)

    # Basic validations
    if prices is None or prices.empty:
        print("No prices loaded.", file=sys.stderr)
        sys.exit(4)

    # Attempt to auto-pad input prices in-memory if rebalance execution
    # would fall beyond the last available trading date. This avoids
    # failing when a rebalance decision occurs on/near the final index
    # and there's no next trading day for execution.
    try:
        import backtest as _bt
    except Exception:
        _bt = None

    def _auto_pad_prices_for_execution(prices_df: pd.DataFrame, pad_mode: str = 'business', max_pads: int = 5):
        if _bt is None or pad_mode == 'none' or max_pads <= 0:
            return prices_df, 0
        df = prices_df.copy()
        pads = 0
        # choose offset type
        if pad_mode == 'business':
            offset_cls = pd.tseries.offsets.BDay
        else:
            offset_cls = pd.tseries.offsets.Day

        while pads < max_pads:
            try:
                rebals = _bt.get_rebalance_schedule(df.index)
            except Exception:
                break
            missing = False
            for R in rebals:
                try:
                    _bt.next_trading_day(df.index, R)
                except ValueError:
                    missing = True
                    break
            if not missing:
                break
            # pad one day (business or calendar) by duplicating last row
            last = df.iloc[-1:].copy()
            nextd = df.index[-1] + offset_cls(1)
            last.index = [nextd]
            df = pd.concat([df, last])
            pads += 1
        return df, pads

    pad_mode = args.pad_mode if hasattr(args, 'pad_mode') else 'business'
    pad_max = args.pad_max_days if hasattr(args, 'pad_max_days') else 5
    prices, _pads = _auto_pad_prices_for_execution(prices, pad_mode=pad_mode, max_pads=pad_max)
    if _pads:
        print(f"Auto-padded prices in-memory with {_pads} extra {pad_mode} day(s) to allow post-rebalance execution.")
        # optionally persist padded file next to input
        try:
            if getattr(args, 'persist_padded', False) and getattr(args, 'prices_file', None):
                from pathlib import Path
                in_path = Path(args.prices_file)
                out_name = in_path.stem + '_padded' + in_path.suffix
                out_path = in_path.with_name(out_name)
                if in_path.suffix.lower() in ('.parquet', '.pq'):
                    prices.to_parquet(out_path)
                else:
                    prices.to_csv(out_path)
                print(f"Persisted padded prices to: {out_path}")
        except Exception as _err:
            print(f"Warning: failed to persist padded file: {_err}", file=sys.stderr)

    # Run backtest
    try:
        import backtest
    except Exception as e:
        print(f"failed to import backtest module: {e}", file=sys.stderr)
        sys.exit(5)

    try:
        res = backtest.run_backtest(prices)
    except Exception as e:
        print(f"backtest.run_backtest failed: {e}", file=sys.stderr)
        sys.exit(6)

    eq = res.get('equity')
    weights = res.get('weights')
    trades = res.get('trades')
    metrics = res.get('metrics') or {}

    print('\n' + '=' * 30)
    print('=== SINGLE RUN SUMMARY ===')
    print('=' * 30)
    start_date = prices.index.min()
    end_date = prices.index.max()
    print(f"Start date: {start_date}")
    print(f"End date:   {end_date}")
    print(f"Total trading days: {len(prices.index)}")
    if eq is not None and not eq.empty:
        print(f"Final equity: {format_f(eq.iloc[-1])}")
    else:
        print("Final equity: n/a")

    print('\nMetrics:')
    for k in ('CAGR', 'MaxDD', 'Sharpe', 'Calmar', 'Worst12M'):
        print(f" - {k}: {format_f(metrics.get(k))}")

    print('\nTrades summary:')
    rebals = res.get('rebalance_schedule') or []
    print(f" - Total rebalances (schedule): {len(rebals)}")

    if isinstance(trades, (list,)):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades if isinstance(trades, pd.DataFrame) else None

    if trades_df is not None and not trades_df.empty:
        traded_count = int(trades_df['traded'].sum()) if 'traded' in trades_df.columns else trades_df.shape[0]
        total_turnover = float(trades_df['turnover'].sum()) if 'turnover' in trades_df.columns else float(0.0)
        total_cost = float(trades_df['cost'].sum()) if 'cost' in trades_df.columns else float(0.0)
    else:
        traded_count = 0
        total_turnover = 0.0
        total_cost = 0.0

    print(f" - Number of actual trades: {traded_count}")
    print(f" - Total turnover: {format_f(total_turnover)}")
    print(f" - Total transaction cost: {format_f(total_cost)}")

    if trades_df is not None and not trades_df.empty:
        print('\nLast 5 trades:')
        # show subset of useful columns if present
        display_cols = [c for c in ['decision_date', 'exec_date', 'traded', 'turnover', 'cost'] if c in trades_df.columns]
        print(trades_df[display_cols].tail(5).to_string(index=False))

    print('\nLast 5 equity values:')
    if eq is not None and not eq.empty:
        print(eq.tail(5).to_string())
    else:
        print('n/a')

    # Optional: monitor
    try:
        import monitor
        mstate = monitor.compute_current_state(prices, eq)
        print('\nMonitor current state:')
        print(f" - drawdown: {format_f(mstate.get('drawdown'))}")
        print(f" - vol_scale: {format_f(mstate.get('vol_scale'))}")
        print(f" - dd_bucket: {mstate.get('dd_bucket')}")
        print(f" - next_rebalance: {mstate.get('next_rebalance')}")
        tw = mstate.get('target_weights')
        if isinstance(tw, pd.Series):
            print('\nTarget weights:')
            print(tw.to_string())
    except Exception:
        pass

    # Optional: benchmark
    try:
        import benchmark
        # attempt 60/40 SPY/BIL if present in prices
        bench_tickers = [t for t in ('SPY', 'BIL') if t in prices.columns]
        if len(bench_tickers) == 2:
            w_b = pd.Series({'SPY': 0.6, 'BIL': 0.4})
            b_eq = benchmark.run_benchmark(prices[bench_tickers], w_b)
            comp = benchmark.compare_vs_benchmark(eq, b_eq)
            print('\nBenchmark comparison (60/40 SPY/BIL):')
            for k, v in comp.items():
                print(f" - {k}: {format_f(v) if isinstance(v, (int, float)) else v}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
