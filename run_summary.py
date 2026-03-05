#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

from src import ui_io


DEFAULT_CACHE_PATH = os.environ.get("QUANT_CACHE_PATH", "data/prices.parquet")


def _import_core_module(module_name: str):
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    src_str = str(src_dir)
    inserted = False
    try:
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
            inserted = True
        return importlib.import_module(module_name)
    finally:
        if inserted:
            try:
                sys.path.remove(src_str)
            except ValueError:
                pass


def _format_value(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _load_cached_prices(cache_path: str) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"cache file not found: {cache_path}")
    if path.suffix.lower() not in (".parquet", ".pq"):
        raise ValueError("cache file must be parquet")

    prices = ui_io.load_prices_path(path)
    validation = _import_core_module("data_validation")
    validation.validate_price_frame(prices)
    if prices.empty:
        raise ValueError("cached prices are empty")
    return prices


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic summary from cached parquet prices")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH, help="Path to cached parquet prices")
    parser.add_argument("--config", default=None, help="JSON string or path to a JSON config file")
    parser.add_argument("--export", action="store_true", help="Export equity.csv, weights.csv, and trades.csv to ./out/")
    return parser.parse_args(argv)


def _export_outputs(result: Dict) -> None:
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = result.get("equity")
    weights = result.get("weights")
    trades = result.get("trades")

    if isinstance(equity, pd.Series):
        equity.to_csv(out_dir / "equity.csv")
    if isinstance(weights, pd.DataFrame):
        weights.to_csv(out_dir / "weights.csv")
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(out_dir / "trades.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        prices = _load_cached_prices(args.cache)
    except Exception as exc:
        print(f"Failed to load cached prices: {exc}", file=sys.stderr)
        return 2

    try:
        config = ui_io.parse_json_config(args.config)
    except Exception as exc:
        print(f"Invalid config: {exc}", file=sys.stderr)
        return 3

    try:
        backtest = _import_core_module("backtest")
        result = backtest.run_backtest(prices, config=config)
    except Exception as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 4

    equity = result.get("equity")
    trades = result.get("trades")
    metrics = result.get("metrics") or {}
    rebalance_schedule = result.get("rebalance_schedule") or []

    if not isinstance(equity, pd.Series) or equity.empty:
        print("Backtest returned an invalid equity series.", file=sys.stderr)
        return 5

    if not isinstance(trades, pd.DataFrame):
        print("Backtest returned an invalid trades table.", file=sys.stderr)
        return 6

    traded_count = int(trades["traded"].sum()) if "traded" in trades.columns and not trades.empty else 0
    total_turnover = float(trades["turnover"].sum()) if "turnover" in trades.columns and not trades.empty else 0.0
    total_cost = float(trades["cost"].sum()) if "cost" in trades.columns and not trades.empty else 0.0
    final_equity = float(equity.iloc[-1])

    print("=== Quant System Summary ===")
    print(f"Cache:        {Path(args.cache).resolve()}")
    print(f"Date range:   {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Days:         {len(prices)}")
    print(f"Final equity: {_format_value(final_equity)}")
    print(f"CAGR:         {_format_value(metrics.get('CAGR'))}")
    print(f"MaxDD:        {_format_value(metrics.get('MaxDD'))}")
    print(f"Sharpe:       {_format_value(metrics.get('Sharpe'))}")
    print(f"Calmar:       {_format_value(metrics.get('Calmar'))}")
    print(f"Worst12M:     {_format_value(metrics.get('Worst12M'))}")
    print(f"Trades:       {traded_count}/{len(rebalance_schedule)} traded")
    print(f"Turnover:     {total_turnover:.4f}")
    print(f"Total cost:   {total_cost:.6f}")

    print("\nLast 5 trades:")
    if trades.empty:
        print("none")
    else:
        display_cols = [col for col in ["decision_date", "exec_date", "traded", "turnover", "cost"] if col in trades.columns]
        print(trades[display_cols].tail(5).to_string(index=False))

    print("\nLast 5 equity values:")
    print(equity.tail(5).to_string())

    if args.export:
        try:
            _export_outputs(result)
        except Exception as exc:
            print(f"Failed to export outputs: {exc}", file=sys.stderr)
            return 7
        print("\nExported equity.csv, weights.csv, and trades.csv to ./out/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
