from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src import ui_io


DEFAULT_TICKERS = ["SPY", "TLT", "BIL", "BTC-USD"]
TICKER_OPTIONS = [
    "SPY",
    "TLT",
    "BIL",
    "BTC-USD",
    "QQQ",
    "GLD",
    "EFA",
    "VNQ",
    "DBC",
    "IEF",
    "SHY",
]


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


@st.cache_data(show_spinner=False)
def load_cached_prices_cached(
    cache_path: str,
    cache_signature: str,
    tickers: tuple[str, ...],
    start: str,
    end: str | None,
    strict_inception: bool,
) -> pd.DataFrame:
    del cache_signature
    prices = ui_io.load_prices_path(cache_path)
    return ui_io.slice_prices(prices, tickers=tickers, start=start, end=end, strict_inception=strict_inception)


@st.cache_data(show_spinner=False)
def run_backtest_cached(prices: pd.DataFrame, prices_signature: str, config_tuple: tuple[tuple[str, Any], ...]) -> Dict:
    del prices_signature
    backtest = _import_core_module("backtest")
    return backtest.run_backtest(prices, config=dict(config_tuple))


@st.cache_data(show_spinner=False)
def run_monitor_cached(prices: pd.DataFrame, equity: pd.Series, prices_signature: str) -> Dict:
    del prices_signature
    monitor = _import_core_module("monitor")
    return monitor.compute_current_state(prices, equity)


@st.cache_data(show_spinner=False)
def run_benchmark_cached(prices: pd.DataFrame, equity: pd.Series, prices_signature: str) -> Dict | None:
    del prices_signature
    try:
        benchmark = _import_core_module("benchmark")
    except Exception:
        return None

    if not all(ticker in prices.columns for ticker in ("SPY", "BIL")):
        return None

    weights = pd.Series({"SPY": 0.6, "BIL": 0.4})
    benchmark_equity = benchmark.run_benchmark(prices[["SPY", "BIL"]], weights)
    comparison = benchmark.compare_vs_benchmark(equity, benchmark_equity)
    return {
        "benchmark_equity": benchmark_equity,
        "comparison": comparison,
    }


def _fmt_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _config_from_sidebar() -> dict:
    st.sidebar.header("Backtest Config")
    target_vol = st.sidebar.number_input("target_vol", min_value=0.0, value=0.12, step=0.01, format="%.4f")
    vol_lookback = st.sidebar.number_input("vol_lookback", min_value=1, value=63, step=1)
    band = st.sidebar.number_input("band", min_value=0.0, value=0.05, step=0.01, format="%.4f")
    tx_cost = st.sidebar.number_input("tx_cost", min_value=0.0, value=0.0015, step=0.0001, format="%.6f")
    ma_window = st.sidebar.number_input("ma_window", min_value=1, value=200, step=1)
    mom_lookback = st.sidebar.number_input("mom_lookback", min_value=1, value=252, step=1)
    top_n = st.sidebar.number_input("top_n", min_value=1, value=4, step=1)

    return {
        "target_vol": float(target_vol),
        "vol_lookback": int(vol_lookback),
        "band": float(band),
        "tx_cost": float(tx_cost),
        "ma_window": int(ma_window),
        "mom_lookback": int(mom_lookback),
        "top_n": int(top_n),
    }


def _get_data_health(prices: pd.DataFrame) -> dict:
    validation = _import_core_module("data_validation")
    missing = validation.report_missing_days(prices)
    inceptions = validation.report_inception_dates(prices)
    short_history = [ticker for ticker in prices.columns if int(prices[ticker].notna().sum()) < 252]
    return {
        "missing": missing,
        "inceptions": inceptions,
        "short_history": short_history,
    }


def _load_cache_only(cache_path: str, tickers: list[str], start: str, end: str | None, strict_inception: bool) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"cache file not found: {cache_path}")
    cache_signature = ui_io.file_signature(path)
    return load_cached_prices_cached(
        cache_path=str(path),
        cache_signature=cache_signature,
        tickers=tuple(tickers),
        start=start,
        end=end,
        strict_inception=strict_inception,
    )


def _update_cache(
    cache_path: str,
    tickers: list[str],
    start: str,
    end: str | None,
    refresh: bool,
    incremental: bool,
    strict_inception: bool,
) -> pd.DataFrame:
    data_mod = _import_core_module("data")
    return data_mod.get_prices(
        tickers=tickers,
        start=start,
        end=end,
        cache_path=cache_path,
        refresh=refresh,
        incremental=incremental,
        strict_inception=strict_inception,
    )


def _store_prices(prices: pd.DataFrame, cache_path: str) -> None:
    st.session_state["loaded_prices"] = prices
    st.session_state["loaded_prices_signature"] = ui_io.dataframe_signature(prices)
    st.session_state["loaded_cache_path"] = cache_path
    st.session_state.pop("last_backtest_result", None)


def _current_prices() -> pd.DataFrame | None:
    return st.session_state.get("loaded_prices")


def _render_data_health(prices: pd.DataFrame) -> None:
    health = _get_data_health(prices)

    st.subheader("Data Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(prices))
    col2.metric("Columns", prices.shape[1])
    col3.metric("Date Range", f"{prices.index.min().date()} to {prices.index.max().date()}")

    st.caption(f"Selected tickers: {', '.join(map(str, prices.columns))}")

    left, right = st.columns(2)
    with left:
        st.write("Missing Values Per Ticker")
        st.dataframe(health["missing"])
    with right:
        st.write("Inception Dates")
        st.dataframe(health["inceptions"].to_frame(name="inception_date"))

    if health["short_history"]:
        st.warning("Tickers with <252 trading days: " + ", ".join(health["short_history"]))


def _render_summary_tab(prices: pd.DataFrame, result: Dict) -> None:
    equity = result.get("equity")
    trades = result.get("trades")
    metrics = result.get("metrics") or {}
    rebalance_schedule = result.get("rebalance_schedule") or []

    final_equity = float(equity.iloc[-1]) if isinstance(equity, pd.Series) and not equity.empty else float("nan")
    traded_count = int(trades["traded"].sum()) if isinstance(trades, pd.DataFrame) and "traded" in trades.columns and not trades.empty else 0
    total_turnover = float(trades["turnover"].sum()) if isinstance(trades, pd.DataFrame) and "turnover" in trades.columns and not trades.empty else 0.0
    total_cost = float(trades["cost"].sum()) if isinstance(trades, pd.DataFrame) and "cost" in trades.columns and not trades.empty else 0.0

    cards = st.columns(6)
    cards[0].metric("CAGR", _fmt_metric(metrics.get("CAGR")))
    cards[1].metric("MaxDD", _fmt_metric(metrics.get("MaxDD")))
    cards[2].metric("Sharpe", _fmt_metric(metrics.get("Sharpe")))
    cards[3].metric("Calmar", _fmt_metric(metrics.get("Calmar")))
    cards[4].metric("Worst12M", _fmt_metric(metrics.get("Worst12M")))
    cards[5].metric("Final Equity", _fmt_metric(final_equity))

    st.write(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    st.write(f"Trading days: {len(prices)}")
    st.write(f"Rebalances traded / total: {traded_count} / {len(rebalance_schedule)}")
    st.write(f"Total turnover: {total_turnover:.4f}")
    st.write(f"Total cost: {total_cost:.6f}")

    export_cols = st.columns(3)
    export_cols[0].download_button("Download equity.csv", ui_io.to_csv_bytes(equity), file_name="equity.csv")
    export_cols[1].download_button("Download weights.csv", ui_io.to_csv_bytes(result.get("weights")), file_name="weights.csv")
    export_cols[2].download_button(
        "Download trades.csv",
        ui_io.to_csv_bytes(result.get("trades"), index=False),
        file_name="trades.csv",
    )


def _render_equity_tab(result: Dict) -> None:
    equity = result.get("equity")
    if not isinstance(equity, pd.Series) or equity.empty:
        st.info("No equity series available.")
        return
    st.line_chart(equity.rename("equity"))
    st.dataframe(equity.tail(50).to_frame(name="equity"))


def _render_weights_tab(result: Dict) -> None:
    weights = result.get("weights")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        st.info("No weights available.")
        return

    min_date = weights.index.min().date()
    max_date = weights.index.max().date()
    selected_range = st.date_input("Date range", value=(min_date, max_date), key="weights_date_range")
    selected_tickers = st.multiselect(
        "Tickers",
        options=list(weights.columns),
        default=list(weights.columns),
        key="weights_ticker_filter",
    )

    filtered = weights.copy()
    if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
        start_dt = pd.Timestamp(selected_range[0])
        end_dt = pd.Timestamp(selected_range[1])
        filtered = filtered.loc[(filtered.index >= start_dt) & (filtered.index <= end_dt)]
    if selected_tickers:
        filtered = filtered.reindex(columns=selected_tickers)

    st.dataframe(filtered.tail(200))
    if "BIL" in weights.columns:
        st.write("BIL Weight")
        st.line_chart(filtered["BIL"] if "BIL" in filtered.columns else weights["BIL"])


def _render_trades_tab(result: Dict) -> None:
    trades = result.get("trades")
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        st.info("No trades recorded.")
        return

    filter_value = st.selectbox("Trade Filter", options=["All", "Traded", "Not Traded"], index=0)
    filtered = trades.copy()
    if filter_value == "Traded":
        filtered = filtered.loc[filtered["traded"] == True]
    elif filter_value == "Not Traded":
        filtered = filtered.loc[filtered["traded"] == False]

    if "diag" in filtered.columns:
        try:
            diag_df = pd.json_normalize(filtered["diag"]).add_prefix("diag_")
            filtered = pd.concat([filtered.drop(columns=["diag"]), diag_df], axis=1)
        except Exception:
            pass

    st.dataframe(filtered)


def _render_monitor_tab(prices: pd.DataFrame, result: Dict, prices_signature: str) -> None:
    equity = result.get("equity")
    if not isinstance(equity, pd.Series) or equity.empty:
        st.info("Monitor requires a valid equity series.")
        return

    try:
        monitor_state = run_monitor_cached(prices, equity, prices_signature)
    except Exception as exc:
        st.error("Monitor calculation failed.")
        st.exception(exc)
        return

    meta = st.columns(6)
    meta[0].metric("As Of", str(pd.Timestamp(monitor_state.get("asof")).date()))
    meta[1].metric("Realized Vol", _fmt_metric(monitor_state.get("realized_vol")))
    meta[2].metric("Vol Scale", _fmt_metric(monitor_state.get("vol_scale")))
    meta[3].metric("Drawdown", _fmt_metric(monitor_state.get("drawdown")))
    meta[4].metric("DD Bucket", str(monitor_state.get("dd_bucket", "n/a")))
    next_rebalance = monitor_state.get("next_rebalance")
    meta[5].metric("Next Rebalance", str(pd.Timestamp(next_rebalance).date()) if next_rebalance is not None else "Unavailable")

    if next_rebalance is None:
        st.warning("No future valid decision date exists in the loaded price cache.")

    alerts = monitor_state.get("alerts") or []
    st.write("Alerts")
    if alerts:
        st.write(alerts)
    else:
        st.write("None")

    target_weights = monitor_state.get("target_weights")
    if isinstance(target_weights, pd.Series):
        st.write("Target Weights")
        st.dataframe(target_weights.to_frame(name="weight"))


def _render_benchmark_tab(prices: pd.DataFrame, result: Dict, prices_signature: str) -> None:
    equity = result.get("equity")
    if not isinstance(equity, pd.Series) or equity.empty:
        st.info("Benchmark requires a valid equity series.")
        return

    try:
        benchmark_result = run_benchmark_cached(prices, equity, prices_signature)
    except Exception as exc:
        st.error("Benchmark run failed.")
        st.exception(exc)
        return

    if not benchmark_result:
        st.info("Benchmark unavailable. Ensure `src/benchmark.py` exists and SPY/BIL are present.")
        return

    comparison = benchmark_result["comparison"]
    cards = st.columns(4)
    cards[0].metric("CAGR Diff", _fmt_metric(comparison.get("CAGR_diff")))
    cards[1].metric("MaxDD Diff", _fmt_metric(comparison.get("MaxDD_diff")))
    cards[2].metric("Sharpe Diff", _fmt_metric(comparison.get("Sharpe_diff")))
    cards[3].metric("Hit Ratio Monthly", _fmt_metric(comparison.get("Hit_ratio_monthly")))

    benchmark_equity = benchmark_result.get("benchmark_equity")
    if isinstance(benchmark_equity, pd.Series) and not benchmark_equity.empty:
        st.line_chart(pd.DataFrame({"system": equity, "benchmark": benchmark_equity}))


def main() -> None:
    st.set_page_config(page_title="Quant System v1", layout="wide")
    st.title("Quant System v1")

    st.sidebar.header("Data Source")
    data_mode = st.sidebar.radio(
        "Mode",
        options=("Use cached file only (no network)", "Fetch/Update from Yahoo into cache (network allowed)"),
        index=0,
    )
    cache_path = st.sidebar.text_input("Cache Path", value="data/prices.parquet")
    tickers = st.sidebar.multiselect("Tickers", options=TICKER_OPTIONS, default=DEFAULT_TICKERS)
    start = st.sidebar.text_input("Start Date", value="2005-01-01")
    end_raw = st.sidebar.text_input("End Date (optional)", value="")
    end = end_raw.strip() or None
    refresh = st.sidebar.checkbox("refresh", value=False)
    incremental = st.sidebar.checkbox("incremental", value=True)
    strict_inception = st.sidebar.checkbox("strict_inception", value=False)

    update_clicked = st.sidebar.button("Update Cache")
    load_clicked = st.sidebar.button("Load Cache")

    config = _config_from_sidebar()
    run_clicked = st.sidebar.button("Run Backtest")

    if update_clicked:
        if data_mode != "Fetch/Update from Yahoo into cache (network allowed)":
            st.error("Switch the Data Source mode to the Yahoo update option before updating the cache.")
        elif not tickers:
            st.error("Select at least one ticker before updating the cache.")
        else:
            try:
                with st.spinner("Updating cache from Yahoo..."):
                    updated_prices = _update_cache(
                        cache_path=cache_path,
                        tickers=tickers,
                        start=start,
                        end=end,
                        refresh=refresh,
                        incremental=incremental,
                        strict_inception=strict_inception,
                    )
                _store_prices(updated_prices, cache_path)
                st.success("Cache updated and loaded.")
            except Exception as exc:
                st.error("Cache update failed.")
                st.exception(exc)

    if load_clicked:
        if not tickers:
            st.error("Select at least one ticker before loading the cache.")
        else:
            try:
                with st.spinner("Loading cached prices..."):
                    cached_prices = _load_cache_only(
                        cache_path=cache_path,
                        tickers=tickers,
                        start=start,
                        end=end,
                        strict_inception=strict_inception,
                    )
                _store_prices(cached_prices, cache_path)
                st.success("Cache loaded.")
            except Exception as exc:
                st.error("Failed to load cached prices.")
                st.exception(exc)

    prices = _current_prices()
    if prices is None:
        st.info("Load the cache or update it from Yahoo before running the backtest.")
        return

    _render_data_health(prices)

    if run_clicked:
        try:
            prices_signature = st.session_state["loaded_prices_signature"]
            result = run_backtest_cached(prices, prices_signature, ui_io.config_to_tuple(config))
            st.session_state["last_backtest_result"] = result
        except Exception as exc:
            st.error("Backtest failed.")
            st.exception(exc)

    result = st.session_state.get("last_backtest_result")
    if result is None:
        st.info("Press `Run Backtest` to generate results.")
        return

    equity = result.get("equity")
    if not isinstance(equity, pd.Series) or equity.empty:
        st.error("Backtest returned an invalid result.")
        return

    prices_signature = st.session_state["loaded_prices_signature"]
    tabs = st.tabs(["Summary", "Equity", "Weights", "Trades", "Monitor", "Benchmark"])

    with tabs[0]:
        _render_summary_tab(prices, result)
    with tabs[1]:
        _render_equity_tab(result)
    with tabs[2]:
        _render_weights_tab(result)
    with tabs[3]:
        _render_trades_tab(result)
    with tabs[4]:
        _render_monitor_tab(prices, result, prices_signature)
    with tabs[5]:
        _render_benchmark_tab(prices, result, prices_signature)


if __name__ == "__main__":
    main()
