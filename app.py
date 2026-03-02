from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from src import ui_io


def _import_core_module(module_name: str):
    """Import a core module from the repo `src/` directory using top-level imports.

    This adds the `src/` folder to `sys.path` temporarily so modules that use
    top-level imports (e.g. `from signals import ...`) work when imported here.
    """
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / 'src'
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
            except Exception:
                pass


st.set_page_config(page_title="Quant System", layout="wide")


@st.cache_data
def load_prices_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return ui_io.load_prices_bytes(file_bytes, filename)


@st.cache_data
def run_backtest_cached(file_bytes: bytes, filename: str, config_tuple: Tuple[Tuple[str, object], ...]):
    prices = ui_io.load_prices_bytes(file_bytes, filename)
    cfg = dict(config_tuple)
    # import core backtest module so its internal top-level imports resolve
    backtest = _import_core_module('backtest')
    return backtest.run_backtest(prices, config=cfg)


def sidebar_inputs() -> Dict:
    st.sidebar.title("Inputs")
    upload = st.sidebar.file_uploader("Upload prices (CSV or Parquet)", type=['csv', 'parquet', 'pq'])

    st.sidebar.markdown("---")
    st.sidebar.header("Config")
    target_vol = st.sidebar.number_input("target_vol", value=0.12, step=0.01, format="%.4f")
    band = st.sidebar.number_input("band", value=0.05, step=0.01, format="%.4f")
    tx_cost = st.sidebar.number_input("tx_cost", value=0.0015, step=0.0001, format="%.6f")
    top_n = st.sidebar.number_input("top_n", value=4, min_value=1, step=1)
    ma_window = st.sidebar.number_input("ma_window", value=200, min_value=1, step=1)
    mom_lookback = st.sidebar.number_input("mom_lookback", value=252, min_value=1, step=1)
    vol_lookback = st.sidebar.number_input("vol_lookback", value=63, min_value=1, step=1)

    pad_mode = st.sidebar.selectbox("Pad mode", options=["business", "calendar", "none"], index=0)
    pad_max_days = st.sidebar.number_input("pad_max_days", value=5, min_value=0, step=1)
    persist = st.sidebar.checkbox("Persist padded (no disk writes unless checked)", value=False)

    run = st.sidebar.button("Run Backtest")

    return {
        'upload': upload,
        'config': {
            'target_vol': float(target_vol),
            'band': float(band),
            'tx_cost': float(tx_cost),
            'top_n': int(top_n),
            'ma_window': int(ma_window),
            'mom_lookback': int(mom_lookback),
            'vol_lookback': int(vol_lookback),
            'pad_mode': pad_mode,
            'pad_max_days': int(pad_max_days),
            'persist_padded': bool(persist),
        },
        'run': run,
    }


def main() -> None:
    inputs = sidebar_inputs()
    upload = inputs['upload']
    cfg = inputs['config']

    st.title("Quant System — Backtest UI")

    if upload is None:
        st.info("Upload a local prices CSV or parquet file in the sidebar to run the backtest.\nEnsure it includes at least a 'BIL' or cash ticker.")
        return

    try:
        file_bytes = upload.read()
    except Exception as e:
        st.error("Failed to read uploaded file")
        st.exception(e)
        return

    # load prices
    try:
        prices = load_prices_cached(file_bytes, upload.name)
    except Exception as e:
        st.error("Failed to load prices file")
        st.exception(e)
        return

    # validation
    v = ui_io.validate_prices(prices, required_tickers=("BIL",))
    st.sidebar.subheader("Validation")
    st.sidebar.write(v)

    st.write(f"File: {upload.name}")
    st.write(f"Rows: {v.get('n_rows')}, Columns: {v.get('n_columns')}")

    if inputs['run']:
        # prepare config tuple for core backtest (exclude UI-only pad keys)
        config_tuple = ui_io.config_to_tuple({k: v for k, v in cfg.items() if not k.startswith('pad_') and k != 'persist_padded'})

        # Optionally auto-pad the prices in-memory if rebalance execution dates
        # would fall beyond the last available trading date (prevents ValueError).
        file_bytes_to_use = file_bytes
        pad_mode = cfg.get('pad_mode', 'business')
        pad_max = int(cfg.get('pad_max_days', 0))
        persist_padded = bool(cfg.get('persist_padded', False))

        if pad_mode != 'none' and pad_max > 0:
            try:
                backtest = _import_core_module('backtest')
                df_pad = prices.copy()
                pads = 0
                offset_cls = pd.tseries.offsets.BDay if pad_mode == 'business' else pd.tseries.offsets.Day
                while pads < pad_max:
                    try:
                        rebals = backtest.get_rebalance_schedule(df_pad.index)
                    except Exception:
                        break
                    missing = False
                    for R in rebals:
                        try:
                            backtest.next_trading_day(df_pad.index, R)
                        except ValueError:
                            missing = True
                            break
                    if not missing:
                        break
                    last = df_pad.iloc[-1:].copy()
                    nextd = df_pad.index[-1] + offset_cls(1)
                    last.index = [nextd]
                    df_pad = pd.concat([df_pad, last])
                    pads += 1
                if pads:
                    st.info(f"Auto-padded prices in-memory with {pads} extra {pad_mode} day(s) to allow post-rebalance execution.")
                    # optionally persist padded file next to input
                    if persist_padded:
                        try:
                            in_path = Path(upload.name)
                            out_name = in_path.stem + '_padded' + in_path.suffix
                            out_path = Path('') / out_name
                            if upload.name.lower().endswith(('.parquet', '.pq')):
                                df_pad.to_parquet(out_path)
                            else:
                                df_pad.to_csv(out_path)
                            st.success(f"Persisted padded prices to: {out_path}")
                        except Exception as _err:
                            st.warning(f"Failed to persist padded file: {_err}")
                    # convert to bytes for cached backtest
                    bio = io.BytesIO()
                    if upload.name.lower().endswith(('.parquet', '.pq')):
                        df_pad.to_parquet(bio)
                        file_bytes_to_use = bio.getvalue()
                    else:
                        s = df_pad.to_csv()
                        file_bytes_to_use = s.encode()
            except Exception:
                # if padding fails, fall back to original bytes and let backtest raise
                file_bytes_to_use = file_bytes

        try:
            with st.spinner('Running backtest...'):
                res = run_backtest_cached(file_bytes_to_use, upload.name, config_tuple)
        except Exception as e:
            st.error("Backtest failed — see details")
            st.exception(e)
            return

        equity = res.get('equity')
        weights = res.get('weights')
        trades = res.get('trades')
        metrics = res.get('metrics') or {}

        # Summary
        cols = st.columns(6)
        cols[0].metric("CAGR", f"{metrics.get('CAGR'):.4f}" if metrics.get('CAGR') is not None else "n/a")
        cols[1].metric("MaxDD", f"{metrics.get('MaxDD'):.4f}" if metrics.get('MaxDD') is not None else "n/a")
        cols[2].metric("Sharpe", f"{metrics.get('Sharpe'):.4f}" if metrics.get('Sharpe') is not None else "n/a")
        cols[3].metric("Calmar", f"{metrics.get('Calmar'):.4f}" if metrics.get('Calmar') is not None else "n/a")
        cols[4].metric("Worst12M", f"{metrics.get('Worst12M'):.4f}" if metrics.get('Worst12M') is not None else "n/a")
        cols[5].metric("Final equity", f"{float(equity.iloc[-1]):.4f}" if equity is not None and not equity.empty else "n/a")

        tot_turn = trades['turnover'].sum() if trades is not None and 'turnover' in trades.columns else 0.0
        tot_cost = trades['cost'].sum() if trades is not None and 'cost' in trades.columns else 0.0
        st.write(f"Total turnover: {tot_turn:.4f} — Total cost: {tot_cost:.6f}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Equity", "Weights", "Trades", "Monitor", "Benchmark"])

        with tab1:
            if equity is not None and not equity.empty:
                st.line_chart(equity.rename('equity'))
                st.subheader("Last 20 equity values")
                st.dataframe(equity.tail(20).to_frame())
            else:
                st.write("No equity data")

        with tab2:
            st.subheader("Weights")
            if weights is None or weights.empty:
                st.write("No weights available")
            else:
                min_d, max_d = weights.index.min(), weights.index.max()
                try:
                    dr = st.date_input("Date range", value=(min_d.date(), max_d.date()))
                except Exception:
                    dr = (min_d.date(), max_d.date())
                sel_tickers = st.multiselect("Tickers", options=list(weights.columns), default=list(weights.columns))
                dfw = weights.copy()
                try:
                    d0 = pd.to_datetime(dr[0])
                    d1 = pd.to_datetime(dr[1])
                    dfw = dfw[(dfw.index >= d0) & (dfw.index <= d1)]
                except Exception:
                    pass
                dfw = dfw[sel_tickers]
                st.dataframe(dfw.tail(50))

        with tab3:
            st.subheader("Trades")
            if trades is None or trades.empty:
                st.write("No trades recorded")
            else:
                tdf = trades.copy()
                if 'diag' in tdf.columns:
                    try:
                        diag_df = pd.json_normalize(tdf['diag']).add_prefix('diag_')
                        tdf = pd.concat([tdf.drop(columns=['diag']), diag_df], axis=1)
                    except Exception:
                        pass
                filter_choice = st.selectbox("Filter", options=["All", "Traded", "Not Traded"])
                if filter_choice == 'Traded':
                    tdf = tdf[tdf['traded'] == True]
                elif filter_choice == 'Not Traded':
                    tdf = tdf[tdf['traded'] == False]
                st.dataframe(tdf)

        with tab4:
            st.subheader("Monitor")
            try:
                monitor = _import_core_module('monitor')
                mstate = monitor.compute_current_state(prices, equity)
                st.json(mstate)
            except Exception as e:
                st.write("Monitor state failed:")
                st.exception(e)

        with tab5:
            st.subheader("Benchmark")
            if all(t in prices.columns for t in ("SPY", "BIL")):
                try:
                    benchmark = _import_core_module('benchmark')
                    w_b = pd.Series({'SPY': 0.6, 'BIL': 0.4})
                    b_eq = benchmark.run_benchmark(prices[['SPY', 'BIL']], w_b)
                    comp = benchmark.compare_vs_benchmark(equity, b_eq)
                    st.json(comp)
                except Exception as e:
                    st.write("Benchmark failed:")
                    st.exception(e)
            else:
                st.write("SPY and BIL not both present in prices — benchmark unavailable")

        # Exports (provide bytes, do not write to disk)
        try:
            eq_bytes = equity.to_csv().encode() if equity is not None else b''
            wt_bytes = weights.to_csv().encode() if weights is not None else b''
            tr_bytes = trades.to_csv(index=False).encode() if trades is not None else b''
            st.download_button("Download equity.csv", eq_bytes, file_name="equity.csv")
            st.download_button("Download weights.csv", wt_bytes, file_name="weights.csv")
            st.download_button("Download trades.csv", tr_bytes, file_name="trades.csv")
        except Exception:
            pass


if __name__ == '__main__':
    main()
