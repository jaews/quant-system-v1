from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import warnings

import pandas as pd

from data_validation import report_inception_dates, validate_price_frame


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for ticker in tickers:
        name = str(ticker).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _as_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _format_date(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _extract_adj_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)

    adj = None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            adj = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
        elif "Adj Close" in raw.columns.get_level_values(-1):
            adj = raw.xs("Adj Close", axis=1, level=-1, drop_level=True)
    elif "Adj Close" in raw.columns:
        adj = raw.loc[:, ["Adj Close"]]

    if adj is None:
        raise ValueError("yfinance response does not contain 'Adj Close'")

    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=tickers[0])
    elif list(adj.columns) == ["Adj Close"] and len(tickers) == 1:
        adj.columns = [tickers[0]]

    normalized_cols: List[str] = []
    for col in adj.columns:
        if isinstance(col, tuple):
            matched = None
            for part in col:
                if str(part) in tickers:
                    matched = str(part)
                    break
            normalized_cols.append(matched or str(col[-1]))
        else:
            normalized_cols.append(str(col))
    adj.columns = normalized_cols

    return adj.reindex(columns=tickers)


def _download_adj_close(tickers: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close data from Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()

    try:
        import yfinance as yf
    except Exception as exc:
        raise ImportError("yfinance is required to download price data") from exc

    raw = yf.download(
        tickers=tickers,
        start=_format_date(start),
        end=_format_date(end),
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=True,
        group_by="column",
    )
    return _extract_adj_close(raw, tickers)


def _forward_fill_within_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if isinstance(df, pd.DataFrame) else None)

    filled = df.copy()
    for col in filled.columns:
        series = filled[col]
        first_valid = series.first_valid_index()
        if first_valid is None:
            continue
        mask = filled.index >= first_valid
        filled.loc[mask, col] = series.loc[mask].ffill()
    return filled


def _clean_prices(df: pd.DataFrame, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Normalize price data while preserving pre-inception NaNs."""
    if df is None or df.empty:
        columns = _normalize_tickers(tickers or [])
        return pd.DataFrame(columns=columns, dtype=float)

    cleaned = df.copy()
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned.loc[~cleaned.index.isna()]
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_convert(None).tz_localize(None)

    cleaned = cleaned.sort_index()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce").astype(float)
    cleaned = _forward_fill_within_history(cleaned)
    cleaned = cleaned.dropna(how="all")

    columns = _normalize_tickers(tickers or cleaned.columns.tolist())
    if columns:
        cleaned = cleaned.reindex(columns=columns)
    return cleaned


def _load_cached_prices(cache_path: str) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame()
    cached = pd.read_parquet(path)
    return _clean_prices(cached)


def _save_cached_prices(prices: pd.DataFrame, cache_path: str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)


def _slice_date_range(prices: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = _as_timestamp(end)
    sliced = prices.loc[prices.index >= start_ts]
    if end_ts is not None:
        sliced = sliced.loc[sliced.index <= end_ts]
    return sliced


def _warn_for_short_history(prices: pd.DataFrame) -> None:
    for col in prices.columns:
        history = int(prices[col].notna().sum())
        if 0 < history < 252:
            warnings.warn(
                f"{col} has only {history} non-null observations (<252 trading days)",
                stacklevel=2,
            )


def _apply_strict_inception(prices: pd.DataFrame, strict_inception: bool) -> pd.DataFrame:
    if not strict_inception or prices.empty:
        return prices

    inceptions = report_inception_dates(prices)
    valid = inceptions.dropna()
    if valid.empty:
        return prices
    return prices.loc[prices.index >= valid.max()]


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_path: str = "data/prices.parquet",
    refresh: bool = False,
    incremental: bool = True,
    strict_inception: bool = True,
) -> pd.DataFrame:
    """Return cleaned adjusted-close prices with deterministic parquet caching.

    When `strict_inception` is True, the returned frame is trimmed so the first date
    is the latest inception date across the requested tickers. This avoids starting a
    backtest before all assets are live. The cache still retains the full downloaded
    history. When False, missing pre-inception values are preserved and downstream
    fallback logic can handle them.
    """
    requested = _normalize_tickers(tickers)
    if not requested:
        raise ValueError("tickers must be a non-empty list")

    cache = pd.DataFrame()
    if not refresh:
        cache = _load_cached_prices(cache_path)

    combined = cache.copy()
    if refresh or cache.empty:
        fetched = _clean_prices(_download_adj_close(requested, start, end), requested)
        combined = fetched
    else:
        missing_tickers = [ticker for ticker in requested if ticker not in combined.columns]
        if missing_tickers:
            missing_history = _clean_prices(_download_adj_close(missing_tickers, start, end), missing_tickers)
            combined = missing_history.combine_first(combined)

        if incremental and not combined.empty:
            last_cached = combined.index.max()
            fetch_end = _as_timestamp(end)
            needs_update = fetch_end is None or last_cached < fetch_end
            if needs_update:
                fetch_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                updated = _clean_prices(_download_adj_close(requested, fetch_start, end), requested)
                if not updated.empty:
                    combined = updated.combine_first(combined)

        combined = _clean_prices(combined, combined.columns.tolist())

    if combined.empty:
        raise ValueError("No price data available for the requested range")

    validate_price_frame(combined)
    _save_cached_prices(combined, cache_path)

    result = combined.reindex(columns=requested)
    result = _slice_date_range(result, start, end)
    result = _clean_prices(result, requested)
    result = _apply_strict_inception(result, strict_inception)

    if result.empty or result.notna().sum().sum() == 0:
        raise ValueError("No price data returned for the requested tickers")

    validate_price_frame(result)
    _warn_for_short_history(result)
    return result


def load_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
    strict_inception: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper around `get_prices` with a per-range cache file."""
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_cache = cache_dir / f"prices_{start}_{end}.parquet"
    legacy_csv_cache = cache_dir / f"prices_{start}_{end}.csv"

    if use_cache and not parquet_cache.exists() and legacy_csv_cache.exists():
        legacy = pd.read_csv(legacy_csv_cache, index_col=0, parse_dates=True)
        legacy = _clean_prices(legacy)
        validate_price_frame(legacy)
        _save_cached_prices(legacy, str(parquet_cache))

    return get_prices(
        tickers=tickers,
        start=start,
        end=end,
        cache_path=str(parquet_cache),
        refresh=not use_cache,
        incremental=False,
        strict_inception=strict_inception,
    )
