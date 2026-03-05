from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def compute_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def file_signature(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    payload = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return compute_sha1(payload)


def dataframe_signature(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    col_hash = "|".join(map(str, df.columns)).encode()
    return compute_sha1(row_hashes + col_hash)


def load_prices_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Load prices from CSV or parquet bytes into a DataFrame with DatetimeIndex."""
    bio = BytesIO(data)
    if filename.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(bio)
    else:
        df = pd.read_csv(bio, index_col=0, parse_dates=True)
    return normalize_prices(df)


def load_prices_path(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prices file not found: {p}")
    if p.suffix.lower() not in (".parquet", ".pq"):
        raise ValueError("cached price files must be parquet")
    df = pd.read_parquet(p)
    return normalize_prices(df)


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("no price data provided")
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()]
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None).tz_localize(None)
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out


def slice_prices(
    df: pd.DataFrame,
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    strict_inception: bool = False,
) -> pd.DataFrame:
    selected = [str(t).strip() for t in tickers if str(t).strip()]
    if not selected:
        raise ValueError("tickers must be non-empty")

    out = normalize_prices(df)
    missing = [ticker for ticker in selected if ticker not in out.columns]
    if missing:
        raise ValueError(f"tickers missing from cache: {', '.join(missing)}")

    out = out.reindex(columns=selected)
    out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]

    if strict_inception and not out.empty:
        first_valid = {col: out[col].first_valid_index() for col in out.columns}
        valid = [pd.Timestamp(dt) for dt in first_valid.values() if dt is not None]
        if valid:
            out = out.loc[out.index >= max(valid)]

    out = out.dropna(how="all")
    if out.empty:
        raise ValueError("no cached rows available for the selected date range")
    return out


def validate_prices(df: pd.DataFrame, required_tickers: Iterable[str] = ("BIL",)) -> Dict[str, object]:
    """Return a dict summarizing validation checks for the prices DataFrame."""
    out: Dict[str, object] = {}
    if df is None:
        out["loaded"] = False
        out["error"] = "no dataframe provided"
        return out

    out["loaded"] = True
    out["is_datetime_index"] = isinstance(df.index, pd.DatetimeIndex)
    out["monotonic_increasing"] = bool(df.index.is_monotonic_increasing)
    out["has_duplicates"] = bool(df.index.duplicated().any())
    out["n_rows"] = int(len(df))
    out["n_columns"] = int(df.shape[1])
    out["missing_values"] = int(df.isna().sum().sum())
    req = list(required_tickers)
    present = [t for t in req if t in df.columns]
    missing = [t for t in req if t not in df.columns]
    out["required_present"] = present
    out["required_missing"] = missing
    return out


def config_to_tuple(cfg: Dict) -> Tuple[Tuple[str, object], ...]:
    """Convert config dict to a sorted, hashable tuple for caching keys."""
    return tuple(sorted(cfg.items()))


def parse_json_config(value: str | None) -> Dict:
    if value is None or not str(value).strip():
        return {}
    text = str(value).strip()
    path = Path(text)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("config must decode to a JSON object")
    return parsed


def to_csv_bytes(df: pd.DataFrame | pd.Series | None, index: bool = True) -> bytes:
    if df is None:
        return b""
    if isinstance(df, pd.Series):
        return df.to_csv().encode("utf-8")
    return df.to_csv(index=index).encode("utf-8")
