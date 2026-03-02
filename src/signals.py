from __future__ import annotations

from typing import Optional

import pandas as pd


def trend_filter(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
	"""Return boolean DataFrame: price > rolling MA(window).

	Keeps NaN where rolling MA is not available.
	"""
	if prices is None or prices.empty:
		return pd.DataFrame(index=getattr(prices, 'index', None), columns=getattr(prices, 'columns', None))
	px = prices.sort_index()
	ma = px.rolling(window=window, min_periods=window).mean()
	elig = px > ma
	elig = elig.mask(ma.isna())
	return elig


def momentum_12m(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
	"""Return percent-change over `lookback` days.
	"""
	if prices is None or prices.empty:
		return pd.DataFrame(index=getattr(prices, 'index', None), columns=getattr(prices, 'columns', None), dtype=float)
	px = prices.sort_index()
	return px.div(px.shift(lookback)) - 1.0


def _aligned_history(prices: pd.DataFrame, as_of: Optional[pd.Timestamp]) -> pd.DataFrame:
	px = prices.sort_index().copy()
	if as_of is not None:
		as_of = pd.Timestamp(as_of)
		# include the as-of session itself; caller is responsible for picking
		# an as-of date that avoids look-ahead.
		px = px.loc[px.index <= as_of]
	# forward-fill after alignment
	px = px.ffill()
	return px


def compute_eligibility_and_momentum(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, ma_window: int = 200, momentum_window: int = 252) -> pd.DataFrame:
	"""Compute eligibility (MA) and momentum (12m) as-of `as_of` without look-ahead.

	Returns DataFrame indexed by ticker with columns `eligible` (bool) and `momentum` (float).
	"""
	hist = _aligned_history(prices, as_of)
	if hist.empty:
		return pd.DataFrame(columns=["eligible", "momentum"])

	elig_table = trend_filter(hist, window=ma_window)
	mom_table = momentum_12m(hist, lookback=momentum_window)

	last_eligibility = elig_table.iloc[-1].fillna(False)
	last_momentum = mom_table.iloc[-1]

	out = pd.DataFrame({"eligible": last_eligibility, "momentum": last_momentum})
	return out

