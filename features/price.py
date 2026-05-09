"""V2 §2a — price-derived features.

All 8 features are trailing-only. Each one filters via
`snap.trades_as_of(...)` or `snap.bookticker_as_of(...)` BEFORE any
computation, so the as-of harness is satisfied by construction.

Features:
  ret_1m, ret_5m, ret_15m, ret_60m
  realized_vol_5m_trailing, realized_vol_30m_trailing
  microprice_imbalance
  spread_bps

V1 features `zscore_ret_30m` and `high_low_range_5m` are intentionally
cut per V2 §2a (overlap with realized_vol; low marginal info).

Reference exchange: Binance (most-liquid BTC perp). The cross-exchange
features in `crossex.py` add the basis spreads.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

from .feature_base import Snapshot, register

REFERENCE_EXCHANGE = "binance"


def _last_price_at_or_before(
    df: pd.DataFrame, ts: datetime, max_age_sec: int = 60
) -> float:
    """Returns the last trade price <= ts, or NaN if no row inside max_age."""
    if df is None or df.empty:
        return float("nan")
    cutoff = ts - timedelta(seconds=max_age_sec)
    eligible = df[df["available_at"] >= cutoff]
    if eligible.empty:
        return float("nan")
    return float(eligible["price"].iloc[-1])


def _price_n_minutes_ago(
    df: pd.DataFrame, ts: datetime, n_minutes: int, tolerance_sec: int = 30
) -> float:
    """Returns the trade price closest to (ts - n_minutes), within tolerance.

    Used as the denominator of return calculations. Tolerance allows
    for thin minutes where no trade happened on the very second.
    """
    if df is None or df.empty:
        return float("nan")
    target = ts - timedelta(minutes=n_minutes)
    lo = target - timedelta(seconds=tolerance_sec)
    hi = target + timedelta(seconds=tolerance_sec)
    window = df[(df["available_at"] >= lo) & (df["available_at"] <= hi)]
    if window.empty:
        return float("nan")
    # Take the row closest to target
    diffs = (window["available_at"] - target).abs()
    idx = diffs.idxmin()
    return float(window.loc[idx, "price"])


def _log_return(p_now: float, p_then: float) -> float:
    if math.isnan(p_now) or math.isnan(p_then) or p_then <= 0 or p_now <= 0:
        return float("nan")
    return math.log(p_now / p_then)


def _trailing_returns(
    df: pd.DataFrame, ts: datetime, lookback_min: int
) -> list[float]:
    """1-minute log returns over the last `lookback_min` minutes,
    using as-of-truncated `df`. Returns empty list if not enough data.
    """
    if df is None or df.empty:
        return []
    cutoff = ts - timedelta(minutes=lookback_min + 1)
    window = df[df["available_at"] >= cutoff].copy()
    if len(window) < 2:
        return []

    # Resample to 1-minute bars using last price within each minute.
    window["minute"] = window["available_at"].dt.floor("1min")
    bars = window.groupby("minute")["price"].last()
    if len(bars) < 2:
        return []
    rets = []
    prev = float(bars.iloc[0])
    for cur in bars.iloc[1:]:
        cur_f = float(cur)
        if prev > 0 and cur_f > 0:
            rets.append(math.log(cur_f / prev))
        prev = cur_f
    return rets


# ─── Returns ─────────────────────────────────────────────────────────────────


@register("ret_1m", category="price",
          description="1-minute log return on Binance BTC, trailing only")
def ret_1m(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    p_now = _last_price_at_or_before(df, ts)
    p_then = _price_n_minutes_ago(df, ts, n_minutes=1)
    return _log_return(p_now, p_then)


@register("ret_5m", category="price",
          description="5-minute log return on Binance BTC, trailing only")
def ret_5m(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    p_now = _last_price_at_or_before(df, ts)
    p_then = _price_n_minutes_ago(df, ts, n_minutes=5)
    return _log_return(p_now, p_then)


@register("ret_15m", category="price",
          description="15-minute log return on Binance BTC, trailing only")
def ret_15m(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    p_now = _last_price_at_or_before(df, ts)
    p_then = _price_n_minutes_ago(df, ts, n_minutes=15)
    return _log_return(p_now, p_then)


@register("ret_60m", category="price",
          description="60-minute log return on Binance BTC, trailing only")
def ret_60m(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    p_now = _last_price_at_or_before(df, ts)
    p_then = _price_n_minutes_ago(df, ts, n_minutes=60)
    return _log_return(p_now, p_then)


# ─── Realized volatility ────────────────────────────────────────────────────


@register("realized_vol_5m_trailing", category="price",
          description="Stdev of 1-min log returns over the trailing 5 minutes")
def realized_vol_5m_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    rets = _trailing_returns(df, ts, lookback_min=5)
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


@register("realized_vol_30m_trailing", category="price",
          description="Stdev of 1-min log returns over the trailing 30 minutes")
def realized_vol_30m_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.trades_as_of(REFERENCE_EXCHANGE, ts)
    rets = _trailing_returns(df, ts, lookback_min=30)
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


# ─── Order book microstructure ──────────────────────────────────────────────


@register("microprice_imbalance", category="price",
          description="(bid_size - ask_size) / (bid_size + ask_size) at decision_ts")
def microprice_imbalance(snap: Snapshot, ts: datetime) -> float:
    df = snap.bookticker_as_of(REFERENCE_EXCHANGE, ts)
    if df is None or df.empty:
        return float("nan")
    last = df.iloc[-1]
    bs = float(last.get("bid_size", float("nan")))
    as_ = float(last.get("ask_size", float("nan")))
    if math.isnan(bs) or math.isnan(as_) or (bs + as_) <= 0:
        return float("nan")
    return (bs - as_) / (bs + as_)


@register("spread_bps", category="price",
          description="(ask - bid) / mid expressed in basis points, trailing")
def spread_bps(snap: Snapshot, ts: datetime) -> float:
    df = snap.bookticker_as_of(REFERENCE_EXCHANGE, ts)
    if df is None or df.empty:
        return float("nan")
    last = df.iloc[-1]
    bid = float(last.get("bid", float("nan")))
    ask = float(last.get("ask", float("nan")))
    if math.isnan(bid) or math.isnan(ask) or bid <= 0 or ask <= 0 or ask <= bid:
        return float("nan")
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 10_000.0
