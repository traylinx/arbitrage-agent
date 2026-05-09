"""V2 §2b — cross-exchange / aggregator-derived features.

Reads from the wide frames produced by `external_metrics_loader`:
  - `coinalyze`: OI, funding, predicted funding, liquidations, long/short pct
  - `coinglass`: aggregate OI, Binance funding, aggregate liquidations USD,
    Binance CVD, aggregate taker volume, global LSR, aggregate orderbook depth

All features are trailing-only and read via `snap.derivatives_as_of(...)`.
The Snapshot's `_filtered` enforces `available_at <= decision_ts` via
searchsorted on the pre-sorted frame, so each feature is leakage-free
by construction.

Two pure-CEX basis features (`binance_coinbase_basis_bps`,
`binance_hyperliquid_perp_basis`) are intentionally NOT registered here
yet — they require Coinbase + Hyperliquid OHLCV backfills which are not
in the V2 v0.6 dataset. They will land alongside those backfills.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

from .feature_base import Snapshot, register


def _last_value(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return float("nan")
    series = df[col].dropna()
    if series.empty:
        return float("nan")
    try:
        return float(series.iloc[-1])
    except (TypeError, ValueError):
        return float("nan")


def _last_price(df: pd.DataFrame) -> float:
    """Last `price` value in a trades-shaped frame (synthetic from klines = close)."""
    return _last_value(df, "price")


def _trailing_zscore(values: list[float]) -> float:
    """z-score of the last element vs the prior elements. NaN if not enough data."""
    if len(values) < 3:
        return float("nan")
    head = values[:-1]
    mean = sum(head) / len(head)
    var = sum((v - mean) ** 2 for v in head) / (len(head) - 1)
    if var <= 0:
        return float("nan")
    return (values[-1] - mean) / math.sqrt(var)


# ─── Open Interest dynamics (Coinalyze, 30-min bars) ─────────────────────────


@register("oi_delta_30m_trailing", category="crossex",
          description="Coinalyze aggregate OI change between the latest two 30-min bars")
def oi_delta_30m_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    if df is None or df.empty or "oi_close" not in df.columns:
        return float("nan")
    series = df.dropna(subset=["oi_close"]).tail(2)
    if len(series) < 2:
        return float("nan")
    return float(series["oi_close"].iloc[-1]) - float(series["oi_close"].iloc[-2])


@register("oi_zscore_24h_trailing", category="crossex",
          description="z-score of the latest aggregate OI vs the prior 24h distribution")
def oi_zscore_24h_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    if df is None or df.empty or "oi_close" not in df.columns:
        return float("nan")
    cutoff = ts - timedelta(hours=24)
    recent = df[df["available_at"] >= cutoff].dropna(subset=["oi_close"])
    if len(recent) < 5:
        return float("nan")
    return _trailing_zscore(recent["oi_close"].tolist())


# ─── Funding term structure (Coinalyze, 1-hour bars) ─────────────────────────


@register("funding_predicted_vs_actual", category="crossex",
          description="Coinalyze predicted_funding - realized_funding (positive = funding rising)")
def funding_predicted_vs_actual(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    pf = _last_value(df, "predicted_funding_close")
    f = _last_value(df, "funding_close")
    if math.isnan(pf) or math.isnan(f):
        return float("nan")
    return pf - f


@register("funding_zscore_24h_trailing", category="crossex",
          description="z-score of the latest aggregate funding rate vs the prior 24h distribution")
def funding_zscore_24h_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    if df is None or df.empty or "funding_close" not in df.columns:
        return float("nan")
    cutoff = ts - timedelta(hours=24)
    recent = df[df["available_at"] >= cutoff].dropna(subset=["funding_close"])
    if len(recent) < 5:
        return float("nan")
    return _trailing_zscore(recent["funding_close"].tolist())


# ─── Liquidation imbalance (Coinalyze, 1-hour bars) ──────────────────────────


@register("liquidation_imbalance_1h_trailing", category="crossex",
          description="(longs - shorts) / (longs + shorts) liquidations from the latest aggregator bar")
def liquidation_imbalance_1h_trailing(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    longs = _last_value(df, "liquidations_long")
    shorts = _last_value(df, "liquidations_short")
    if math.isnan(longs) or math.isnan(shorts):
        return float("nan")
    if (longs + shorts) <= 0:
        return float("nan")
    return (longs - shorts) / (longs + shorts)


# ─── Long/short positioning (Coinalyze, 30-min bars) ─────────────────────────


@register("long_short_ratio_zscore_24h", category="crossex",
          description="z-score of the latest long/short account ratio vs prior 24h")
def long_short_ratio_zscore_24h(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinalyze", ts)
    if df is None or df.empty or "long_short_ratio" not in df.columns:
        return float("nan")
    cutoff = ts - timedelta(hours=24)
    recent = df[df["available_at"] >= cutoff].dropna(subset=["long_short_ratio"])
    if len(recent) < 5:
        return float("nan")
    return _trailing_zscore(recent["long_short_ratio"].tolist())


# ─── Cross-source funding agreement (Coinalyze vs Coinglass) ─────────────────


@register("funding_aggregator_disagreement", category="crossex",
          description="Coinalyze aggregate funding minus Coinglass Binance funding "
                      "(non-zero indicates Binance is leading or lagging the average)")
def funding_aggregator_disagreement(snap: Snapshot, ts: datetime) -> float:
    ca = snap.derivatives_as_of("coinalyze", ts)
    cg = snap.derivatives_as_of("coinglass", ts)
    f_ca = _last_value(ca, "funding_close")
    f_cg = _last_value(cg, "funding_binance_close")
    if math.isnan(f_ca) or math.isnan(f_cg):
        return float("nan")
    return f_ca - f_cg


# ─── Coinglass: Binance taker flow + aggregate orderbook depth ───────────────


@register("binance_taker_buy_sell_imbalance", category="crossex",
          description="(taker_buy_vol - taker_sell_vol) / (buy + sell) on Binance over the latest 4h bar")
def binance_taker_buy_sell_imbalance(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinglass", ts)
    buy = _last_value(df, "binance_taker_buy_vol")
    sell = _last_value(df, "binance_taker_sell_vol")
    if math.isnan(buy) or math.isnan(sell):
        return float("nan")
    total = buy + sell
    if total <= 0:
        return float("nan")
    return (buy - sell) / total


@register("orderbook_depth_imbalance", category="crossex",
          description="(bids_usd - asks_usd) / (bids + asks) from coinglass aggregate orderbook")
def orderbook_depth_imbalance(snap: Snapshot, ts: datetime) -> float:
    df = snap.derivatives_as_of("coinglass", ts)
    bids = _last_value(df, "depth_bids_usd")
    asks = _last_value(df, "depth_asks_usd")
    if math.isnan(bids) or math.isnan(asks):
        return float("nan")
    total = bids + asks
    if total <= 0:
        return float("nan")
    return (bids - asks) / total


# ─── Cross-exchange basis (binance vs coinbase / bybit / hyperliquid) ────────


@register("binance_coinbase_basis_bps", category="crossex",
          description="(binance_close - coinbase_close) / coinbase_close in bps; trailing")
def binance_coinbase_basis_bps(snap: Snapshot, ts: datetime) -> float:
    bn = _last_price(snap.trades_as_of("binance", ts))
    cb = _last_price(snap.trades_as_of("coinbase", ts))
    if math.isnan(bn) or math.isnan(cb) or cb <= 0:
        return float("nan")
    return ((bn - cb) / cb) * 10_000.0


@register("binance_bybit_basis_bps", category="crossex",
          description="(binance_close - bybit_close) / bybit_close in bps; both perp; trailing")
def binance_bybit_basis_bps(snap: Snapshot, ts: datetime) -> float:
    bn = _last_price(snap.trades_as_of("binance", ts))
    by = _last_price(snap.trades_as_of("bybit", ts))
    if math.isnan(bn) or math.isnan(by) or by <= 0:
        return float("nan")
    return ((bn - by) / by) * 10_000.0


@register("binance_hyperliquid_basis_bps", category="crossex",
          description="(binance_close - hyperliquid_close) / hyperliquid_close in bps; "
                      "binance is centralized perp, hyperliquid is decentralized perp; trailing")
def binance_hyperliquid_basis_bps(snap: Snapshot, ts: datetime) -> float:
    bn = _last_price(snap.trades_as_of("binance", ts))
    hl = _last_price(snap.trades_as_of("hyperliquid", ts))
    if math.isnan(bn) or math.isnan(hl) or hl <= 0:
        return float("nan")
    return ((bn - hl) / hl) * 10_000.0
