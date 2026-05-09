"""external_metrics_loader.py — load Coinalyze + Coinglass historical
parquet shards and produce one wide-merged frame per source.

Shard layout (produced by scripts/backfill_external_metrics.py):

    parquet-cache/external_metrics/
      source=coinalyze/series={oi,funding,predicted_funding,liquidations,long_short}/date=YYYY-MM-DD.parquet
      source=coinglass/source=coinglass/series={oi_aggregated,funding_binance,liquidations_aggregated,
                                                cvd_binance,taker_volume_aggregated,
                                                global_lsr_binance,orderbook_depth_aggregated}/date=YYYY-MM-DD.parquet

Each per-series frame has its own column shape (Coinalyze uses t,o,h,l,c
or t,l,s; Coinglass uses open/close/various USD aggregates). This loader
renames + outer-merges all series into a single dataframe per source,
keyed on `available_at`, with forward-fill across mixed intervals so a
30-min OI bar persists into the next 1-hour funding bar.

Forward-fill is applied; back-fill is NEVER applied (that would be
future leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_BASE = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/external_metrics"
DEFAULT_CROSS_EX_BASE = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/cross_exchange_klines"


def _load_series(base: Path, source: str, series: str) -> pd.DataFrame:
    """Read every daily shard for one (source, series) into one frame.

    Returns an empty frame if the directory is missing or no shards exist.
    Output is sorted by `available_at`, deduped on `available_at`.
    """
    dirpath = base / f"source={source}" / f"series={series}"
    if not dirpath.exists():
        return pd.DataFrame(columns=["available_at"])
    shards = sorted(dirpath.glob("*.parquet"))
    if not shards:
        return pd.DataFrame(columns=["available_at"])
    parts: list[pd.DataFrame] = []
    for shard in shards:
        try:
            parts.append(pd.read_parquet(shard))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["available_at"])
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["available_at"])
    df = df.sort_values("available_at")
    df = df.drop_duplicates(subset=["available_at"], keep="last")
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    df = df.reset_index(drop=True)
    return df


def _select_rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Project + rename columns. Drops rows where the mapped subset is all-NaN."""
    if df.empty:
        return pd.DataFrame(columns=["available_at"] + list(mapping.values()))
    have = [c for c in mapping.keys() if c in df.columns]
    if not have:
        return pd.DataFrame(columns=["available_at"])
    out = df[["available_at"] + have].rename(columns={c: mapping[c] for c in have})
    return out


def _outer_merge(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["available_at"])
    merged = parts[0]
    for p in parts[1:]:
        merged = pd.merge(merged, p, on="available_at", how="outer")
    merged = merged.sort_values("available_at").reset_index(drop=True)
    cols_to_ffill = [c for c in merged.columns if c != "available_at"]
    if cols_to_ffill:
        merged[cols_to_ffill] = merged[cols_to_ffill].ffill()
    merged["available_at"] = pd.to_datetime(merged["available_at"], utc=True)
    return merged


def load_coinalyze_wide(base: Path = DEFAULT_BASE) -> pd.DataFrame:
    """Returns a single frame with columns:
        available_at,
        oi_open, oi_high, oi_low, oi_close,
        funding_close, predicted_funding_close,
        liquidations_long, liquidations_short,
        long_short_ratio, long_pct, short_pct
    """
    parts: list[pd.DataFrame] = []
    parts.append(_select_rename(
        _load_series(base, "coinalyze", "oi"),
        {"o": "oi_open", "h": "oi_high", "l": "oi_low", "c": "oi_close"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinalyze", "funding"),
        {"c": "funding_close"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinalyze", "predicted_funding"),
        {"c": "predicted_funding_close"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinalyze", "liquidations"),
        {"l": "liquidations_long", "s": "liquidations_short"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinalyze", "long_short"),
        {"r": "long_short_ratio", "l": "long_pct", "s": "short_pct"},
    ))
    return _outer_merge(parts)


def load_coinglass_wide(base: Path = DEFAULT_BASE) -> pd.DataFrame:
    """Returns a single frame with columns:
        available_at,
        oi_agg_open, oi_agg_high, oi_agg_low, oi_agg_close,
        funding_binance_close,
        liq_long_usd, liq_short_usd,
        binance_taker_buy_vol, binance_taker_sell_vol, binance_cvd,
        taker_buy_agg_usd, taker_sell_agg_usd,
        lsr_long_pct, lsr_short_pct, lsr_ratio,
        depth_bids_usd, depth_bids_qty, depth_asks_usd, depth_asks_qty
    """
    parts: list[pd.DataFrame] = []
    parts.append(_select_rename(
        _load_series(base, "coinglass", "oi_aggregated"),
        {"open": "oi_agg_open", "high": "oi_agg_high",
         "low": "oi_agg_low", "close": "oi_agg_close"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "funding_binance"),
        {"close": "funding_binance_close"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "liquidations_aggregated"),
        {"aggregated_long_liquidation_usd": "liq_long_usd",
         "aggregated_short_liquidation_usd": "liq_short_usd"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "cvd_binance"),
        {"taker_buy_vol": "binance_taker_buy_vol",
         "taker_sell_vol": "binance_taker_sell_vol",
         "cum_vol_delta": "binance_cvd"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "taker_volume_aggregated"),
        {"aggregated_buy_volume_usd": "taker_buy_agg_usd",
         "aggregated_sell_volume_usd": "taker_sell_agg_usd"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "global_lsr_binance"),
        {"global_account_long_percent": "lsr_long_pct",
         "global_account_short_percent": "lsr_short_pct",
         "global_account_long_short_ratio": "lsr_ratio"},
    ))
    parts.append(_select_rename(
        _load_series(base, "coinglass", "orderbook_depth_aggregated"),
        {"aggregated_bids_usd": "depth_bids_usd",
         "aggregated_bids_quantity": "depth_bids_qty",
         "aggregated_asks_usd": "depth_asks_usd",
         "aggregated_asks_quantity": "depth_asks_qty"},
    ))
    return _outer_merge(parts)


def load_all(base: Path = DEFAULT_BASE) -> dict[str, pd.DataFrame]:
    """Load every source as a wide frame. Empty frames if data is missing."""
    return {
        "coinalyze": load_coinalyze_wide(base),
        "coinglass": load_coinglass_wide(base),
    }


# ─── Cross-exchange 1m OHLCV (Coinbase, Bybit, Hyperliquid) ──────────────────


def load_cross_exchange_trades(
    exchange: str, base: Path = DEFAULT_CROSS_EX_BASE,
) -> pd.DataFrame:
    """Load 1m OHLCV shards for one exchange, return as a 'trades'-shaped
    frame (one row per minute, available_at = close_time, price = close,
    size = volume).

    This frame matches what `klines_to_synthetic_trades` produces from the
    Binance fapi data, so feature code that does
    `snap.trades_as_of(exchange, ts)` works uniformly.

    Returns an empty DataFrame if no shards exist.
    """
    dirpath = base / f"exchange={exchange}"
    if not dirpath.exists():
        return pd.DataFrame()
    shards = sorted(dirpath.glob("*.parquet"))
    if not shards:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for shard in shards:
        try:
            parts.append(pd.read_parquet(shard))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["available_at"])
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    df = df.sort_values("available_at")
    df = df.drop_duplicates(subset=["open_time_ms"], keep="last")
    # Build the output frame from the Series (NOT .values) so the tz-aware
    # dtype is preserved. .values strips the UTC tz, and Snapshot._filtered
    # then can't compare tz-naive vs tz-aware decision_ts → empty filter →
    # all basis features were NaN. Caught 2026-05-09 in v0.7 rebuild.
    out = df[["available_at", "close", "volume"]].copy()
    out = out.rename(columns={"close": "price", "volume": "size"})
    out["side"] = "close"
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    return out.reset_index(drop=True)


def load_cross_exchange_all(
    exchanges: tuple[str, ...] = ("coinbase", "bybit", "hyperliquid"),
    base: Path = DEFAULT_CROSS_EX_BASE,
) -> dict[str, pd.DataFrame]:
    """Load every requested exchange. Skips entries with no data."""
    out: dict[str, pd.DataFrame] = {}
    for ex in exchanges:
        df = load_cross_exchange_trades(ex, base)
        if not df.empty:
            out[ex] = df
    return out


if __name__ == "__main__":
    import sys
    sources = load_all()
    for src, df in sources.items():
        print(f"[{src}]  rows={len(df)}  cols={list(df.columns)}")
        if not df.empty:
            print(f"  range: {df['available_at'].min()} → {df['available_at'].max()}")
            non_nan = {c: int(df[c].notna().sum()) for c in df.columns if c != "available_at"}
            print(f"  non_nan: {non_nan}")
    cx = load_cross_exchange_all()
    for ex, df in cx.items():
        print(f"[cx:{ex}]  rows={len(df)}  range: {df['available_at'].min()} → {df['available_at'].max()}")
    sys.exit(0)
