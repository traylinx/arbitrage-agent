#!/usr/bin/env python3.11
"""
backfill_external_metrics.py — pull HISTORICAL Coinalyze + Coinglass time
series for the full PM label window (~211 days) using the user's paid
API keys (read from `makakoo secret`).

This fills in the 8 cross-ex features that were NaN in the v0 baseline
(funding term structure, funding zscore, OI deltas, liquidation imbalance,
funding diffs, basis, etc).

Output:
    data/arbitrage-agent/v2/parquet-cache/external_metrics/
      source=coinalyze/series={oi,funding,liquidations,long_short}/
        date=YYYY-MM-DD.parquet
      source=coinglass/series={oi,funding,liquidations,cvd,taker_volume,lsr}/
        date=YYYY-MM-DD.parquet

Each row has at minimum:
  available_at  (UTC datetime, equal to bar close)
  value         (the metric value, float)
  + per-source extra columns (open/high/low for OHLC series)

Usage:
    python3.11 scripts/backfill_external_metrics.py
    python3.11 scripts/backfill_external_metrics.py --since 2025-10-09
    python3.11 scripts/backfill_external_metrics.py --only coinalyze
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/external_metrics"

COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINALYZE_BASE = "https://api.coinalyze.net"

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; arbitrage-backfill/1.0)"}


def get_secret(name: str) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    try:
        cp = subprocess.run(
            ["makakoo", "secret", "get", name],
            text=True, capture_output=True, timeout=8,
        )
        v = cp.stdout.strip()
        if cp.returncode == 0 and v:
            return v
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


# ─── Coinalyze ─────────────────────────────────────────────────────────────


def coinalyze_history(
    api_key: str, path: str, symbol: str, start_ts: int, end_ts: int,
    interval: str = "30min",
) -> list[dict]:
    """Walk Coinalyze history endpoint in chunks (their API caps the window).

    Returns flattened rows: [{available_at, t, o, h, l, c}, ...]
    """
    rows: list[dict] = []
    # Coinalyze caps at ~7 days for 30min interval; chunk by 6 days to be safe.
    chunk_seconds = 6 * 86400
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)
        try:
            r = requests.get(
                COINALYZE_BASE + path,
                params={
                    "api_key": api_key,
                    "symbols": symbol,
                    "interval": interval,
                    "from": cursor,
                    "to": chunk_end,
                },
                headers=DEFAULT_HEADERS,
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"  ! coinalyze {path} chunk {cursor}: {e}", file=sys.stderr)
            cursor = chunk_end
            continue

        if isinstance(data, list) and data:
            history = data[0].get("history", [])
            for row in history:
                if "t" in row:
                    rows.append({
                        "available_at": datetime.fromtimestamp(int(row["t"]), tz=timezone.utc),
                        **{k: float(v) if isinstance(v, (int, float, str)) and str(v) not in ("", "None") else None
                           for k, v in row.items()},
                    })
        cursor = chunk_end
        time.sleep(0.05)  # gentle on rate limit
    return rows


def backfill_coinalyze_all(
    api_key: str, symbol: str, start_ts: int, end_ts: int,
    out_root: Path,
) -> dict[str, int]:
    """Pull every historical endpoint we care about."""
    series_specs = [
        ("oi", "/v1/open-interest-history", "30min"),
        ("funding", "/v1/funding-rate-history", "1hour"),
        ("predicted_funding", "/v1/predicted-funding-rate-history", "1hour"),
        ("liquidations", "/v1/liquidation-history", "1hour"),
        ("long_short", "/v1/long-short-ratio-history", "30min"),
    ]
    out: dict[str, int] = {}
    for name, path, interval in series_specs:
        print(f"[coinalyze] fetching {name} ({path}, {interval})...")
        t0 = time.time()
        rows = coinalyze_history(api_key, path, symbol, start_ts, end_ts, interval)
        if not rows:
            print(f"  → 0 rows (skipped)")
            out[name] = 0
            continue
        df = pd.DataFrame(rows).drop_duplicates(subset=["available_at"]).sort_values("available_at")
        df["source"] = "coinalyze"
        df["series"] = name
        df["symbol"] = symbol
        df["interval"] = interval
        # Write daily shards
        df["date_iso"] = df["available_at"].dt.strftime("%Y-%m-%d")
        for day, group in df.groupby("date_iso"):
            day_dir = out_root / "source=coinalyze" / f"series={name}"
            day_dir.mkdir(parents=True, exist_ok=True)
            shard = day_dir / f"date={day}.parquet"
            if shard.exists():
                existing = pd.read_parquet(shard)
                merged = pd.concat([existing, group.drop(columns=["date_iso"])],
                                   ignore_index=True)
                merged = merged.drop_duplicates(subset=["available_at"], keep="last")
            else:
                merged = group.drop(columns=["date_iso"])
            pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                           shard, compression="snappy")
        elapsed = time.time() - t0
        print(f"  → {len(df)} rows in {elapsed:.1f}s")
        out[name] = len(df)
    return out


# ─── Coinglass ─────────────────────────────────────────────────────────────


def coinglass_get(api_key: str, path: str, params: dict, max_retries: int = 3) -> Optional[dict]:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(
                COINGLASS_BASE + path,
                params=params,
                headers={**DEFAULT_HEADERS, "accept": "application/json", "CG-API-KEY": api_key},
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"  ! coinglass {path}: {e}", file=sys.stderr)
                return None
            time.sleep(backoff)
            backoff *= 2
    return None


def coinglass_history(
    api_key: str, path: str, symbol: str, interval: str,
    start_ts: int, end_ts: int, extra_params: Optional[dict] = None,
) -> list[dict]:
    """Coinglass v4 history endpoints generally support `start_time`/`end_time`
    in milliseconds + `interval` + `limit`. Server caps to ~4500 rows per call.
    Walk in 4000-row chunks. extra_params adds per-endpoint required keys
    (e.g. `exchange_list`, `exchange`, `unit`).
    """
    rows: list[dict] = []
    interval_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                        "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 14400)
    chunk_rows = 4000
    chunk_seconds = chunk_rows * interval_seconds

    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_time": cursor * 1000,
            "end_time": chunk_end * 1000,
            "limit": chunk_rows,
        }
        if extra_params:
            params.update(extra_params)
        d = coinglass_get(api_key, path, params)
        if d is None:
            cursor = chunk_end
            continue
        if str(d.get("code")) != "0":
            print(f"  coinglass {path}: code={d.get('code')} msg={str(d.get('msg', ''))[:80]}",
                  file=sys.stderr)
            cursor = chunk_end
            continue
        data = d.get("data") or []
        if not data:
            cursor = chunk_end
            continue
        # Format varies per endpoint. We try to normalize:
        for entry in data:
            if isinstance(entry, dict) and "time" in entry:
                ts_ms = int(entry["time"])
                rec = {
                    "available_at": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                }
                for k, v in entry.items():
                    if k == "time":
                        continue
                    try:
                        rec[k] = float(v) if v is not None else None
                    except (TypeError, ValueError):
                        rec[k] = v
                rows.append(rec)
        cursor = chunk_end
        time.sleep(0.05)
    return rows


def backfill_coinglass_all(
    api_key: str, start_ts: int, end_ts: int, out_root: Path,
) -> dict[str, int]:
    """Pull every historical endpoint we care about, with per-endpoint
    extra params discovered from existing btc_external_metrics.py.
    """
    EXCHANGE_LIST = "Binance,OKX,Bybit"
    # name, path, symbol, interval, extra_params
    series_specs = [
        ("oi_aggregated", "/api/futures/open-interest/aggregated-history",
         "BTC", "4h", {}),
        ("liquidations_aggregated", "/api/futures/liquidation/aggregated-history",
         "BTC", "4h", {"exchange_list": EXCHANGE_LIST}),
        ("cvd_binance", "/api/futures/cvd/history",
         "BTCUSDT", "4h", {"exchange": "Binance"}),
        ("taker_volume_aggregated", "/api/futures/aggregated-taker-buy-sell-volume/history",
         "BTC", "4h", {"exchange_list": EXCHANGE_LIST, "unit": "usd"}),
        ("global_lsr_binance", "/api/futures/global-long-short-account-ratio/history",
         "BTCUSDT", "4h", {"exchange": "Binance"}),
        ("funding_binance", "/api/futures/funding-rate/history",
         "BTCUSDT", "4h", {"exchange": "Binance"}),
        ("orderbook_depth_aggregated", "/api/futures/orderbook/aggregated-ask-bids-history",
         "BTC", "4h", {"exchange_list": EXCHANGE_LIST}),
    ]
    out: dict[str, int] = {}
    for name, path, symbol, interval, extra in series_specs:
        print(f"[coinglass] fetching {name} ({path}, {interval}) extra={extra}...")
        t0 = time.time()
        rows = coinglass_history(api_key, path, symbol, interval, start_ts, end_ts, extra)
        if not rows:
            print(f"  → 0 rows (skipped)")
            out[name] = 0
            continue
        df = pd.DataFrame(rows).drop_duplicates(subset=["available_at"]).sort_values("available_at")
        df["source"] = "coinglass"
        df["series"] = name
        df["interval"] = interval
        df["date_iso"] = df["available_at"].dt.strftime("%Y-%m-%d")
        for day, group in df.groupby("date_iso"):
            day_dir = out_root / "source=coinglass" / f"series={name}"
            day_dir.mkdir(parents=True, exist_ok=True)
            shard = day_dir / f"date={day}.parquet"
            if shard.exists():
                existing = pd.read_parquet(shard)
                merged = pd.concat([existing, group.drop(columns=["date_iso"])],
                                   ignore_index=True)
                merged = merged.drop_duplicates(subset=["available_at"], keep="last")
            else:
                merged = group.drop(columns=["date_iso"])
            pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                           shard, compression="snappy")
        elapsed = time.time() - t0
        print(f"  → {len(df)} rows in {elapsed:.1f}s")
        out[name] = len(df)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Coinalyze + Coinglass historical metrics.")
    p.add_argument("--since", default="2025-10-09",
                   help="ISO date YYYY-MM-DD (default = PM 15m horizon start)")
    p.add_argument("--until", default="",
                   help="ISO date YYYY-MM-DD (default = now)")
    p.add_argument("--only", choices=["coinalyze", "coinglass", "both"], default="both")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--symbol-coinalyze", default="BTCUSDT_PERP.A")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    if args.until:
        end = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    days = (end - start).days
    print(f"=== backfill window: {start.isoformat()} → {end.isoformat()} ({days} days) ===")
    print()

    summary = {}
    if args.only in ("coinalyze", "both"):
        ca_key = get_secret("COINALYZE_API_KEY")
        if not ca_key:
            print("[error] no COINALYZE_API_KEY", file=sys.stderr)
        else:
            print(f"=== Coinalyze (key length {len(ca_key)}) ===")
            summary["coinalyze"] = backfill_coinalyze_all(
                ca_key, args.symbol_coinalyze, start_ts, end_ts, args.out,
            )
            print()

    if args.only in ("coinglass", "both"):
        cg_key = get_secret("COINGLASS_API_KEY")
        if not cg_key:
            print("[error] no COINGLASS_API_KEY", file=sys.stderr)
        else:
            print(f"=== Coinglass (key length {len(cg_key)}) ===")
            summary["coinglass"] = backfill_coinglass_all(
                cg_key, start_ts, end_ts, args.out,
            )
            print()

    print("=== summary ===")
    for source, by_series in summary.items():
        for series, n in by_series.items():
            print(f"  {source:11s} / {series:30s} {n:>7,} rows")
    print()
    print(f"output root: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
