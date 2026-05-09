#!/usr/bin/env python3.11
"""
backfill_cross_exchange_klines.py — pull historical 1m BTC OHLCV from
Coinbase + Bybit + Hyperliquid for the full PM label window (~211 days).

These three exchanges are FREE (no auth) and unlock the basis features
that were dropped in v0.6:
  - binance_coinbase_basis_bps  (spot leader; best institutional anchor)
  - binance_bybit_basis_bps     (perp; second-largest centralized exchange)
  - binance_hyperliquid_perp_basis_bps  (decentralized perp; whale liquidity)

Output:
    data/arbitrage-agent/v2/parquet-cache/cross_exchange_klines/
      exchange={coinbase,bybit,hyperliquid}/
        date=YYYY-MM-DD.parquet

Each row: open_time_ms, open_time, open, high, low, close, volume,
          close_time, close_time_ms, available_at, exchange.

Usage:
    python3.11 scripts/backfill_cross_exchange_klines.py
    python3.11 scripts/backfill_cross_exchange_klines.py --since 2025-10-09
    python3.11 scripts/backfill_cross_exchange_klines.py --only coinbase
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/cross_exchange_klines"
DEFAULT_PM_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/polymarket_resolutions"

USER_AGENT = "Mozilla/5.0 (compatible; arbitrage-backfill/1.0)"
HEADERS = {"User-Agent": USER_AGENT}

COINBASE_BASE = "https://api.exchange.coinbase.com"
BYBIT_BASE = "https://api.bybit.com"
HYPERLIQUID_BASE = "https://api.hyperliquid.xyz/info"


# ─── Persisting one exchange's klines as daily shards ────────────────────────


def _write_daily_shards(df: pd.DataFrame, out_root: Path, exchange: str) -> int:
    """Group by UTC date and write idempotent parquet shards."""
    if df.empty:
        return 0
    df = df.dropna(subset=["available_at"]).copy()
    df["date_iso"] = pd.to_datetime(df["available_at"], utc=True).dt.strftime("%Y-%m-%d")
    written = 0
    for day, group in df.groupby("date_iso"):
        day_dir = out_root / f"exchange={exchange}"
        day_dir.mkdir(parents=True, exist_ok=True)
        shard = day_dir / f"date={day}.parquet"
        merged = group.drop(columns=["date_iso"])
        if shard.exists():
            existing = pd.read_parquet(shard)
            merged = pd.concat([existing, merged], ignore_index=True)
        merged = merged.drop_duplicates(subset=["open_time_ms"], keep="last")
        merged = merged.sort_values("open_time_ms").reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                       shard, compression="snappy")
        written += len(group)
    return written


# ─── Coinbase: GET candles, granularity=60, max 300 candles per req ──────────


async def fetch_coinbase_chunk(
    session: aiohttp.ClientSession, start_ms: int, end_ms: int,
    sem: asyncio.Semaphore, max_retries: int = 3,
) -> list[dict]:
    """One Coinbase /products/BTC-USD/candles call; ≤ 300 rows."""
    url = f"{COINBASE_BASE}/products/BTC-USD/candles"
    params = {
        "granularity": 60,
        "start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    backoff = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, headers=HEADERS, timeout=15) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    if resp.status >= 500:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    if resp.status != 200:
                        return []
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == max_retries - 1:
                    return []
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            if not isinstance(data, list):
                return []
            # Coinbase returns [time, low, high, open, close, volume] reverse-chrono.
            rows: list[dict] = []
            for entry in data:
                if not (isinstance(entry, list) and len(entry) >= 6):
                    continue
                t = int(entry[0])
                rows.append({
                    "open_time_ms": t * 1000,
                    "open_time": datetime.fromtimestamp(t, tz=timezone.utc),
                    "low": float(entry[1]),
                    "high": float(entry[2]),
                    "open": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                    "close_time": datetime.fromtimestamp(t + 60, tz=timezone.utc),
                    "close_time_ms": (t + 60) * 1000,
                    "available_at": datetime.fromtimestamp(t + 60, tz=timezone.utc),
                    "exchange": "coinbase",
                })
            return rows
    return []


async def backfill_coinbase(
    start_ts: int, end_ts: int, out_root: Path, concurrency: int = 4,
) -> int:
    """Walk Coinbase candles in 300-min chunks (300 candles × 60s)."""
    chunk_seconds = 300 * 60
    chunks = []
    cursor = start_ts
    while cursor < end_ts:
        chunks.append((cursor, min(cursor + chunk_seconds, end_ts)))
        cursor += chunk_seconds
    print(f"[coinbase] {len(chunks)} chunks (300m each)")

    sem = asyncio.Semaphore(concurrency)
    total_rows = 0
    daily_buffer: list[dict] = []
    async with aiohttp.ClientSession() as session:
        # Fetch in batches of `concurrency * 4` to keep memory bounded
        batch = concurrency * 8
        for i in range(0, len(chunks), batch):
            tasks = [
                fetch_coinbase_chunk(session, s * 1000, e * 1000, sem)
                for (s, e) in chunks[i:i + batch]
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for rows in results:
                daily_buffer.extend(rows)
                total_rows += len(rows)
            if i % (batch * 4) == 0:
                print(f"  coinbase progress: {min(i + batch, len(chunks))}/{len(chunks)} "
                      f"chunks, {total_rows:,} rows")
            # Periodically flush to parquet to avoid building a huge in-mem df
            if len(daily_buffer) >= 50_000:
                df = pd.DataFrame(daily_buffer)
                _write_daily_shards(df, out_root, "coinbase")
                daily_buffer = []
    if daily_buffer:
        df = pd.DataFrame(daily_buffer)
        _write_daily_shards(df, out_root, "coinbase")
    print(f"[coinbase] → {total_rows:,} rows")
    return total_rows


# ─── Bybit: GET v5/market/kline, max 1000 candles per req ────────────────────


async def fetch_bybit_chunk(
    session: aiohttp.ClientSession, start_ms: int, end_ms: int,
    sem: asyncio.Semaphore, max_retries: int = 3,
) -> list[dict]:
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "1",
        "start": start_ms,
        "end": end_ms,
        "limit": 1000,
    }
    backoff = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, headers=HEADERS, timeout=15) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    if resp.status != 200:
                        return []
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == max_retries - 1:
                    return []
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            if not isinstance(data, dict) or data.get("retCode") != 0:
                return []
            entries = (data.get("result") or {}).get("list") or []
            rows: list[dict] = []
            for entry in entries:
                if not (isinstance(entry, list) and len(entry) >= 6):
                    continue
                t = int(entry[0]) // 1000  # bybit returns ms
                rows.append({
                    "open_time_ms": t * 1000,
                    "open_time": datetime.fromtimestamp(t, tz=timezone.utc),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                    "close_time": datetime.fromtimestamp(t + 60, tz=timezone.utc),
                    "close_time_ms": (t + 60) * 1000,
                    "available_at": datetime.fromtimestamp(t + 60, tz=timezone.utc),
                    "exchange": "bybit",
                })
            return rows
    return []


async def backfill_bybit(
    start_ts: int, end_ts: int, out_root: Path, concurrency: int = 4,
) -> int:
    """Walk Bybit klines in 1000-minute chunks."""
    chunk_seconds = 1000 * 60
    chunks = []
    cursor = start_ts
    while cursor < end_ts:
        chunks.append((cursor, min(cursor + chunk_seconds, end_ts)))
        cursor += chunk_seconds
    print(f"[bybit] {len(chunks)} chunks (1000m each)")

    sem = asyncio.Semaphore(concurrency)
    total_rows = 0
    daily_buffer: list[dict] = []
    async with aiohttp.ClientSession() as session:
        batch = concurrency * 8
        for i in range(0, len(chunks), batch):
            tasks = [
                fetch_bybit_chunk(session, s * 1000, e * 1000, sem)
                for (s, e) in chunks[i:i + batch]
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for rows in results:
                daily_buffer.extend(rows)
                total_rows += len(rows)
            if i % (batch * 4) == 0:
                print(f"  bybit progress: {min(i + batch, len(chunks))}/{len(chunks)} "
                      f"chunks, {total_rows:,} rows")
            if len(daily_buffer) >= 50_000:
                df = pd.DataFrame(daily_buffer)
                _write_daily_shards(df, out_root, "bybit")
                daily_buffer = []
    if daily_buffer:
        df = pd.DataFrame(daily_buffer)
        _write_daily_shards(df, out_root, "bybit")
    print(f"[bybit] → {total_rows:,} rows")
    return total_rows


# ─── Hyperliquid: POST info type=candleSnapshot, max 5000 candles per req ────


async def fetch_hyperliquid_chunk(
    session: aiohttp.ClientSession, start_ms: int, end_ms: int,
    sem: asyncio.Semaphore, max_retries: int = 3,
) -> list[dict]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    backoff = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                async with session.post(HYPERLIQUID_BASE, json=payload, headers=HEADERS, timeout=20) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    if resp.status != 200:
                        return []
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == max_retries - 1:
                    return []
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            if not isinstance(data, list):
                return []
            rows: list[dict] = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                t_open_ms = int(entry.get("t", 0) or 0)
                t_close_ms = int(entry.get("T", 0) or 0)
                if not t_open_ms:
                    continue
                rows.append({
                    "open_time_ms": t_open_ms,
                    "open_time": datetime.fromtimestamp(t_open_ms / 1000, tz=timezone.utc),
                    "open": float(entry.get("o", 0.0) or 0.0),
                    "high": float(entry.get("h", 0.0) or 0.0),
                    "low": float(entry.get("l", 0.0) or 0.0),
                    "close": float(entry.get("c", 0.0) or 0.0),
                    "volume": float(entry.get("v", 0.0) or 0.0),
                    "close_time_ms": t_close_ms or t_open_ms + 60_000,
                    "close_time": datetime.fromtimestamp(
                        (t_close_ms or t_open_ms + 60_000) / 1000, tz=timezone.utc),
                    "available_at": datetime.fromtimestamp(
                        (t_close_ms or t_open_ms + 60_000) / 1000, tz=timezone.utc),
                    "exchange": "hyperliquid",
                })
            return rows
    return []


async def backfill_hyperliquid(
    start_ts: int, end_ts: int, out_root: Path, concurrency: int = 3,
) -> int:
    """Walk Hyperliquid candles in 5000-minute chunks (their server cap)."""
    chunk_seconds = 5000 * 60
    chunks = []
    cursor = start_ts
    while cursor < end_ts:
        chunks.append((cursor, min(cursor + chunk_seconds, end_ts)))
        cursor += chunk_seconds
    print(f"[hyperliquid] {len(chunks)} chunks (5000m each)")

    sem = asyncio.Semaphore(concurrency)
    total_rows = 0
    daily_buffer: list[dict] = []
    async with aiohttp.ClientSession() as session:
        batch = concurrency * 6
        for i in range(0, len(chunks), batch):
            tasks = [
                fetch_hyperliquid_chunk(session, s * 1000, e * 1000, sem)
                for (s, e) in chunks[i:i + batch]
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for rows in results:
                daily_buffer.extend(rows)
                total_rows += len(rows)
            if i % (batch * 4) == 0:
                print(f"  hyperliquid progress: {min(i + batch, len(chunks))}/{len(chunks)} "
                      f"chunks, {total_rows:,} rows")
            if len(daily_buffer) >= 50_000:
                df = pd.DataFrame(daily_buffer)
                _write_daily_shards(df, out_root, "hyperliquid")
                daily_buffer = []
    if daily_buffer:
        df = pd.DataFrame(daily_buffer)
        _write_daily_shards(df, out_root, "hyperliquid")
    print(f"[hyperliquid] → {total_rows:,} rows")
    return total_rows


# ─── Time-window discovery ───────────────────────────────────────────────────


def derive_window(pm_root: Path) -> tuple[int, int]:
    """Auto-derive the start/end times from the PM resolutions backfill."""
    files = sorted(pm_root.rglob("*.parquet"))
    if not files:
        # fall back to default 211-day horizon
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=211)
        return int(start.timestamp()), int(end.timestamp())
    df = pd.concat([pd.read_parquet(f, columns=["window_start"]) for f in files], ignore_index=True)
    s = int(df["window_start"].min())
    e = int(df["window_start"].max()) + 60 * 30  # buffer
    return s, e


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill 1m BTC OHLCV from Coinbase + Bybit + Hyperliquid.")
    p.add_argument("--since", default="",
                   help="ISO date YYYY-MM-DD (default: derive from PM resolutions)")
    p.add_argument("--until", default="",
                   help="ISO date YYYY-MM-DD (default: now)")
    p.add_argument("--only", choices=["coinbase", "bybit", "hyperliquid", "all"],
                   default="all")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pm-root", type=Path, default=DEFAULT_PM_ROOT)
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> dict[str, int]:
    args.out.mkdir(parents=True, exist_ok=True)

    if args.since:
        start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        if args.until:
            end = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc)
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    else:
        start_ts, end_ts = derive_window(args.pm_root)
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    days = (end - start).days
    print(f"=== window: {start.isoformat()} → {end.isoformat()} ({days} days) ===")
    print()

    summary: dict[str, int] = {}

    if args.only in ("coinbase", "all"):
        summary["coinbase"] = await backfill_coinbase(
            start_ts, end_ts, args.out, args.concurrency,
        )
    if args.only in ("bybit", "all"):
        summary["bybit"] = await backfill_bybit(
            start_ts, end_ts, args.out, args.concurrency,
        )
    if args.only in ("hyperliquid", "all"):
        summary["hyperliquid"] = await backfill_hyperliquid(
            start_ts, end_ts, args.out, max(2, args.concurrency // 2),
        )

    return summary


def main() -> int:
    args = parse_args()
    summary = asyncio.run(main_async(args))
    print()
    print("=== summary ===")
    for ex, n in summary.items():
        print(f"  {ex:11s} {n:>9,} rows")
    print(f"output root: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
