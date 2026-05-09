#!/usr/bin/env python3.11
"""
backfill_binance_klines.py — Phase 1a follow-on for SPRINT-ARBITRAGE-ML-V1.

Walks Binance fapi `/fapi/v1/klines` to backfill 1m BTCUSDT perp OHLCV
for the time range covering our Polymarket label corpus. Writes daily
parquet shards keyed by date, idempotent on (open_time).

The output joins to `polymarket_resolutions/` rows by `decision_ts`
to produce per-decision price-derived features (ret_*, realized_vol_*,
microprice_imbalance from book is NaN-only here — needs live capture).

Output layout:
    data/arbitrage-agent/v2/parquet-cache/
      binance_klines/
        symbol=BTCUSDT/
          interval=1m/
            date=YYYY-MM-DD.parquet

Usage:
    python3.11 backfill_binance_klines.py                 # full range from PM corpus
    python3.11 backfill_binance_klines.py --start 2026-04-01
    python3.11 backfill_binance_klines.py --smoke         # last 2 hours only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FAPI_BASE = "https://fapi.binance.com"
SYMBOL_DEFAULT = "BTCUSDT"
INTERVAL_DEFAULT = "1m"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; arbitrage-backfill/1.0)"}

DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/binance_klines"

# Per-request weight (fapi /klines): 5 weight, 2400/min cap = 480 req/min ceiling.
# We use a conservative 8 concurrent + 0.05s jitter between chunks.


@dataclass
class BackfillStats:
    candles: int = 0
    requests: int = 0
    errors: int = 0
    earliest: Optional[int] = None  # ms
    latest: Optional[int] = None  # ms
    shards_written: int = 0


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 4,
) -> list[list]:
    """Fetch up to `limit` klines in [start_ms, end_ms). Returns raw arrays."""
    url = f"{FAPI_BASE}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    backoff = 0.5
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, headers=HEADERS, timeout=30) as r:
                if r.status == 200:
                    return await r.json()
                if r.status in (418, 429):
                    # Banned or rate-limited; respect Retry-After if present
                    retry_after = float(r.headers.get("Retry-After", str(backoff)))
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue
                if r.status >= 500:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                # Other 4xx → raise
                txt = await r.text()
                raise RuntimeError(f"klines HTTP {r.status}: {txt[:200]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"klines failed after {max_retries} retries: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
    raise RuntimeError("klines exhausted retries silently")


def klines_to_df(rows: list[list]) -> pd.DataFrame:
    """Convert raw kline arrays from Binance to a typed DataFrame."""
    if not rows:
        return pd.DataFrame()
    cols = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "n_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base_volume", "taker_buy_quote_volume"):
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df["available_at"] = df["close_time"]  # candle is "available" once closed
    df = df.drop(columns=["_ignore"])
    return df


def shard_path_for(out_root: Path, symbol: str, interval: str, date_iso: str) -> Path:
    return (out_root / f"symbol={symbol}" / f"interval={interval}"
            / f"date={date_iso}.parquet")


def write_shard(out_root: Path, symbol: str, interval: str,
                date_iso: str, df: pd.DataFrame) -> Path:
    """Append (idempotent on open_time_ms) to the daily parquet shard."""
    p = shard_path_for(out_root, symbol, interval, date_iso)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time_ms"], keep="last")
        combined = combined.sort_values("open_time_ms").reset_index(drop=True)
    else:
        combined = df.sort_values("open_time_ms").reset_index(drop=True)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, p, compression="snappy")
    return p


async def backfill_range(
    session: aiohttp.ClientSession,
    out_root: Path,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_size_minutes: int,
    concurrency: int,
) -> BackfillStats:
    """Walk [start_dt, end_dt) in chunks of `chunk_size_minutes`."""
    stats = BackfillStats()
    sem = asyncio.Semaphore(concurrency)

    # Build list of (start_ms, end_ms) chunks
    chunks: list[tuple[int, int]] = []
    cur = start_dt
    chunk_delta = timedelta(minutes=chunk_size_minutes)
    while cur < end_dt:
        nxt = min(cur + chunk_delta, end_dt)
        chunks.append((int(cur.timestamp() * 1000), int(nxt.timestamp() * 1000)))
        cur = nxt
    print(f"[binance] {symbol} {interval}: {len(chunks)} chunks "
          f"({(end_dt - start_dt).total_seconds() / 86400:.1f}d window)")

    async def fetch_one(start_ms: int, end_ms: int):
        async with sem:
            try:
                rows = await fetch_klines(session, symbol, interval, start_ms, end_ms)
                return rows
            except Exception as e:
                stats.errors += 1
                print(f"  ! chunk {start_ms}: {e}")
                return []

    # Process all chunks; sort & write per day
    pending: dict[str, list[pd.DataFrame]] = {}
    last_print = time.time()
    chunk_concurrency_batch = max(40, concurrency * 4)
    for batch_start in range(0, len(chunks), chunk_concurrency_batch):
        batch = chunks[batch_start:batch_start + chunk_concurrency_batch]
        results = await asyncio.gather(*(fetch_one(s, e) for s, e in batch))
        stats.requests += len(results)

        for rows in results:
            if not rows:
                continue
            df = klines_to_df(rows)
            if df.empty:
                continue
            stats.candles += len(df)
            mn = int(df["open_time_ms"].min())
            mx = int(df["open_time_ms"].max())
            stats.earliest = mn if stats.earliest is None else min(stats.earliest, mn)
            stats.latest = mx if stats.latest is None else max(stats.latest, mx)

            # Group by UTC date
            df["date_iso"] = df["open_time"].dt.strftime("%Y-%m-%d")
            for day, group in df.groupby("date_iso"):
                pending.setdefault(day, []).append(group.drop(columns=["date_iso"]))

        # Flush periodically to avoid memory blow-up
        if sum(sum(len(d) for d in v) for v in pending.values()) > 5000:
            for day, parts in pending.items():
                merged = pd.concat(parts, ignore_index=True)
                write_shard(out_root, symbol, interval, day, merged)
                stats.shards_written += 1
            pending.clear()

        now = time.time()
        if now - last_print > 5:
            print(f"  [binance] {batch_start + len(batch)}/{len(chunks)} chunks "
                  f"candles={stats.candles} errors={stats.errors}")
            last_print = now

    # Final flush
    for day, parts in pending.items():
        merged = pd.concat(parts, ignore_index=True)
        write_shard(out_root, symbol, interval, day, merged)
        stats.shards_written += 1

    return stats


def detect_label_range(pm_root: Path) -> Optional[tuple[datetime, datetime]]:
    """Find min/max window_start across the polymarket_resolutions parquet
    shards, so the OHLCV backfill covers exactly the labeled range.
    """
    if not pm_root.exists():
        return None
    files = list(pm_root.rglob("*.parquet"))
    if not files:
        return None
    mn_ts: Optional[int] = None
    mx_ts: Optional[int] = None
    for f in files:
        df = pd.read_parquet(f, columns=["window_start"])
        if df.empty:
            continue
        f_mn = int(df["window_start"].min())
        f_mx = int(df["window_start"].max())
        mn_ts = f_mn if mn_ts is None else min(mn_ts, f_mn)
        mx_ts = f_mx if mx_ts is None else max(mx_ts, f_mx)
    if mn_ts is None:
        return None
    # Add a 1h pad on both ends so trailing windows have data.
    start = datetime.fromtimestamp(mn_ts, tz=timezone.utc) - timedelta(hours=1)
    end = datetime.fromtimestamp(mx_ts, tz=timezone.utc) + timedelta(hours=1)
    return (start, end)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Binance fapi klines for ML training.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output dir (default: {DEFAULT_OUT})")
    p.add_argument("--symbol", default=SYMBOL_DEFAULT)
    p.add_argument("--interval", default=INTERVAL_DEFAULT,
                   help="Binance kline interval (default: 1m)")
    p.add_argument("--start", default="",
                   help="ISO date YYYY-MM-DD; default = derived from PM label range")
    p.add_argument("--end", default="",
                   help="ISO date YYYY-MM-DD; default = derived from PM label range")
    p.add_argument("--chunk-minutes", type=int, default=1000,
                   help="Minutes per request (default 1000 = max for 1m interval)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent requests (default 8; well under 480/min budget)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: last 2 hours of klines only")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(hours=2)
        print(f"[smoke] last 2h: {start.isoformat()} → {end.isoformat()}")
    elif args.start and args.end:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    else:
        # Derive from PM resolutions
        pm_root = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/polymarket_resolutions"
        rng = detect_label_range(pm_root)
        if rng is None:
            print("[error] no PM resolutions found and no --start/--end given", file=sys.stderr)
            return 2
        start, end = rng
        print(f"[binance] derived range from PM corpus: {start.isoformat()} → {end.isoformat()}")
        if args.start:
            start = max(start, datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc))

    timeout = aiohttp.ClientTimeout(total=45)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        stats = await backfill_range(
            session=session,
            out_root=args.out,
            symbol=args.symbol,
            interval=args.interval,
            start_dt=start,
            end_dt=end,
            chunk_size_minutes=args.chunk_minutes,
            concurrency=args.concurrency,
        )
    elapsed = time.time() - t0
    print()
    print("=== summary ===")
    print(f"  symbol={args.symbol} interval={args.interval}")
    print(f"  candles_written={stats.candles}")
    print(f"  shards_written={stats.shards_written}")
    print(f"  requests={stats.requests} errors={stats.errors}")
    if stats.earliest is not None:
        e_dt = datetime.fromtimestamp(stats.earliest / 1000, tz=timezone.utc)
        l_dt = datetime.fromtimestamp(stats.latest / 1000, tz=timezone.utc)
        print(f"  range: {e_dt.isoformat()} → {l_dt.isoformat()}")
    print(f"  elapsed: {elapsed:.1f}s")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
