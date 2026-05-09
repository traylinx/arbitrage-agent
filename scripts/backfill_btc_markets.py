#!/usr/bin/env python3.11
"""
backfill_btc_markets.py — Phase 1a deliverable for SPRINT-ARBITRAGE-ML-V1.

Walks Polymarket gamma-api backwards in time fetching every resolved
`btc-updown-{tf}m-{window_unix}` market it can find. Persists each market
as a row in a daily parquet shard for downstream label_builder consumption.

Output layout (matches SPEC-V2 §1 collector schema):
    data/arbitrage-agent/v2/parquet-cache/
      polymarket_resolutions/
        date=YYYY-MM-DD/
          tf=5m.parquet
          tf=15m.parquet
        _state.json    # checkpoint: last_completed_window per tf

Usage:
    python3.11 backfill_btc_markets.py                 # default: walk back until 200 consecutive 404s
    python3.11 backfill_btc_markets.py --max-windows 500
    python3.11 backfill_btc_markets.py --tf 5          # 5m only
    python3.11 backfill_btc_markets.py --concurrency 5 # gentler on gamma-api
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GAMMA_API = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; arbitrage-backfill/1.0)"}

DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/polymarket_resolutions"
STATE_FILE_NAME = "_state.json"


@dataclass
class BackfillStats:
    tf_minutes: int
    hits: int = 0
    misses: int = 0
    errors: int = 0
    consecutive_misses: int = 0
    earliest_window: Optional[int] = None
    latest_window: Optional[int] = None
    rows_buffered: int = 0
    shards_written: int = 0
    by_day_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_hit(self, window: int):
        self.hits += 1
        self.consecutive_misses = 0
        if self.earliest_window is None or window < self.earliest_window:
            self.earliest_window = window
        if self.latest_window is None or window > self.latest_window:
            self.latest_window = window

    def record_miss(self):
        self.misses += 1
        self.consecutive_misses += 1


def slug_for(tf_minutes: int, window_start: int) -> str:
    return f"btc-updown-{tf_minutes}m-{window_start}"


async def fetch_market(
    session: aiohttp.ClientSession, slug: str, max_retries: int = 4
) -> Optional[dict]:
    """Fetch one closed market by slug. Returns None on miss; raises on hard error."""
    url = f"{GAMMA_API}/markets"
    params = {"slug": slug, "closed": "true"}
    backoff = 0.5

    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, headers=HEADERS, timeout=15) as r:
                if r.status == 200:
                    data = await r.json()
                    if isinstance(data, list) and data:
                        return data[0]
                    return None  # empty list = closed market not found at this slug
                if r.status == 429:
                    await asyncio.sleep(backoff + (attempt * 0.5))
                    backoff *= 2
                    continue
                if r.status >= 500:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                # 4xx other than 429 = real miss
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"fetch_market({slug}) failed: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


def decode_outcome(market: dict) -> Optional[dict]:
    """Extract the V2 label-builder fields from a gamma-api market dict.

    Returns None if the market lacks resolution data we need.
    """
    status = market.get("umaResolutionStatus")
    if status != "resolved":
        return None
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = None
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            prices = None
    if not (isinstance(outcomes, list) and isinstance(prices, list) and len(outcomes) == 2 == len(prices)):
        return None

    # outcomePrices is e.g. ["1", "0"] or ["0", "1"]; element with "1" is the winner
    try:
        prices_f = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    if max(prices_f) < 0.99:
        return None  # no clean winner — disputed or void

    winner_idx = prices_f.index(max(prices_f))
    winner_label = outcomes[winner_idx]
    binary_label_up_won = 1 if winner_label.lower() == "up" else 0

    clob_ids = market.get("clobTokenIds")
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except Exception:
            clob_ids = None
    if not (isinstance(clob_ids, list) and len(clob_ids) == 2):
        clob_up = clob_down = ""
    else:
        clob_up, clob_down = clob_ids[0], clob_ids[1]

    fee_schedule = market.get("feeSchedule") or {}
    if isinstance(fee_schedule, str):
        try:
            fee_schedule = json.loads(fee_schedule)
        except Exception:
            fee_schedule = {}

    return {
        "slug": market.get("slug", ""),
        "condition_id": market.get("conditionId", ""),
        "clob_token_id_up": str(clob_up),
        "clob_token_id_down": str(clob_down),
        "winner": winner_label,
        "binary_label_up_won": binary_label_up_won,
        "outcome_prices": json.dumps(prices),
        "event_start_time": market.get("eventStartTime", ""),
        "end_date": market.get("endDate", ""),
        "closed_time": str(market.get("closedTime", "")),
        "accepting_orders_ts": market.get("acceptingOrdersTimestamp", ""),
        "volume": float(market.get("volume", 0.0) or 0.0),
        "volume_clob": float(market.get("volumeClob", 0.0) or 0.0),
        "best_bid": float(market.get("bestBid", 0.0) or 0.0),
        "best_ask": float(market.get("bestAsk", 0.0) or 0.0),
        "spread": float(market.get("spread", 0.0) or 0.0),
        "order_min_size": int(market.get("orderMinSize", 0) or 0),
        "tick_size": float(market.get("orderPriceMinTickSize", 0.01) or 0.01),
        "fee_rate": float(fee_schedule.get("rate", 0.0) or 0.0),
        "fee_rebate_rate": float(fee_schedule.get("rebateRate", 0.0) or 0.0),
        "fee_taker_only": bool(fee_schedule.get("takerOnly", False)),
        "resolution_source": market.get("resolutionSource", ""),
        "question": market.get("question", ""),
    }


def date_for_window(window_unix: int) -> str:
    return datetime.fromtimestamp(window_unix, tz=timezone.utc).strftime("%Y-%m-%d")


def write_shard(out_root: Path, tf_minutes: int, date_iso: str, rows: list[dict]) -> Path:
    """Append rows to the daily parquet shard. Idempotent on (tf, date, window_start)."""
    day_dir = out_root / f"date={date_iso}"
    day_dir.mkdir(parents=True, exist_ok=True)
    shard_path = day_dir / f"tf={tf_minutes}m.parquet"

    new_df = pd.DataFrame(rows)
    if shard_path.exists():
        existing = pd.read_parquet(shard_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["window_start"], keep="last")
    else:
        combined = new_df

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, shard_path, compression="snappy")
    return shard_path


def load_state(out_root: Path) -> dict:
    p = out_root / STATE_FILE_NAME
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_state(out_root: Path, state: dict) -> None:
    p = out_root / STATE_FILE_NAME
    p.write_text(json.dumps(state, indent=2))


async def backfill_tf(
    session: aiohttp.ClientSession,
    out_root: Path,
    tf_minutes: int,
    start_window: int,
    max_windows: int,
    miss_cutoff: int,
    concurrency: int,
    flush_every: int,
    state: dict,
) -> BackfillStats:
    """Walk windows backwards from start_window until cutoff hit or max reached."""
    stats = BackfillStats(tf_minutes=tf_minutes)
    sem = asyncio.Semaphore(concurrency)
    window_sec = tf_minutes * 60

    # Align start_window down to this tf's boundary (15m = XX:00/15/30/45;
    # 5m = XX:00/05/10/...). Caller may pass a 5m-aligned ts.
    start_window = (start_window // window_sec) * window_sec

    # Resume from checkpoint if present
    resume_key = f"tf{tf_minutes}_oldest_seen"
    if resume_key in state:
        start_window = min(start_window, state[resume_key] - window_sec)

    # Build candidate window list
    candidates = [start_window - i * window_sec for i in range(max_windows)]
    print(f"[tf={tf_minutes}m] backfilling up to {len(candidates)} windows starting at "
          f"{datetime.fromtimestamp(start_window, tz=timezone.utc).isoformat()}")

    buffer: dict[str, list[dict]] = defaultdict(list)  # date_iso -> rows
    consecutive = 0
    last_print = time.time()

    async def fetch_one(window: int):
        async with sem:
            slug = slug_for(tf_minutes, window)
            try:
                m = await fetch_market(session, slug)
            except Exception as e:
                stats.errors += 1
                print(f"  ! {slug}: {e}")
                return window, None
            return window, m

    # Process in chunks so we can short-circuit on miss-cutoff
    chunk_size = max(20, concurrency * 3)
    i = 0
    while i < len(candidates):
        chunk = candidates[i : i + chunk_size]
        results = await asyncio.gather(*(fetch_one(w) for w in chunk))
        # results are in order
        for window, m in results:
            if m is None:
                stats.record_miss()
                consecutive += 1
                continue
            decoded = decode_outcome(m)
            if decoded is None:
                stats.record_miss()
                consecutive += 1
                continue
            decoded["tf_minutes"] = tf_minutes
            decoded["window_start"] = window
            decoded["window_start_iso"] = datetime.fromtimestamp(
                window, tz=timezone.utc
            ).isoformat()
            decoded["fetched_at"] = datetime.now(timezone.utc).isoformat()
            day = date_for_window(window)
            buffer[day].append(decoded)
            stats.record_hit(window)
            stats.rows_buffered += 1
            stats.by_day_rows[day] += 1
            consecutive = 0

        # Flush if buffer big enough or progress milestone
        if stats.rows_buffered >= flush_every:
            for day, rows in buffer.items():
                write_shard(out_root, tf_minutes, day, rows)
                stats.shards_written += 1
            buffer.clear()
            stats.rows_buffered = 0

        # Periodic status
        now = time.time()
        if now - last_print > 5:
            print(f"  [tf={tf_minutes}m] processed {i + len(chunk)}/{len(candidates)} windows "
                  f"hits={stats.hits} misses={stats.misses} errors={stats.errors} "
                  f"earliest={stats.earliest_window}")
            last_print = now

        # Cutoff: stop if we've seen too many consecutive misses (likely past PM history)
        if consecutive >= miss_cutoff:
            print(f"[tf={tf_minutes}m] cutoff hit: {consecutive} consecutive misses, stopping")
            break

        i += chunk_size

    # Final flush
    for day, rows in buffer.items():
        write_shard(out_root, tf_minutes, day, rows)
        stats.shards_written += 1
    buffer.clear()

    # Update state
    if stats.earliest_window is not None:
        state[resume_key] = stats.earliest_window
    save_state(out_root, state)

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Polymarket BTC up/down markets.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output dir (default: {DEFAULT_OUT})")
    p.add_argument("--tf", default="5,15",
                   help="Comma-separated timeframes in minutes (default: 5,15)")
    p.add_argument("--start-window", type=int, default=0,
                   help="Unix ts to start walking backwards from (0 = now - 1h, aligned)")
    p.add_argument("--max-windows", type=int, default=60_000,
                   help="Max windows to attempt per tf (default: 60000 ~ 200d at 5m)")
    p.add_argument("--miss-cutoff", type=int, default=200,
                   help="Stop after N consecutive misses (default: 200)")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Concurrent gamma-api requests (default: 10)")
    p.add_argument("--flush-every", type=int, default=500,
                   help="Buffer rows before parquet flush (default: 500)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: fetch only last 24 windows of each tf")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    state = load_state(args.out)

    if args.smoke:
        args.max_windows = 24
        args.miss_cutoff = 24
        print(f"[smoke] limited to {args.max_windows} windows per tf")

    if args.start_window == 0:
        # Align to current 5m boundary, back off 1h to avoid unresolved markets
        args.start_window = (int(time.time()) - 3600) // 300 * 300

    tfs = [int(x) for x in args.tf.split(",") if x.strip()]
    print(f"backfill: out={args.out} tfs={tfs} start={args.start_window} "
          f"({datetime.fromtimestamp(args.start_window, tz=timezone.utc).isoformat()}) "
          f"max={args.max_windows} cutoff={args.miss_cutoff} concurrency={args.concurrency}")
    print(f"resume state: {state or '<empty>'}")
    print()

    timeout = aiohttp.ClientTimeout(total=20)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        all_stats: list[BackfillStats] = []
        for tf in tfs:
            t0 = time.time()
            stats = await backfill_tf(
                session=session,
                out_root=args.out,
                tf_minutes=tf,
                start_window=args.start_window,
                max_windows=args.max_windows,
                miss_cutoff=args.miss_cutoff,
                concurrency=args.concurrency,
                flush_every=args.flush_every,
                state=state,
            )
            elapsed = time.time() - t0
            print(f"[tf={tf}m] done in {elapsed:.1f}s  hits={stats.hits} "
                  f"misses={stats.misses} errors={stats.errors} "
                  f"shards={stats.shards_written} earliest_window={stats.earliest_window}")
            if stats.earliest_window is not None:
                print(f"  earliest UTC: "
                      f"{datetime.fromtimestamp(stats.earliest_window, tz=timezone.utc).isoformat()}")
            all_stats.append(stats)

    print()
    print("=== summary ===")
    for s in all_stats:
        days = len(s.by_day_rows)
        total_rows = sum(s.by_day_rows.values()) + s.rows_buffered
        print(f"  tf={s.tf_minutes}m: hits={s.hits} misses={s.misses} errors={s.errors} "
              f"days={days} rows_in_shards={total_rows - s.rows_buffered}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
