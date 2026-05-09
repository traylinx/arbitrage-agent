#!/usr/bin/env python3.11
"""live_shadow.py — V2 §5 Phase 1: paper shadow runner.

Runs the trained baseline_v0 model against LIVE Polymarket BTC markets
and the LIVE Binance perp feed, computing the V2 decision for each
active 5m/15m market window, AND logging:
  - the decision the bot would make
  - the actual yes_ask / no_ask at decision_ts (this is the missing
    piece backfill couldn't give us)
  - market metadata so we can join to resolution later

NOTHING IS PLACED. This is paper-only. After the run accumulates 24-48h
of decisions, we can join `live_shadow_decisions.jsonl` against the
window's eventual resolution and compute REAL production-time edge.

Output:
    data/arbitrage-agent/v2/state/live_shadow/decisions.jsonl

Each row is one decision per (slug, decision_ts). The same window is
NOT re-decided — we lock in at decision_ts = window_start + 60s.

Usage:
    python3.11 -m scripts.live_shadow                # both 5m + 15m
    python3.11 -m scripts.live_shadow --tf 5         # 5m only
    python3.11 -m scripts.live_shadow --once         # one cycle, exit
    python3.11 -m scripts.live_shadow --duration 3600  # run 1 hour
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Path bootstrap so this script can be run as `python -m scripts.live_shadow`
# AND find sibling modules.
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

import features.price  # noqa: F401  populate registry
import features.crossex  # noqa: F401
from features.feature_base import build_feature_vector
from features.predictor_adapter import load_predictor
from features.label_builder import (
    klines_to_synthetic_trades,
    klines_to_synthetic_bookticker,
    build_global_snapshot,
)
from features.external_metrics_loader import (
    DEFAULT_BASE as EXT_METRICS_BASE,
    DEFAULT_CROSS_EX_BASE,
    load_coinalyze_wide,
    load_coinglass_wide,
    load_cross_exchange_all,
)
from decision_engine import (
    ModelDecisionEngine, PMOrderbook, Wallet,
)

GAMMA_API = "https://gamma-api.polymarket.com"
FAPI_BASE = "https://fapi.binance.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; arbitrage-shadow/1.0)"}

DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/state/live_shadow"
DECISION_OFFSET_SEC = 60   # match training: decision_ts = window_start + 60s
KLINE_LOOKBACK_MINUTES = 90
DERIV_REFRESH_SEC = 1800   # re-load external_metrics parquet every 30 min


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ─── Polymarket fetch ───────────────────────────────────────────────────────


def fetch_active_btc_market(tf_minutes: int, window_start: int) -> Optional[dict]:
    """Look up the BTC up/down market for this exact window, if accepting orders."""
    slug = f"btc-updown-{tf_minutes}m-{window_start}"
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"slug": slug},
            headers=HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        m = data[0]
        return m
    except (requests.RequestException, ValueError):
        return None


def parse_pm_book(market: dict) -> Optional[PMOrderbook]:
    """Build a PMOrderbook from gamma-api market metadata.

    gamma-api gives us `bestBid`/`bestAsk` which is the YES side. The NO
    side derives from inverse pricing (yes_ask + no_ask ≈ 1). For shadow
    we use this approximation; live PM CLOB books from Phase 1b will
    give us true two-sided quotes.
    """
    yes_bid = float(market.get("bestBid", 0.0) or 0.0)
    yes_ask = float(market.get("bestAsk", 0.0) or 0.0)
    if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
        return None
    clob_ids = market.get("clobTokenIds")
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except Exception:
            clob_ids = None
    if not (isinstance(clob_ids, list) and len(clob_ids) == 2):
        clob_ids = ["", ""]
    return PMOrderbook(
        yes_token_id=str(clob_ids[0]),
        no_token_id=str(clob_ids[1]),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=max(0.0, 1.0 - yes_ask),
        no_ask=min(1.0, 1.0 - yes_bid),
        tick_size=float(market.get("orderPriceMinTickSize", 0.01) or 0.01),
    )


# ─── Binance fetch ──────────────────────────────────────────────────────────


def fetch_recent_klines(minutes: int = KLINE_LOOKBACK_MINUTES) -> pd.DataFrame:
    """Fetch the last `minutes` of BTCUSDT 1m klines from Binance fapi."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - minutes * 60_000
    try:
        r = requests.get(
            f"{FAPI_BASE}/fapi/v1/klines",
            params={"symbol": "BTCUSDT", "interval": "1m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            headers=HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        cols = [
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "quote_volume", "n_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        df["available_at"] = df["close_time"]
        df = df.drop(columns=["_ignore"]).sort_values("open_time_ms").reset_index(drop=True)
        return df
    except (requests.RequestException, ValueError):
        return pd.DataFrame()


# ─── Decision logging ───────────────────────────────────────────────────────


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_already_decided(path: Path) -> set[str]:
    """Avoid double-logging the same (slug, decision_ts) tuple."""
    if not path.exists():
        return set()
    seen = set()
    try:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(f"{rec.get('slug')}|{rec.get('decision_ts')}")
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return seen


# ─── Main loop ───────────────────────────────────────────────────────────────


_deriv_cache: dict = {"loaded_at": 0.0, "frames": {}}
_cross_ex_cache: dict = {"loaded_at": 0.0, "frames": {}}


def get_derivatives() -> dict[str, pd.DataFrame]:
    """Load coinalyze + coinglass wide frames from parquet, cached for
    DERIV_REFRESH_SEC. Returns empty dict if no shards exist (no error).
    """
    global _deriv_cache
    now = time.time()
    if now - _deriv_cache["loaded_at"] < DERIV_REFRESH_SEC and _deriv_cache["frames"]:
        return _deriv_cache["frames"]
    frames: dict[str, pd.DataFrame] = {}
    try:
        ca = load_coinalyze_wide(EXT_METRICS_BASE)
        if not ca.empty:
            frames["coinalyze"] = ca
        cg = load_coinglass_wide(EXT_METRICS_BASE)
        if not cg.empty:
            frames["coinglass"] = cg
    except Exception as e:
        print(f"[{now_utc().isoformat()}] external_metrics load failed: {e}")
    _deriv_cache = {"loaded_at": now, "frames": frames}
    if frames:
        print(f"[{now_utc().isoformat()}] external_metrics refreshed: "
              f"{ {k: len(v) for k, v in frames.items()} }")
    return frames


def get_cross_ex_trades() -> dict[str, pd.DataFrame]:
    """Load coinbase + bybit + hyperliquid 1m klines as 'trades' frames,
    cached for DERIV_REFRESH_SEC. Empty dict if no shards exist.
    """
    global _cross_ex_cache
    now = time.time()
    if now - _cross_ex_cache["loaded_at"] < DERIV_REFRESH_SEC and _cross_ex_cache["frames"]:
        return _cross_ex_cache["frames"]
    frames: dict[str, pd.DataFrame] = {}
    try:
        frames = load_cross_exchange_all(base=DEFAULT_CROSS_EX_BASE)
    except Exception as e:
        print(f"[{now_utc().isoformat()}] cross_exchange load failed: {e}")
    _cross_ex_cache = {"loaded_at": now, "frames": frames}
    if frames:
        print(f"[{now_utc().isoformat()}] cross_exchange refreshed: "
              f"{ {k: len(v) for k, v in frames.items()} }")
    return frames


def cycle(predictors: dict[int, object], decisions_path: Path,
          already_decided: set, fees_pct: float = 0.0) -> int:
    """One scan cycle. Returns number of new decisions logged."""
    klines = fetch_recent_klines()
    if klines.empty:
        print(f"[{now_utc().isoformat()}] no klines — skipping cycle")
        return 0
    derivatives = get_derivatives()
    cross_ex = get_cross_ex_trades()
    snap = build_global_snapshot(klines, derivatives=derivatives,
                                 cross_exchange_trades=cross_ex)

    new_decisions = 0
    now_ts = int(time.time())
    for tf, predictor in predictors.items():
        engine = ModelDecisionEngine(
            predictor=predictor,
            edge_threshold=0.04,
            kelly_multiplier=0.25,
            fees_maker_pct=fees_pct,
            assume_maker_execution=True,
            max_open_orders=999,  # shadow doesn't track real open orders
            max_wallet_age_sec=999_999,  # shadow doesn't track wallet
            min_balance_usdc=0.0,
        )
        window_sec = tf * 60
        # Decision moment = window_start + 60s. So we fire on the window
        # whose window_start was 60-90s ago. Pick the most recent one
        # whose decision_ts has passed.
        target_window = (now_ts - DECISION_OFFSET_SEC) // window_sec * window_sec
        decision_ts = datetime.fromtimestamp(target_window + DECISION_OFFSET_SEC,
                                             tz=timezone.utc)

        slug = f"btc-updown-{tf}m-{target_window}"
        key = f"{slug}|{decision_ts.isoformat()}"
        if key in already_decided:
            continue

        market = fetch_active_btc_market(tf, target_window)
        if market is None:
            print(f"[{now_utc().isoformat()}] {slug}: market not found")
            continue

        book = parse_pm_book(market)
        # Wallet is a stub for shadow mode
        wallet = Wallet(usdc_balance=10.0, open_orders=0, last_refreshed_age_sec=0.0)

        fv = build_feature_vector(snap, decision_ts)
        d = engine.decide(fv, book, wallet)

        record = {
            "tf_minutes": tf,
            "slug": slug,
            "window_start": target_window,
            "decision_ts": decision_ts.isoformat(),
            "logged_at": now_utc().isoformat(),
            "decision": {
                "action": d.action,
                "side": d.side,
                "fraction": d.fraction,
                "edge": d.edge,
                "p_model": d.p_model,
                "p_market": d.p_market,
                "reason": d.reason,
            },
            "book": {
                "yes_bid": book.yes_bid if book else None,
                "yes_ask": book.yes_ask if book else None,
                "no_bid": book.no_bid if book else None,
                "no_ask": book.no_ask if book else None,
                "tick_size": book.tick_size if book else None,
            } if book else None,
            "features": dict(fv.values),
            "market_meta": {
                "accepting_orders": market.get("acceptingOrders", False),
                "closed": market.get("closed", False),
                "end_date": market.get("endDate"),
                "volume": float(market.get("volume", 0.0) or 0.0),
                "condition_id": market.get("conditionId"),
                "fee_rate": float(
                    json.loads(market["feeSchedule"]).get("rate", 0.0)
                    if isinstance(market.get("feeSchedule"), str)
                    else (market.get("feeSchedule") or {}).get("rate", 0.0)
                ),
            },
            "schema_hash": fv.schema_hash[:16],
        }
        append_jsonl(decisions_path, record)
        already_decided.add(key)
        new_decisions += 1

        action = d.action
        side = d.side or "—"
        # Recover raw p_up: for YES bets d.p_model = p_up; for NO bets it's 1 - p_up
        if d.p_model is None:
            raw_p_up = float("nan")
        elif d.side == "NO":
            raw_p_up = 1.0 - d.p_model
        else:
            raw_p_up = d.p_model
        ya = book.yes_ask if book else 0.0
        na = book.no_ask if book else 0.0
        edge = d.edge if d.edge is not None else 0.0
        print(f"[{now_utc().strftime('%H:%M:%S')}] {tf}m {slug.split('-')[-1]} "
              f"p_up={raw_p_up:.3f} yes_ask={ya:.3f} no_ask={na:.3f} → "
              f"{action} {side} edge={edge:+.4f} ({d.reason[:40]})")

    return new_decisions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 §5 paper shadow runner.")
    p.add_argument("--tf", default="5,15", help="Comma-separated tfs to shadow (default 5,15)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--interval-sec", type=int, default=30,
                   help="Cycle interval in seconds (default 30)")
    p.add_argument("--duration", type=int, default=0,
                   help="Run for N seconds, then exit. 0 = run forever.")
    p.add_argument("--once", action="store_true", help="One cycle, then exit")
    p.add_argument("--fees-pct", type=float, default=0.0,
                   help="Fees fraction (0 for maker, 0.02 for taker)")
    p.add_argument("--model", choices=["logreg", "xgb", "lgbm"], default="logreg",
                   help="Which model artifacts to load (default logreg = baseline_v0/{tf}m/, "
                        "xgb = baseline_v0/{tf}m_xgb/)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    decisions_path = args.out / "decisions.jsonl"

    tfs = [int(x) for x in args.tf.split(",") if x.strip()]
    print(f"[live_shadow] tfs={tfs} interval={args.interval_sec}s out={decisions_path}")

    predictors = {}
    for tf in tfs:
        try:
            p = load_predictor(tf, model_kind=args.model)
            predictors[tf] = p
            print(f"  loaded {tf}m {args.model} predictor: {p.info()}")
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    already = load_already_decided(decisions_path)
    print(f"  resume: {len(already)} prior decisions in log")

    stop = {"flag": False}
    def _sig(_n, _f):
        stop["flag"] = True
        print("\n[live_shadow] caught signal — finishing current cycle...")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t0 = time.time()
    cycles = 0
    decisions = 0
    while not stop["flag"]:
        n = cycle(predictors, decisions_path, already, fees_pct=args.fees_pct)
        decisions += n
        cycles += 1
        if args.once:
            break
        if args.duration > 0 and (time.time() - t0) >= args.duration:
            break
        for _ in range(args.interval_sec):
            if stop["flag"]:
                break
            time.sleep(1)

    elapsed = time.time() - t0
    print()
    print(f"[live_shadow] {cycles} cycles, {decisions} new decisions, {elapsed:.0f}s")
    print(f"[live_shadow] decisions log: {decisions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
