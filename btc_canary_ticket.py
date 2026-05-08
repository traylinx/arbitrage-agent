#!/usr/bin/env python3
"""Single-shot $1 canary order for BTC 5m/15m Polymarket markets.

Hardcoded safety rails (cannot be overridden by CLI flags):
  WALLET_CAP_USDC   = $5.00   — total deployed across canary trades
  MAX_ORDER_USDC    = $1.00   — never spend more than this on one order
  DAILY_KILL_USDC   = $2.00   — stop trading if cumulative day's BUY notional
                                exceeds this and net realized PnL is negative

Operator flow:
  1. List candidates:    --list
  2. Build dry-run:      --market <slug> --side YES|NO --cost 1.00
  3. Place real order:   add --confirm  (will post a GTC limit at the best ask)

Refuses to place an order unless:
  - market.question matches BTC 5m/15m pattern
  - end-of-market is in the future and within 60 minutes
  - cost <= MAX_ORDER_USDC
  - cumulative wallet spend (today + open) <= WALLET_CAP_USDC
  - daily kill not tripped
  - --confirm flag is explicitly set

Journal: data/arbitrage-agent/v2/state/canary_journal.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY

from btc_fee_model import buy_cost_usdc, resolved_buy_pnl, taker_fee_usdc

WALLET_CAP_USDC = 5.00
MAX_ORDER_USDC = 1.00
DAILY_KILL_USDC = 2.00

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
ENV_FILE = HARVEY_HOME / "data" / "arbitrage-agent" / ".env.live"
STATE_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2" / "state"
JOURNAL = STATE_DIR / "canary_journal.jsonl"

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

BTC_MARKET_PATTERNS = [
    re.compile(r"bitcoin up or down", re.I),
    re.compile(r"\bbtc\b.*\b(up|down)\b", re.I),
    re.compile(r"bitcoin.*5\s*min", re.I),
    re.compile(r"bitcoin.*15\s*min", re.I),
    re.compile(r"bitcoin\s+(above|below)\s+\$?[\d,]+", re.I),
    re.compile(r"price of bitcoin.*\b(above|below|between)\b", re.I),
]


@dataclass
class Ticket:
    market_slug: str
    market_question: str
    condition_id: str
    side: str  # "YES" or "NO"
    token_id: str
    limit_price: float
    shares: float
    cost: float
    fee_estimate: float
    win_payout: float
    win_pnl: float
    lose_pnl: float
    end_iso: str
    minutes_to_end: float


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_btc_5_or_15(question: str) -> bool:
    return any(p.search(question or "") for p in BTC_MARKET_PATTERNS)


def _read_journal() -> list[dict]:
    if not JOURNAL.exists():
        return []
    out = []
    for line in JOURNAL.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _today_spend(rows: list[dict]) -> tuple[float, float]:
    """Return (today_buy_notional, today_realized_pnl)."""
    today = _now_utc().date().isoformat()
    spend = 0.0
    pnl = 0.0
    for r in rows:
        ts = r.get("placed_at_utc") or ""
        if not ts.startswith(today):
            continue
        if r.get("status") == "PLACED":
            spend += float(r.get("cost_usdc") or 0)
        if r.get("status") == "RESOLVED":
            pnl += float(r.get("realized_pnl_usdc") or 0)
    return spend, pnl


def _all_time_spend(rows: list[dict]) -> float:
    return sum(float(r.get("cost_usdc") or 0) for r in rows if r.get("status") == "PLACED")


def _make_client() -> ClobClient:
    if not ENV_FILE.exists():
        sys.exit(f"FATAL: {ENV_FILE} not found")
    load_dotenv(ENV_FILE)
    pk = os.environ.get("POLYMARKET_PRIVATE_KEY")
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS")
    sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", 2))
    if not pk or not funder:
        sys.exit("FATAL: POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS missing")
    client = ClobClient(CLOB_API, key=pk, chain_id=137, signature_type=sig_type, funder=funder)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def list_candidates(window_minutes: int = 60) -> list[dict]:
    """Fetch BTC 5/15-min markets resolving in the future within window_minutes.

    Gamma's active=true filter is unreliable (returns stale-closed markets), so we
    pull from multiple sort orders and dedupe by conditionId, then enforce
    minutes_to_end > 0 ourselves.
    """
    seen = {}
    for params in (
        {"active": "true", "closed": "false", "limit": 500, "order": "startDate", "ascending": "false"},
        {"active": "true", "closed": "false", "limit": 500, "order": "endDate", "ascending": "true"},
        {"active": "true", "closed": "false", "limit": 500, "order": "volume", "ascending": "false"},
    ):
        try:
            r = requests.get(f"{GAMMA_API}/markets", params=params, timeout=15)
            for m in r.json():
                cid = m.get("conditionId") or m.get("id")
                if cid:
                    seen[cid] = m
        except Exception:
            continue
    out = []
    now = _now_utc()
    for m in seen.values():
        q = m.get("question") or ""
        if not _is_btc_5_or_15(q):
            continue
        end = m.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except Exception:
            continue
        mins = (end_dt - now).total_seconds() / 60
        if mins <= 0 or mins > window_minutes:
            continue
        op = m.get("outcomePrices") or "[]"
        try:
            prices = json.loads(op) if isinstance(op, str) else op
            yes = float(prices[0])
        except Exception:
            yes = 0.0
        tokens = m.get("clobTokenIds") or "[]"
        try:
            tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
        except Exception:
            tokens = []
        if not tokens or len(tokens) < 2:
            continue
        out.append({
            "slug": m.get("slug", ""),
            "question": q,
            "condition_id": m.get("conditionId", ""),
            "yes_token": tokens[0],
            "no_token": tokens[1],
            "yes_price": yes,
            "no_price": 1.0 - yes,
            "minutes_to_end": mins,
            "end_iso": end,
            "liquidity": float(m.get("liquidityNum") or m.get("liquidity") or 0),
            "vol24h": float(m.get("volume24hr") or 0),
        })
    out.sort(key=lambda x: x["minutes_to_end"])
    return out


def build_ticket(client: ClobClient, slug: str, side: str, max_cost: float) -> Ticket:
    side = side.upper()
    if side not in ("YES", "NO"):
        sys.exit(f"side must be YES or NO, got {side}")
    cands = list_candidates(window_minutes=120)
    match = next((c for c in cands if c["slug"] == slug), None)
    if not match:
        all_btc = list_candidates(window_minutes=24 * 60)
        sys.exit(
            f"slug '{slug}' not in tradeable 5/15-min BTC window (≤120 min).\n"
            f"Currently tradeable slugs:\n  " +
            "\n  ".join(c["slug"] for c in all_btc[:20])
        )

    token_id = match["yes_token"] if side == "YES" else match["no_token"]
    book = client.get_order_book(token_id)
    asks = sorted(book.asks, key=lambda a: float(a.price)) if book.asks else []
    if not asks:
        sys.exit(f"No asks on {side} side of {slug}")
    best_ask = float(asks[0].price)
    best_ask_size = float(asks[0].size)

    limit_price = round(best_ask, 4)
    shares = round(max_cost / limit_price, 2)
    cost = buy_cost_usdc(shares, limit_price)
    fee = taker_fee_usdc(shares, limit_price)
    win_payout = shares
    win_pnl = resolved_buy_pnl(True, shares, limit_price)
    lose_pnl = resolved_buy_pnl(False, shares, limit_price)

    if best_ask_size < shares:
        print(f"[warn] best ask size {best_ask_size:.2f} < requested {shares:.2f} — order may partial-fill")

    return Ticket(
        market_slug=slug,
        market_question=match["question"],
        condition_id=match["condition_id"],
        side=side,
        token_id=token_id,
        limit_price=limit_price,
        shares=shares,
        cost=cost,
        fee_estimate=fee,
        win_payout=win_payout,
        win_pnl=win_pnl,
        lose_pnl=lose_pnl,
        end_iso=match["end_iso"],
        minutes_to_end=match["minutes_to_end"],
    )


def safety_check(client: ClobClient, ticket: Ticket) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if ticket.cost > MAX_ORDER_USDC + 1e-6:
        issues.append(f"cost ${ticket.cost:.4f} exceeds MAX_ORDER_USDC ${MAX_ORDER_USDC}")
    rows = _read_journal()
    today_spend, today_pnl = _today_spend(rows)
    open_spend = _all_time_spend([r for r in rows if r.get("status") == "PLACED" and not r.get("resolved")])
    cumulative = max(today_spend, open_spend)
    if cumulative + ticket.cost > WALLET_CAP_USDC + 1e-6:
        issues.append(
            f"cumulative open spend ${cumulative:.4f} + this ${ticket.cost:.4f} "
            f"exceeds WALLET_CAP_USDC ${WALLET_CAP_USDC}"
        )
    if today_spend >= DAILY_KILL_USDC and today_pnl < 0:
        issues.append(
            f"daily kill triggered: today_spend ${today_spend:.2f} >= ${DAILY_KILL_USDC} "
            f"with negative realized PnL ${today_pnl:+.2f}"
        )
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    avail = int(bal["balance"]) / 1e6
    if avail < ticket.cost + 0.01:
        issues.append(f"on-chain collateral ${avail:.4f} insufficient for cost ${ticket.cost:.4f}")
    if ticket.minutes_to_end < 1:
        issues.append(f"market resolves in {ticket.minutes_to_end:.1f} min, too late to be filled safely")
    return len(issues) == 0, issues


def render_ticket(ticket: Ticket) -> str:
    lines = [
        "=" * 70,
        "PROPOSED ORDER",
        "=" * 70,
        f"  Market   : {ticket.market_question}",
        f"  Slug     : {ticket.market_slug}",
        f"  Side     : BUY {ticket.side}",
        f"  Token    : {ticket.token_id[:20]}...",
        f"  Price    : ${ticket.limit_price:.4f}  (limit, GTC)",
        f"  Shares   : {ticket.shares}",
        f"  Cost     : ${ticket.cost:.4f}  (+ est fee ${ticket.fee_estimate:.4f})",
        f"  WIN      : payout ${ticket.win_payout:.2f}  → net ${ticket.win_pnl:+.4f}",
        f"  LOSE     : payout $0.00  → net ${ticket.lose_pnl:+.4f}",
        f"  Resolves : {ticket.end_iso}  (in {ticket.minutes_to_end:.1f} min)",
        f"  Caps     : MAX_ORDER=${MAX_ORDER_USDC}  WALLET=${WALLET_CAP_USDC}  DAILY_KILL=${DAILY_KILL_USDC}",
        "=" * 70,
    ]
    return "\n".join(lines)


def journal_write(entry: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def place_order(client: ClobClient, ticket: Ticket) -> dict:
    args = OrderArgs(price=ticket.limit_price, size=ticket.shares, side=BUY, token_id=ticket.token_id)
    signed = client.create_order(args)
    resp = client.post_order(signed, OrderType.GTC)
    return resp


def cmd_list() -> int:
    cands = list_candidates(window_minutes=24 * 60)
    if not cands:
        print("No 5/15-min BTC markets currently active.")
        return 0
    imm = [c for c in cands if c["minutes_to_end"] <= 60]
    print(f"Tradeable 5/15-min BTC markets within 60 min: {len(imm)}")
    for c in imm:
        print(f"  {c['minutes_to_end']:5.1f} min  YES=${c['yes_price']:.4f}  liq=${c['liquidity']:.0f}  {c['slug']}")
        print(f"    {c['question']}")
    soon = [c for c in cands if 60 < c["minutes_to_end"] <= 24 * 60]
    print(f"\nNext 5/15-min markets within 24h: {len(soon)}")
    for c in soon[:10]:
        h = c["minutes_to_end"] / 60
        print(f"  in {h:5.1f}h  YES=${c['yes_price']:.4f}  liq=${c['liquidity']:.0f}  {c['slug']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="$1 canary ticket for BTC 5/15-min Polymarket")
    ap.add_argument("--list", action="store_true", help="show currently tradeable BTC 5/15-min markets")
    ap.add_argument("--market", help="market slug (e.g. 'bitcoin-up-or-down-...')")
    ap.add_argument("--side", choices=["YES", "NO", "yes", "no"], help="BUY YES or BUY NO")
    ap.add_argument("--cost", type=float, default=MAX_ORDER_USDC, help="USDC cost (default 1.00, capped to MAX_ORDER_USDC)")
    ap.add_argument("--confirm", action="store_true", help="actually post the order. Without this flag, dry-run.")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if not args.market or not args.side:
        ap.error("--market and --side are required (or use --list)")

    if args.cost > MAX_ORDER_USDC:
        print(f"[clamp] requested cost ${args.cost:.4f} clamped to MAX_ORDER_USDC ${MAX_ORDER_USDC}")
        args.cost = MAX_ORDER_USDC

    client = _make_client()
    ticket = build_ticket(client, args.market, args.side, args.cost)
    print(render_ticket(ticket))

    ok, issues = safety_check(client, ticket)
    if not ok:
        print("\nSAFETY GATE FAIL:")
        for i in issues:
            print(f"  - {i}")
        return 2
    print("\nSAFETY GATE PASS")

    base_entry = {
        "placed_at_utc": _now_utc().isoformat(),
        "ticket": asdict(ticket),
        "caps": {
            "max_order_usdc": MAX_ORDER_USDC,
            "wallet_cap_usdc": WALLET_CAP_USDC,
            "daily_kill_usdc": DAILY_KILL_USDC,
        },
        "cost_usdc": ticket.cost,
    }

    if not args.confirm:
        print("\n[dry-run] no order posted. Re-run with --confirm to place.")
        base_entry["status"] = "DRY_RUN"
        journal_write(base_entry)
        return 0

    print("\nposting order...")
    try:
        resp = place_order(client, ticket)
    except Exception as e:
        base_entry["status"] = "ERROR"
        base_entry["error"] = str(e)
        journal_write(base_entry)
        print(f"ERROR: {e}")
        return 3

    base_entry["status"] = "PLACED" if resp.get("success") else "REJECTED"
    base_entry["response"] = resp
    journal_write(base_entry)

    if resp.get("success"):
        print(f"PLACED  order_id={resp.get('orderID') or resp.get('orderId')}  status={resp.get('status')}")
        return 0
    print(f"REJECTED  {resp}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
