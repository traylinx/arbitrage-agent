#!/usr/bin/env python3
"""BTC sniper live-canary GO/NO-GO report.

Reads latest freeze snapshot + journal rows after that snapshot. This is not a
profit guarantee; it is a mechanical gate to prevent live promotion while code,
fees, or params are still unstable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
STATE_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2" / "state"
FREEZE_FILE = STATE_DIR / "freezes" / "latest.json"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
CLOB_PRICE_SOURCE = "clob_book_ask_depth"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def max_drawdown(equity_steps: list[float], starting_equity: float = 20.0) -> float:
    peak = float(starting_equity)
    dd = 0.0
    for x in equity_steps:
        peak = max(peak, x)
        if peak > 0:
            dd = max(dd, (peak - x) / peak)
    return dd


def load_rows(start: datetime, *, require_clob: bool = True) -> tuple[list[dict], int]:
    rows = []
    excluded_non_clob = 0
    if not JOURNAL_FILE.exists():
        return rows, excluded_non_clob
    for line in JOURNAL_FILE.read_text().splitlines():
        try:
            r = json.loads(line)
            dt = datetime.fromisoformat(r.get("placed_at", ""))
        except Exception:
            continue
        if dt >= start and r.get("mode") == "paper" and "btc_delta" in r:
            if require_clob and r.get("price_source") != CLOB_PRICE_SOURCE:
                excluded_non_clob += 1
                continue
            rows.append(r)
    return rows, excluded_non_clob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=80)
    ap.add_argument("--min-wilson", type=float, default=0.55)
    ap.add_argument("--min-pnl", type=float, default=20.0)
    ap.add_argument("--max-dd", type=float, default=0.25)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-non-clob", action="store_true", help="include legacy/gamma fallback rows; never use for live")
    ap.add_argument("--allow-dynamic-changed", action="store_true", help="do not require dynamic params to match freeze")
    args = ap.parse_args()
    require_clob = not args.allow_non_clob
    require_dynamic_unchanged = not args.allow_dynamic_changed

    freeze = json.loads(FREEZE_FILE.read_text())
    start = datetime.fromisoformat(freeze["created_at"])
    rows, excluded_non_clob = load_rows(start, require_clob=require_clob)
    wins = sum(1 for r in rows if r.get("won"))
    losses = sum(1 for r in rows if r.get("won") is False)
    pnl_steps = [float(r.get("pnl") or 0.0) for r in rows]
    pnl = sum(pnl_steps)
    equity = []
    cur = 20.0
    for x in pnl_steps:
        cur += x
        equity.append(cur)
    wr = wins / len(rows) if rows else 0.0
    wilson = wilson_lower(wins, len(rows))
    dd = max_drawdown(equity, starting_equity=20.0)

    code_ok = True
    code_diff = []
    for rel, expected in freeze.get("code_hashes_sha256", {}).items():
        path = HARVEY_HOME / rel
        got = sha256_file(path) if path.exists() else "missing"
        if got != expected:
            code_ok = False
            code_diff.append(rel)

    params = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    dyn_ok = freeze.get("dynamic_params") == {k: params.get(k) for k in freeze.get("dynamic_fields", [])}

    gates = {
        "frozen_code_hashes_match": code_ok,
        "clob_realistic_rows_only": require_clob,
        "dynamic_params_unchanged_since_freeze": (dyn_ok if require_dynamic_unchanged else True),
        "min_trades": len(rows) >= args.min_trades,
        "wilson_wr_lower": wilson >= args.min_wilson,
        "positive_pnl": pnl >= args.min_pnl,
        "drawdown": dd <= args.max_dd,
    }
    go = all(gates.values())
    report = {
        "verdict": "GO_CANARY" if go else "NO_GO",
        "scope": "technical live canary only, not full-size trading",
        "freeze_start": freeze["created_at"],
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "wilson_wr_lower_95": wilson,
        "pnl": pnl,
        "max_drawdown": dd,
        "gates": gates,
        "code_diff": code_diff,
        "dynamic_params": {k: params.get(k) for k in freeze.get("dynamic_fields", [])},
        "dynamic_params_changed_during_training": not dyn_ok,
        "require_clob": require_clob,
        "excluded_non_clob_rows": excluded_non_clob,
        "canary_limits_if_go": {"wallet_cap_usdc": 5, "max_order_usdc": 1, "daily_kill_loss_usdc": 2},
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"VERDICT: {report['verdict']}")
        print(f"trades={len(rows)} W={wins} L={losses} WR={wr:.1%} Wilson95Lo={wilson:.1%} PnL=${pnl:+.2f} maxDD={dd:.1%} clob_only={require_clob} excluded_non_clob={excluded_non_clob}")
        for k, v in gates.items():
            print(f"{'PASS' if v else 'FAIL'} {k}")
        if go:
            print("Allowed next step: $5 wallet-cap / $1 order-cap canary only.")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
