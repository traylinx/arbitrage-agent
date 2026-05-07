#!/usr/bin/env python3
"""Create an auditable frozen-strategy snapshot for BTC paper/live promotion.

This does not place orders. It records:
- exact strategy contract version
- hashes of frozen code files
- whitelisted dynamic params only
- current readiness/trade counters if available
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from btc_param_contract import DYNAMIC_PARAM_FIELDS, STRATEGY_VERSION, read_params_file

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
STATE_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2" / "state"
BEST_PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
FREEZE_DIR = STATE_DIR / "freezes"

FROZEN_CODE_FILES = [
    SRC_DIR / "btc_fee_model.py",
    SRC_DIR / "btc_param_contract.py",
    SRC_DIR / "btc_paper_fast.py",
    SRC_DIR / "btc_backtest_autoresearch.py",
    SRC_DIR / "autoimprove_v3.py",
    SRC_DIR / "btc_paper_fast_watchdog.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def journal_stats() -> dict:
    import json as _json
    rows = []
    if not JOURNAL_FILE.exists():
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pnl": 0.0}
    for line in JOURNAL_FILE.read_text().splitlines():
        try:
            r = _json.loads(line)
        except Exception:
            continue
        if r.get("mode") == "paper" and "btc_delta" in r:
            rows.append(r)
    wins = sum(1 for r in rows if r.get("won"))
    losses = sum(1 for r in rows if r.get("won") is False)
    pnl = sum(float(r.get("pnl") or 0) for r in rows)
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(rows) if rows else 0.0,
        "pnl": pnl,
    }


def build_snapshot(label: str, mode: str) -> dict:
    params = read_params_file(BEST_PARAMS_FILE, mode="paper")
    code_hashes = {str(p.relative_to(HARVEY_HOME)): sha256_file(p) for p in FROZEN_CODE_FILES if p.exists()}
    dynamic = {k: params[k] for k in DYNAMIC_PARAM_FIELDS}
    return {
        "label": label,
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "created_ts": int(time.time()),
        "strategy_version": STRATEGY_VERSION,
        "dynamic_fields": list(DYNAMIC_PARAM_FIELDS),
        "dynamic_params": dynamic,
        "fixed_params": {k: v for k, v in params.items() if k not in DYNAMIC_PARAM_FIELDS and k not in {"metrics", "rejected_fields"}},
        "code_hashes_sha256": code_hashes,
        "journal_stats_all_btc_paper": journal_stats(),
        "promotion_rule": "Only dynamic_params may change after this snapshot; code_hash changes require a new freeze.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="paper-freeze")
    ap.add_argument("--mode", choices=["paper", "live-candidate"], default="paper")
    args = ap.parse_args()
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot(args.label, args.mode)
    out = FREEZE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.label}.json"
    out.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    latest = FREEZE_DIR / "latest.json"
    latest.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    print(out)
    print("dynamic", snap["dynamic_params"])
    print("stats", snap["journal_stats_all_btc_paper"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
