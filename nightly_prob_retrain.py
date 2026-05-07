#!/usr/bin/env python3
"""
Phase 4: Nightly Probability Model Retrain
==========================================
Runs every night at ~02:00 UTC via launchd/cron.
  1. Build/append training dataset from new journal entries
  2. Retrain model on full dataset
  3. Run backtest on last 7 days
  4. If AUC improved AND 7d expected WR > 55% → promote new model
  5. Else keep current model and log why

Usage:
  python3 nightly_prob_retrain.py [--dry-run]
  # Add to crontab:
  0 2 * * * cd /path/to/src && python3 nightly_prob_retrain.py >> logs/nightly_retrain.log 2>&1
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR    = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
MODEL_DIR   = DATA_DIR / "model"
JOURNAL_FILE = DATA_DIR / "state" / "intraday_journal.jsonl"
MODEL_PKL    = MODEL_DIR / "btc_prob_model_current.pkl"
MODEL_META   = MODEL_DIR / "btc_prob_model_current_meta.json"
REPORT_FILE  = MODEL_DIR / "backtest_report.jsonl"
LOG_FILE     = DATA_DIR / "logs" / "nightly_prob_retrain.log"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Load previous model meta ─────────────────────────────────────────────────
def load_previous_meta() -> dict:
    if not MODEL_META.exists():
        return {}
    with MODEL_META.open() as f:
        return json.load(f)


def load_previous_journal_trades() -> set:
    """Return set of (window_start, market_id) already in dataset."""
    rows = []
    ds_file = MODEL_DIR / "train_dataset.jsonl"
    if ds_file.exists():
        with ds_file.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    continue
    return {(r["meta"]["window_start"], r["meta"]["market_id"]) for r in rows if "meta" in r}


# ── Check if new journal entries exist ───────────────────────────────────────
def count_unprocessed_journal_entries() -> int:
    processed = load_previous_journal_trades()
    count = 0
    if not JOURNAL_FILE.exists():
        return 0
    with JOURNAL_FILE.open() as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("mode") != "paper" or t.get("won") is None:
                    continue
                key = (t.get("window_start"), t.get("market_id"))
                if key not in processed:
                    count += 1
            except:
                continue
    return count


# ── Retrain pipeline ─────────────────────────────────────────────────────────
def retrain(dry_run: bool = False):
    log("=== NIGHTLY PROB RETRAIN START ===")

    # Count new entries
    new_count = count_unprocessed_journal_entries()
    log(f"New journal entries: {new_count}")

    if new_count == 0 and not dry_run:
        prev_meta = load_previous_meta()
        log(f"No new entries. Current model: AUC={prev_meta.get('test_auc','?')} "
            f"WR={prev_meta.get('test_wr','?')}")
        log("Nothing to retrain.")
        return

    # Step 1: Build dataset
    log("Step 1: Building dataset...")
    sys.path.insert(0, str(Path(__file__).parent))
    from btc_prob_dataset import build_dataset
    rows = build_dataset(append=True)
    log(f"Dataset size: {len(rows)} rows")

    if len(rows) < 100:
        log("Dataset too small (< 100 rows). Skipping retrain.")
        return

    # Step 2: Train model
    log("Step 2: Training model...")
    from btc_probability_model import train_model
    new_meta = train_model()

    new_auc = new_meta.get("test_auc", 0)
    new_wr  = new_meta.get("test_wr", 0)
    new_cal = new_meta.get("test_cal_err", 1)

    log(f"New model: AUC={new_auc:.3f}, WR={new_wr:.1%}, CalErr={new_cal:.3f}")

    # Step 3: Backtest
    log("Step 3: Running backtest...")
    from btc_prob_backtest import backtest
    report = backtest(edge_threshold=new_meta.get("best_threshold", 0.05))

    # Step 4: Promotion decision
    prev_meta = load_previous_meta()
    prev_auc  = prev_meta.get("test_auc", 0)

    should_promote = (
        new_auc > prev_auc + 0.005  # AUC improved by at least 0.5%
        and new_wr >= 0.55
        and new_cal < 0.10
    )

    if dry_run:
        log(f"[DRY RUN] Would {'PROMOTE' if should_promote else 'SKIP (no improvement)'}")
        return

    if should_promote:
        log(f"✓ PROMOTING new model: AUC {prev_auc:.3f} → {new_auc:.3f}, WR={new_wr:.1%}")
        # Update meta
        with MODEL_META.open("w") as f:
            json.dump(new_meta, f, indent=2)
        # Save versioned copy
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        versioned_pkl = MODEL_DIR / f"btc_prob_model_v{version}.pkl"
        if MODEL_PKL.exists():
            import shutil
            shutil.copy(MODEL_PKL, versioned_pkl)
        log(f"Saved versioned model: {versioned_pkl.name}")
    else:
        reason = []
        if new_auc <= prev_auc + 0.005:
            reason.append(f"AUC not improved ({new_auc:.3f} vs prev {prev_auc:.3f})")
        if new_wr < 0.55:
            reason.append(f"WR={new_wr:.1%} < 55%")
        if new_cal >= 0.10:
            reason.append(f"CalErr={new_cal:.3f} >= 0.10")
        log(f"✗ NOT PROMOTING: {'; '.join(reason)}")
        log(f"  Keeping previous model.")

    log("=== NIGHTLY RETRAIN DONE ===\n")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Run without saving")
    args = p.parse_args()
    retrain(dry_run=args.dry_run)
