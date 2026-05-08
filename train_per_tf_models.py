#!/usr/local/opt/python@3.11/bin/python3.11
"""Train one probability model per timeframe.

Reads the unified train_dataset.jsonl (all tfs mixed) and writes:
  - btc_prob_model_5m.pkl   (filter: features.tf_minutes == 5.0)
  - btc_prob_model_15m.pkl  (filter: features.tf_minutes == 15.0)

Each model is trained against ONLY its timeframe's rows, so 5m and 15m
patterns no longer compete for the same parameter space.

Sebastian's note 2026-05-07:
  > Probability model is still shared. That limitation is real.
  > Yes: split probability model per timeframe. That's the next bottleneck.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
MODEL_DIR = DATA_DIR / "model"
MASTER_DATASET = MODEL_DIR / "train_dataset.jsonl"

sys.path.insert(0, str(SRC_DIR))


def filter_dataset(master: Path, out: Path, tf_minutes: float) -> int:
    rows_in = rows_out = 0
    with master.open() as fi, out.open("w") as fo:
        for line in fi:
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows_in += 1
            feats = d.get("features", {}) or {}
            row_tf = feats.get("tf_minutes")
            if row_tf is None:
                row_tf = (d.get("meta", {}) or {}).get("window_tf")
            try:
                if abs(float(row_tf) - tf_minutes) > 1e-6:
                    continue
            except Exception:
                continue
            fo.write(line if line.endswith("\n") else line + "\n")
            rows_out += 1
    print(f"[filter] tf={tf_minutes}m: {rows_out}/{rows_in} rows kept → {out.name}")
    return rows_out


def train_for_tf(tf: int) -> None:
    out_dataset = MODEL_DIR / f"train_dataset_{tf}m.jsonl"
    out_model = MODEL_DIR / f"btc_prob_model_{tf}m.pkl"
    out_meta = MODEL_DIR / f"btc_prob_model_{tf}m_meta.json"
    out_base = MODEL_DIR / f"btc_prob_model_{tf}m"

    n = filter_dataset(MASTER_DATASET, out_dataset, float(tf))
    if n < 50:
        print(f"[skip] tf={tf}m has only {n} rows; need ≥50 to train. Skipping.")
        return

    # Monkey-patch the trainer's module-level paths so train_model() writes
    # into the per-tf slots without us having to fork the script.
    import btc_probability_model as M

    original = (
        M.DATASET_FILE,
        M.MODEL_PKL,
        M.MODEL_META,
        M.MODEL_OUT,
    )
    try:
        M.DATASET_FILE = out_dataset
        M.MODEL_PKL = out_model
        M.MODEL_META = out_meta
        M.MODEL_OUT = out_base
        # Lower min_samples for the smaller per-tf dataset
        # (15m has ~480 rows, 5m has ~1086; default min was 100).
        # Keep min at 50 to allow training on small slices.
        # We patch the default by wrapping load_dataset.
        original_load = M.load_dataset
        M.load_dataset = lambda min_samples=50: original_load(min_samples=min_samples)
        try:
            print(f"\n{'='*60}\nTRAINING tf={tf}m → {out_model.name}\n{'='*60}")
            M.train_model()
            print(f"[done] tf={tf}m model saved to {out_model}")
        finally:
            M.load_dataset = original_load
    finally:
        M.DATASET_FILE, M.MODEL_PKL, M.MODEL_META, M.MODEL_OUT = original


def main() -> int:
    if not MASTER_DATASET.exists():
        print(f"FATAL: master dataset missing at {MASTER_DATASET}. Run btc_prob_dataset.py first.")
        return 1
    for tf in (5, 15):
        train_for_tf(tf)
    print("\n[summary] per-tf models written:")
    for tf in (5, 15):
        p = MODEL_DIR / f"btc_prob_model_{tf}m.pkl"
        if p.exists():
            print(f"  ✅ {p.name}  ({p.stat().st_size} bytes)")
        else:
            print(f"  ❌ {p.name}  (missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
