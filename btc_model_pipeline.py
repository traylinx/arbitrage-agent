#!/usr/bin/env python3
"""
BTC Model Pipeline — One-shot: collect → label → train → backtest.

Usage:
  # 1. Collect features for N hours (e.g., 24-48h minimum, 7 days preferred)
  python btc_feature_engine.py &

  # 2. After collection, run the full pipeline:
  python btc_model_pipeline.py

  # 3. Start paper trading with the trained model:
  python btc_edge_paper.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], label: str) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[pipeline] {label} exited with code {rc}")
    return rc


def main():
    # Ensure labels are computed on whatever feature data exists
    rc = run([sys.executable, str(SRC_DIR / "btc_feature_engine.py"), "--oneshot-labels"], "compute_labels")
    if rc != 0:
        print("[pipeline] Warning: label computation had issues")

    # Train models
    rc = run([sys.executable, str(SRC_DIR / "btc_model_trainer.py")], "train")
    if rc != 0:
        print("[pipeline] Training failed.")
        return

    # Backtest on holdout
    rc = run([sys.executable, str(SRC_DIR / "btc_backtest_model.py")], "backtest")
    if rc != 0:
        print("[pipeline] Backtest failed.")
        return

    print("\n=== Pipeline complete ===")
    print("Next step: review model_report.json, then run btc_edge_paper.py")


if __name__ == "__main__":
    main()
