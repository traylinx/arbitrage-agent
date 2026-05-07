#!/usr/local/opt/python@3.11/bin/python3.11
"""
Auto-Improve v3 — Unified improvement loop.

1. Runs btc_backtest_autoresearch.py to analyze legacy sniper params against real journal trades
2. Logs results to Brain journal
3. Runs every 30 minutes via launchd

No GBM fantasy. No broken AI auth. Just journal replay.
Analysis-only by default. Runtime promotion is owned by the model-gated
new-data loop, not this legacy replay path.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from btc_param_contract import DYNAMIC_PARAM_FIELDS, PAPER_BOUNDS, dynamic_fingerprint, read_params_file

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
LEGACY_PARAMS_FILE = STATE_DIR / "legacy_sniper_best_params.json"
BRAIN_JOURNAL = HARVEY_HOME / "data" / "Brain" / "journals" / (datetime.now().strftime("%Y_%m_%d") + ".md")

LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR / "autoimprove_v3.log", "a") as f:
        f.write(line + "\n")


def brain_log(msg: str) -> None:
    """Append to today's Brain journal."""
    BRAIN_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(BRAIN_JOURNAL, "a") as f:
        f.write(f"- {msg}\n")


def count_journal_trades() -> tuple[int, int]:
    if not JOURNAL_FILE.exists():
        return 0, 0
    total = 0
    clob = 0
    with open(JOURNAL_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                if row.get("price_source") == "clob_book_ask_depth":
                    clob += 1
            except Exception:
                pass
    return total, clob


def load_best_sniper_params() -> dict:
    f = STATE_DIR / "sniper_best_params.json"
    if f.exists():
        return read_params_file(f, mode="paper")
    return {}


def load_legacy_analysis_params() -> dict:
    if LEGACY_PARAMS_FILE.exists():
        return read_params_file(LEGACY_PARAMS_FILE, mode="paper")
    return {}


def run_backtest() -> dict:
    """Run btc_backtest_autoresearch and return legacy analysis params."""
    script = Path(__file__).parent / "btc_backtest_autoresearch.py"
    log(f"Running backtest: {script}")
    env = os.environ.copy()
    env.setdefault("ITERATIONS", "150")
    env.setdefault("BOOT_ITERS", "500")
    env["BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS"] = "0"
    env["BTC_LEGACY_BEST_PARAMS_FILE"] = str(LEGACY_PARAMS_FILE)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        log(f"Backtest exit code: {result.returncode}")
        if result.stdout:
            # Log last 10 lines
            for line in result.stdout.strip().split("\n")[-10:]:
                log(f"  > {line}")
    except Exception as e:
        log(f"Backtest failed: {e}")
        return {}

    # Read the generated legacy analysis params. Do not read runtime params here:
    # this loop must not mistake model-gated production config for its output.
    return load_legacy_analysis_params()


def main():
    log("=" * 60)
    log("AUTO-IMPROVE v3 START")
    log("=" * 60)

    n_trades, n_clob = count_journal_trades()
    log(f"Journal trades: {n_trades} total; {n_clob} CLOB-realistic")

    if n_clob < 20:
        log("Too few CLOB-realistic trades for backtest (< 20). Skipping.")
        return

    old_params = load_best_sniper_params()
    old_delta = old_params.get("delta_thresh", "?")
    old_conf = old_params.get("conf_thresh", "?")

    new_params = run_backtest()

    if not new_params:
        log("No params produced. Aborting.")
        return

    new_delta = new_params.get("delta_thresh", "?")
    new_conf = new_params.get("conf_thresh", "?")
    new_score = new_params.get("best_score", "?")

    log(f"Old params: delta={old_delta} conf={old_conf}")
    log(f"Legacy analysis params: delta={new_delta} conf={new_conf} score={new_score}")

    # Final paper-training contract gate. Live-readiness is stricter elsewhere.
    # This loop is paper-only and may deploy only whitelisted dynamic thresholds.
    conf_floor = PAPER_BOUNDS["conf_thresh"][0]
    delta_floor = PAPER_BOUNDS["delta_thresh"][0]
    ens_floor = PAPER_BOUNDS["ens_thresh"][0]
    if float(new_conf) < conf_floor:
        log(f"CONTRACT FAIL: conf_thresh={new_conf} < paper floor={conf_floor}. NOT deploying.")
        return
    if float(new_params.get("delta_thresh", 999)) < delta_floor:
        log(f"CONTRACT FAIL: delta_thresh below paper floor={delta_floor}. NOT deploying.")
        return
    if float(new_params.get("ens_thresh", 999)) < ens_floor:
        log(f"CONTRACT FAIL: ens_thresh below paper floor={ens_floor}. NOT deploying.")
        return

    old_fp = dynamic_fingerprint(old_params) if old_params else None
    new_fp = dynamic_fingerprint(new_params)
    if new_fp != old_fp:
        log(
            "ANALYSIS ONLY: legacy replay suggests dynamic="
            f"{dict(zip(DYNAMIC_PARAM_FIELDS, new_fp))}; runtime promotion stays model-gated."
        )
        brain_log(
            f"[Auto-Improve v3] Legacy replay analysis only: delta={new_delta} conf={new_conf} "
            f"score={new_score} (from {n_clob} CLOB-realistic / {n_trades} total journal trades). "
            "Runtime promotion remains owned by model-gated new-data loop."
        )
    else:
        log("No dynamic-param change. Keeping current params.")

    log("DONE")


if __name__ == "__main__":
    main()
