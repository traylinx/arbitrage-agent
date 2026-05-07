#!/usr/bin/env python3
"""Curate BTC paper swarms using live-valid CLOB evidence only.

This is a paper-only self-improvement loop:
- kill alive workers that are halted or negative on strict CLOB executable fills
- keep positive/zero live-valid workers running for more samples
- launch a focused contrarian-band wave when active paper workers fall below target

Never starts live trading. Never keeps Gamma fallback PnL as promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import btc_parallel_paper_lab as lab

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
STATE_FILE = STATE_DIR / "btc_live_valid_swarm_curator_state.json"
LOG_FILE = LOG_DIR / "btc_live_valid_swarm_curator.log"
PYTHON = Path(os.environ.get("BTC_PAPER_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

RUN_NAME_TOKENS = (
    "v12_contra",
    "v11_focus",
    "v10_clob_highdelta",
    "v8_focus",
    "strict_min_poly_v7",
)

WAVE_SPECS = (
    ("v13_curated_contra_e010_d16_s070", "0.10", "16.0", "0.70"),
    ("v13_curated_contra_e012_d16_s070", "0.12", "16.0", "0.70"),
    ("v13_curated_contra_e010_d14_s070", "0.10", "14.0", "0.70"),
    ("v13_curated_contra_e012_d14_s075", "0.12", "14.0", "0.75"),
)


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 2_000_000:
            LOG_FILE.replace(LOG_DIR / "btc_live_valid_swarm_curator.log.1")
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE_FILE)


def run_dirs() -> list[Path]:
    if not lab.LAB_ROOT.exists():
        return []
    runs = [
        p
        for p in lab.LAB_ROOT.iterdir()
        if p.is_dir()
        and (p / "manifest.json").exists()
        and any(token in p.name for token in RUN_NAME_TOKENS)
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def rows_for(run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [lab.summarize_strategy(e) for e in manifest.get("strategies", [])]
    for row in rows:
        row["run_dir"] = str(run_dir)
        row["run_name"] = run_dir.name
    return rows


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs():
        try:
            rows.extend(rows_for(run_dir))
        except Exception as e:
            log(f"read failed run={run_dir}: {type(e).__name__}: {e}")
    return rows


def is_bad_alive(row: dict[str, Any]) -> bool:
    if not row.get("alive"):
        return False
    live_trades = int(row.get("live_valid_trades") or 0)
    live_pnl = float(row.get("live_valid_pnl") or 0.0)
    return bool(row.get("halted")) or (live_trades > 0 and live_pnl <= 0.0)


def is_active_good_or_sampling(row: dict[str, Any]) -> bool:
    if not row.get("alive") or row.get("halted"):
        return False
    live_trades = int(row.get("live_valid_trades") or 0)
    live_pnl = float(row.get("live_valid_pnl") or 0.0)
    return live_trades == 0 or live_pnl > 0.0


def terminate_worker(row: dict[str, Any], dry_run: bool) -> bool:
    pid = int(row.get("pid") or 0)
    if pid <= 0:
        return False
    msg = (
        f"kill run={row.get('run_name')} strategy={row.get('name')} pid={pid} "
        f"live={int(row.get('live_valid_trades') or 0)} "
        f"vPnL={float(row.get('live_valid_pnl') or 0):+.2f} halt={row.get('halted')}"
    )
    if dry_run:
        log("DRY " + msg)
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            log(f"kill failed pid={pid}: {type(e).__name__}: {e}")
            return False
    log(msg)
    return True


def launch_wave(dry_run: bool) -> list[Path]:
    model = DATA_DIR / "model" / "btc_prob_model_candidate_best.pkl"
    only = "firehose_gamma_probe,aggressive_clob_floor"
    base_env = os.environ.copy()
    base_env.update(
        {
            "POLYMARKET_LIVE_TRADING": "0",
            "BTC_PAPER_ONLY": "1",
            "BTC_FAST_GA_ENABLED": "0",
            "BTC_LAB_MAX_OPEN_BTC_TRADES": "1",
            "BTC_LAB_MAX_OPEN_BTC_TRADES_PER_TF": "1",
            "BTC_LAB_MAX_TOTAL_BTC_EXPOSURE_PCT": "0.16",
            "BTC_LAB_ALLOW_CROSS_TF_CORRELATED_OPEN": "0",
            "BTC_LAB_PROB_MIN_POLY_PRICE": "0.40",
            "BTC_LAB_PROB_HARD_POLY_CAP": "0.56",
            "BTC_LAB_PROB_POLY_PRICE_CEILING": "0.55",
            "BTC_LAB_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE": "0",
            "BTC_LAB_CLOB_GAMMA_GAP_OVERRIDE_PROB": "0.95",
            "BTC_LAB_REQUIRE_CLOB_QUOTE": "1",
            "BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT": "0",
            "BTC_LAB_EXTERNAL_BULL_SCORE_MIN": "-0.20",
            "BTC_LAB_EXTERNAL_BULL_SCORE_MAX": "-0.05",
        }
    )
    launched: list[Path] = []
    for label, edge, delta, shrink in WAVE_SPECS:
        cmd = [
            str(PYTHON),
            str(SRC_DIR / "btc_parallel_paper_lab.py"),
            "--label",
            label,
            "--only",
            only,
            "--profile",
            "model_gate",
            "--model-path",
            str(model),
            "--model-edge",
            edge,
            "--model-delta",
            delta,
            "--prob-shrink",
            shrink,
            "--duration",
            "21600",
            "--capital",
            "20",
            "--loop-sleep",
            "2",
            "--market-check",
            "5",
            "--stagger",
            "0.2",
        ]
        log("launch " + " ".join(cmd))
        if dry_run:
            continue
        res = subprocess.run(
            cmd,
            cwd=str(SRC_DIR),
            env=base_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        if res.returncode != 0:
            log(f"launch failed label={label} rc={res.returncode} out={res.stdout[-400:]}")
            continue
        first = (res.stdout.splitlines() or [""])[0].strip()
        if first:
            launched.append(Path(first))
            log(f"launched {first}")
        time.sleep(1)
    return launched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-active", type=int, default=int(os.environ.get("BTC_LIVE_VALID_CURATOR_TARGET_ACTIVE", "16")))
    ap.add_argument("--cooldown-seconds", type=int, default=int(os.environ.get("BTC_LIVE_VALID_CURATOR_COOLDOWN", "900")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = collect_rows()
    bad_rows = [r for r in rows if is_bad_alive(r)]
    active_rows = [r for r in rows if is_active_good_or_sampling(r)]
    for row in bad_rows:
        terminate_worker(row, args.dry_run)

    # Recompute active count after cull intent.
    active_after = max(0, len(active_rows) - len([r for r in bad_rows if r in active_rows]))
    state = read_state()
    now = int(time.time())
    last_launch = int(state.get("last_launch_ts") or 0)
    should_launch = active_after < args.target_active and (now - last_launch) >= args.cooldown_seconds
    log(
        f"summary rows={len(rows)} active={len(active_rows)} bad={len(bad_rows)} "
        f"active_after={active_after} target={args.target_active} "
        f"cooldown_left={max(0, args.cooldown_seconds - (now - last_launch))}"
    )
    if should_launch:
        launched = launch_wave(args.dry_run)
        state.update(
            {
                "last_launch_ts": now,
                "last_launch_at": datetime.now().isoformat(),
                "last_launch_count": len(launched),
                "last_launch_dirs": [str(p) for p in launched],
            }
        )
        if not args.dry_run:
            write_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
