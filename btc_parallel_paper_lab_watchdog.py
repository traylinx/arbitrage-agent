#!/usr/bin/env python3
"""Self-heal the isolated BTC parallel paper lab.

Launchd runs this every few minutes. It keeps paper-only strategy workers alive
for continuous fake-money data collection. It never starts live trading.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import btc_parallel_paper_lab as lab

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
LOG_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2" / "logs"
LOG_FILE = LOG_DIR / "btc_parallel_paper_lab_watchdog.log"
SRC_DIR = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("BTC_PAPER_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 2_000_000:
        LOG_FILE.replace(LOG_DIR / "btc_parallel_paper_lab_watchdog.log.1")
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def latest_rows() -> tuple[Path | None, list[dict]]:
    try:
        run_dir = lab.resolve_run_dir("latest")
        rows = [lab.summarize_strategy(e) for e in __import__("json").loads((run_dir / "manifest.json").read_text()).get("strategies", [])]
        return run_dir, rows
    except Exception as e:
        log(f"latest read failed: {type(e).__name__}: {e}")
        return None, []


def _run_dirs() -> list[Path]:
    if not lab.LAB_ROOT.exists():
        return []
    return sorted(
        [p for p in lab.LAB_ROOT.iterdir() if p.is_dir() and (p / "manifest.json").exists()],
        key=lambda p: p.stat().st_mtime,
    )


def rows_for(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return [lab.summarize_strategy(e) for e in manifest.get("strategies", [])]


def run_profiles(run_dir: Path | None) -> set[str]:
    if not run_dir:
        return set()
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except Exception:
        return set()
    profiles = {
        str(entry.get("profile") or "explore")
        for entry in manifest.get("strategies", [])
        if isinstance(entry, dict)
    }
    return profiles or {"explore"}


def is_model_gate_run(run_dir: Path | None) -> bool:
    return "model_gate" in run_profiles(run_dir)


def aggregate(rows: list[dict]) -> dict:
    trades = sum(int(r.get("trades") or 0) for r in rows)
    wins = sum(int(r.get("wins") or 0) for r in rows)
    losses = sum(int(r.get("losses") or 0) for r in rows)
    pnl = sum(float(r.get("pnl") or 0.0) for r in rows)
    live_valid_trades = sum(int(r.get("live_valid_trades") or 0) for r in rows)
    live_valid_wins = sum(int(r.get("live_valid_wins") or 0) for r in rows)
    live_valid_losses = sum(int(r.get("live_valid_losses") or 0) for r in rows)
    live_valid_pnl = sum(float(r.get("live_valid_pnl") or 0.0) for r in rows)
    alive = sum(1 for r in rows if r.get("alive"))
    active = sum(1 for r in rows if r.get("alive") and not r.get("halted"))
    halted = sum(1 for r in rows if r.get("halted"))
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "wr": wins / trades if trades else 0.0,
        "pnl": pnl,
        "live_valid_trades": live_valid_trades,
        "live_valid_wins": live_valid_wins,
        "live_valid_losses": live_valid_losses,
        "live_valid_wr": live_valid_wins / live_valid_trades if live_valid_trades else 0.0,
        "live_valid_pnl": live_valid_pnl,
        "alive": alive,
        "active": active,
        "halted": halted,
    }


def promote_best_active(min_alive: int, min_trades: int, min_pnl: float) -> Path | None:
    """Promote a healthy active run to LATEST instead of launching duplicates."""
    candidates: list[tuple[float, float, int, Path]] = []
    for run_dir in _run_dirs():
        try:
            rows = rows_for(run_dir)
            a = aggregate(rows)
        except Exception:
            continue
        if a["active"] < min_alive:
            continue
        if a["live_valid_trades"] < min_trades:
            continue
        if a["live_valid_pnl"] < min_pnl:
            continue
        candidates.append((a["live_valid_pnl"], a["live_valid_wr"], a["live_valid_trades"], run_dir))
    if not candidates:
        return None
    _, _, _, best = sorted(candidates)[-1]
    (lab.LAB_ROOT / "LATEST").write_text(str(best) + "\n")
    log(f"promoted healthy active lab: {best}")
    return best


def stop_run(run_dir: Path | None) -> None:
    if not run_dir:
        return
    try:
        log(f"stopping stale/halted lab: {run_dir}")
        lab.stop(run_dir)
    except Exception as e:
        log(f"stop stale lab failed: {type(e).__name__}: {e}")


def launch_new(args: argparse.Namespace) -> None:
    label = f"constant_fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        str(PYTHON),
        str(SRC_DIR / "btc_parallel_paper_lab.py"),
        "--n", str(args.target),
        "--duration", str(args.duration),
        "--capital", str(args.capital),
        "--loop-sleep", str(args.loop_sleep),
        "--market-check", str(args.market_check),
        "--stagger", str(args.stagger),
        "--label", label,
    ]
    out = LOG_DIR / f"btc_parallel_paper_lab_launch_{label}.log"
    log("launching new paper lab: " + " ".join(cmd))
    with out.open("ab", buffering=0) as f:
        subprocess.Popen(cmd, cwd=str(SRC_DIR), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=int(os.environ.get("BTC_PARALLEL_LAB_TARGET", "12")))
    ap.add_argument("--min-alive", type=int, default=int(os.environ.get("BTC_PARALLEL_LAB_MIN_ALIVE", "10")))
    ap.add_argument("--replace-halted", action="store_true", default=os.environ.get("BTC_PARALLEL_LAB_REPLACE_HALTED", "1") != "0")
    ap.add_argument("--duration", type=int, default=int(os.environ.get("BTC_PARALLEL_LAB_DURATION", str(8 * 3600))))
    ap.add_argument("--capital", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_CAPITAL", "20")))
    ap.add_argument("--loop-sleep", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_LOOP_SLEEP", "2")))
    ap.add_argument("--market-check", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_MARKET_CHECK", "5")))
    ap.add_argument("--stagger", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_STAGGER", "0.5")))
    ap.add_argument("--fail-min-trades", type=int, default=int(os.environ.get("BTC_PARALLEL_LAB_FAIL_MIN_TRADES", "12")))
    ap.add_argument("--fail-pnl", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_FAIL_PNL", "-15")))
    ap.add_argument("--fail-wr", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_FAIL_WR", "0.35")))
    ap.add_argument("--promote-min-trades", type=int, default=int(os.environ.get("BTC_PARALLEL_LAB_PROMOTE_MIN_TRADES", "40")))
    ap.add_argument("--promote-min-pnl", type=float, default=float(os.environ.get("BTC_PARALLEL_LAB_PROMOTE_MIN_PNL", "0")))
    args = ap.parse_args()

    run_dir, rows = latest_rows()
    a = aggregate(rows)
    log(
        f"status run={run_dir} alive={a['alive']}/{len(rows)} active={a['active']} "
        f"halted={a['halted']} trades={a['trades']} wr={a['wr']:.1%} pnl={a['pnl']:+.2f} "
        f"live_trades={a['live_valid_trades']} live_wr={a['live_valid_wr']:.1%} "
        f"live_pnl={a['live_valid_pnl']:+.2f}"
    )

    failed = (
        a["live_valid_trades"] >= args.fail_min_trades
        and (a["live_valid_pnl"] <= args.fail_pnl or a["live_valid_wr"] <= args.fail_wr)
    )
    model_gate = is_model_gate_run(run_dir)
    if failed:
        log(
            f"aggregate fail guard tripped: live_trades={a['live_valid_trades']} "
            f"live_wr={a['live_valid_wr']:.1%} live_pnl={a['live_valid_pnl']:+.2f}; "
            f"thresholds trades>={args.fail_min_trades} "
            f"wr<={args.fail_wr:.1%} pnl<={args.fail_pnl:+.2f}"
        )
        if model_gate:
            log("preserving model-gated canary after fail guard; no default exploration replacement")
            return 0
        if args.replace_halted:
            stop_run(run_dir)
        promoted = promote_best_active(args.min_alive, args.promote_min_trades, args.promote_min_pnl)
        if promoted:
            return 0
        launch_new(args)
        return 0

    # Curated labs may intentionally run fewer than the default target workers.
    # Do not kill a healthy 3-4 worker winner cohort just because global
    # min-alive is 10; scale the liveness requirement to manifest size.
    required_alive = min(args.min_alive, max(1, len(rows)))
    if a["active"] < required_alive:
        if model_gate:
            log(
                f"preserving model-gated canary despite halted workers: "
                f"alive={a['alive']}/{len(rows)} active={a['active']} halted={a['halted']} "
                f"trades={a['trades']} wr={a['wr']:.1%} pnl={a['pnl']:+.2f}"
            )
            return 0
        mature_profitable = (
            a["live_valid_trades"] >= args.promote_min_trades
            and a["live_valid_pnl"] >= args.promote_min_pnl
        )
        if mature_profitable:
            log(
                f"preserving under-active profitable lab: live_trades={a['live_valid_trades']} "
                f"live_wr={a['live_valid_wr']:.1%} live_pnl={a['live_valid_pnl']:+.2f}; "
                f"launching supplemental workers"
            )
            launch_new(args)
            return 0
        if args.replace_halted:
            stop_run(run_dir)
        promoted = promote_best_active(args.min_alive, args.promote_min_trades, args.promote_min_pnl)
        if not promoted:
            launch_new(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
