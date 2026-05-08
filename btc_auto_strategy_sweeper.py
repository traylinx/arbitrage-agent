#!/usr/bin/env python3
"""Continuous fake-money strategy/parameter sweeper for BTC paper labs.

Purpose:
  - Keep testing different strategy variables while live trading is locked.
  - Launch only isolated btc_paper_fast.py PAPER workers via btc_parallel_paper_lab.
  - Rotate parameter grids + coarse regime filters so autoimprove_v3 has fresh
    out-of-sample evidence instead of overfitting one lucky window.

This script never starts btc_sniper_live.py and inherits the lab launcher safety
guards that strip live credentials from worker environments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import btc_parallel_paper_lab as lab
from btc_parallel_paper_lab import StrategySpec


HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
STATE_FILE = STATE_DIR / "auto_strategy_sweeper_state.json"
LOG_FILE = LOG_DIR / "btc_auto_strategy_sweeper.log"
LIVE_KILL_SWITCH = STATE_DIR / "live_trading_disabled.json"


BASES: list[dict[str, Any]] = [
    # Proven but regressing. Keep testing neighborhoods, not exact clones.
    dict(name="firehose_clob_quality", delta=9.0, conf=0.50, ens=0.35, spend=0.14, max_bet=0.18, dd=0.16, clob=True),
    dict(name="aggressive_clob_floor", delta=8.0, conf=0.45, ens=0.30, spend=0.16, max_bet=0.20, dd=0.20, clob=True),
    dict(name="firehose_gamma_probe", delta=8.0, conf=0.45, ens=0.30, spend=0.18, max_bet=0.20, dd=0.20, clob=False),
    # Conservative anchors.
    dict(name="balanced_contract", delta=11.8, conf=0.45, ens=0.50, spend=0.15, max_bet=0.20, dd=0.20, clob=True),
    dict(name="quality_mid", delta=15.0, conf=0.62, ens=0.62, spend=0.12, max_bet=0.18, dd=0.16, clob=True),
    dict(name="confidence_hunter", delta=10.0, conf=0.82, ens=0.40, spend=0.10, max_bet=0.15, dd=0.12, clob=True),
]

REGIME_FILTERS: list[dict[str, str]] = [
    # No filter: benchmark generalization.
    {},
    # Bearish external/orderflow score only. Tests whether short-window edge is
    # regime-conditional instead of universally valid.
    {
        "BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT": "0",
        "BTC_LAB_EXTERNAL_BULL_SCORE_MAX": "-0.10",
    },
    # Bullish external/orderflow score only.
    {
        "BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT": "0",
        "BTC_LAB_EXTERNAL_BULL_SCORE_MIN": "0.10",
    },
    # Chop/neutral only.
    {
        "BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT": "0",
        "BTC_LAB_EXTERNAL_BULL_SCORE_MIN": "-0.10",
        "BTC_LAB_EXTERNAL_BULL_SCORE_MAX": "0.10",
    },
]


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"generation": 0, "launched": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"generation": 0, "launched": []}


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def active_lab_workers() -> int:
    if not lab.LAB_ROOT.exists():
        return 0
    count = 0
    for manifest_path in lab.LAB_ROOT.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for entry in manifest.get("strategies", []):
            try:
                if lab._pid_alive(int(entry.get("pid") or 0)):
                    count += 1
            except Exception:
                pass
    return count


def _kill_worker(pid: int, reason: str, name: str) -> bool:
    if not pid or not lab._pid_alive(pid):
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return False
    log(f"pruned worker pid={pid} name={name} reason={reason}")
    return True


def _manifest_entries() -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    if not lab.LAB_ROOT.exists():
        return entries
    for manifest_path in lab.LAB_ROOT.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for entry in manifest.get("strategies", []):
            entries.append((manifest_path, entry))
    return entries


def prune_bad_workers(args: argparse.Namespace) -> int:
    """Stop fake-money workers that already proved bad enough.

    This is exploration hygiene, not risk management. A bad paper worker that
    keeps running blocks new variable tests. Kill it, keep the labels, launch a
    different variant next sweep.
    """
    if getattr(args, "no_prune", False):
        return 0
    killed = 0
    now = time.time()
    for manifest_path, entry in _manifest_entries():
        pid = int(entry.get("pid") or 0)
        if not lab._pid_alive(pid):
            continue
        try:
            summary = lab.summarize_strategy(entry)
        except Exception as exc:
            log(f"prune skip: cannot summarize pid={pid} manifest={manifest_path}: {exc}")
            continue
        name = str(summary.get("name") or entry.get("name") or "unknown")
        trades = int(summary.get("live_valid_trades") or summary.get("trades") or 0)
        wins = int(summary.get("live_valid_wins") or summary.get("wins") or 0)
        pnl = float(summary.get("live_valid_pnl") if summary.get("live_valid_trades") is not None else summary.get("pnl") or 0.0)
        wr = wins / trades if trades else 0.0
        halted = bool(summary.get("halted"))
        log_file = Path(str(entry.get("log_file") or ""))
        idle_minutes = 0.0
        if log_file.exists():
            idle_minutes = max(0.0, (now - log_file.stat().st_mtime) / 60.0)

        reason = ""
        if halted and trades >= 1:
            reason = f"halted trades={trades} WR={wr:.1%} PnL=${pnl:+.2f}"
        elif trades >= args.prune_min_trades and wr < args.prune_min_wr:
            reason = f"low_WR trades={trades} WR={wr:.1%} < {args.prune_min_wr:.1%} PnL=${pnl:+.2f}"
        elif trades >= args.prune_min_trades and pnl < args.prune_min_pnl:
            reason = f"low_PnL trades={trades} WR={wr:.1%} PnL=${pnl:+.2f} < ${args.prune_min_pnl:+.2f}"
        elif trades == 0 and idle_minutes >= args.prune_idle_minutes:
            reason = f"idle_no_trades idle={idle_minutes:.0f}m"

        if reason and _kill_worker(pid, reason, name):
            killed += 1
    if killed:
        # Give SIGTERM a short window so active count reflects pruning before
        # launch/refill decision.
        time.sleep(1.5)
    return killed


def previous_rank_bias() -> list[str]:
    """Return recent winners by name. Used only to weight generation, not gate."""
    rows: dict[str, dict[str, float]] = {}
    cutoff = time.time() - 4 * 3600
    for manifest_path in lab.LAB_ROOT.glob("*/manifest.json"):
        try:
            if manifest_path.stat().st_mtime < cutoff:
                continue
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for entry in manifest.get("strategies", []):
            name = str(entry.get("name") or "unknown")
            journal = Path(entry.get("journal_file", ""))
            if not journal.exists():
                continue
            bucket = rows.setdefault(name, {"n": 0, "w": 0, "pnl": 0.0})
            for line in journal.read_text().splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("price_source") != "clob_book_ask_depth" or "pnl" not in e:
                    continue
                bucket["n"] += 1
                bucket["w"] += 1 if e.get("won") is True or float(e.get("pnl") or 0) > 0 else 0
                bucket["pnl"] += float(e.get("pnl") or 0)
    ranked = []
    for name, r in rows.items():
        n = r["n"]
        if n < 8:
            continue
        wr = r["w"] / n
        score = r["pnl"] + wr * 5.0 + math.log1p(n)
        ranked.append((score, name))
    return [name for _, name in sorted(ranked, reverse=True)[:3]]


def make_specs(generation: int, n: int) -> list[StrategySpec]:
    rng = random.Random(1778160000 + generation)
    winners = set(previous_rank_bias())
    bases = list(BASES)
    bases.sort(key=lambda b: (0 if b["name"] in winners else 1, b["name"]))

    specs: list[StrategySpec] = []
    # Deterministic rotating grid around each base. Enough variance to learn;
    # bounded enough to avoid pure random lottery-tail firehosing.
    delta_jitters = [-2.0, 0.0, 2.0, 4.0]
    conf_jitters = [-0.05, 0.0, 0.08, 0.15]
    ens_jitters = [-0.08, 0.0, 0.12, 0.25]
    spend_mults = [0.75, 1.0, 1.2]
    for i in range(max(n * 2, 24)):
        base = bases[(i + generation) % len(bases)]
        dj = delta_jitters[(generation + i) % len(delta_jitters)]
        cj = conf_jitters[(generation * 2 + i) % len(conf_jitters)]
        ej = ens_jitters[(generation * 3 + i) % len(ens_jitters)]
        sm = spend_mults[(generation + i * 2) % len(spend_mults)]
        # Add small deterministic noise so repeated generations do not clone.
        delta = max(6.0, min(24.0, base["delta"] + dj + rng.uniform(-0.6, 0.6)))
        conf = max(0.35, min(0.88, base["conf"] + cj + rng.uniform(-0.015, 0.015)))
        ens = max(0.25, min(0.88, base["ens"] + ej + rng.uniform(-0.02, 0.02)))
        spend = max(0.08, min(0.20, base["spend"] * sm))
        max_bet = max(0.12, min(0.24, base["max_bet"] * sm))
        # Alternate CLOB requirement for gamma probe only; keep others executable.
        require_clob = bool(base["clob"])
        if base["name"] == "firehose_gamma_probe" and (generation + i) % 3 == 0:
            require_clob = True
        name = (
            f"sweep_g{generation:03d}_{i+1:02d}_"
            f"{base['name']}_d{delta:.1f}_c{conf:.2f}_e{ens:.2f}"
        )
        specs.append(
            StrategySpec(
                name=name,
                delta_thresh=round(delta, 3),
                conf_thresh=round(conf, 3),
                ens_thresh=round(ens, 3),
                spend_ratio=round(spend, 3),
                max_bet_pct=round(max_bet, 3),
                risk_halt_drawdown_pct=float(base["dd"]),
                max_open=1,
                cohort=f"sweep:{base['name']}",
                require_clob_quote=require_clob,
                loop_sleep=2.0,
                market_check=5.0,
                status_seconds=15.0,
            )
        )
        if len(specs) >= n:
            break
    return specs


def launch_generation(args: argparse.Namespace) -> Path | None:
    pruned = prune_bad_workers(args)
    if pruned:
        log(f"pruned {pruned} bad fake-money workers before refill check")
    active = active_lab_workers()
    if active >= args.max_active_lab:
        log(f"skip launch: active_lab_workers={active} >= max_active_lab={args.max_active_lab}")
        return None

    state = read_state()
    generation = int(state.get("generation") or 0) + 1
    specs = make_specs(generation, args.n)
    regime = REGIME_FILTERS[generation % len(REGIME_FILTERS)]

    # Keep sweeps fake-money and regime-filtered. Environment applies only to
    # this launch process and is mapped by btc_parallel_paper_lab into workers.
    old_env = {k: os.environ.get(k) for k in regime}
    try:
        for k, v in regime.items():
            os.environ[k] = v
        os.environ.setdefault("BTC_LAB_EXPLORATION_MIN_EDGE", "0.04")
        os.environ.setdefault("BTC_LAB_EXPLORATION_MIN_CONF", "0.35")
        os.environ.setdefault("BTC_LAB_EXPLORATION_MIN_POLY", "0.20")
        os.environ.setdefault("BTC_LAB_EXPLORATION_MAX_POLY", "0.82")
        os.environ.setdefault("BTC_LAB_EXPLORATION_MAX_SPEND", "1.50")
        os.environ.setdefault("BTC_LAB_EXPLORATION_DELTA_FLOOR", "0.50")
        os.environ.setdefault("BTC_LAB_EXPLORATION_MAX_SLIPPAGE_BPS", "750")
        os.environ.setdefault("BTC_LAB_MAX_OPEN_BTC_TRADES", "3")
        os.environ.setdefault("BTC_LAB_MAX_OPEN_BTC_TRADES_PER_TF", "2")
        os.environ.setdefault("BTC_LAB_ALLOW_CROSS_TF_CORRELATED_OPEN", "1")
        os.environ.setdefault("BTC_LAB_MAX_TOTAL_BTC_EXPOSURE_PCT", "0.60")
        os.environ.setdefault("BTC_LAB_REQUIRE_CLOB_QUOTE", "1")

        lab.STRATEGIES = specs
        ns = Namespace(
            label=f"auto_sweep_g{generation:03d}",
            n=len(specs),
            only="",
            profile=args.profile,
            model_path=str(DATA_DIR / "model" / "btc_prob_model_current.pkl"),
            model_edge=args.model_edge,
            model_delta=args.model_delta,
            prob_shrink=args.prob_shrink,
            duration=args.duration,
            capital=args.capital,
            loop_sleep=args.loop_sleep,
            market_check=args.market_check,
            stagger=args.stagger,
        )
        run_dir = lab.launch(ns)
    finally:
        for k, old in old_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

    state["generation"] = generation
    state.setdefault("launched", []).append(
        {
            "generation": generation,
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            "n": len(specs),
            "profile": args.profile,
            "regime_filter": regime,
            "live_kill_switch_active": LIVE_KILL_SWITCH.exists(),
        }
    )
    state["launched"] = state["launched"][-50:]
    write_state(state)
    log(
        f"launched generation={generation} workers={len(specs)} profile={args.profile} "
        f"run_dir={run_dir} regime_filter={regime or 'none'} live_kill_switch_active={LIVE_KILL_SWITCH.exists()}"
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-launch fake-money BTC strategy/param sweeps")
    p.add_argument("--n", type=int, default=12, help="workers per generation")
    p.add_argument("--max-active-lab", type=int, default=18, help="skip launch when this many lab workers are alive")
    p.add_argument("--duration", type=int, default=2 * 3600)
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--loop-sleep", type=float, default=2.0)
    p.add_argument("--market-check", type=float, default=5.0)
    p.add_argument("--stagger", type=float, default=0.2)
    p.add_argument("--profile", choices=("explore", "model_gate"), default="explore")
    p.add_argument("--model-edge", type=float, default=0.02)
    p.add_argument("--model-delta", type=float, default=1.0)
    p.add_argument("--prob-shrink", type=float, default=0.65)
    p.add_argument("--no-prune", action="store_true", help="do not stop bad/halted paper workers before refill")
    p.add_argument("--prune-min-trades", type=int, default=8, help="minimum CLOB-valid trades before low-WR/low-PnL pruning")
    p.add_argument("--prune-min-wr", type=float, default=0.52, help="prune workers below this WR after --prune-min-trades")
    p.add_argument("--prune-min-pnl", type=float, default=-2.0, help="prune workers below this PnL after --prune-min-trades")
    p.add_argument("--prune-idle-minutes", type=float, default=90.0, help="prune zero-trade workers idle this long")
    p.add_argument("--status", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        state = read_state()
        print(
            json.dumps(
                {
                    "active_lab_workers": active_lab_workers(),
                    "state": state,
                    "live_kill_switch_active": LIVE_KILL_SWITCH.exists(),
                    "log_file": str(LOG_FILE),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    launch_generation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
