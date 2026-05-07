#!/usr/bin/env python3
"""Parallel BTC paper autoresearch supervisor.

Runs multiple isolated btc_backtest_autoresearch.py workers against the same
real BTC paper journal, then commits only the best normalized candidate.

Paper-only. No Polymarket orders. No live wallet access.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from btc_param_contract import DYNAMIC_PARAM_FIELDS, STRATEGY_VERSION, params_file_lock, read_params_file, write_params_file

import btc_backtest_autoresearch as bt


HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
BEST_PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
LOG_FILE = LOG_DIR / "btc_parallel_autoresearch.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 2_000_000:
        LOG_FILE.replace(LOG_DIR / "btc_parallel_autoresearch.log.1")
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def freeze_after_commit(label: str) -> None:
    try:
        res = subprocess.run(
            [sys.executable, str(SRC_DIR / "btc_freeze_strategy.py"), "--label", label, "--mode", "paper"],
            cwd=str(SRC_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        log(f"freeze-after-commit rc={res.returncode} out={res.stdout.strip()[-240:]} err={res.stderr.strip()[-240:]}")
    except Exception as e:
        log(f"freeze-after-commit error: {type(e).__name__}: {e}")


def dynamic(params: dict[str, Any]) -> dict[str, float]:
    return {k: float(params[k]) for k in DYNAMIC_PARAM_FIELDS}


def load_current() -> dict[str, Any]:
    return read_params_file(BEST_PARAMS_FILE, mode="paper")


def normalized_result(params: dict[str, Any], trades: list[dict[str, Any]]) -> dict[str, Any]:
    result = bt.backtest_score(trades, params)
    return {
        "params": dynamic(params),
        "score": float(result["score"]),
        "metrics": {
            "n": int(result.get("n", 0)),
            "wins": int(result.get("wins", 0)),
            "losses": int(result.get("losses", 0)),
            "wr": float(result.get("wr", 0.0)),
            "pnl": float(result.get("pnl", 0.0)),
        },
    }


def placed_sort_key(trade: dict[str, Any]) -> str:
    return str(trade.get("placed_at") or "")


def split_train_holdout(trades: list[dict[str, Any]], holdout_pct: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(trades, key=placed_sort_key)
    if len(ordered) < 50 or holdout_pct <= 0:
        return ordered, []
    holdout_n = max(10, int(round(len(ordered) * holdout_pct)))
    holdout_n = min(holdout_n, len(ordered) - 20)
    return ordered[:-holdout_n], ordered[-holdout_n:]


def write_journal(path: Path, trades: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for trade in trades:
            f.write(json.dumps(trade, sort_keys=True) + "\n")


def holdout_pass(candidate: dict[str, Any], baseline: dict[str, Any], min_trades: int) -> tuple[bool, str]:
    cm = candidate["metrics"]
    bm = baseline["metrics"]
    if cm["n"] < min_trades:
        return False, f"holdout_n {cm['n']} < {min_trades}"
    if cm["pnl"] < 0:
        return False, f"holdout_pnl {cm['pnl']:.2f} < 0"
    wr_floor = max(0.55, bm["wr"] - 0.02)
    if cm["wr"] < wr_floor:
        return False, f"holdout_wr {cm['wr']:.3f} < floor {wr_floor:.3f}"
    return True, "holdout_ok"


def run_worker(idx: int, args: argparse.Namespace, current: dict[str, Any]) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"btc-parallel-{idx}-", dir=str(DATA_DIR / "tmp")))
    state = root / "state"
    logs = root / "logs"
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    local_best = state / "sniper_best_params.json"
    write_params_file(local_best, current, mode="paper")

    env = os.environ.copy()
    env.update({
        "HARVEY_HOME": str(HARVEY_HOME),
        "BTC_STATE_DIR": str(state),
        "BTC_LOG_DIR": str(logs),
        "BTC_JOURNAL_FILE": str(args.train_journal_file),
        "BTC_BEST_PARAMS_FILE": str(local_best),
        "BTC_EVOLUTION_TSV": str(state / "backtest_evolution.tsv"),
        "ITERATIONS": str(args.iterations),
        "BOOT_ITERS": str(args.boot_iters),
        "MUTATION_RATE": str(args.mutation_rates[idx % len(args.mutation_rates)]),
        "PYTHONUNBUFFERED": "1",
    })
    if args.min_win_rate is not None:
        env["MIN_WIN_RATE"] = str(args.min_win_rate)

    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(SRC_DIR / "btc_backtest_autoresearch.py")],
        cwd=str(SRC_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )

    params = read_params_file(local_best, mode="paper")
    tail = "\n".join(proc.stdout.splitlines()[-20:])
    if not args.keep_temps:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "worker": idx,
        "rc": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "mutation_rate": env["MUTATION_RATE"],
        "params": params,
        "stdout_tail": tail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("PARALLEL_WORKERS", "6")))
    ap.add_argument("--iterations", type=int, default=int(os.environ.get("ITERATIONS_PER_WORKER", "300")))
    ap.add_argument("--boot-iters", type=int, default=int(os.environ.get("BOOT_ITERS_PER_WORKER", "250")))
    ap.add_argument("--min-score-gap", type=float, default=float(os.environ.get("MIN_SCORE_GAP", "1.0")))
    ap.add_argument("--mutation-rates", default=os.environ.get("MUTATION_RATES", "0.20,0.35,0.50,0.65"))
    ap.add_argument("--min-win-rate", type=float, default=None)
    ap.add_argument("--holdout-pct", type=float, default=float(os.environ.get("HOLDOUT_PCT", "0.20")))
    ap.add_argument("--holdout-min-trades", type=int, default=int(os.environ.get("HOLDOUT_MIN_TRADES", "10")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("PARALLEL_WORKER_TIMEOUT", "180")))
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--freeze-on-commit", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-temps", action="store_true")
    args = ap.parse_args()

    args.workers = max(1, min(args.workers, 16))
    args.mutation_rates = [float(x) for x in args.mutation_rates.split(",") if x.strip()]
    (DATA_DIR / "tmp").mkdir(parents=True, exist_ok=True)

    trades = bt.load_btc_trades()
    if len(trades) < 20:
        raise SystemExit("too few BTC paper trades for parallel autoresearch")
    train_trades, holdout_trades = split_train_holdout(trades, args.holdout_pct)
    train_journal_file = DATA_DIR / "tmp" / f"parallel_train_{int(time.time())}.jsonl"
    write_journal(train_journal_file, train_trades)
    args.train_journal_file = train_journal_file

    current_before = load_current()
    current_norm = normalized_result(current_before, train_trades)
    current_all_norm = normalized_result(current_before, trades)
    current_holdout_norm = normalized_result(current_before, holdout_trades) if holdout_trades else None
    log(
        "PARALLEL AUTORESEARCH start "
        f"workers={args.workers} iterations={args.iterations} trades={len(trades)} train={len(train_trades)} holdout={len(holdout_trades)} "
        f"current_train_score={current_norm['score']:.2f} "
        f"current={current_norm['params']}"
    )

    raw_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_worker, i, args, current_before) for i in range(args.workers)]
        for fut in as_completed(futures):
            raw = fut.result()
            norm = normalized_result(raw["params"], train_trades)
            raw["normalized"] = norm
            raw["holdout"] = normalized_result(raw["params"], holdout_trades) if holdout_trades else None
            raw_results.append(raw)
            log(
                f"worker={raw['worker']} rc={raw['rc']} seconds={raw['seconds']} "
                f"score={norm['score']:.2f} params={norm['params']} metrics={norm['metrics']}"
            )

    best_raw = max(raw_results, key=lambda r: r["normalized"]["score"])
    best_norm = best_raw["normalized"]
    best_holdout = best_raw.get("holdout")
    holdout_ok, holdout_reason = (
        holdout_pass(best_holdout, current_holdout_norm, args.holdout_min_trades)
        if holdout_trades and current_holdout_norm and best_holdout
        else (True, "no_holdout")
    )

    # Re-read current at commit time because other paper-only optimizers may
    # have published a new candidate while workers were running.
    current_after = load_current()
    current_after_norm = normalized_result(current_after, train_trades)
    current_after_all_norm = normalized_result(current_after, trades)
    current_after_holdout_norm = normalized_result(current_after, holdout_trades) if holdout_trades else None
    improved = best_norm["score"] > current_after_norm["score"] + args.min_score_gap and holdout_ok
    committed = False
    if args.commit:
        with params_file_lock(BEST_PARAMS_FILE):
            current_after = load_current()
            current_after_norm = normalized_result(current_after, train_trades)
            current_after_holdout_norm = normalized_result(current_after, holdout_trades) if holdout_trades else None
            if holdout_trades and current_after_holdout_norm and best_holdout:
                holdout_ok, holdout_reason = holdout_pass(best_holdout, current_after_holdout_norm, args.holdout_min_trades)
            improved = best_norm["score"] > current_after_norm["score"] + args.min_score_gap and holdout_ok
            if improved:
                out = dict(best_raw["params"])
                out.update({
                    "best_score": best_norm["score"],
                    "generation": int(time.time()),
                    "strategy_version": STRATEGY_VERSION,
                    "source": "btc_parallel_backtest_orchestrator",
                    "metrics": best_norm["metrics"],
                    "holdout_metrics": best_holdout["metrics"] if best_holdout else None,
                    "holdout_reason": holdout_reason,
                })
                write_params_file(BEST_PARAMS_FILE, out, mode="paper")
                committed = True
                log(f"COMMIT best_train_score={best_norm['score']:.2f} holdout={holdout_reason} params={best_norm['params']}")
                if args.freeze_on_commit:
                    freeze_after_commit("parallel-autoresearch-param-update")
            else:
                log(
                    f"NO COMMIT commit=True improved=False "
                    f"best_train={best_norm['score']:.2f} current_after_train={current_after_norm['score']:.2f} "
                    f"holdout={holdout_reason}"
                )
    else:
        log(
            f"NO COMMIT commit=False improved={improved} "
            f"best_train={best_norm['score']:.2f} current_after_train={current_after_norm['score']:.2f} "
            f"holdout={holdout_reason}"
        )

    summary = {
        "mode": "paper_only_no_orders",
        "workers": args.workers,
        "iterations_per_worker": args.iterations,
        "trades": len(trades),
        "train_trades": len(train_trades),
        "holdout_trades": len(holdout_trades),
        "current_before": current_norm,
        "current_before_all": current_all_norm,
        "current_before_holdout": current_holdout_norm,
        "current_after": current_after_norm,
        "current_after_all": current_after_all_norm,
        "current_after_holdout": current_after_holdout_norm,
        "best": best_norm,
        "best_holdout": best_holdout,
        "best_worker": best_raw["worker"],
        "holdout_ok": holdout_ok,
        "holdout_reason": holdout_reason,
        "improved": improved,
        "committed": committed,
        "worker_results": [
            {
                "worker": r["worker"],
                "rc": r["rc"],
                "seconds": r["seconds"],
                "mutation_rate": r["mutation_rate"],
                "normalized": r["normalized"],
                "holdout": r.get("holdout"),
            }
            for r in sorted(raw_results, key=lambda x: x["worker"])
        ],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.keep_temps:
        try:
            train_journal_file.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
