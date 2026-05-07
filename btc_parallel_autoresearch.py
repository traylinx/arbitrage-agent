#!/usr/bin/env python3
"""Parallel BTC paper autoresearch.

Runs independent in-memory optimizer searches against the BTC intraday journal
(5m/15m paper trades only). Workers never write. Parent writes
sniper_best_params.json only when the best worker candidate improves the current
backtest score by MIN_SCORE_GAP.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from btc_backtest_autoresearch import (
    BEST_PARAMS_FILE,
    JOURNAL_FILE,
    backtest_score,
    load_best_params,
)
from btc_param_contract import (
    DYNAMIC_PARAM_FIELDS,
    PAPER_BOUNDS,
    STRATEGY_VERSION,
    locked_write_params_file,
)

PARALLEL_WORKERS = max(1, int(os.environ.get("PARALLEL_WORKERS", str(os.cpu_count() or 2))))
ITERATIONS_PER_WORKER = max(1, int(os.environ.get("ITERATIONS_PER_WORKER", "2000")))
MIN_SCORE_GAP = float(os.environ.get("MIN_SCORE_GAP", "5.0"))
RESTART_EVERY = max(1, int(os.environ.get("RESTART_EVERY", "200")))
MUTATION_RATE = float(os.environ.get("MUTATION_RATE", "0.35"))
RANDOM_ACCEPT_RATE = float(os.environ.get("RANDOM_ACCEPT_RATE", "0.03"))


def _jsonable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": float(result.get("score", -9999.0)),
        "pnl": float(result.get("pnl", 0.0)),
        "wins": int(result.get("wins", 0)),
        "losses": int(result.get("losses", 0)),
        "n": int(result.get("n", 0)),
        "wr": float(result.get("wr", 0.0)),
    }


def load_btc_window_trades(path: Path = JOURNAL_FILE) -> list[dict[str, Any]]:
    """Load BTC paper trades from intraday_journal.jsonl, restricted to 5m/15m."""
    trades: list[dict[str, Any]] = []
    if not path.exists():
        return trades

    with path.open() as handle:
        for line in handle:
            try:
                trade = json.loads(line)
            except Exception:
                continue

            if trade.get("mode") != "paper":
                continue
            if int(trade.get("window_tf", 0) or 0) not in (5, 15):
                continue
            if not all(key in trade for key in ("btc_delta", "conf", "won", "pnl")):
                continue
            trades.append(trade)
    return trades


def _round_param(key: str, value: float) -> float:
    if key == "delta_thresh":
        return round(value, 2)
    return round(value, 4)


def _random_params(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    for key in DYNAMIC_PARAM_FIELDS:
        lo, hi = PAPER_BOUNDS[key]
        candidate[key] = _round_param(key, rng.uniform(lo, hi))
    candidate["name"] = f"parallel_restart_{int(time.time())}_{rng.randrange(1_000_000)}"
    return candidate


def _mutate_params(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    for key in DYNAMIC_PARAM_FIELDS:
        if rng.random() >= MUTATION_RATE:
            continue
        lo, hi = PAPER_BOUNDS[key]
        span = hi - lo
        current = float(candidate.get(key, lo))
        # Local gaussian step; occasional broader jump for independent search diversity.
        sigma = span * (0.08 if rng.random() > 0.15 else 0.25)
        candidate[key] = _round_param(key, min(hi, max(lo, rng.gauss(current, sigma))))
    candidate["name"] = f"parallel_mut_{int(time.time())}_{rng.randrange(1_000_000)}"
    return candidate


def _worker_search(
    worker_id: int,
    trades: list[dict[str, Any]],
    starting_params: dict[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Run one independent in-memory hill-climb/random-restart search."""
    rng = random.Random(seed + worker_id * 1_000_003)

    # Give each worker a distinct initial basin, while preserving worker 0 as current baseline.
    current = copy.deepcopy(starting_params) if worker_id == 0 else _random_params(starting_params, rng)
    current_result = _jsonable_result(backtest_score(trades, current))
    best = copy.deepcopy(current)
    best_result = current_result

    evaluated = 1
    for i in range(1, iterations + 1):
        if i % RESTART_EVERY == 0:
            candidate = _random_params(best, rng)
        else:
            candidate = _mutate_params(current, rng)

        result = _jsonable_result(backtest_score(trades, candidate))
        evaluated += 1

        if result["score"] > best_result["score"]:
            best = copy.deepcopy(candidate)
            best_result = result
            current = copy.deepcopy(candidate)
            current_result = result
        elif result["score"] > current_result["score"] or rng.random() < RANDOM_ACCEPT_RATE:
            current = copy.deepcopy(candidate)
            current_result = result

    return {
        "worker_id": worker_id,
        "evaluated": evaluated,
        "params": {key: best[key] for key in DYNAMIC_PARAM_FIELDS},
        "result": best_result,
    }


def _make_output_payload(params: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(params)
    payload.update(
        {
            "best_score": result["score"],
            "generation": int(time.time()),
            "strategy_version": STRATEGY_VERSION,
            "source": "btc_parallel_autoresearch",
            "metrics": {
                "n": result.get("n", 0),
                "wins": result.get("wins", 0),
                "losses": result.get("losses", 0),
                "wr": result.get("wr", 0.0),
                "pnl": result.get("pnl", 0.0),
            },
        }
    )
    return payload


def main() -> int:
    started = time.time()
    trades = load_btc_window_trades()
    if not trades:
        print(json.dumps({"ok": False, "error": f"no 5m/15m paper BTC trades found in {JOURNAL_FILE}"}, sort_keys=True))
        return 1

    current_params = load_best_params()
    current_result = _jsonable_result(backtest_score(trades, current_params))

    seed = int.from_bytes(os.urandom(8), "big")
    worker_count = PARALLEL_WORKERS
    worker_summaries: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_worker_search, wid, trades, current_params, ITERATIONS_PER_WORKER, seed)
            for wid in range(worker_count)
        ]
        for future in as_completed(futures):
            worker_summaries.append(future.result())

    best_worker = max(worker_summaries, key=lambda item: item["result"]["score"])
    best_params = {**current_params, **best_worker["params"], "name": "parallel_best"}
    best_result = _jsonable_result(backtest_score(trades, best_params))
    improvement = best_result["score"] - current_result["score"]
    should_write = improvement >= MIN_SCORE_GAP

    written = False
    if should_write:
        locked_write_params_file(BEST_PARAMS_FILE, _make_output_payload(best_params, best_result), mode="paper")
        written = True

    summary = {
        "ok": True,
        "paper_only": True,
        "journal": str(JOURNAL_FILE),
        "params_file": str(BEST_PARAMS_FILE),
        "trades": len(trades),
        "workers": worker_count,
        "iterations_per_worker": ITERATIONS_PER_WORKER,
        "evaluated": sum(int(w["evaluated"]) for w in worker_summaries),
        "min_score_gap": MIN_SCORE_GAP,
        "current": {
            "params": {key: current_params[key] for key in DYNAMIC_PARAM_FIELDS},
            "result": current_result,
        },
        "best": {
            "worker_id": best_worker["worker_id"],
            "params": {key: best_params[key] for key in DYNAMIC_PARAM_FIELDS},
            "result": best_result,
        },
        "improvement": improvement,
        "written": written,
        "duration_sec": round(time.time() - started, 3),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
