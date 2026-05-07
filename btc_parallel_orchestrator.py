#!/usr/bin/env python3
"""Parallel supervisor for BTC paper backtest autoresearch.

PAPER ONLY. This script never talks to trading venues and never places orders.
It runs btc_backtest_autoresearch.py in isolated temporary HARVEY_HOME trees so
workers do not race on the real sniper_best_params.json/backtest_evolution.tsv.

Default mode is dry-run: workers write only to temp state dirs, then the best
candidate is reported. Use --commit to atomically write the selected params to
the real PAPER params file through btc_param_contract.write_params_file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from btc_param_contract import locked_write_params_file, read_params_file

SRC_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SRC_DIR / "btc_backtest_autoresearch.py"
REAL_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
REAL_DATA_DIR = REAL_HOME / "data" / "arbitrage-agent" / "v2"
REAL_STATE_DIR = REAL_DATA_DIR / "state"
REAL_JOURNAL_FILE = REAL_STATE_DIR / "intraday_journal.jsonl"
REAL_BEST_PARAMS_FILE = REAL_STATE_DIR / "sniper_best_params.json"


@dataclass
class WorkerResult:
    worker_id: int
    returncode: int
    mutation_rate: float
    temp_home: Path
    params_file: Path
    stdout_tail: str
    stderr_tail: str
    params: dict[str, Any] | None
    error: str | None = None

    @property
    def score(self) -> float:
        if not self.params:
            return float("-inf")
        try:
            return float(self.params.get("best_score", float("-inf")))
        except Exception:
            return float("-inf")


def _tail(text: str, limit: int = 6000) -> str:
    return text[-limit:] if len(text) > limit else text


def _mutation_rates(workers: int, base: float) -> list[float]:
    """Spread workers across nearby mutation rates without needing worker args."""
    if workers <= 1:
        return [base]
    # Center around base, broaden enough to make parallel workers meaningfully different.
    lo = max(0.05, base * 0.55)
    hi = min(0.95, max(base * 1.75, base + 0.20))
    step = (hi - lo) / max(workers - 1, 1)
    return [round(lo + i * step, 4) for i in range(workers)]


def _prepare_temp_home(root: Path, worker_id: int) -> Path:
    """Create isolated Makakoo-like data tree for one worker."""
    temp_home = root / f"worker_{worker_id:02d}_home"
    state_dir = temp_home / "data" / "arbitrage-agent" / "v2" / "state"
    logs_dir = temp_home / "data" / "arbitrage-agent" / "v2" / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not REAL_JOURNAL_FILE.exists():
        raise FileNotFoundError(f"missing journal: {REAL_JOURNAL_FILE}")
    shutil.copy2(REAL_JOURNAL_FILE, state_dir / "intraday_journal.jsonl")

    # Seed each worker from current real params if available. If absent, worker falls back.
    if REAL_BEST_PARAMS_FILE.exists():
        shutil.copy2(REAL_BEST_PARAMS_FILE, state_dir / "sniper_best_params.json")

    return temp_home


async def _run_worker(worker_id: int, temp_home: Path, iterations: int, mutation_rate: float) -> WorkerResult:
    env = os.environ.copy()
    env["HARVEY_HOME"] = str(temp_home)
    env["ITERATIONS"] = str(iterations)
    env["MUTATION_RATE"] = str(mutation_rate)
    # User asked for different ITERATIONS/MUTATION env; current worker script reads MUTATION_RATE.
    # Set MUTATION too for forward compatibility if btc_backtest_autoresearch grows that alias.
    env["MUTATION"] = str(mutation_rate)
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(WORKER_SCRIPT),
        cwd=str(SRC_DIR),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    params_file = temp_home / "data" / "arbitrage-agent" / "v2" / "state" / "sniper_best_params.json"
    params: dict[str, Any] | None = None
    error: str | None = None
    if params_file.exists():
        try:
            params = read_params_file(params_file, mode="paper")
        except Exception as exc:  # corrupted candidate should not kill supervisor
            error = f"failed to read candidate params: {exc}"
    else:
        error = "candidate params file missing"

    if proc.returncode != 0:
        error = (error + "; " if error else "") + f"worker exited {proc.returncode}"

    return WorkerResult(
        worker_id=worker_id,
        returncode=int(proc.returncode or 0),
        mutation_rate=mutation_rate,
        temp_home=temp_home,
        params_file=params_file,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        params=params,
        error=error,
    )


def _candidate_summary(result: WorkerResult) -> dict[str, Any]:
    p = result.params or {}
    metrics = p.get("metrics") if isinstance(p.get("metrics"), dict) else {}
    return {
        "worker": result.worker_id,
        "returncode": result.returncode,
        "mutation_rate": result.mutation_rate,
        "score": result.score,
        "delta_thresh": p.get("delta_thresh"),
        "conf_thresh": p.get("conf_thresh"),
        "ens_thresh": p.get("ens_thresh"),
        "metrics": metrics,
        "params_file": str(result.params_file),
        "error": result.error,
    }


def _current_params() -> dict[str, Any] | None:
    if not REAL_BEST_PARAMS_FILE.exists():
        return None
    return read_params_file(REAL_BEST_PARAMS_FILE, mode="paper")


def _score(params: dict[str, Any] | None) -> float:
    if not params:
        return float("-inf")
    try:
        return float(params.get("best_score", float("-inf")))
    except Exception:
        return float("-inf")


def _build_commit_payload(winner: WorkerResult) -> dict[str, Any]:
    if not winner.params:
        raise ValueError("winner has no params")
    payload = dict(winner.params)
    payload["source"] = "btc_parallel_orchestrator"
    metrics = dict(payload.get("metrics") or {})
    metrics["parallel_worker"] = winner.worker_id
    metrics["parallel_mutation_rate"] = winner.mutation_rate
    metrics["parallel_committed_at"] = int(time.time())
    payload["metrics"] = metrics
    return payload


async def _amain(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"missing worker script: {WORKER_SCRIPT}")

    base_mutation = float(os.environ.get("MUTATION_RATE", os.environ.get("MUTATION", "0.35")))
    mutation_rates = _mutation_rates(args.workers, base_mutation)

    keep_temps = bool(args.keep_temps or args.commit)
    temp_ctx = tempfile.TemporaryDirectory(prefix="btc_parallel_orchestrator_")
    root = Path(temp_ctx.name)
    try:
        temp_homes = [_prepare_temp_home(root, i) for i in range(args.workers)]
        results = await asyncio.gather(*[
            _run_worker(i, temp_homes[i], args.iterations, mutation_rates[i])
            for i in range(args.workers)
        ])

        valid = [r for r in results if r.params is not None]
        winner = max(valid, key=lambda r: r.score) if valid else None
        current = _current_params()
        current_score = _score(current)
        committed = False
        written: dict[str, Any] | None = None
        commit_reason = "dry_run"

        if args.commit:
            if winner is None:
                commit_reason = "no_valid_candidate"
            elif winner.score <= current_score:
                commit_reason = f"winner_score_not_above_current ({winner.score:.6g} <= {current_score:.6g})"
            else:
                payload = _build_commit_payload(winner)
                written = locked_write_params_file(REAL_BEST_PARAMS_FILE, payload, mode="paper")
                committed = True
                commit_reason = "committed"

        report = {
            "paper_only": True,
            "commit_requested": bool(args.commit),
            "committed": committed,
            "commit_reason": commit_reason,
            "workers": args.workers,
            "iterations_per_worker": args.iterations,
            "real_params_file": str(REAL_BEST_PARAMS_FILE),
            "current_score": current_score,
            "winner": _candidate_summary(winner) if winner else None,
            "candidates": [_candidate_summary(r) for r in sorted(results, key=lambda r: r.score, reverse=True)],
            "temp_root": str(root) if keep_temps else None,
            "written": written,
        }
        return report
    finally:
        if keep_temps:
            # TemporaryDirectory cleanup would remove evidence; detach by disabling cleanup.
            temp_ctx._finalizer.detach()  # type: ignore[attr-defined]
        else:
            temp_ctx.cleanup()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run btc_backtest_autoresearch.py in isolated parallel PAPER workers."
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    parser.add_argument("--iterations", type=int, default=2000, help="ITERATIONS per worker")
    parser.add_argument("--commit", action="store_true", help="write best candidate to real sniper_best_params.json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON report")
    parser.add_argument("--keep-temps", action="store_true", help="keep temp worker state dirs for inspection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report = asyncio.run(_amain(args))
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("BTC parallel orchestrator complete (PAPER ONLY)")
        print(f"workers={report['workers']} iterations_per_worker={report['iterations_per_worker']}")
        print(f"real_params_file={report['real_params_file']}")
        print(f"current_score={report['current_score']}")
        if report["winner"]:
            w = report["winner"]
            print(
                "winner="
                f"worker {w['worker']} score={w['score']:.2f} "
                f"delta={w['delta_thresh']} conf={w['conf_thresh']} ens={w['ens_thresh']}"
            )
        print(f"committed={report['committed']} reason={report['commit_reason']}")
        if report.get("temp_root"):
            print(f"temp_root={report['temp_root']}")
        for c in report["candidates"]:
            err = f" error={c['error']}" if c.get("error") else ""
            print(f"candidate worker={c['worker']} rc={c['returncode']} score={c['score']:.2f}{err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
