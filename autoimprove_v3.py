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
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from btc_param_contract import (
    DYNAMIC_PARAM_FIELDS,
    PAPER_BOUNDS,
    dynamic_fingerprint,
    locked_write_params_file,
    read_params_file,
)

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
LAB_ROOT = DATA_DIR / "parallel_lab"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
LEGACY_PARAMS_FILE = STATE_DIR / "legacy_sniper_best_params.json"
SPLIT_PARAMS_FILES = (
    STATE_DIR / "sniper_best_params_5m.json",
    STATE_DIR / "sniper_best_params_15m.json",
    STATE_DIR / "sniper_best_params.json",
)
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
    paths = [JOURNAL_FILE]
    extra_globs = os.environ.get("BTC_EXTRA_JOURNAL_GLOBS", "")
    for raw in [x.strip() for x in extra_globs.split(os.pathsep) if x.strip()]:
        paths.extend(Path(p) for p in __import__("glob").glob(os.path.expanduser(raw)))
    total = 0
    clob = 0
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = (
                        row.get("placed_at"),
                        row.get("agent_id"),
                        row.get("window_start"),
                        row.get("window_tf"),
                        row.get("direction"),
                        row.get("pnl"),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    total += 1
                    if row.get("price_source") == "clob_book_ask_depth":
                        clob += 1
                except Exception:
                    pass
    return total, clob


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _wilson_lower_bound(wins: int, n: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson lower confidence bound.

    z=1.64485 is ~90% one-sided. This is deliberately stricter than raw WR but
    not as slow as 95% two-sided for fake-money paper promotion.
    """
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _lab_cutoff_ts() -> float:
    hours = _float(os.environ.get("BTC_LAB_PROMOTION_LOOKBACK_HOURS"), 4.0)
    return time.time() - max(0.1, hours) * 3600


def scan_lab_strategy_results(window_tf: int | None = None) -> list[dict]:
    """Group recent isolated lab journals by strategy name.

    This is the fast improvement loop Sebastian asked for: test many strategy
    variables in fake-money workers, score only executable CLOB fills, then
    promote the best dynamic thresholds to paper agents. Never touches live.
    """
    if not LAB_ROOT.exists():
        return []
    cutoff = _lab_cutoff_ts()
    grouped: dict[str, dict] = {}
    manifests = sorted(LAB_ROOT.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for manifest_path in manifests:
        try:
            if manifest_path.stat().st_mtime < cutoff:
                continue
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for entry in manifest.get("strategies", []):
            try:
                spec = entry.get("spec") or {}
                name = str(entry.get("name") or spec.get("name") or "unknown")
                rows = [
                    r
                    for r in _read_jsonl(Path(entry["journal_file"]))
                    if r.get("price_source") == "clob_book_ask_depth"
                    and (window_tf is None or int(r.get("window_tf") or 0) == int(window_tf))
                ]
            except Exception:
                continue
            if not rows:
                continue
            bucket = grouped.setdefault(
                name,
                {
                    "name": name,
                    "spec": spec,
                    "rows": [],
                    "sources": set(),
                },
            )
            bucket["rows"].extend(rows)
            bucket["sources"].add(str(manifest_path.parent))

    out = []
    for item in grouped.values():
        rows = item["rows"]
        n = len(rows)
        wins = sum(1 for r in rows if r.get("won") is True or _float(r.get("pnl")) > 0)
        losses = sum(1 for r in rows if _float(r.get("pnl")) < 0)
        pnl = sum(_float(r.get("pnl")) for r in rows)
        wr = wins / n if n else 0.0
        wr_lb = _wilson_lower_bound(wins, n)
        avg_pnl = pnl / n if n else 0.0
        # Reward real CLOB PnL, enough sample, and robust WR lower bound; punish
        # losses and tiny-sample lucky streaks. This is paper promotion only,
        # not live readiness.
        score = pnl + wr_lb * 12.0 + avg_pnl * min(n, 40) * 0.25 - losses * 0.50 + min(n, 40) * 0.06
        out.append(
            {
                "name": item["name"],
                "spec": item["spec"],
                "n": n,
                "wins": wins,
                "losses": losses,
                "wr": wr,
                "wr_lb": wr_lb,
                "pnl": pnl,
                "avg_pnl": avg_pnl,
                "score": score,
                "sources": sorted(item["sources"]),
            }
        )
    out.sort(key=lambda r: (r["score"], r["wr_lb"], r["pnl"], r["n"]), reverse=True)
    return out


def _env_threshold(name: str, default: float, tf: int | None = None) -> float:
    if tf is not None:
        raw = os.environ.get(f"{name}_{tf}M")
        if raw is not None:
            return _float(raw, default)
    return _float(os.environ.get(name), default)


def _promote_one_lab_strategy(tf: int | None, target: Path) -> dict | None:
    results = scan_lab_strategy_results(window_tf=tf)
    label = f"{tf}m" if tf is not None else "global"
    if not results:
        log(f"Lab promotion {label}: no recent CLOB-valid lab trades.")
        return None
    for i, r in enumerate(results[:8], 1):
        log(
            f"Lab rank {label} #{i}: {r['name']} n={r['n']} W={r['wins']} L={r['losses']} "
            f"WR={r['wr']:.1%} LB90={r['wr_lb']:.1%} PnL=${r['pnl']:+.2f} score={r['score']:.2f}"
        )

    best = results[0]
    # 15m resolves slower, but n=4 100% winners already overfit in live paper.
    # Require enough resolved CLOB fills plus a Wilson lower bound before
    # writing params to the split paper agents.
    default_min_n = 8 if tf == 15 else (12 if tf == 5 else 20)
    min_n = int(_env_threshold("BTC_LAB_PROMOTION_MIN_TRADES", default_min_n, tf=tf))
    min_wr = _env_threshold("BTC_LAB_PROMOTION_MIN_WR", 0.60, tf=tf)
    min_wr_lb = _env_threshold("BTC_LAB_PROMOTION_MIN_WR_LB", 0.50, tf=tf)
    min_pnl = _env_threshold("BTC_LAB_PROMOTION_MIN_PNL", 0.0, tf=tf)
    if best["n"] < min_n or best["wr"] < min_wr or best["wr_lb"] < min_wr_lb or best["pnl"] <= min_pnl:
        log(
            f"Lab promotion {label} rejected: "
            f"best={best['name']} n={best['n']} WR={best['wr']:.1%} LB90={best['wr_lb']:.1%} "
            f"PnL=${best['pnl']:+.2f}; requires n>={min_n} WR>={min_wr:.1%} "
            f"LB90>={min_wr_lb:.1%} PnL>${min_pnl:.2f}"
        )
        return None

    spec = dict(best.get("spec") or {})
    payload = {
        "name": f"lab_promoted_{best['name']}",
        "delta_thresh": spec.get("delta_thresh"),
        "conf_thresh": spec.get("conf_thresh"),
        "ens_thresh": spec.get("ens_thresh"),
        "best_score": best["score"],
        "metrics": {
            "n": best["n"],
            "wins": best["wins"],
            "losses": best["losses"],
            "wr": best["wr"],
            "wr_lb": best["wr_lb"],
            "pnl": best["pnl"],
            "avg_pnl": best["avg_pnl"],
            "sources": best["sources"][:4],
        },
        "source": "autoimprove_v3.lab_strategy_scan",
        "generation": int(time.time()),
    }
    clean = locked_write_params_file(target, payload, mode="paper")
    log(
        f"LAB PROMOTION DEPLOYED TO PAPER {label}: {best['name']} -> {target.name} "
        f"delta={payload['delta_thresh']} conf={payload['conf_thresh']} ens={payload['ens_thresh']} "
        f"n={best['n']} WR={best['wr']:.1%} LB90={best['wr_lb']:.1%} PnL=${best['pnl']:+.2f}"
    )
    brain_log(
        f"[[BTC Auto-Improve]] promoted {label} paper params from lab strategy `{best['name']}` "
        f"(n={best['n']}, WR={best['wr']:.1%}, LB90={best['wr_lb']:.1%}, PnL=${best['pnl']:+.2f}) "
        "to split paper agents; live remains kill-switched."
    )
    return {"best": best, "written": [str(target)], "params": clean, "tf": tf}


def promote_best_lab_strategy() -> dict | None:
    """Promote best paper-only lab strategy separately for 5m and 15m.

    Important: the agents are separate processes and their edge can differ by
    market window. Do not smear a 5m winner onto 15m or vice versa.
    """
    promotions = []
    for tf, target in (
        (5, STATE_DIR / "sniper_best_params_5m.json"),
        (15, STATE_DIR / "sniper_best_params_15m.json"),
        (None, STATE_DIR / "sniper_best_params.json"),
    ):
        promoted = _promote_one_lab_strategy(tf, target)
        if promoted:
            promotions.append(promoted)
    return {"promotions": promotions} if promotions else None


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

    promote_best_lab_strategy()

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
