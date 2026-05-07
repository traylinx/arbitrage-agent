#!/usr/local/opt/python@3.11/bin/python3.11
"""
BTC Backtest Autoresearch — legacy journal replay optimizer.

Unlike btc_autoresearch.py (which evolves the wrong genome against wrong markets),
this one:
  1. Loads ACTUAL historical BTC sniper trades from intraday_journal.jsonl
  2. Evolves SniperParams (the real param struct the live sniper uses)
  3. Replays each trade through the filter to see which would have fired
  4. Scores by realized PnL + WR + trade count
  5. Writes winner to legacy_sniper_best_params.json by default.
     Runtime params are only touched with BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS=1.

Because the backtest is instant (no live API calls), we can run 10,000+ experiments
per minute vs the old 1-per-10-minutes rate.

Usage:
    /usr/local/bin/python3.11 agents/arbitrage-agent/btc_backtest_autoresearch.py
    # or with options:
    ITERATIONS=5000 MUTATION_RATE=0.4 ./btc_backtest_autoresearch.py

Output:
    legacy_sniper_best_params.json — analysis result by default
    sniper_best_params.json        — updated only with BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS=1
    backtest_evolution.tsv         — one row per experiment
"""

import copy
import glob
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from btc_fee_model import resolved_buy_pnl
from btc_param_contract import (
    DEFAULT_DYNAMIC_PARAMS,
    DYNAMIC_PARAM_FIELDS,
    FIXED_EXECUTION_PARAMS,
    PAPER_BOUNDS,
    STRATEGY_VERSION,
    params_file_lock,
    read_params_file,
    write_params_file,
)

# ── Paths ────────────────────────────────────────────────────────────────────
HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = Path(os.path.expanduser(os.environ.get("BTC_STATE_DIR", str(DATA_DIR / "state"))))
LOG_DIR = Path(os.path.expanduser(os.environ.get("BTC_LOG_DIR", str(DATA_DIR / "logs"))))
JOURNAL_FILE = Path(os.path.expanduser(os.environ.get("BTC_JOURNAL_FILE", str(STATE_DIR / "intraday_journal.jsonl"))))
BEST_PARAMS_FILE = Path(os.path.expanduser(os.environ.get("BTC_BEST_PARAMS_FILE", str(STATE_DIR / "sniper_best_params.json"))))
LEGACY_BEST_PARAMS_FILE = Path(os.path.expanduser(os.environ.get("BTC_LEGACY_BEST_PARAMS_FILE", str(STATE_DIR / "legacy_sniper_best_params.json"))))
EVOLUTION_TSV = Path(os.path.expanduser(os.environ.get("BTC_EVOLUTION_TSV", str(STATE_DIR / "backtest_evolution.tsv"))))
EXTRA_JOURNAL_FILES = os.environ.get("BTC_EXTRA_JOURNAL_FILES", "")
EXTRA_JOURNAL_GLOBS = os.environ.get("BTC_EXTRA_JOURNAL_GLOBS", "")

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ITERATIONS = int(os.environ.get("ITERATIONS", "2000"))
MUTATION_RATE = float(os.environ.get("MUTATION_RATE", "0.35"))
RESTART_EVERY = int(os.environ.get("RESTART_EVERY", "200"))  # jump out of local minima
MIN_TRADES = int(os.environ.get("MIN_TRADES", "15"))  # reject overfit single-trade wins
REPLAY_REQUIRE_CLOB = os.environ.get("BTC_REPLAY_REQUIRE_CLOB", "1") != "0"
CLOB_PRICE_SOURCE = "clob_book_ask_depth"
WRITE_RUNTIME_PARAMS = os.environ.get("BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS", "0") == "1"
PARAM_WRITE_TARGET = BEST_PARAMS_FILE if WRITE_RUNTIME_PARAMS else LEGACY_BEST_PARAMS_FILE

# ── GUARDRAILS: parameter bounds & quality gates ─────────────────────────────
CONF_FLOOR   = float(os.environ.get("CONF_FLOOR", str(PAPER_BOUNDS["conf_thresh"][0])))   # paper exploration floor
CONF_CEIL    = float(os.environ.get("CONF_CEIL",  str(PAPER_BOUNDS["conf_thresh"][1])))   # sanity ceiling
CONF_STEP_MAX = float(os.environ.get("CONF_STEP_MAX", "0.10")) # max change per generation
DELTA_FLOOR  = float(os.environ.get("DELTA_FLOOR", str(PAPER_BOUNDS["delta_thresh"][0])))
DELTA_CEIL   = float(os.environ.get("DELTA_CEIL", str(PAPER_BOUNDS["delta_thresh"][1])))
ENS_FLOOR    = float(os.environ.get("ENS_FLOOR", str(PAPER_BOUNDS["ens_thresh"][0])))
ENS_CEIL     = float(os.environ.get("ENS_CEIL", str(PAPER_BOUNDS["ens_thresh"][1])))
ENS_STEP_MAX = 0.10
MIN_WIN_RATE = float(os.environ.get("MIN_WIN_RATE", "0.60")) # reject below 60% WR
MIN_SCORE_GAP = 5.0  # new score must beat best by this to commit (prevents noise fits)
HOLDOUT_PCT  = 0.20  # 20% of trades held out for statistical significance
BOOT_ITERS   = int(os.environ.get("BOOT_ITERS", "5000"))  # bootstrap iterations for CI
CI_LEVEL     = 0.95  # confidence interval level

# ── Statistical helpers ──────────────────────────────────────────────────────
def _bootstrap_ci(values: list[float], iters: int, level: float) -> tuple[float, float]:
    """Return (lo, hi) bootstrap CI for a list of per-trade PnLs."""
    if len(values) < 3:
        return (-float('inf'), float('inf'))  # not enough data
    n = len(values)
    boot_means = []
    for _ in range(iters):
        sample = random.choices(values, k=n)
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(iters * (1 - level) / 2)]
    hi = boot_means[int(iters * (1 + level) / 2)]
    return (lo, hi)


def _ci_overlap(ci_a: tuple, ci_b: tuple) -> bool:
    """Return True if two CIs overlap."""
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def apply_guardrails(params: dict, best_so_far: dict, trade_sample: list[dict]) -> tuple[bool, str]:
    """
    Returns (passes, reason). Reject param changes that smell like overfitting.
    Checks:
      1. Step size: no param drifted more than step_max from best_so_far
      2. Floor/ceiling: all params within defined bounds
      3. Trade count: enough trades to be statistically meaningful
      4. Win rate: minimum viable WR gate
      5. Bootstrap CI: improvement over baseline must be statistically significant
    """
    # 1. Step size check
    for key, step_max in [("conf_thresh", CONF_STEP_MAX), ("ens_thresh", ENS_STEP_MAX)]:
        if key in best_so_far:
            delta = abs(params[key] - best_so_far[key])
            if delta > step_max:
                return False, f"STEP_{key} {delta:.3f} > {step_max} (drift too aggressive)"

    # 2. Floor/ceiling
    for key, floor, ceil in [
        ("conf_thresh", CONF_FLOOR, CONF_CEIL),
        ("ens_thresh", ENS_FLOOR, ENS_CEIL),
        ("delta_thresh", DELTA_FLOOR, DELTA_CEIL),
    ]:
        if params[key] < floor - 1e-9 or params[key] > ceil + 1e-9:
            return False, f"BOUNDS_{key} {params[key]:.3f} outside [{floor},{ceil}]"

    # 3. Trade count gate (enforced at scoring level too — double-gate)
    # done in backtest_score

    # 4. Win rate gate — reject below MIN_WIN_RATE
    fired = [t for t in trade_sample
             if would_fire(t, params["delta_thresh"], params["conf_thresh"], params["ens_thresh"])]
    if len(fired) < MIN_TRADES:
        return False, f"LOW_TRADES n={len(fired)} < {MIN_TRADES}"
    wins = sum(1 for t in fired if t.get("won"))
    wr = wins / len(fired)
    if wr < MIN_WIN_RATE:
        return False, f"LOW_WR {wr:.1%} < {MIN_WIN_RATE:.0%}"

    # 5. Bootstrap CI significance: new params must beat current best
    # by more than CI overlap
    if best_so_far:
        current_fired = [t for t in trade_sample
                         if would_fire(t, best_so_far["delta_thresh"],
                                       best_so_far["conf_thresh"], best_so_far["ens_thresh"])]
        if len(current_fired) >= 10:
            current_pnls = [trade_pnl(t) for t in current_fired]
            new_pnls     = [trade_pnl(t) for t in fired]
            ci_curr = _bootstrap_ci(current_pnls, BOOT_ITERS, CI_LEVEL)
            ci_new  = _bootstrap_ci(new_pnls,     BOOT_ITERS, CI_LEVEL)
            if _ci_overlap(ci_curr, ci_new):
                return False, f"CI_OVERLAP cur={ci_curr[0]:.3f}–{ci_curr[1]:.3f} " \
                              f"new={ci_new[0]:.3f}–{ci_new[1]:.3f} (not significant)"

    return True, "pass"


def holdout_validate(params: dict, trades: list[dict]) -> dict:
    """
    Split trades 80/20. Score on 80%% (train), require consistent direction on 20%% (val).
    Returns a dict with train/val scores for gated acceptance.
    """
    n = len(trades)
    if n < 30:
        return {"train": None, "val": None, "valid": False}
    cutoff = int(n * (1 - HOLDOUT_PCT))
    shuffled = trades[:]
    random.shuffle(shuffled)
    train_trades = shuffled[:cutoff]
    val_trades   = shuffled[cutoff:]

    train_result = backtest_score(train_trades, params)
    val_result   = backtest_score(val_trades,   params)

    # Gate: val PnL must be non-negative (never bet on a strategy that loses OOS)
    valid = val_result["pnl"] >= 0
    return {
        "train": train_result,
        "val":   val_result,
        "valid": valid,
        "val_wr": val_result["wr"],
        "val_pnl": val_result["pnl"],
    }


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR / "btc_backtest_autoresearch.log", "a") as f:
        f.write(line + "\n")


# ── Load historical BTC trades ────────────────────────────────────────────────
def _split_path_list(raw: str) -> list[str]:
    raw = (raw or "").replace(",", os.pathsep)
    return [x.strip() for x in raw.split(os.pathsep) if x.strip()]


def journal_paths() -> list[Path]:
    """Main paper journal plus optional isolated lab journals for research.

    Live GO/NO-GO still reads only the main journal. Extra lab journals are for
    offline parameter research; they never execute orders and never bypass live
    promotion gates.
    """
    paths = [JOURNAL_FILE]
    for raw in _split_path_list(EXTRA_JOURNAL_FILES):
        paths.append(Path(os.path.expanduser(raw)))
    for pat in _split_path_list(EXTRA_JOURNAL_GLOBS):
        paths.extend(Path(x) for x in glob.glob(os.path.expanduser(pat)))
    out: list[Path] = []
    seen = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _trade_key(t: dict) -> tuple:
    return (
        t.get("placed_at"),
        t.get("strategy"),
        t.get("window_start"),
        t.get("window_tf"),
        t.get("direction"),
        t.get("btc_delta"),
        t.get("pnl"),
    )


def load_btc_trades() -> list[dict]:
    """Load replayable BTC trades.

    Default is CLOB-realistic only. Old journal rows without executable CLOB
    ask-depth pricing inflated profitability and must not drive live promotion.
    Set BTC_REPLAY_REQUIRE_CLOB=0 only for historical research comparisons.
    """
    trades = []
    excluded_non_clob = 0
    missing = 0
    seen = set()
    sources = journal_paths()
    for path in sources:
        if not path.exists():
            missing += 1
            continue
        with open(path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if not ("btc_delta" in t and "conf" in t and "won" in t and "pnl" in t):
                        continue
                    if REPLAY_REQUIRE_CLOB and t.get("price_source") != CLOB_PRICE_SOURCE:
                        excluded_non_clob += 1
                        continue
                    key = _trade_key(t)
                    if key in seen:
                        continue
                    seen.add(key)
                    trades.append(t)
                except Exception:
                    continue
    log(f"Replay sources: files={len(sources)} missing={missing} trades={len(trades)}")
    if REPLAY_REQUIRE_CLOB:
        log(f"Replay filter: CLOB-only enabled; excluded_non_clob={excluded_non_clob}")
    return trades


# ── Hour bias (synced with btc_sniper_live.py) ───────────────────────────────
HOUR_CONFLICT_DOWN = {17}
HOUR_CONFLICT_UP = {20}
HOUR_PENALIZE_UP = {12}
HOUR_PENALIZE_DOWN = {4, 13}
HOUR_SKIP_ALL = {21, 5}
HOUR_BOOST_UP = {8, 9, 10, 11, 15, 17, 18, 19, 22}
HOUR_BOOST_DOWN = {0, 9, 12, 14, 20, 23}


def _hour_adjust_conf(base_conf: float, direction: str, hour: int) -> float:
    """Apply same UTC hour biases as live sniper."""
    if hour in HOUR_SKIP_ALL:
        return 0.0
    if direction == "Down" and hour in HOUR_CONFLICT_DOWN:
        return 0.0
    if direction == "Up" and hour in HOUR_CONFLICT_UP:
        return 0.0
    penalty = 0.0
    if direction == "Up" and hour in HOUR_PENALIZE_UP:
        penalty = 0.30
    if direction == "Down" and hour in HOUR_PENALIZE_DOWN:
        penalty = 0.30
    boost = 0.0
    if direction == "Up" and hour in HOUR_BOOST_UP:
        boost = 0.20
    if direction == "Down" and hour in HOUR_BOOST_DOWN:
        boost = 0.20
    return max(0.0, base_conf - penalty + boost)


# ── Replay filter ─────────────────────────────────────────────────────────────
def would_fire(trade: dict, delta_thresh: float, conf_thresh: float, ens_thresh: float) -> bool:
    """Mirror btc_sniper_live.py::_check_signal filter logic.

    Fire condition:
      (|btc_delta| >= delta_thresh OR poly_conv >= 0.90)
      AND (conf >= conf_thresh OR poly_conv >= 0.90)
      AND conf >= ens_thresh  (else ensemble would have returned Neutral)
      AND passes hour bias filter
    """
    btc_delta = abs(trade["btc_delta"])
    conf = trade["conf"]
    poly_price = trade.get("poly_price", 0.5)
    poly_conv = abs(poly_price - 0.5) * 2
    direction = trade.get("direction", "Up")
    placed = trade.get("placed_at", "")
    try:
        hour = int(placed[11:13]) if len(placed) >= 13 else 0
    except Exception:
        hour = 0

    # Apply hour bias (same as live sniper)
    conf = _hour_adjust_conf(conf, direction, hour)

    # Poly conviction override: if market is extreme, always fires
    if poly_conv >= 0.90:
        # ens_thresh still applies — ensemble must have returned a direction
        return conf >= ens_thresh

    # Normal path: all three filters must pass
    if btc_delta < delta_thresh:
        return False
    if conf < conf_thresh:
        return False
    if conf < ens_thresh:
        return False
    return True


def trade_pnl(trade: dict) -> float:
    """Use exact current fee model for all replay, even for old journal rows."""
    try:
        if "size" in trade and "poly_price" in trade:
            return resolved_buy_pnl(bool(trade.get("won")), float(trade["size"]), float(trade["poly_price"]))
        # Fallback for older resolved rows that only stored spend/price.
        price = float(trade.get("poly_price", 0.5) or 0.5)
        spend = float(trade.get("spend", 0.0) or 0.0)
        shares = spend / price if price > 0 else 0.0
        return resolved_buy_pnl(bool(trade.get("won")), shares, price)
    except Exception:
        return float(trade.get("pnl", 0.0) or 0.0)

# ── Score a parameter set ─────────────────────────────────────────────────────
def backtest_score(trades: list[dict], params: dict) -> dict:
    """Replay all trades through filter. Return aggregate score."""
    delta_thresh = params["delta_thresh"]
    conf_thresh = params["conf_thresh"]
    ens_thresh = params["ens_thresh"]

    fired = [t for t in trades if would_fire(t, delta_thresh, conf_thresh, ens_thresh)]
    n = len(fired)
    if n == 0:
        return {"score": -9999.0, "pnl": 0.0, "wins": 0, "losses": 0, "n": 0, "wr": 0.0}

    wins = sum(1 for t in fired if t.get("won"))
    losses = n - wins
    pnl = sum(trade_pnl(t) for t in fired)
    wr = wins / n

    # ── Guards ──────────────────────────────────────────────────────────────
    penalty = 0.0
    if n < MIN_TRADES:
        penalty = (MIN_TRADES - n) * 5.0  # penalise single-trade luck
    if wr < MIN_WIN_RATE:
        penalty += 200.0  # hard gate: WR below 60% is disqualified

    # ── Sharpe-ratio normalisation (prevents score inflation from overtrading) ─
    # Raw PnL bonus but normalised by trade frequency so more trades don't
    # mechanically inflate the score.  log(1+n) stays as a mild coverage signal.
    pnls = [trade_pnl(t) for t in fired]
    mean_pnl = sum(pnls) / n
    std_pnl   = math.sqrt(sum((x - mean_pnl) ** 2 for x in pnls) / max(n - 1, 1)) or 1e-9
    sharpe_like = mean_pnl / std_pnl * math.sqrt(min(n, 50))  # cap freq contribution

    # PnL component (stabilised) + WR + sharpe-like quality + coverage - penalties
    score = (
        pnl * 0.5           # reduced PnL weight (was 1.5 — inflated low-conf trades)
        + wr * 30.0         # WR is primary quality signal
        + sharpe_like * 5.0 # quality-per-trade normalisation
        + math.log(1 + n) * 2.0  # reduced coverage bonus (was 8.0)
        - penalty
    )

    return {
        "score": score,
        "pnl": pnl,
        "wins": wins,
        "losses": losses,
        "n": n,
        "wr": wr,
    }


# ── Param loading / mutation / saving ─────────────────────────────────────────
def load_best_params() -> dict:
    """Load current frozen-contract params from sniper_best_params.json."""
    if BEST_PARAMS_FILE.exists():
        return read_params_file(BEST_PARAMS_FILE, mode="paper")
    return {
        **FIXED_EXECUTION_PARAMS,
        **DEFAULT_DYNAMIC_PARAMS,
        "name": "backtest_init",
        "strategy_version": STRATEGY_VERSION,
    }


def mutate_params(params: dict, rate: float = MUTATION_RATE) -> dict:
    """Mutate the three replayable threshold params."""
    new = copy.deepcopy(params)
    new["name"] = f"bt_{datetime.now().strftime('%H%M%S')}_{random.randint(1000,9999)}"

    # Only replayable threshold params are mutable. Sizing/risk stays frozen.
    mutations = [
        ("delta_thresh", DELTA_FLOOR, DELTA_CEIL, 1.0),
        ("conf_thresh", CONF_FLOOR, CONF_CEIL, 0.05),
        ("ens_thresh", ENS_FLOOR, ENS_CEIL, 0.05),
    ]
    for attr, lo, hi, step in mutations:
        if random.random() < rate:
            val = round(random.uniform(lo, hi) / step) * step
            new[attr] = round(val, 2)
    return new


def random_restart_params(base: dict) -> dict:
    """Jump to a random point in the search space to escape local minima."""
    new = copy.deepcopy(base)
    new["name"] = f"btrestart_{datetime.now().strftime('%H%M%S')}"
    # Bound restarts within guardrail floors/ceilings (prevents re-found boundary exploit)
    new["delta_thresh"] = round(random.uniform(DELTA_FLOOR, DELTA_CEIL) / 1.0) * 1.0
    new["conf_thresh"]  = round(random.uniform(CONF_FLOOR,  CONF_CEIL)  / 0.05) * 0.05
    new["ens_thresh"]   = round(random.uniform(ENS_FLOOR,  ENS_CEIL)   / 0.05) * 0.05
    return new


def save_best_params(params: dict, result: dict) -> None:
    """Persist winning params.

    Default is analysis-only: write to legacy_sniper_best_params.json so stale
    launchd optimizers cannot overwrite the model-gated runtime config.
    Set BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS=1 for explicit runtime writes.
    """
    out = copy.deepcopy(params)
    out.update({
        "best_score": result["score"],
        "generation": int(time.time()),
        "strategy_version": STRATEGY_VERSION,
        "source": "btc_backtest_autoresearch",
        "metrics": {
            "n": result.get("n", 0),
            "wins": result.get("wins", 0),
            "losses": result.get("losses", 0),
            "wr": result.get("wr", 0.0),
            "pnl": result.get("pnl", 0.0),
        },
    })
    if not WRITE_RUNTIME_PARAMS:
        with params_file_lock(LEGACY_BEST_PARAMS_FILE):
            if LEGACY_BEST_PARAMS_FILE.exists():
                try:
                    existing = read_params_file(LEGACY_BEST_PARAMS_FILE, mode="paper")
                    unchanged = all(
                        existing.get(k) == out.get(k)
                        for k in ["delta_thresh", "conf_thresh", "ens_thresh"]
                    ) and abs(float(existing.get("best_score", -999999)) - float(out.get("best_score", -999998))) < 1e-9
                    if unchanged:
                        return
                except Exception:
                    pass
            write_params_file(LEGACY_BEST_PARAMS_FILE, out, mode="paper")
        return

    with params_file_lock(BEST_PARAMS_FILE):
        # Skip write if params unchanged (prevents mtime churn on stability gate)
        if BEST_PARAMS_FILE.exists():
            try:
                existing = read_params_file(BEST_PARAMS_FILE, mode="paper")
                unchanged = all(
                    existing.get(k) == out.get(k)
                    for k in ["delta_thresh", "conf_thresh", "ens_thresh"]
                ) and abs(float(existing.get("best_score", -999999)) - float(out.get("best_score", -999998))) < 1e-9
                if unchanged:
                    return
                # Race guard: several paper-only optimizers may run in parallel.
                # Never let a slower worker overwrite a currently better params file.
                trades = load_btc_trades()
                if trades:
                    existing_result = backtest_score(trades, existing)
                    if existing_result["score"] > result["score"] + 0.01:
                        log(
                            "SKIP write: existing params score "
                            f"{existing_result['score']:.2f} > candidate {result['score']:.2f}"
                        )
                        return
            except Exception:
                pass
        write_params_file(BEST_PARAMS_FILE, out, mode="paper")


# ── TSV logging ───────────────────────────────────────────────────────────────
def init_tsv() -> None:
    if not EVOLUTION_TSV.exists():
        with open(EVOLUTION_TSV, "w") as f:
            f.write("iter\tdelta\tconf\tens\tn\twins\tlosses\twr\tpnl\tscore\tstatus\n")


def log_tsv(i: int, p: dict, r: dict, status: str) -> None:
    with open(EVOLUTION_TSV, "a") as f:
        f.write(
            f"{i}\t{p['delta_thresh']}\t{p['conf_thresh']}\t{p['ens_thresh']}\t"
            f"{r['n']}\t{r['wins']}\t{r['losses']}\t{r['wr']:.3f}\t"
            f"{r['pnl']:.2f}\t{r['score']:.2f}\t{status}\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("BTC BACKTEST AUTORESEARCH — evolves SniperParams against real historical trades")
    log(f"  Iterations: {ITERATIONS}")
    log(f"  Mutation rate: {MUTATION_RATE}")
    log(f"  Restart every: {RESTART_EVERY}")
    log(f"  Min trades: {MIN_TRADES}")
    log(f"  Replay require CLOB: {REPLAY_REQUIRE_CLOB}")
    log(f"  Runtime write enabled: {WRITE_RUNTIME_PARAMS}")
    log(f"  Param write target: {PARAM_WRITE_TARGET}")
    log(f"  [GUARDRAILS] CONF_FLOOR={CONF_FLOOR} CONF_STEP_MAX={CONF_STEP_MAX}")
    log(f"  [GUARDRAILS] MIN_WIN_RATE={MIN_WIN_RATE:.0%} HOLDOUT_PCT={HOLDOUT_PCT:.0%}")
    log(f"  [GUARDRAILS] CONF_CEIL={CONF_CEIL} BOOT_ITERS={BOOT_ITERS} CI={CI_LEVEL:.0%}")
    log("=" * 70)

    # Load historical trades
    trades = load_btc_trades()
    if len(trades) < 20:
        log(f"ERROR: Only {len(trades)} BTC trades in journal. Need 20+. Aborting.")
        return

    log(f"Loaded {len(trades)} historical BTC trades from journal")

    total_pnl_all = sum(trade_pnl(t) for t in trades)
    total_wins = sum(1 for t in trades if t.get("won"))
    log(f"Baseline (no filter): n={len(trades)} WR={total_wins/len(trades):.0%} PnL=${total_pnl_all:+.2f}")

    # Initial params
    current = load_best_params()
    current_result = backtest_score(trades, current)
    log(
        f"Starting params: delta={current['delta_thresh']} conf={current['conf_thresh']} ens={current['ens_thresh']}"
    )
    log(
        f"  → n={current_result['n']} WR={current_result['wr']:.0%} "
        f"PnL=${current_result['pnl']:+.2f} score={current_result['score']:.2f}"
    )

    best = copy.deepcopy(current)
    best_result = current_result

    init_tsv()

    keeps = 0
    restarts = 0
    guard_rejects = 0
    holdout_rejects = 0
    step_rejects = 0
    log_tsv(0, current, current_result, "init")

    for i in range(1, ITERATIONS + 1):
        # Random restart periodically to escape local minima
        if i % RESTART_EVERY == 0:
            candidate = random_restart_params(best)
            restarts += 1
        else:
            candidate = mutate_params(current)

        # ── GUARDRAIL 1: step size (max change from best so far) ──────────────
        step_ok = all(
            abs(candidate[k] - best[k]) <= step_max
            for k, step_max in [("conf_thresh", CONF_STEP_MAX), ("ens_thresh", ENS_STEP_MAX)]
            if k in best
        )
        if not step_ok:
            step_rejects += 1
            log_tsv(i, candidate, {"score": -9998, "n": 0, "wr": 0.0, "pnl": 0.0,
                                   "wins": 0, "losses": 0}, "STEP_REJECT")
            continue  # skip — aggressive drift

        # ── GUARDRAIL 2: parameter bounds ─────────────────────────────────────
        passes_guard, guard_reason = apply_guardrails(candidate, best, trades)
        if not passes_guard:
            guard_rejects += 1
            log_tsv(i, candidate, {"score": -9998, "n": 0, "wr": 0.0, "pnl": 0.0,
                                   "wins": 0, "losses": 0}, f"GARD_REJECT({guard_reason})")
            continue

        result = backtest_score(trades, candidate)

        # Score must beat best by MIN_SCORE_GAP to commit (prevents noise fits)
        if result["score"] > best_result["score"] + MIN_SCORE_GAP:

            # ── GUARDRAIL 3: holdout validation before committing ──────────────
            if len(trades) >= 30:
                ho = holdout_validate(candidate, trades)
                if not ho["valid"]:
                    holdout_rejects += 1
                    log(f"[{i}] HOLDOUT_REJECT val_pnl={ho['val_pnl']:+.2f} val_wr={ho['val_wr']:.0%}")
                    log_tsv(i, candidate, result, f"HOLDOUT_REJECT(val_pnl={ho['val_pnl']:+.2f})")
                    # Do NOT update best/current — keep searching
                    continue
                else:
                    log(f"[{i}] HOLDOUT OK  val_wr={ho['val_wr']:.0%} val_pnl={ho['val_pnl']:+.2f}")

            best = copy.deepcopy(candidate)
            best_result = result
            current = copy.deepcopy(candidate)
            keeps += 1
            log(
                f"[{i}] KEEP delta={candidate['delta_thresh']} conf={candidate['conf_thresh']:.2f} "
                f"ens={candidate['ens_thresh']:.2f} → n={result['n']} WR={result['wr']:.0%} "
                f"PnL=${result['pnl']:+.2f} score={result['score']:.2f}"
            )
            save_best_params(best, best_result)
            log_tsv(i, candidate, result, "keep")
        else:
            # Hill climbing: probabilistically accept slightly worse to explore
            if random.random() < 0.05 and result["score"] > best_result["score"] - 10:
                current = copy.deepcopy(candidate)
                log_tsv(i, candidate, result, "accept_worse")
            else:
                log_tsv(i, candidate, result, "discard")

    # ── Final report ─────────────────────────────────────────────────────────
    log("=" * 70)
    log("BACKTEST AUTORESEARCH COMPLETE")
    log(f"  Iterations:    {ITERATIONS}")
    log(f"  Keeps:          {keeps}")
    log(f"  Restarts:       {restarts}")
    log(f"  Step rejects:   {step_rejects}")
    log(f"  Guard rejects:  {guard_rejects}")
    log(f"  Holdout rejects:{holdout_rejects}")
    log("=" * 70)
    log(
        f"BEST: delta={best['delta_thresh']} conf={best['conf_thresh']} ens={best['ens_thresh']}"
    )
    log(
        f"  → n={best_result['n']} WR={best_result['wr']:.0%} "
        f"PnL=${best_result['pnl']:+.2f} score={best_result['score']:.2f}"
    )
    save_best_params(best, best_result)
    log(f"Written to {PARAM_WRITE_TARGET}")
    log(f"TSV log: {EVOLUTION_TSV}")


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage: btc_backtest_autoresearch.py\n"
            "Env-driven legacy journal replay optimizer.\n"
            "Default writes analysis to state/legacy_sniper_best_params.json only.\n"
            "Set BTC_LEGACY_AUTORESEARCH_WRITE_PARAMS=1 to update runtime sniper_best_params.json.\n"
        )
        sys.exit(0)

    random.seed()

    def ctrlc_handler(sig, frame):
        log("\nCaught Ctrl+C, exiting")
        sys.exit(0)

    signal.signal(signal.SIGINT, ctrlc_handler)
    signal.signal(signal.SIGTERM, ctrlc_handler)

    main()
