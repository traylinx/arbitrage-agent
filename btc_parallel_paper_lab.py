#!/usr/bin/env python3
"""Launch and inspect isolated BTC paper-trading strategy workers.

Only starts btc_paper_fast.py in PAPER mode. It never calls the live runner and
never has wallet/order credentials. Each strategy writes its own journal, params,
and log file so experiments cannot poison the main paper session.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
LAB_ROOT = DATA_DIR / "parallel_lab"
SRC_DIR = Path(__file__).resolve().parent
PAPER_RUNNER = SRC_DIR / "btc_paper_fast.py"
PYTHON = Path(os.environ.get("BTC_PAPER_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


SAFE_PAPER_ENV = {
    # Explicit fake-money markers for workers and any future shared helpers.
    "BTC_PAPER_ONLY": "1",
    "BTC_FAKE_MONEY_ONLY": "1",
    "BTC_LAB_FAKE_MONEY_ONLY": "1",
    # Allow fake-money label collection when the trained probability model is
    # absent. btc_paper_fast.py requires these flags plus --parallel-lab before
    # it can bypass MODEL_DOWN, so live/main code remains conservative.
    "BTC_PAPER_EXPLORATION": "1",
    "BTC_ALLOW_PAPER_EXPLORATION": "1",
    # After the 17:07-17:35 CEST holdout failed in a choppy regime, default
    # fake-money exploration now needs some edge/confidence instead of firing
    # every barely-positive label candidate.
    "BTC_EXPLORATION_MIN_EDGE": "0.08",
    "BTC_EXPLORATION_MIN_CONF": "0.45",
    # Avoid lottery-tail contracts and thin-book fills. Bad 17:29 CEST run
    # bought ~2c asks with ~900bps slippage and bled immediately.
    "BTC_EXPLORATION_MIN_POLY": "0.25",
    "BTC_EXPLORATION_MAX_POLY": "0.70",
    "BTC_EXPLORATION_MAX_SLIPPAGE_BPS": "250",
    "BTC_EXPLORATION_MAX_SPEND": "2.50",
    "BTC_EXPLORATION_DELTA_FLOOR": "1.00",
    "POLYMARKET_LIVE_TRADING": "0",
    # Low-odds contracts looked attractive to the model but lost as a group in
    # the first post-fallback wave. Treat <=20c executable asks as late-window
    # lottery tails for default fake-money training too.
    "BTC_PROB_MIN_POLY_PRICE": "0.20",
    "BTC_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE": "0",
    "BTC_CLOB_GAMMA_GAP_OVERRIDE_PROB": "0.90",
    # Lab workers must not fight the main trainer or reload global params.
    "BTC_FAST_GA_ENABLED": "0",
    "BTC_DISABLE_PARAM_RELOAD": "1",
    # Current trained model passed offline backtest but failed fresh paper.
    # Keep default lab in heuristic exploration mode until clean holdout proves
    # the model edge live; model-specific labs can override this explicitly.
    "BTC_PROB_GATE_DISABLED": "1",
    # Poison pill: live gate expects the exact canary ACK, not this value.
    "BTC_LIVE_CANARY_ACK": "PAPER_LAB_NO_LIVE",
}

PROFILE_ENV_KEYS = (
    "BTC_PAPER_EXPLORATION",
    "BTC_ALLOW_PAPER_EXPLORATION",
    "BTC_PROB_GATE_DISABLED",
    "BTC_IGNORE_MODEL_PROBATION",
    "BTC_PROB_MODEL_PATH",
    "BTC_PROB_EDGE_THRESHOLD",
    "BTC_PROB_SHRINK",
    "BTC_FORCE_DELTA_THRESHOLD",
    "BTC_DELTA_HINT_SIGNAL",
    "BTC_DELTA_HINT_MIN_CONF",
)

LIVE_CREDENTIAL_ENV_KEYS = (
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_FUNDER_ADDRESS",
    "POLYMARKET_SIGNATURE_TYPE",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLY_KEY",
    "POLY_SECRET",
    "CLOB_API_KEY",
    "CLOB_API_SECRET",
    "CLOB_API_PASSPHRASE",
)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    delta_thresh: float
    conf_thresh: float
    ens_thresh: float
    spend_ratio: float = 0.15
    max_bet_pct: float = 0.20
    risk_halt_drawdown_pct: float = 0.20
    max_open: int = 1
    cohort: str = "baseline"
    require_clob_quote: bool = True
    loop_sleep: float | None = None
    market_check: float | None = None
    status_seconds: float = 30.0

    def overrides(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "delta_thresh": self.delta_thresh,
            "conf_thresh": self.conf_thresh,
            "ens_thresh": self.ens_thresh,
            "spend_ratio": self.spend_ratio,
            "max_bet_pct": self.max_bet_pct,
        }


STRATEGIES: list[StrategySpec] = [
    StrategySpec("quality_strict", 18.0, 0.70, 0.72, 0.10, 0.15, 0.12, cohort="quality"),
    StrategySpec("quality_mid", 15.0, 0.62, 0.62, 0.12, 0.18, 0.16, cohort="quality"),
    StrategySpec("quality_mid_clob_guard", 14.0, 0.60, 0.58, 0.12, 0.16, 0.14, cohort="quality"),
    StrategySpec("balanced_contract", 11.8, 0.45, 0.50, 0.15, 0.20, 0.20, cohort="balanced"),
    StrategySpec("momentum_ensemble", 12.0, 0.52, 0.78, 0.12, 0.18, 0.16, cohort="momentum"),
    StrategySpec("confidence_hunter", 10.0, 0.82, 0.40, 0.10, 0.15, 0.12, cohort="confidence"),
    StrategySpec("aggressive_clob_floor", 8.0, 0.45, 0.30, 0.16, 0.20, 0.20, cohort="aggressive"),
    StrategySpec(
        "firehose_gamma_probe",
        8.0,
        0.45,
        0.30,
        0.18,
        0.20,
        0.20,
        cohort="firehose",
        require_clob_quote=False,
        loop_sleep=2.0,
        market_check=5.0,
        status_seconds=15.0,
    ),
    StrategySpec("slow_high_signal", 24.0, 0.58, 0.84, 0.10, 0.15, 0.12, cohort="breakout"),
    StrategySpec("mid_fast_mix", 13.0, 0.48, 0.42, 0.14, 0.20, 0.18, cohort="mixed"),
    StrategySpec(
        "firehose_confidence_flip",
        9.0,
        0.45,
        0.85,
        0.16,
        0.20,
        0.20,
        cohort="firehose",
        require_clob_quote=False,
        loop_sleep=2.0,
        market_check=5.0,
        status_seconds=15.0,
    ),
    StrategySpec(
        "firehose_clob_quality",
        9.0,
        0.50,
        0.35,
        0.14,
        0.18,
        0.16,
        cohort="firehose",
        require_clob_quote=True,
        loop_sleep=2.0,
        market_check=5.0,
        status_seconds=15.0,
    ),
    # Winner canaries showed that firehose_gamma_probe and
    # firehose_confidence_flip are the only variants producing enough positive
    # sample. These guarded clones force an executable CLOB quote and use smaller
    # bet/DD limits so we can separate real executable edge from Gamma fallback
    # optimism before any live-money discussion.
    StrategySpec(
        "firehose_gamma_clob_guard",
        8.0,
        0.45,
        0.30,
        0.14,
        0.16,
        0.14,
        cohort="firehose_guard",
        require_clob_quote=True,
        loop_sleep=2.0,
        market_check=5.0,
        status_seconds=15.0,
    ),
    StrategySpec(
        "firehose_confidence_clob_guard",
        9.0,
        0.45,
        0.85,
        0.14,
        0.16,
        0.14,
        cohort="firehose_guard",
        require_clob_quote=True,
        loop_sleep=2.0,
        market_check=5.0,
        status_seconds=15.0,
    ),
    # Quarantine slots: available for explicit --n > 12 experiments, not part of
    # the default 12-worker army because live paper evidence showed either no
    # trades or early drawdown halts.
    StrategySpec("tight_risk_probe", 8.0, 0.55, 0.45, 0.10, 0.12, 0.10, cohort="quarantine"),
    StrategySpec("wide_delta_loose_ens", 22.0, 0.45, 0.30, 0.10, 0.15, 0.08, cohort="quarantine"),
]


def _now_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_run_dir() -> Path | None:
    if not LAB_ROOT.exists():
        return None
    runs = [p for p in LAB_ROOT.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    return sorted(runs)[-1] if runs else None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _journal_status_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose optional status fields already emitted by btc_paper_fast journals."""
    out: dict[str, Any] = {}
    if not rows:
        return out

    # Match btc_live_go_nogo.py exactly: a trade is live-valid only when the
    # executable fill price came from the CLOB ask-depth quote. A Gamma fallback
    # row may still carry a token id, but that is not enough for live-canary
    # promotion.
    live_rows = [r for r in rows if r.get("price_source") == "clob_book_ask_depth"]
    gamma_rows = [r for r in rows if r.get("price_source") == "gamma_outcomePrices_fallback"]
    if live_rows:
        live_wins = sum(1 for r in live_rows if r.get("won") is True)
        live_losses = sum(1 for r in live_rows if r.get("won") is False)
        live_pnl = sum(float(r.get("pnl") or 0) for r in live_rows)
        out["live_valid_trades"] = len(live_rows)
        out["live_valid_wins"] = live_wins
        out["live_valid_losses"] = live_losses
        out["live_valid_wr"] = live_wins / len(live_rows)
        out["live_valid_pnl"] = live_pnl
        out["clob_ratio"] = len(live_rows) / len(rows)
    else:
        out["live_valid_trades"] = 0
        out["live_valid_wins"] = 0
        out["live_valid_losses"] = 0
        out["live_valid_wr"] = 0.0
        out["live_valid_pnl"] = 0.0
        out["clob_ratio"] = 0.0
    if gamma_rows:
        out["gamma_fallback_trades"] = len(gamma_rows)
        out["gamma_fallback_pnl"] = sum(float(r.get("pnl") or 0) for r in gamma_rows)

    price_sources = sorted({str(r["price_source"]) for r in rows if r.get("price_source")})
    if price_sources:
        out["price_sources"] = price_sources
        out["last_price_source"] = next((r.get("price_source") for r in reversed(rows) if r.get("price_source")), None)

    clob_trades = sum(1 for r in rows if r.get("price_source") == "clob_book_ask_depth")
    if clob_trades:
        out["clob_trades"] = clob_trades

    decisions = [r.get("prob_decision") for r in rows if isinstance(r.get("prob_decision"), dict)]
    edges = [_float_or_none(d.get("edge")) for d in decisions]
    edges = [e for e in edges if e is not None]
    if edges:
        out["avg_edge"] = sum(edges) / len(edges)
        out["last_edge"] = edges[-1]

    if decisions:
        last_decision = decisions[-1]
        for src, dst in (
            ("model_prob", "last_model_prob"),
            ("market_prob", "last_market_prob"),
            ("prob_up", "last_prob_up"),
            ("prob_down", "last_prob_down"),
            ("gate_reason", "last_gate_reason"),
        ):
            value = last_decision.get(src)
            if value is not None:
                out[dst] = value

    for src, dst in (
        ("external_bull_score", "last_external_bull_score"),
        ("exploration_mode", "last_exploration_mode"),
        ("exploration_reason", "last_exploration_reason"),
        ("clob_best_ask", "last_clob_best_ask"),
        ("clob_slippage_bps", "last_clob_slippage_bps"),
    ):
        value = next((r.get(src) for r in reversed(rows) if r.get(src) is not None), None)
        if value is not None:
            out[dst] = value

    return out


def _tail_status(path: Path) -> tuple[str, bool]:
    if not path.exists():
        return "no log yet", False
    halted = False
    last = ""
    try:
        lines = path.read_text(errors="ignore").splitlines()[-250:]
    except Exception:
        return "log unreadable", False
    for line in lines:
        if "RISK HALT" in line or " HALT=" in line:
            halted = True
        if " elapsed=" in line or "FINAL:" in line or "BET " in line or "RESOLVED" in line:
            last = line
    return last or (lines[-1] if lines else "empty log"), halted


def summarize_strategy(entry: dict[str, Any]) -> dict[str, Any]:
    journal = Path(entry["journal_file"])
    log_file = Path(entry["log_file"])
    rows = _read_jsonl(journal)
    pnl = sum(float(r.get("pnl") or 0) for r in rows)
    wins = sum(1 for r in rows if r.get("won") is True)
    losses = sum(1 for r in rows if r.get("won") is False)
    capital = float(entry.get("capital") or 20.0)
    equity = capital
    peak = capital
    max_dd = 0.0
    for r in rows:
        equity += float(r.get("pnl") or 0)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    last, halted = _tail_status(log_file)
    pid = int(entry.get("pid") or 0)
    summary = {
        "name": entry.get("name"),
        "pid": pid,
        "alive": _pid_alive(pid),
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(rows) if rows else 0.0,
        "pnl": pnl,
        "equity": equity,
        "max_dd": max_dd,
        "halted": halted,
        "last": last[-180:],
        "journal_file": str(journal),
        "log_file": str(log_file),
    }
    summary.update(_journal_status_fields(rows))
    return summary


def _format_status_fields(row: dict[str, Any]) -> str:
    parts = []
    if row.get("last_price_source"):
        parts.append(f"src={row['last_price_source']}")
    if row.get("live_valid_trades") is not None:
        parts.append(
            f"live={int(row.get('live_valid_trades') or 0)} "
            f"vPnL={float(row.get('live_valid_pnl') or 0):+.2f}"
        )
    if row.get("clob_trades") is not None:
        parts.append(f"clob={row['clob_trades']}")
    if row.get("last_edge") is not None:
        parts.append(f"edge={float(row['last_edge']):+.1%}")
    if row.get("last_gate_reason"):
        parts.append(f"gate={row['last_gate_reason']}")
    return (" | " + " ".join(parts)) if parts else ""


def report(run_dir: Path, json_out: bool = False) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [summarize_strategy(e) for e in manifest.get("strategies", [])]
    if json_out:
        print(json.dumps({"run_dir": str(run_dir), "strategies": rows}, indent=2))
        return rows
    print(f"run_dir={run_dir}")
    print("strategy                 pid     alive trades  W/L    WR    pnl    live vPnL    eq    maxDD halt")
    print("-" * 106)
    for r in rows:
        print(
            f"{r['name']:<24} {r['pid']:<7} {str(r['alive']):<5} "
            f"{r['trades']:<6} {r['wins']}/{r['losses']:<4} {r['wr']*100:>5.1f}% "
            f"{r['pnl']:>+7.2f} {int(r.get('live_valid_trades') or 0):>5} "
            f"{float(r.get('live_valid_pnl') or 0):>+7.2f} "
            f"{r['equity']:>6.2f} {r['max_dd']*100:>6.1f}% {str(r['halted']):<5}"
        )
    print("\nlast activity:")
    for r in rows:
        print(f"- {r['name']}: {r['last']}{_format_status_fields(r)}")
    return rows


def build_worker_env(
    base_env: dict[str, str],
    spec: StrategySpec,
    args: argparse.Namespace,
    log_dir: Path,
    log_file: Path,
    journal_file: Path,
    params_file: Path,
) -> dict[str, str]:
    env = dict(base_env)
    for key in LIVE_CREDENTIAL_ENV_KEYS:
        env.pop(key, None)

    profile = getattr(args, "profile", "explore")
    model_path = Path(
        getattr(
            args,
            "model_path",
            DATA_DIR / "model" / "btc_prob_model_candidate_best.pkl",
        )
    )
    if not model_path.exists():
        model_path = DATA_DIR / "model" / "btc_prob_model_current.pkl"
    model_edge = float(getattr(args, "model_edge", 0.02))
    prob_shrink = float(getattr(args, "prob_shrink", 0.65))

    loop_sleep = spec.loop_sleep if spec.loop_sleep is not None else args.loop_sleep
    market_check = spec.market_check if spec.market_check is not None else args.market_check
    env.update(
        {
            **SAFE_PAPER_ENV,
            "BTC_STRATEGY_NAME": spec.name,
            "BTC_LAB_COHORT": spec.cohort,
            "BTC_LOG_DIR": str(log_dir),
            "BTC_PAPER_LOG_FILE": str(log_file),
            "BTC_JOURNAL_FILE": str(journal_file),
            "BTC_BEST_PARAMS_FILE": str(params_file),
            "BTC_PARAM_OVERRIDES_JSON": json.dumps(spec.overrides()),
            "BTC_REQUIRE_CLOB_QUOTE": "1" if spec.require_clob_quote else "0",
            "BTC_MAX_OPEN_BTC_TRADES": str(spec.max_open),
            "BTC_RISK_HALT_DRAWDOWN_PCT": str(spec.risk_halt_drawdown_pct),
            "BTC_PAPER_CAPITAL": str(args.capital),
            # Parallel workers intentionally poll slower than the main runner by
            # default. Firehose specs opt into their own faster fake-money cadence.
            "BTC_LOOP_SLEEP_SECONDS": str(loop_sleep),
            "BTC_MARKET_CHECK_SECONDS": str(market_check),
            "BTC_STATUS_SECONDS": str(spec.status_seconds),
        }
    )
    # Manual swarm launches can widen fake-money exposure without changing the
    # baked strategy table. Keep the knobs lab-scoped so launchd/main settings
    # do not accidentally alter every historical strategy.
    for source_key, target_key in (
        ("BTC_LAB_MAX_OPEN_BTC_TRADES", "BTC_MAX_OPEN_BTC_TRADES"),
        ("BTC_LAB_MAX_OPEN_BTC_TRADES_PER_TF", "BTC_MAX_OPEN_BTC_TRADES_PER_TF"),
        ("BTC_LAB_MAX_TOTAL_BTC_EXPOSURE_PCT", "BTC_MAX_TOTAL_BTC_EXPOSURE_PCT"),
        ("BTC_LAB_ALLOW_CROSS_TF_CORRELATED_OPEN", "BTC_ALLOW_CROSS_TF_CORRELATED_OPEN"),
        ("BTC_LAB_PROB_MIN_POLY_PRICE", "BTC_PROB_MIN_POLY_PRICE"),
        ("BTC_LAB_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE", "BTC_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE"),
        ("BTC_LAB_CLOB_GAMMA_GAP_OVERRIDE_PROB", "BTC_CLOB_GAMMA_GAP_OVERRIDE_PROB"),
        ("BTC_LAB_REQUIRE_CLOB_QUOTE", "BTC_REQUIRE_CLOB_QUOTE"),
        ("BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT", "BTC_REQUIRE_EXTERNAL_FLOW_AGREEMENT"),
        ("BTC_LAB_EXTERNAL_BULL_SCORE_MIN", "BTC_EXTERNAL_BULL_SCORE_MIN"),
        ("BTC_LAB_EXTERNAL_BULL_SCORE_MAX", "BTC_EXTERNAL_BULL_SCORE_MAX"),
        ("BTC_LAB_PROB_HARD_POLY_CAP", "BTC_PROB_HARD_POLY_CAP"),
        ("BTC_LAB_PROB_POLY_PRICE_CEILING", "BTC_PROB_POLY_PRICE_CEILING"),
    ):
        if base_env.get(source_key):
            env[target_key] = base_env[source_key]
    if profile == "model_gate":
        # Fake-money canary that tests the trained probability model directly.
        # No heuristic exploration fallback, no live creds, no production
        # probation mutation. BTC_IGNORE_MODEL_PROBATION is scoped to this
        # isolated paper worker only so we can validate the candidate safely.
        env.update(
            {
                "BTC_PAPER_EXPLORATION": "0",
                "BTC_ALLOW_PAPER_EXPLORATION": "0",
                "BTC_PROB_GATE_DISABLED": "0",
                "BTC_IGNORE_MODEL_PROBATION": "1",
                "BTC_PROB_MODEL_PATH": str(model_path),
                "BTC_PROB_EDGE_THRESHOLD": str(model_edge),
                "BTC_PROB_SHRINK": str(prob_shrink),
                "BTC_FORCE_DELTA_THRESHOLD": str(getattr(args, "model_delta", 1.0)),
                "BTC_DELTA_HINT_SIGNAL": "1",
                "BTC_DELTA_HINT_MIN_CONF": "0.45",
            }
        )
    elif profile != "explore":
        raise SystemExit(f"unknown --profile: {profile}")
    return env


def launch(args: argparse.Namespace) -> Path:
    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LAB_ROOT / f"{_now_label()}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = STRATEGIES[: args.n]
    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        by_name = {spec.name: spec for spec in STRATEGIES}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(f"unknown strategy in --only: {', '.join(missing)}")
        selected = [by_name[name] for name in wanted]
    strategies_manifest: list[dict[str, Any]] = []

    for idx, spec in enumerate(selected):
        strat_dir = run_dir / f"{idx+1:02d}_{spec.name}"
        strat_dir.mkdir(parents=True, exist_ok=True)
        log_dir = strat_dir / "logs"
        state_dir = strat_dir / "state"
        log_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        params_file = state_dir / "sniper_best_params.json"
        journal_file = state_dir / "intraday_journal.jsonl"
        log_file = log_dir / "btc_sniper_paper_fast.log"
        params_file.write_text(
            json.dumps(
                {
                    **spec.overrides(),
                    "strategy_version": "btc-window-sniper-frozen-v1",
                    "mode": "paper",
                    "generation": int(time.time()),
                    "source": "btc_parallel_paper_lab.seed",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        env = build_worker_env(os.environ.copy(), spec, args, log_dir, log_file, journal_file, params_file)
        out_path = strat_dir / "supervisor.out"
        out_f = out_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            [str(PYTHON), str(PAPER_RUNNER), str(args.duration), "--parallel-lab"],
            cwd=str(SRC_DIR),
            env=env,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (strat_dir / "pid").write_text(str(proc.pid) + "\n")
        strategies_manifest.append(
            {
                "name": spec.name,
                "pid": proc.pid,
                "directory": str(strat_dir),
                "journal_file": str(journal_file),
                "log_file": str(log_file),
                "params_file": str(params_file),
                "supervisor_out": str(out_path),
                "capital": args.capital,
                "duration_seconds": args.duration,
                "spec": asdict(spec),
                "cohort": spec.cohort,
                "paper_only_env": {key: env[key] for key in sorted(SAFE_PAPER_ENV)},
                "profile": args.profile,
                "profile_env": {key: env[key] for key in PROFILE_ENV_KEYS if key in env},
                "require_clob_quote": spec.require_clob_quote,
            }
        )
        time.sleep(args.stagger)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "paper_only": True,
        "runner": str(PAPER_RUNNER),
        "duration_seconds": args.duration,
        "capital": args.capital,
        "loop_sleep": args.loop_sleep,
        "market_check": args.market_check,
        "strategies": strategies_manifest,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (LAB_ROOT / "LATEST").write_text(str(run_dir) + "\n")
    print(str(run_dir))
    return run_dir


def stop(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for entry in manifest.get("strategies", []):
        pid = int(entry.get("pid") or 0)
        if _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
            print(f"stopped {entry.get('name')} pid={pid}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC paper parallel strategy lab")
    p.add_argument("--label", default="paper_lab", help="run label suffix")
    p.add_argument("--n", type=int, default=8, choices=range(1, len(STRATEGIES) + 1), metavar=f"1..{len(STRATEGIES)}")
    p.add_argument("--only", default="", help="comma-separated strategy names; overrides --n")
    p.add_argument("--profile", choices=("explore", "model_gate"), default="explore")
    p.add_argument(
        "--model-path",
        default=str(DATA_DIR / "model" / "btc_prob_model_candidate_best.pkl"),
        help="model pickle for --profile model_gate; falls back to current model if missing",
    )
    p.add_argument("--model-edge", type=float, default=0.02, help="edge threshold for --profile model_gate")
    p.add_argument("--model-delta", type=float, default=1.0, help="BTC delta trigger for --profile model_gate")
    p.add_argument("--prob-shrink", type=float, default=0.65, help="probability shrink factor for --profile model_gate")
    p.add_argument("--duration", type=int, default=8 * 3600)
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--loop-sleep", type=float, default=5.0)
    p.add_argument("--market-check", type=float, default=15.0)
    p.add_argument("--stagger", type=float, default=1.5)
    p.add_argument("--report", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--run-dir", default="latest")
    return p.parse_args()


def resolve_run_dir(raw: str) -> Path:
    if raw == "latest":
        latest_file = LAB_ROOT / "LATEST"
        if latest_file.exists():
            return Path(latest_file.read_text().strip())
        latest = _latest_run_dir()
        if latest:
            return latest
        raise SystemExit("no parallel lab run found")
    return Path(os.path.expanduser(raw)).resolve()


def main() -> int:
    args = parse_args()
    if args.stop:
        stop(resolve_run_dir(args.run_dir))
        return 0
    if args.report:
        report(resolve_run_dir(args.run_dir), json_out=args.json)
        return 0
    run_dir = launch(args)
    # Immediate liveness smoke.
    time.sleep(2)
    report(run_dir, json_out=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
