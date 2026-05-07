#!/usr/bin/env python3
"""BTC Polymarket paper-trader Gym harness.

Purpose: send the trading app "to gym" on demand and get:
- current paper/live-readiness score
- concrete improvement backlog
- evidence paths
- machine-readable JSON + human Markdown report

No orders. No wallet. Read-only except reports + optional Gym flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from btc_fee_model import resolved_buy_pnl
from btc_param_contract import DYNAMIC_PARAM_FIELDS, read_params_file


HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
REPORT_DIR = HARVEY_HOME / "data" / "reports" / "gym" / "btc-polymarket-trader"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
FREEZE_FILE = STATE_DIR / "freezes" / "latest.json"
RUN_UNTIL_FILE = STATE_DIR / "btc_paper_fast_run_until.ts"
CHILD_LOG = LOG_DIR / "btc_paper_fast_watchdog_child.log"


def _now() -> datetime:
    return datetime.now()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _trade_pnl(row: dict[str, Any]) -> float:
    try:
        if "size" in row and "poly_price" in row:
            return resolved_buy_pnl(bool(row.get("won")), float(row["size"]), float(row["poly_price"]))
        price = float(row.get("poly_price", 0.5) or 0.5)
        spend = float(row.get("spend", 0.0) or 0.0)
        shares = spend / price if price > 0 else 0.0
        return resolved_buy_pnl(bool(row.get("won")), shares, price)
    except Exception:
        return float(row.get("pnl") or 0.0)


def load_trades(start: datetime | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not JOURNAL_FILE.exists():
        return rows
    for line in JOURNAL_FILE.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("mode") != "paper" or "btc_delta" not in row:
            continue
        if int(row.get("window_tf", 0) or 0) not in (5, 15):
            continue
        placed = _parse_dt(row.get("placed_at"))
        if start and placed and placed < start:
            continue
        rows.append(row)
    return rows


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for r in rows if r.get("won") is True)
    losses = sum(1 for r in rows if r.get("won") is False)
    pnl = sum(_trade_pnl(r) for r in rows)
    pnls = [_trade_pnl(r) for r in rows]
    wr = wins / len(rows) if rows else 0.0
    equity = 20.0
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    mean = sum(pnls) / len(pnls) if pnls else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in pnls) / max(len(pnls) - 1, 1)) if len(pnls) > 1 else 0.0
    sharpe_like = mean / std * math.sqrt(min(len(pnls), 50)) if std else 0.0
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "pnl": pnl,
        "max_drawdown": max_dd,
        "sharpe_like": sharpe_like,
    }


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def launchctl_labels() -> dict[str, bool]:
    labels = [
        "com.makakoo.arbitrage.btcpaperfast.watchdog",
        "com.makakoo.arbitrage.parallelresearch",
        "com.makakoo.arbitrage.btc.telegramreporter",
        "com.makakoo.arbitrage.readiness",
        "com.makakoo.arbitrage.autoimprove",
    ]
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        out = ""
    return {label: (label in out) for label in labels}


def latest_status_line() -> str:
    if not CHILD_LOG.exists():
        return "no child log"
    lines = CHILD_LOG.read_text(errors="replace").splitlines()[-400:]
    for line in reversed(lines):
        if "elapsed=" in line and "Eq=$" in line:
            return line
    return lines[-1] if lines else "empty child log"


def scan_order_safety() -> dict[str, Any]:
    paper = (SRC_DIR / "btc_paper_fast.py").read_text(errors="replace")
    orchestrator = (SRC_DIR / "btc_parallel_backtest_orchestrator.py").read_text(errors="replace")
    risky = []
    for pat in [r"requests\\.post", r"create_order", r"submit_order", r"place_order", r"py_clob_client"]:
        if re.search(pat, paper, flags=re.I):
            risky.append(pat)
    return {
        "paper_trader_uses_live_market_data": all(s in paper for s in ["BINANCE_REST", "GAMMA_API", "requests.get"]),
        "paper_trader_uses_clob_executable_fills": all(s in paper for s in ["CLOB_API", "fetch_clob_buy_quote", "clob_book_ask_depth"]),
        "optimizer_uses_chronological_holdout": all(s in orchestrator for s in ["split_train_holdout", "holdout_pass", "BTC_JOURNAL_FILE"]),
        "paper_trader_order_post_patterns": risky,
        "paper_trader_no_real_order_posts": not risky,
    }


def build_report() -> dict[str, Any]:
    freeze = _load_json(FREEZE_FILE, {}) or {}
    freeze_start = _parse_dt(freeze.get("created_at"))
    params = read_params_file(PARAMS_FILE, mode="paper") if PARAMS_FILE.exists() else {}
    all_rows = load_trades(None)
    clean_rows = load_trades(freeze_start)
    all_s = stats(all_rows)
    clean_s = stats(clean_rows)
    all_clob_fills = sum(1 for r in all_rows if r.get("price_source") == "clob_book_ask_depth")
    clean_clob_fills = sum(1 for r in clean_rows if r.get("price_source") == "clob_book_ask_depth")
    labels = launchctl_labels()
    run_until = None
    if RUN_UNTIL_FILE.exists():
        try:
            run_until = datetime.fromtimestamp(int(float(RUN_UNTIL_FILE.read_text().strip()))).isoformat()
        except Exception:
            pass
    safety = scan_order_safety()
    clean_wilson = wilson_lower(clean_s["wins"], clean_s["trades"])

    gates = {
        "paper_real_market_data": safety["paper_trader_uses_live_market_data"],
        "paper_no_real_order_posts": safety["paper_trader_no_real_order_posts"],
        "paper_clob_executable_fill_code": safety["paper_trader_uses_clob_executable_fills"],
        "trader_running": bool(pgrep(str(SRC_DIR / "btc_paper_fast.py"))),
        "watchdog_running": bool(pgrep(str(SRC_DIR / "btc_paper_fast_watchdog.py"))),
        "parallelresearch_enabled": labels["com.makakoo.arbitrage.parallelresearch"],
        "optimizer_holdout_gate": safety["optimizer_uses_chronological_holdout"],
        "telegram_enabled": labels["com.makakoo.arbitrage.btc.telegramreporter"],
        "clean_min_trades_80": clean_s["trades"] >= 80,
        "clean_clob_executable_fills": clean_s["trades"] > 0 and clean_clob_fills == clean_s["trades"],
        "clean_wilson_55": clean_wilson >= 0.55,
        "clean_positive_pnl": clean_s["pnl"] > 0,
        "clean_drawdown_25": clean_s["max_drawdown"] <= 0.25,
        "readiness_total_500": all_s["trades"] >= 500,
    }

    backlog: list[dict[str, Any]] = []
    if not gates["clean_min_trades_80"]:
        backlog.append({
            "priority": "P0",
            "title": "Collect more clean post-freeze paper trades",
            "why": f"Need 80 clean trades; have {clean_s['trades']}. No live canary before this gate.",
            "action": "Keep 6h validation running until enough 5m/15m resolved trades exist.",
        })
    if not gates["clean_wilson_55"]:
        backlog.append({
            "priority": "P0",
            "title": "Improve statistical confidence",
            "why": f"Wilson lower WR is {clean_wilson:.1%}; need >=55%.",
            "action": "Tune thresholds only if validation WR/PnL improves; otherwise keep params stable.",
        })
    if not gates["clean_drawdown_25"]:
        backlog.append({
            "priority": "P0",
            "title": "Reduce drawdown before canary",
            "why": f"Clean max drawdown is {clean_s['max_drawdown']:.1%}; max allowed 25%.",
            "action": "Add drawdown-aware scoring/kill switch in paper before live canary.",
        })
    if not gates["paper_clob_executable_fill_code"]:
        backlog.append({
            "priority": "P1",
            "title": "Paper fills should use CLOB executable prices",
            "why": "Current paper trader uses Gamma outcomePrices; this is real market data but not the exact executable bid/ask/depth path.",
            "action": "Add CLOB orderbook fetch and simulate crossing spread/slippage before live.",
        })
    elif not gates["clean_clob_executable_fills"]:
        backlog.append({
            "priority": "P0",
            "title": "Validate with post-CLOB clean paper trades",
            "why": f"Clean freeze has {clean_clob_fills}/{clean_s['trades']} trades priced from CLOB ask depth.",
            "action": "When current virtual positions are flat, restart trader with new CLOB fill code, create a new freeze, and collect clean validation trades.",
        })
    if not gates["optimizer_holdout_gate"]:
        backlog.append({
            "priority": "P1",
            "title": "Make optimizer train/validation split deterministic",
            "why": "Parallel workers re-score whole journal; stronger Gym should require temporal holdout pass before param deploy.",
            "action": "Deploy candidate only if latest 20% chronological holdout PnL >= 0 and WR lower bound does not degrade.",
        })
    backlog.extend([
        {
            "priority": "P2",
            "title": "Stabilize single training authority",
            "why": "Process should keep parallelresearch as sole optimizer; internal FastGA restart-when-flat is pending if current process still has open virtual positions.",
            "action": "After flat restart, verify btc_paper_fast logs FastGA disabled.",
        },
    ])

    score = 0
    score += 10 if gates["paper_real_market_data"] else 0
    score += 15 if gates["paper_no_real_order_posts"] else 0
    score += 10 if gates["paper_clob_executable_fill_code"] else 0
    score += 10 if gates["trader_running"] and gates["watchdog_running"] else 0
    score += 10 if gates["parallelresearch_enabled"] else 0
    score += 5 if gates["telegram_enabled"] else 0
    score += 15 if gates["clean_positive_pnl"] else 0
    score += 10 if gates["clean_drawdown_25"] else 0
    score += 10 if gates["clean_wilson_55"] else 0
    score += 5 if gates["clean_min_trades_80"] else 0

    verdict = "LIVE_CANARY_ALLOWED" if all([
        gates["paper_real_market_data"],
        gates["paper_no_real_order_posts"],
        gates["paper_clob_executable_fill_code"],
        gates["clean_clob_executable_fills"],
        gates["clean_min_trades_80"],
        gates["clean_wilson_55"],
        gates["clean_positive_pnl"],
        gates["clean_drawdown_25"],
    ]) else "KEEP_TRAINING"

    return {
        "generated_at": _now().isoformat(),
        "app": "btc-polymarket-trader",
        "mode": "paper_real_market_data_no_real_orders",
        "score": score,
        "verdict": verdict,
        "run_until": run_until,
        "status_line": latest_status_line(),
        "dynamic_params": {k: params.get(k) for k in DYNAMIC_PARAM_FIELDS},
        "strategy_version": params.get("strategy_version"),
        "freeze": {
            "label": freeze.get("label"),
            "created_at": freeze.get("created_at"),
        },
        "stats": {
            "clean_since_freeze": clean_s | {"wilson_lower_95": clean_wilson, "clob_fills": clean_clob_fills},
            "all_btc_paper": all_s | {"clob_fills": all_clob_fills},
        },
        "gates": gates,
        "safety": safety,
        "backlog": backlog,
        "evidence": {
            "journal": str(JOURNAL_FILE),
            "params": str(PARAMS_FILE),
            "freeze": str(FREEZE_FILE),
            "child_log": str(CHILD_LOG),
            "parallel_log": str(LOG_DIR / "btc_parallel_autoresearch.log"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# BTC Polymarket Trader Gym Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Verdict: **{report['verdict']}**",
        f"- Gym score: **{report['score']}/100**",
        f"- Mode: `{report['mode']}`",
        f"- Run until: `{report.get('run_until')}`",
        "",
        "## Stats",
        "",
        f"- Clean since freeze: `{report['stats']['clean_since_freeze']['trades']}` trades, "
        f"{report['stats']['clean_since_freeze']['wins']}W/{report['stats']['clean_since_freeze']['losses']}L, "
        f"WR `{report['stats']['clean_since_freeze']['wr']:.1%}`, "
        f"Wilson95Lo `{report['stats']['clean_since_freeze']['wilson_lower_95']:.1%}`, "
        f"PnL `${report['stats']['clean_since_freeze']['pnl']:+.2f}`, "
        f"DD `{report['stats']['clean_since_freeze']['max_drawdown']:.1%}`",
        f"- Clean CLOB executable fills: `{report['stats']['clean_since_freeze'].get('clob_fills', 0)}` / `{report['stats']['clean_since_freeze']['trades']}`",
        f"- All BTC paper: `{report['stats']['all_btc_paper']['trades']}` trades, "
        f"{report['stats']['all_btc_paper']['wins']}W/{report['stats']['all_btc_paper']['losses']}L, "
        f"WR `{report['stats']['all_btc_paper']['wr']:.1%}`, "
        f"PnL `${report['stats']['all_btc_paper']['pnl']:+.2f}`, "
        f"CLOB fills `{report['stats']['all_btc_paper'].get('clob_fills', 0)}`",
        "",
        "## Gates",
        "",
    ]
    for k, v in report["gates"].items():
        lines.append(f"- [{'x' if v else ' '}] `{k}`")
    lines.extend(["", "## Improvement backlog", ""])
    for item in report["backlog"]:
        lines.extend([
            f"### {item['priority']} — {item['title']}",
            f"- Why: {item['why']}",
            f"- Action: {item['action']}",
            "",
        ])
    lines.extend(["## Evidence", ""])
    for k, v in report["evidence"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"{stamp}_gym_report.json"
    md_path = REPORT_DIR / f"{stamp}_gym_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    latest_json = REPORT_DIR / "latest.json"
    latest_md = REPORT_DIR / "latest.md"
    latest_json.write_text(json_path.read_text())
    latest_md.write_text(md_path.read_text())
    return json_path, md_path


def flag_to_gym(report: dict[str, Any], md_path: Path) -> bool:
    try:
        sys.path.insert(0, str(HARVEY_HOME / "plugins" / "lib-harvey-core" / "src"))
        from core.gym.flag import add_flag  # type: ignore
        return bool(add_flag(
            reason=(
                "BTC Polymarket trader app gym audit: "
                f"verdict={report['verdict']} score={report['score']}/100; "
                "use backlog to improve paper/live-readiness process before real-money canary."
            ),
            skill="trading/btc-polymarket-agent",
            context=md_path.read_text()[:8192],
            cmd_label="btc_trading_gym audit",
        ))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print JSON report")
    ap.add_argument("--flag-gym", action="store_true", help="also add a Gym manual flag with the backlog")
    args = ap.parse_args()
    report = build_report()
    json_path, md_path = write_reports(report)
    gym_flagged = flag_to_gym(report, md_path) if args.flag_gym else False
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path), "latest": str(REPORT_DIR / "latest.md")}
    report["gym_flagged"] = gym_flagged
    # Rewrite with final paths included.
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "latest.json").write_text(json_path.read_text())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
        print(f"Latest: {REPORT_DIR / 'latest.md'}")
        print(f"Gym flagged: {gym_flagged}")
    return 0 if report["verdict"] in {"KEEP_TRAINING", "LIVE_CANARY_ALLOWED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
