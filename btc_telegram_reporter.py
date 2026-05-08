#!/usr/bin/env python3
"""Send BTC paper-trainer status reports to Telegram.

Default target: Olibia Telegram group from ~/.claude/channels/telegram/access.json.
No trading. Reporting only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
RUN_UNTIL_FILE = STATE_DIR / "btc_paper_fast_run_until.ts"
JOURNAL_FILE = STATE_DIR / "intraday_journal.jsonl"
LIVE_KILL_SWITCH_FILE = STATE_DIR / "live_trading_disabled.json"
FREEZE_LATEST = STATE_DIR / "freezes" / "latest.json"
REPORT_LOG = LOG_DIR / "btc_telegram_reporter.log"
TELEGRAM_STATE_DIR = Path(os.path.expanduser(os.environ.get("TELEGRAM_STATE_DIR", "~/.claude/channels/telegram")))
ACCESS_FILE = TELEGRAM_STATE_DIR / "access.json"
ENV_FILE = TELEGRAM_STATE_DIR / ".env"
CHILD_LOG = LOG_DIR / "btc_paper_fast_watchdog_child.log"
MANAGED_LOG_GLOB = "btc_paper_fast_6h_*"
SPLIT_AGENT_LOG_GLOB = "btc_paper_fast_*m.log"
WATCHDOG_MAIN_MARKER = "--watchdog-main"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        REPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
        if REPORT_LOG.exists() and REPORT_LOG.stat().st_size > 2_000_000:
            REPORT_LOG.replace(LOG_DIR / "btc_telegram_reporter.log.1")
        with REPORT_LOG.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_token() -> str:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    if tok.strip():
        return tok.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"TELEGRAM_BOT_TOKEN missing in env and {ENV_FILE}")


def pick_chat_id() -> str:
    env_chat = os.environ.get("BTC_REPORT_CHAT_ID") or os.environ.get("TELEGRAM_REPORT_CHAT_ID")
    if env_chat:
        return env_chat.strip()
    if ACCESS_FILE.exists():
        data = json.loads(ACCESS_FILE.read_text())
        groups = data.get("groups") or {}
        # Prefer Telegram supergroup/channel id shape.
        for gid in groups:
            if str(gid).startswith("-100"):
                return str(gid)
        for gid in groups:
            return str(gid)
        allow = data.get("allowFrom") or []
        if allow:
            return str(allow[0])
    raise RuntimeError("No Telegram chat id found; set BTC_REPORT_CHAT_ID")


def pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def process_matches(pattern: str) -> list[tuple[int, str]]:
    """Return (pid, command) rows matching pattern.

    `pgrep -f` can only give pid lists on some macOS setups. The reporter needs
    command lines so parallel-lab workers do not masquerade as the main trader.
    """
    rows: list[tuple[int, str]] = []
    try:
        # `eww` includes environment assignments on macOS, needed to report
        # env-scoped canary params instead of stale frozen params.
        out = subprocess.check_output(["ps", "eww", "-axo", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return rows
    for line in out.splitlines():
        line = line.strip()
        if not line or pattern not in line:
            continue
        try:
            pid_raw, cmd = line.split(None, 1)
            pid = int(pid_raw)
        except Exception:
            continue
        if pid == os.getpid():
            continue
        rows.append((pid, cmd))
    return rows


def env_from_command(cmd: str) -> dict[str, str]:
    """Extract launchd/process env assignments from a `ps` command line.

    macOS `ps eww -o command` appends environment assignments after argv. The
    reporter must prefer those live values over the frozen best-params file; the
    latter is often stale while a paper canary is intentionally testing env-
    scoped overrides.
    """
    env: dict[str, str] = {}
    # Plain split intentionally preserves JSON quotes in
    # BTC_PARAM_OVERRIDES_JSON={...}; `shlex.split` would strip them and make
    # the value invalid JSON.
    parts = cmd.split()
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.startswith(("BTC_", "POLYMARKET_")):
            env[key] = value
    return env


def live_trader_config(trader_rows: list[tuple[int, str]], fallback_params: dict) -> dict:
    """Return user-facing config for the active main trader.

    Parallel lab workers are excluded; if no live env is available, falls back
    to the historical best params file.
    """
    main_rows = [(pid, cmd) for pid, cmd in trader_rows if "--parallel-lab" not in cmd]
    main_rows.sort(key=lambda row: (WATCHDOG_MAIN_MARKER not in row[1], row[0]))
    env: dict[str, str] = {}
    pid = None
    cmd = ""
    if main_rows:
        pid, cmd = main_rows[0]
        env = env_from_command(cmd)

    overrides: dict = {}
    raw_overrides = env.get("BTC_PARAM_OVERRIDES_JSON", "")
    if raw_overrides:
        try:
            loaded = json.loads(raw_overrides)
            if isinstance(loaded, dict):
                overrides = loaded
        except Exception:
            overrides = {}

    def pick_float(key: str, default: float = 0.0) -> float:
        try:
            return float(overrides.get(key, fallback_params.get(key, default)))
        except Exception:
            return default

    return {
        "pid": pid,
        "source": "live-env" if env else "params-file",
        "strategy": env.get("BTC_STRATEGY_NAME")
        or str(overrides.get("name") or fallback_params.get("strategy_version") or "?"),
        "delta_thresh": pick_float("delta_thresh"),
        "conf_thresh": pick_float("conf_thresh"),
        "ens_thresh": pick_float("ens_thresh"),
        "edge_threshold": env.get("BTC_PROB_EDGE_THRESHOLD", ""),
        "clob_required": env.get("BTC_REQUIRE_CLOB_QUOTE", ""),
        "risk_halt": env.get("BTC_RISK_HALT_DRAWDOWN_PCT", ""),
        "max_open": env.get("BTC_MAX_OPEN_BTC_TRADES", ""),
        "max_open_tf": env.get("BTC_MAX_OPEN_BTC_TRADES_PER_TF", ""),
        "max_total_exposure": env.get("BTC_MAX_TOTAL_BTC_EXPOSURE_PCT", ""),
        "cross_tf_corr": env.get("BTC_ALLOW_CROSS_TF_CORRELATED_OPEN", ""),
        "prob_shrink": env.get("BTC_PROB_SHRINK", ""),
        "force_delta": env.get("BTC_FORCE_DELTA_THRESHOLD", ""),
    }


def latest_log_path() -> Path | None:
    candidates = [p for p in [CHILD_LOG, LOG_DIR / "btc_sniper_paper_fast.log"] if p.exists()]
    candidates.extend(LOG_DIR.glob(MANAGED_LOG_GLOB))
    candidates.extend(LOG_DIR.glob(SPLIT_AGENT_LOG_GLOB))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None

    def status_mtime(path: Path) -> float:
        try:
            lines = path.read_text(errors="replace").splitlines()[-500:]
            for line in reversed(lines):
                if "elapsed=" in line and "Cash=$" in line and "Eq=$" in line:
                    return path.stat().st_mtime
        except Exception:
            pass
        return 0.0

    # Prefer logs that contain trader status lines. Plain mtime can be polluted
    # by tests/helper probes that write skip-only messages to the shared log.
    return max(candidates, key=lambda p: (status_mtime(p), p.stat().st_mtime))


def latest_status_line() -> str:
    p = latest_log_path()
    if not p:
        return "no trader log yet"
    lines = p.read_text(errors="replace").splitlines()[-250:]
    newest = lines[-1] if lines else ""
    newest_ts = None
    if newest:
        try:
            newest_ts = datetime.strptime(newest.split("]", 1)[0].lstrip("["), "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            newest_ts = p.stat().st_mtime
    for line in reversed(lines):
        if "elapsed=" in line and "Cash=$" in line and "Eq=$" in line:
            try:
                ts_raw = line.split("]", 1)[0].lstrip("[")
                status_ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                status_ts = p.stat().st_mtime
            age = int(time.time() - status_ts)
            if age > 180 and newest and newest_ts and newest_ts > status_ts + 60:
                return f"{newest} [WAITING_FIRST_STATUS previous_status_age={age}s]"
            stale = f" [STALE_LOG age={age}s]" if age > 180 else ""
            return line + stale
    return lines[-1] if lines else "empty trader log"


def load_params() -> dict:
    if PARAMS_FILE.exists():
        return json.loads(PARAMS_FILE.read_text())
    return {}


def run_until_text() -> str:
    if not RUN_UNTIL_FILE.exists():
        return "unknown"
    try:
        ts = int(float(RUN_UNTIL_FILE.read_text().strip()))
        remaining = max(0, ts - int(time.time()))
        return f"{datetime.fromtimestamp(ts).strftime('%H:%M CEST')} ({remaining//60}m left)"
    except Exception:
        return "unknown"


def freeze_start() -> datetime | None:
    try:
        d = json.loads(FREEZE_LATEST.read_text())
        return datetime.fromisoformat(d["created_at"])
    except Exception:
        return None


def journal_stats(
    start: datetime | None = None,
    agent_id: str | None = None,
    window_tf: int | None = None,
    mode: str | None = "paper",
) -> dict:
    rows = []
    paths = [JOURNAL_FILE]
    if mode == "live":
        # Split live agents write separate ledgers. The shared journal may still
        # contain older live rows; include both, but do not count unfilled limit
        # orders as fills/losses.
        paths.extend(sorted(JOURNAL_FILE.parent.glob("intraday_journal_live_*.jsonl")))
    existing = [p for p in paths if p.exists()]
    if not existing:
        return {"n": 0, "wins": 0, "losses": 0, "wr": 0.0, "pnl": 0.0}

    def as_float(value, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except Exception:
            return default

    seen: set[str] = set()
    for path in existing:
        for line in path.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
                dt = datetime.fromisoformat(r.get("placed_at", ""))
            except Exception:
                continue
            if start and dt < start:
                continue
            if agent_id and r.get("agent_id") != agent_id:
                continue
            if window_tf is not None:
                try:
                    if int(r.get("window_tf") or 0) != int(window_tf):
                        continue
                except Exception:
                    continue
            if mode is not None and r.get("mode") != mode:
                continue
            if "btc_delta" not in r:
                continue
            # Split live ledgers store attempted GTC orders too. Unfilled orders
            # are not fills and must not be counted as losses.
            pnl = as_float(r.get("pnl"))
            if mode == "live" and r.get("filled") is False and abs(pnl) < 1e-9:
                continue
            key = str(r.get("order_id") or "") or "|".join(
                str(r.get(k, "")) for k in ("agent_id", "window_start", "window_tf", "direction", "placed_at")
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    wins = sum(1 for r in rows if r.get("won") is True or as_float(r.get("pnl")) > 0)
    losses = sum(1 for r in rows if as_float(r.get("pnl")) < 0 or str(r.get("exit_reason", "")).upper() == "LOSS")
    pnl = sum(float(r.get("pnl") or 0.0) for r in rows)
    return {"n": len(rows), "wins": wins, "losses": losses, "wr": wins / len(rows) if rows else 0.0, "pnl": pnl}


def live_kill_switch_text() -> str:
    if not LIVE_KILL_SWITCH_FILE.exists():
        return ""
    try:
        data = json.loads(LIVE_KILL_SWITCH_FILE.read_text())
    except Exception:
        return f"Live kill switch: ACTIVE ({LIVE_KILL_SWITCH_FILE})"
    reason = str(data.get("reason") or "manual kill switch active")
    balance = data.get("current_clob_balance_usdc")
    open_orders = data.get("open_orders_checked")
    restart = data.get("restart_condition") or "explicit approval required"
    parts = ["Live kill switch: ACTIVE"]
    if balance is not None:
        parts.append(f"CLOB cash_at_stop=${float(balance):.4f}")
    if open_orders is not None:
        parts.append(f"open_orders_at_stop={open_orders}")
    parts.append(f"reason={reason}")
    parts.append(f"restart={restart}")
    return " | ".join(parts)


def _arg_value(cmd: str, name: str) -> str:
    m = re.search(rf"(?:^|\\s){re.escape(name)}\\s+([^\\s]+)", cmd)
    return m.group(1) if m else ""


def split_agent_rows(trader_rows: list[tuple[int, str]]) -> list[dict]:
    rows = []
    for pid, cmd in trader_rows:
        if "--parallel-lab" in cmd:
            continue
        agent_id = _arg_value(cmd, "--agent-id")
        tf_raw = _arg_value(cmd, "--timeframes")
        if not agent_id:
            env = env_from_command(cmd)
            agent_id = env.get("BTC_AGENT_ID", "")
            tf_raw = tf_raw or env.get("BTC_TIMEFRAMES", "")
        if not agent_id or not tf_raw:
            continue
        try:
            tf = int(str(tf_raw).split(",", 1)[0].removesuffix("m"))
        except Exception:
            continue
        log_raw = _arg_value(cmd, "--log-file")
        log_path = Path(log_raw) if log_raw else LOG_DIR / f"btc_paper_fast_{tf}m.log"
        rows.append({"pid": pid, "cmd": cmd, "agent_id": agent_id, "tf": tf, "log": log_path})
    rows.sort(key=lambda r: (r["tf"], r["agent_id"]))
    return rows


def agent_run_until_text(tf: int) -> str:
    path = STATE_DIR / f"btc_paper_fast_{tf}m_run_until.ts"
    if not path.exists():
        return "unknown"
    try:
        ts = int(float(path.read_text().strip()))
        remaining = max(0, ts - int(time.time()))
        return f"{datetime.fromtimestamp(ts).strftime('%H:%M CEST')} ({remaining//60}m left)"
    except Exception:
        return "unknown"


def status_line_for_log(path: Path) -> str:
    if not path.exists():
        return "no trader log yet"
    lines = path.read_text(errors="replace").splitlines()[-250:]
    for line in reversed(lines):
        if "elapsed=" in line and "Cash=$" in line and "Eq=$" in line:
            try:
                ts_raw = line.split("]", 1)[0].lstrip("[")
                status_ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                status_ts = path.stat().st_mtime
            age = int(time.time() - status_ts)
            stale = f" [STALE_LOG age={age}s]" if age > 180 else ""
            return line + stale
    return lines[-1] if lines else "empty trader log"


def split_agent_report_lines(agent_rows: list[dict], start: datetime | None) -> list[str]:
    if not agent_rows:
        return []
    lines = ["Split agents:"]
    for row in agent_rows:
        since = journal_stats(start, agent_id=row["agent_id"], window_tf=row["tf"])
        lines.append(
            f"- {row['agent_id']} {row['tf']}m: OK pid={row['pid']} run_until={agent_run_until_text(row['tf'])} "
            f"since_freeze={since['n']} trades {since['wins']}W/{since['losses']}L WR={since['wr']:.1%} PnL=${since['pnl']:+.2f}"
        )
        lines.append(f"  {status_line_for_log(row['log'])}")
    return lines


def live_agent_report_lines(live_rows: list[tuple[int, str]], start: datetime | None) -> list[str]:
    rows = [(pid, cmd) for pid, cmd in live_rows if "--live" in cmd and "--parallel-lab" not in cmd]
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    lines = ["Live agents:"]
    for pid, cmd in rows:
        tf_raw = _arg_value(cmd, "--timeframes") or env_from_command(cmd).get("BTC_TIMEFRAMES", "?")
        try:
            tf = int(str(tf_raw).split(",", 1)[0].removesuffix("m"))
        except Exception:
            tf = None
        stats = journal_stats(start, window_tf=tf, mode="live") if tf else journal_stats(start, mode="live")
        lines.append(
            f"- live tf={tf_raw}m pid={pid}: since_freeze={stats['n']} fills "
            f"{stats['wins']}W/{stats['losses']}L WR={stats['wr']:.1%} PnL=${stats['pnl']:+.2f}"
        )
    return lines


def build_report(test: bool = False) -> str:
    params = load_params()
    trader_rows = process_matches(str(SRC_DIR / "btc_paper_fast.py"))
    live_rows = process_matches(str(SRC_DIR / "btc_sniper_live.py"))
    trader_pids = [pid for pid, cmd in trader_rows if "--parallel-lab" not in cmd]
    live_pids = [pid for pid, cmd in live_rows if "--live" in cmd and "--parallel-lab" not in cmd]
    lab_pids = [pid for pid, cmd in trader_rows if "--parallel-lab" in cmd]
    wd_pids = [pid for pid, _ in process_matches(str(SRC_DIR / "btc_paper_fast_watchdog.py"))]
    start = freeze_start()
    since = journal_stats(start)
    all_stats = journal_stats(None)
    status = latest_status_line()
    cfg = live_trader_config(trader_rows, params)
    agent_lines = split_agent_report_lines(split_agent_rows(trader_rows), start)
    live_lines = live_agent_report_lines(live_rows, start)
    agent_block = ("\n" + "\n".join(agent_lines)) if agent_lines else ""
    live_block = ("\n" + "\n".join(live_lines)) if live_lines else ""
    run_until = "split-managed (see Split agents)" if agent_lines else run_until_text()
    live_stats = journal_stats(start, mode="live")
    all_live_stats = journal_stats(None, mode="live")
    kill_text = live_kill_switch_text()
    live_label = "Local live journal (wallet audit pending)" if kill_text else "Live"
    all_live_label = "All local live journal" if kill_text else "All BTC live"
    if live_pids:
        mode_line = "PAPER REPORT + LIVE AGENTS RUNNING"
    elif kill_text:
        mode_line = "PAPER ONLY — LIVE KILL SWITCH ACTIVE"
    else:
        mode_line = "PAPER ONLY, no real orders"
    prefix = "TEST " if test else ""
    return (
        f"{prefix}BTC paper trainer report — {datetime.now().strftime('%H:%M CEST')}\n"
        f"Mode: {mode_line}\n"
        f"Trader: {'OK' if trader_pids else 'DOWN'} {trader_pids[:2]} | Live: {'OK' if live_pids else 'DOWN'} {live_pids[:2]} | Watchdog: {'OK' if wd_pids else 'DOWN'} {wd_pids[:2]} | Lab: {len(lab_pids)} workers\n"
        f"Run until: {run_until}\n"
        f"Params: delta={float(cfg.get('delta_thresh',0)):.3f} conf={float(cfg.get('conf_thresh',0)):.3f} ens={float(cfg.get('ens_thresh',0)):.3f} ({cfg.get('source')})\n"
        f"Live cfg: strategy={cfg.get('strategy')} edge={cfg.get('edge_threshold') or '?'} clob={cfg.get('clob_required') or '?'} dd={cfg.get('risk_halt') or '?'} open={cfg.get('max_open') or '?'} tf_cap={cfg.get('max_open_tf') or '?'} exp={cfg.get('max_total_exposure') or '?'} cross_tf={cfg.get('cross_tf_corr') or '0'} shrink={cfg.get('prob_shrink') or '?'} force_delta={cfg.get('force_delta') or '?'}\n"
        f"Score: {float(params.get('best_score',0)):.2f} | Frozen strategy: {params.get('strategy_version','?')}\n"
        f"Current status: {status}\n"
        f"{kill_text + chr(10) if kill_text else ''}"
        f"{agent_block}\n"
        f"{live_block}\n"
        f"Since freeze: {since['n']} trades, {since['wins']}W/{since['losses']}L, WR={since['wr']:.1%}, PnL=${since['pnl']:+.2f}\n"
        f"{live_label} since freeze: {live_stats['n']} fills, {live_stats['wins']}W/{live_stats['losses']}L, WR={live_stats['wr']:.1%}, PnL=${live_stats['pnl']:+.2f}\n"
        f"All BTC paper: {all_stats['n']} trades, {all_stats['wins']}W/{all_stats['losses']}L, WR={all_stats['wr']:.1%}, PnL=${all_stats['pnl']:+.2f}\n"
        f"{all_live_label}: {all_live_stats['n']} fills, {all_live_stats['wins']}W/{all_live_stats['losses']}L, WR={all_live_stats['wr']:.1%}, PnL=${all_live_stats['pnl']:+.2f}\n"
        f"Final GO/NO-GO report scheduled after current validation run."
    )


def send_telegram(text: str, chat_id: str | None = None) -> dict:
    token = read_token()
    chat_id = chat_id or pick_chat_id()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode()
    parsed = json.loads(body)
    if not parsed.get("ok"):
        raise RuntimeError(parsed)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--chat-id")
    ap.add_argument("--print", action="store_true", dest="print_only")
    args = ap.parse_args()
    text = build_report(test=args.test)
    if args.print_only:
        print(text)
        return 0
    result = send_telegram(text, chat_id=args.chat_id)
    ids = []
    if isinstance(result.get("result"), dict):
        ids.append(result["result"].get("message_id"))
    log(f"sent chat_id={args.chat_id or pick_chat_id()} message_ids={ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
