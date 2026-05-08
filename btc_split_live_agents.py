#!/usr/bin/env python3
"""Start/stop/status BTC LIVE sniper as two isolated real-money agents.

Default split:
  - btc-5m-live  -> only Polymarket BTC 5m markets,  configurable bankroll cap
  - btc-15m-live -> only Polymarket BTC 15m markets, configurable bankroll cap

Each agent runs btc_sniper_live.py --live with its own --timeframes filter,
its own log + journal + PID, and its own BTC_MAX_BANKROLL_USDC cap so the
two processes share the same on-chain wallet without one starving the other.

The two agents READ the same wallet balance from CLOB. The bankroll cap is
the upper bound each will deploy at any moment; do not double-count it as
"$X per agent on top of $Y wallet" — pick caps that sum to <= wallet * 0.95.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
RUNNER = SRC_DIR / "btc_sniper_live.py"
PYTHON = Path(os.environ.get("BTC_LIVE_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
ENV_FILE = HARVEY_HOME / "data" / "arbitrage-agent" / ".env.live"
LIVE_KILL_SWITCH_FILE = Path(os.path.expanduser(os.environ.get(
    "BTC_LIVE_KILL_SWITCH_FILE",
    str(STATE_DIR / "live_trading_disabled.json"),
)))


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    timeframe: int
    bankroll_cap: float

    @property
    def pid_file(self) -> Path:
        return STATE_DIR / f"btc_sniper_live_{self.timeframe}m.pid"

    @property
    def run_until_file(self) -> Path:
        return STATE_DIR / f"btc_sniper_live_{self.timeframe}m_run_until.ts"

    @property
    def journal_file(self) -> Path:
        return STATE_DIR / f"intraday_journal_live_{self.timeframe}m.jsonl"

    @property
    def trader_log(self) -> Path:
        return LOG_DIR / f"btc_sniper_live_{self.timeframe}m.log"

    @property
    def stdout_log(self) -> Path:
        return LOG_DIR / f"btc_sniper_live_{self.timeframe}m.stdout.log"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def process_command(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def find_agent_pid(spec: AgentSpec) -> int | None:
    pid = read_pid(spec.pid_file)
    if pid and pid_alive(pid):
        cmd = process_command(pid)
        if str(RUNNER) in cmd and f"--timeframes {spec.timeframe}" in cmd and "--live" in cmd:
            return pid

    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_pid, cmd = line.split(None, 1)
            candidate_pid = int(raw_pid)
        except Exception:
            continue
        if (
            str(RUNNER) in cmd
            and f"--timeframes {spec.timeframe}" in cmd
            and "--live" in cmd
        ):
            spec.pid_file.write_text(str(candidate_pid))
            return candidate_pid
    return None


def load_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_FILE.exists():
        return out
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def arm_kill_switch(reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_KILL_SWITCH_FILE.write_text(json.dumps({
        "disabled": True,
        "reason": reason,
        "ts": datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    print(f"LIVE KILL SWITCH ARMED: {LIVE_KILL_SWITCH_FILE} — {reason}")


def cancel_clob_open_orders() -> None:
    """Best-effort live order cancel via the same CLOB client as the trader."""
    env_loaded = load_env_file()
    sub_env = os.environ.copy()
    sub_env.update(env_loaded)
    code = (
        "import sys, json; sys.path.insert(0, "
        f"{json.dumps(str(SRC_DIR))}); "
        "from btc_sniper_live import CLOBClient; "
        "c=CLOBClient(); "
        "orders=c.get_open_orders(); "
        "print(f'OPEN_ORDERS_BEFORE={len(orders)}'); "
        "\nfor o in orders:\n"
        "    oid=o.get('id') or o.get('order_id') or o.get('orderID')\n"
        "    print('CANCEL', oid)\n"
        "    c.cancel_order(oid)\n"
        "orders2=c.get_open_orders(); print(f'OPEN_ORDERS_AFTER={len(orders2)}')\n"
    )
    try:
        out = subprocess.run(
            [str(PYTHON), "-c", code],
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.stdout.strip():
            print(out.stdout.strip())
        if out.stderr.strip():
            print(out.stderr.strip())
    except Exception as exc:
        print(f"WARN: could not cancel CLOB open orders: {exc}")


def start_agent(
    spec: AgentSpec,
    duration: int,
    fill_premium_bps: int,
    abs_size_cap: float,
    min_spend: float,
    max_trade_cost: float,
    stop_after_first_loss: bool,
    max_bet_pct: float | None,
    spend_ratio: float | None,
    min_seconds_left: float | None,
    external_required: bool,
    max_filled_losses: int,
    max_drawdown: float,
    min_wr_trades: int,
    min_wr: float,
    allow_live_param_mutation: bool,
) -> int:
    existing = find_agent_pid(spec)
    if existing:
        print(f"{spec.agent_id}: already running pid={existing}")
        return existing

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_until = int(time.time()) + duration
    spec.run_until_file.write_text(str(run_until))

    env = os.environ.copy()
    env.update(load_env_file())  # POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS
    env.update(
        {
            "HARVEY_HOME": str(HARVEY_HOME),
            "PYTHONUNBUFFERED": "1",
            "BTC_AGENT_ID": spec.agent_id,
            "BTC_TIMEFRAMES": str(spec.timeframe),
            "BTC_MAX_BANKROLL_USDC": f"{spec.bankroll_cap:.2f}",
            "BTC_LOG_FILE": str(spec.trader_log),
            "BTC_JOURNAL_FILE": str(spec.journal_file),
            "BTC_BEST_PARAMS_FILE": str(
                STATE_DIR / f"sniper_best_params_{spec.timeframe}m.json"
            ),
            "BTC_FILL_PREMIUM_BPS": str(fill_premium_bps),
            "BTC_ABS_SIZE_CAP": str(abs_size_cap),
            "BTC_MIN_SPEND": str(min_spend),
            "BTC_MAX_TRADE_COST": str(max_trade_cost),
            "BTC_STOP_AFTER_FIRST_LOSS": "1" if stop_after_first_loss else "0",
            "BTC_LIVE_REQUIRE_EXTERNAL_CONTEXT": "1" if external_required else "0",
            "BTC_LIVE_MAX_FILLED_LOSSES": str(max_filled_losses),
            "BTC_LIVE_MAX_DRAWDOWN_USDC": str(max_drawdown),
            "BTC_LIVE_MIN_WR_TRADES": str(min_wr_trades),
            "BTC_LIVE_MIN_WR": str(min_wr),
            "BTC_LIVE_ALLOW_PARAM_MUTATION": "1" if allow_live_param_mutation else "0",
            "BTC_RUN_UNTIL_TS": str(run_until),
            "BTC_LIVE_GO": env.get("BTC_LIVE_GO", "1"),
            "BTC_CANARY_CONFIRMED": env.get("BTC_CANARY_CONFIRMED", "1"),
        }
    )
    if max_bet_pct is not None:
        env["BTC_MAX_BET_PCT_OVERRIDE"] = str(max_bet_pct)
    if spend_ratio is not None:
        env["BTC_SPEND_RATIO_OVERRIDE"] = str(spend_ratio)
    if min_seconds_left is not None:
        env["BTC_MIN_SECONDS_LEFT"] = str(min_seconds_left)
    # Do not auto-inject the NO_GO bypass/ACK. Those values must come from the
    # operator environment for this exact launch, after the live preflight.
    # Otherwise removing the kill switch would silently turn this split runner
    # into a gate bypass.
    for key in ("BTC_LIVE_OPERATOR_OVERRIDE", "BTC_LIVE_CANARY_ACK"):
        if env.get(key):
            env[key] = env[key]
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--live",
        "--timeframes",
        str(spec.timeframe),
        "--duration",
        str(duration),
    ]
    with spec.stdout_log.open("ab", buffering=0) as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SRC_DIR),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    spec.pid_file.write_text(str(proc.pid))
    print(
        f"{spec.agent_id}: started pid={proc.pid} tf={spec.timeframe}m "
        f"cap=${spec.bankroll_cap:.2f} premium={fill_premium_bps}bps "
        f"abs_cap={abs_size_cap}sh min_spend=${min_spend} max_order=${max_trade_cost} "
        f"stop_after_first_loss={stop_after_first_loss} max_bet={max_bet_pct} spend_ratio={spend_ratio} "
        f"min_left={min_seconds_left} external_required={external_required} "
        f"max_losses={max_filled_losses} max_drawdown=${max_drawdown:.2f} min_wr={min_wr:.0%}/{min_wr_trades} "
        f"live_mutation={allow_live_param_mutation} "
        f"until={datetime.fromtimestamp(run_until).strftime('%Y-%m-%d %H:%M:%S')} "
        f"log={spec.trader_log}"
    )
    return proc.pid


def stop_agent(spec: AgentSpec) -> None:
    pid = find_agent_pid(spec)
    if not pid:
        print(f"{spec.agent_id}: not running")
        return
    print(f"{spec.agent_id}: stopping pid={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):
        if not pid_alive(pid):
            return
        time.sleep(0.25)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        print(f"{spec.agent_id}: SIGKILL pid={pid}")


def tail_status(path: Path) -> str:
    if not path.exists():
        return "no log"
    try:
        lines = path.read_text(errors="replace").splitlines()[-400:]
    except Exception as exc:
        return f"log read error: {exc}"
    for line in reversed(lines):
        if "trades=" in line and "Bk=$" in line:
            return line
    return lines[-1] if lines else "empty log"


def journal_summary(path: Path) -> str:
    if not path.exists():
        return "journal: none"
    filled = []
    try:
        for raw in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if row.get("exit_reason") == "unfilled" or row.get("filled") is False:
                continue
            if row.get("won") is True or (row.get("won") is False and float(row.get("pnl") or 0) < 0):
                filled.append(row)
    except Exception as exc:
        return f"journal: read error {exc}"
    if not filled:
        return "journal strict filled: n=0"
    wins = sum(1 for r in filled if r.get("won") is True)
    losses = len(filled) - wins
    pnl = sum(float(r.get("pnl") or 0) for r in filled)
    wr = wins / max(len(filled), 1)
    return f"journal strict filled: n={len(filled)} W={wins} L={losses} WR={wr:.0%} PnL=${pnl:+.2f}"


def status_agent(spec: AgentSpec) -> None:
    pid = find_agent_pid(spec)
    run_until = read_pid(spec.run_until_file)
    remaining = max(0, run_until - int(time.time())) if run_until else None
    state = "OK" if pid else "DOWN"
    print(
        f"{spec.agent_id}: {state} pid={pid or '-'} tf={spec.timeframe}m "
        f"cap=${spec.bankroll_cap:.2f} "
        f"remaining={(str(remaining // 60) + 'm') if remaining is not None else '?'}"
    )
    print(f"  {tail_status(spec.trader_log)}")
    print(f"  {journal_summary(spec.journal_file)}")


def specs(args) -> list[AgentSpec]:
    total = float(args.cap_total)
    cap_5m = args.cap_5m if args.cap_5m is not None else round(total / 2.0, 2)
    cap_15m = args.cap_15m if args.cap_15m is not None else round(total - cap_5m, 2)
    all_specs = [
        AgentSpec("btc-5m-live", 5, float(cap_5m)),
        AgentSpec("btc-15m-live", 15, float(cap_15m)),
    ]
    if getattr(args, "only", "both") == "5":
        return [all_specs[0]]
    if getattr(args, "only", "both") == "15":
        return [all_specs[1]]
    return all_specs


def assert_funder() -> None:
    env = load_env_file()
    funder = env.get("POLYMARKET_FUNDER_ADDRESS", "").strip().strip('"')
    pk = env.get("POLYMARKET_PRIVATE_KEY", "").strip().strip('"')
    if not funder or not pk:
        print(f"FATAL: {ENV_FILE} missing POLYMARKET_FUNDER_ADDRESS / POLYMARKET_PRIVATE_KEY")
        sys.exit(2)


def assert_live_not_disabled() -> None:
    if not LIVE_KILL_SWITCH_FILE.exists():
        return
    if os.environ.get("BTC_LIVE_KILL_SWITCH_OVERRIDE") == "I_UNDERSTAND_REAL_MONEY_LOSS_RISK":
        print(f"WARN: kill switch override accepted for this launch: {LIVE_KILL_SWITCH_FILE}")
        return
    reason = ""
    try:
        reason = json.loads(LIVE_KILL_SWITCH_FILE.read_text()).get("reason", "")
    except Exception:
        reason = LIVE_KILL_SWITCH_FILE.read_text(errors="replace").strip()[:240]
    suffix = f": {reason}" if reason else ""
    print(f"FATAL: live kill switch active at {LIVE_KILL_SWITCH_FILE}{suffix}")
    print("Refusing to start real-money agents. Audit first; then remove the file intentionally.")
    sys.exit(6)


def query_clob_balance() -> float | None:
    """Source of truth for Polymarket bankroll: the CLOB SDK itself.

    Raw ERC-20 balanceOf on the proxy reads $0 even when the UI shows real
    cash, because Polymarket settles via the Conditional Tokens framework
    against a Safe-managed escrow that doesn't sit as plain USDC.e on the
    funder address. The CLOBClient asks Polymarket directly via
    update_balance_allowance/get_balance_allowance — same mechanism the
    bot uses for sizing — so this is the only honest pre-flight check.
    """
    env_loaded = load_env_file()
    sub_env = os.environ.copy()
    sub_env.update(env_loaded)
    code = (
        "import sys; sys.path.insert(0, "
        f"{json.dumps(str(SRC_DIR))}); "
        "import btc_sniper_live as M; "
        "c = M.CLOBClient(); "
        "print(f'BALANCE={c.balance:.6f}')"
    )
    try:
        out = subprocess.run(
            [str(PYTHON), "-c", code],
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    for line in (out.stdout or "").splitlines():
        if line.startswith("BALANCE="):
            try:
                return float(line.split("=", 1)[1])
            except Exception:
                continue
    return None


def assert_wallet_funded(spec_total: float) -> None:
    env = load_env_file()
    funder = env.get("POLYMARKET_FUNDER_ADDRESS", "").strip().strip('"')
    if not funder:
        print("FATAL: POLYMARKET_FUNDER_ADDRESS missing — cannot verify funding")
        sys.exit(3)
    usdc = query_clob_balance()
    if usdc is None:
        print(
            f"FATAL: could not query CLOB SDK balance for {funder[:10]}... — "
            f"refusing to launch live without confirmation. Use --skip-wallet-check "
            f"to override (NOT RECOMMENDED)."
        )
        sys.exit(4)
    if usdc < spec_total:
        print(
            f"FATAL: CLOB bankroll {funder[:10]}... is ${usdc:.4f} USDC, "
            f"agents request ${spec_total:.2f}. Top up before launching."
        )
        sys.exit(5)
    print(f"WALLET OK: {funder[:10]}... has ${usdc:.2f} USDC (need ${spec_total:.2f})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Manage split BTC 5m/15m LIVE agents")
    p.add_argument("command", choices=("start", "stop", "restart", "status"))
    p.add_argument("--only", choices=("5", "15", "both"), default="both")
    p.add_argument("--duration", type=int, default=21600, help="run duration seconds")
    p.add_argument("--cap-total", type=float, default=10.0, help="combined bankroll cap USDC")
    p.add_argument("--cap-5m", type=float, default=None)
    p.add_argument("--cap-15m", type=float, default=None)
    p.add_argument("--fill-premium-bps", type=int, default=200,
                   help="cross-spread premium in bps for limit orders (200 = 2%%)")
    p.add_argument("--abs-size-cap", type=float, default=10.0,
                   help="hard upper bound on shares per trade (defense)")
    p.add_argument("--min-spend", type=float, default=2.00,
                   help="minimum cost per trade in USDC")
    p.add_argument("--max-trade-cost", type=float, default=3.00,
                   help="hard upper bound on USDC cost per order")
    p.add_argument("--max-bet-pct", type=float, default=None,
                   help="override params.max_bet_pct for this live launch")
    p.add_argument("--spend-ratio", type=float, default=None,
                   help="override params.spend_ratio for this live launch")
    p.add_argument("--min-seconds-left", type=float, default=None,
                   help="minimum seconds left in 5m/15m market before placing an order")
    p.add_argument("--external-optional", action="store_true",
                   help="allow live trade if external context is unavailable")
    p.add_argument("--stop-after-first-loss", action="store_true",
                   help="stop live process and cancel open orders after first resolved loss")
    p.add_argument("--max-filled-losses", type=int, default=2,
                   help="circuit-breaker: stop live after N filled losses (default 2)")
    p.add_argument("--max-drawdown", type=float, default=2.75,
                   help="circuit-breaker: stop live after this USDC loss (default 2.75)")
    p.add_argument("--min-wr-trades", type=int, default=4,
                   help="circuit-breaker: evaluate WR floor after N filled trades")
    p.add_argument("--min-wr", type=float, default=0.55,
                   help="circuit-breaker: stop when live WR drops below this floor")
    p.add_argument("--allow-live-param-mutation", action="store_true",
                   help="dangerous: allow live process to mutate params instead of shadow-only")
    p.add_argument("--skip-wallet-check", action="store_true")
    args = p.parse_args(argv)

    selected = specs(args)

    if args.command == "status":
        if LIVE_KILL_SWITCH_FILE.exists():
            print(f"LIVE KILL SWITCH ACTIVE: {LIVE_KILL_SWITCH_FILE}")
        for spec in selected:
            status_agent(spec)
        return 0

    if args.command == "stop":
        arm_kill_switch("manual stop via btc_split_live_agents.py stop")
        cancel_clob_open_orders()
        for spec in selected:
            stop_agent(spec)
        return 0

    if args.command == "restart":
        for spec in selected:
            stop_agent(spec)

    if args.command in ("start", "restart"):
        assert_live_not_disabled()
        assert_funder()
        if not args.skip_wallet_check:
            assert_wallet_funded(sum(s.bankroll_cap for s in selected))
        for spec in selected:
            start_agent(
                spec,
                args.duration,
                args.fill_premium_bps,
                args.abs_size_cap,
                args.min_spend,
                args.max_trade_cost,
                args.stop_after_first_loss,
                args.max_bet_pct,
                args.spend_ratio,
                args.min_seconds_left,
                not args.external_optional,
                args.max_filled_losses,
                args.max_drawdown,
                args.min_wr_trades,
                args.min_wr,
                args.allow_live_param_mutation,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
