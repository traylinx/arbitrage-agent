#!/usr/bin/env python3
"""Start/stop/status BTC paper sniper as two isolated agents.

Default split:
  - btc-5m  -> only Polymarket BTC 5m markets, $10 paper bankroll
  - btc-15m -> only Polymarket BTC 15m markets, $10 paper bankroll

No live orders. This launcher only starts btc_paper_fast.py in paper mode with
agent-specific logs, pid files, params files, and timeframe filters.
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
RUNNER = SRC_DIR / "btc_paper_fast.py"
PYTHON = Path(os.environ.get("BTC_PAPER_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    timeframe: int
    capital: float

    @property
    def slug(self) -> str:
        return self.agent_id.replace("/", "_")

    @property
    def pid_file(self) -> Path:
        return STATE_DIR / f"btc_paper_fast_{self.timeframe}m.pid"

    @property
    def run_until_file(self) -> Path:
        return STATE_DIR / f"btc_paper_fast_{self.timeframe}m_run_until.ts"

    @property
    def params_file(self) -> Path:
        return STATE_DIR / f"sniper_best_params_{self.timeframe}m.json"

    @property
    def trader_log(self) -> Path:
        return LOG_DIR / f"btc_paper_fast_{self.timeframe}m.log"

    @property
    def stdout_log(self) -> Path:
        return LOG_DIR / f"btc_paper_fast_{self.timeframe}m.stdout.log"


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
        if str(RUNNER) in cmd and f"--agent-id {spec.agent_id}" in cmd and f"--timeframes {spec.timeframe}" in cmd:
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
            and f"--agent-id {spec.agent_id}" in cmd
            and f"--timeframes {spec.timeframe}" in cmd
            and "--parallel-lab" not in cmd
        ):
            spec.pid_file.write_text(str(candidate_pid))
            return candidate_pid
    return None


def live_sniper_pids() -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        if "btc_sniper_live.py" not in line or "--live" not in line:
            continue
        try:
            pids.append(int(line.strip().split(None, 1)[0]))
        except Exception:
            pass
    return pids


def start_agent(spec: AgentSpec, duration: int, fast_ga: bool, include_live_training: bool, live_trade_weight: float) -> int:
    existing = find_agent_pid(spec)
    if existing:
        print(f"{spec.agent_id}: already running pid={existing}")
        return existing

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_until = int(time.time()) + duration
    spec.run_until_file.write_text(str(run_until))

    env = os.environ.copy()
    # $10 split bankroll must still clear Polymarket's 5-share practical
    # minimum. The frozen params file uses max_bet_pct=0.20, which caps a $10
    # agent at $2.00 and makes every normal 50c/5-share paper entry impossible.
    # Keep this override process-local and paper-only; params files stay frozen.
    param_overrides = {
        "max_bet_pct": 0.35,
        "spend_ratio": 0.20,
    }
    env.update(
        {
            "HARVEY_HOME": str(HARVEY_HOME),
            "PYTHONUNBUFFERED": "1",
            "BTC_AGENT_ID": spec.agent_id,
            "BTC_TIMEFRAMES": str(spec.timeframe),
            "BTC_PAPER_CAPITAL": str(spec.capital),
            "BTC_BEST_PARAMS_FILE": str(spec.params_file),
            "BTC_PAPER_LOG_FILE": str(spec.trader_log),
            "BTC_RUN_UNTIL_TS": str(run_until),
            "BTC_PARAM_OVERRIDES_JSON": json.dumps(param_overrides, separators=(",", ":")),
            # Keep one training authority unless explicitly requested.
            "BTC_FAST_GA_ENABLED": "1" if fast_ga else "0",
            # Paper processes never place live orders, but their optimizer can
            # learn from resolved live fills when Sebastian explicitly enables it.
            "BTC_INCLUDE_LIVE_TRAINING": "1" if include_live_training else "0",
            "BTC_LIVE_TRADE_WEIGHT": str(live_trade_weight),
        }
    )
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--duration",
        str(duration),
        "--timeframes",
        str(spec.timeframe),
        "--agent-id",
        spec.agent_id,
        "--capital",
        str(spec.capital),
        "--best-params-file",
        str(spec.params_file),
        "--log-file",
        str(spec.trader_log),
    ]
    if include_live_training:
        cmd.append("--include-live-training")
        cmd.extend(["--live-trade-weight", str(live_trade_weight)])
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
        f"{spec.agent_id}: started pid={proc.pid} tf={spec.timeframe}m capital=${spec.capital:.2f} "
        f"fast_ga={fast_ga} include_live_training={include_live_training} live_weight={live_trade_weight:g} "
        f"until={datetime.fromtimestamp(run_until).strftime('%Y-%m-%d %H:%M:%S')} log={spec.trader_log}"
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
    for _ in range(20):
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
        lines = path.read_text(errors="replace").splitlines()[-300:]
    except Exception as exc:
        return f"log read error: {exc}"
    for line in reversed(lines):
        if "elapsed=" in line and "Cash=$" in line and "Eq=$" in line:
            return line
    return lines[-1] if lines else "empty log"


def status_agent(spec: AgentSpec) -> None:
    pid = find_agent_pid(spec)
    run_until = read_pid(spec.run_until_file)
    remaining = max(0, run_until - int(time.time())) if run_until else None
    state = "OK" if pid else "DOWN"
    print(
        f"{spec.agent_id}: {state} pid={pid or '-'} tf={spec.timeframe}m capital=${spec.capital:.2f} "
        f"remaining={(str(remaining//60)+'m') if remaining is not None else '?'} params={spec.params_file}"
    )
    print(f"  {tail_status(spec.trader_log)}")


def specs(args) -> list[AgentSpec]:
    total = float(args.capital_total)
    cap_5m = args.capital_5m if args.capital_5m is not None else round(total / 2.0, 2)
    cap_15m = args.capital_15m if args.capital_15m is not None else round(total - cap_5m, 2)
    return [
        AgentSpec("btc-5m", 5, float(cap_5m)),
        AgentSpec("btc-15m", 15, float(cap_15m)),
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Manage split BTC 5m/15m paper agents")
    p.add_argument("command", choices=("start", "stop", "restart", "status"))
    p.add_argument("--duration", type=int, default=21600, help="run duration for start/restart")
    p.add_argument("--capital-total", type=float, default=20.0)
    p.add_argument("--capital-5m", type=float, default=None)
    p.add_argument("--capital-15m", type=float, default=None)
    p.add_argument("--fast-ga", action="store_true", help="enable in-process per-timeframe FastGA")
    p.add_argument("--include-live-training", action="store_true", help="let paper FastGA learn from resolved live fills too")
    p.add_argument("--live-trade-weight", type=float, default=1.0, help="weight assigned to live fills in FastGA scoring")
    args = p.parse_args(argv)

    selected = specs(args)
    if args.command in ("start", "restart"):
        live = live_sniper_pids()
        if live:
            print(f"WARNING: live btc_sniper_live.py --live process still running: {live}")

    if args.command == "status":
        for spec in selected:
            status_agent(spec)
        live = live_sniper_pids()
        if live:
            print(f"WARNING: live btc_sniper_live.py --live process running: {live}")
        return 0

    if args.command in ("stop", "restart"):
        for spec in selected:
            stop_agent(spec)

    if args.command in ("start", "restart"):
        for spec in selected:
            start_agent(spec, args.duration, args.fast_ga, args.include_live_training, args.live_trade_weight)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
