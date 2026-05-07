#!/usr/bin/env python3
"""BTC paper fast watchdog.

Keeps the $20 BTC 5m/15m paper trader alive for a bounded run, checks every
BTC_WATCH_INTERVAL seconds, and nudges auto-improve/readiness launchd jobs if stale.
No live orders. Paper-only process: btc_paper_fast.py.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from btc_param_contract import DYNAMIC_PARAM_FIELDS, locked_write_params_file, read_params_file

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "/Users/sebastian/MAKAKOO")))
SRC_DIR = HARVEY_HOME / "plugins" / "agent-arbitrage-agent" / "src"
TRADER = SRC_DIR / "btc_paper_fast.py"
WATCHDOG_MAIN_MARKER = "--watchdog-main"
PYTHON = Path(os.environ.get("BTC_PAPER_PYTHON", "/usr/local/opt/python@3.11/bin/python3.11"))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
PID_FILE = STATE_DIR / "btc_paper_fast_watchdog.pid"
TRADER_PID_FILE = STATE_DIR / "btc_paper_fast_6h.pid"
RUN_UNTIL_FILE = STATE_DIR / "btc_paper_fast_run_until.ts"
BEST_PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
WATCHDOG_LOG = LOG_DIR / "btc_paper_fast_watchdog.log"
CHILD_LOG = LOG_DIR / "btc_paper_fast_watchdog_child.log"
TRADER_LOG = LOG_DIR / "btc_sniper_paper_fast.log"
INTERVAL = int(float(os.environ.get("BTC_WATCH_INTERVAL", "60")))
STALE_STATUS_SECONDS = int(float(os.environ.get("BTC_TRADER_STALE_STATUS_SECONDS", "300")))
CAPITAL = os.environ.get("BTC_PAPER_CAPITAL", "20.0")
DEFAULT_DURATION = int(float(os.environ.get("BTC_PAPER_DURATION", "21600")))
AUTOIMPROVE_LOG = LOG_DIR / "autoimprove_v3.log"
READINESS_LOG = LOG_DIR / "readiness_monitor.log"
UID = os.getuid()

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def now_s() -> int:
    return int(time.time())


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    try:
        if WATCHDOG_LOG.exists() and WATCHDOG_LOG.stat().st_size > 2_000_000:
            WATCHDOG_LOG.replace(LOG_DIR / "btc_paper_fast_watchdog.log.1")
        with WATCHDOG_LOG.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_run_until() -> int:
    env_ts = os.environ.get("BTC_RUN_UNTIL_TS")
    if env_ts:
        ts = int(float(env_ts))
        RUN_UNTIL_FILE.write_text(str(ts))
        return ts
    if RUN_UNTIL_FILE.exists():
        try:
            return int(float(RUN_UNTIL_FILE.read_text().strip()))
        except Exception:
            pass
    ts = now_s() + DEFAULT_DURATION
    RUN_UNTIL_FILE.write_text(str(ts))
    return ts


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
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def find_trader_pid() -> int | None:
    # Prefer pid file if still alive and command is actually the BTC paper trader.
    if TRADER_PID_FILE.exists():
        try:
            pid = int(TRADER_PID_FILE.read_text().strip())
            cmd = process_command(pid)
            exe = cmd.split(maxsplit=1)[0].lower() if cmd else ""
            if pid_alive(pid) and str(TRADER) in cmd and WATCHDOG_MAIN_MARKER in cmd and "python" in exe:
                return pid
        except Exception:
            pass

    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        exe = cmd.split(maxsplit=1)[0].lower() if cmd else ""
        if (
            str(TRADER) in cmd
            and WATCHDOG_MAIN_MARKER in cmd
            and "btc_paper_fast_watchdog" not in cmd
            and "python" in exe
        ):
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            TRADER_PID_FILE.write_text(str(pid))
            return pid
    return None


def launch_trader(remaining: int) -> int:
    env = os.environ.copy()
    env["HARVEY_HOME"] = str(HARVEY_HOME)
    env["BTC_PAPER_CAPITAL"] = CAPITAL
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PYTHON), str(TRADER), str(max(60, remaining)), WATCHDOG_MAIN_MARKER]
    log(f"START trader remaining={remaining}s capital=${CAPITAL} cmd={' '.join(cmd)}")
    child_f = CHILD_LOG.open("ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(SRC_DIR),
        env=env,
        stdout=child_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    TRADER_PID_FILE.write_text(str(proc.pid))
    return proc.pid


def latest_status_mtime() -> int | None:
    """Newest trader status timestamp across stdout-child and direct log files.

    File mtime alone is not enough: tests and helper probes can write skip lines
    into btc_sniper_paper_fast.log. Only status lines prove the main loop is
    still making progress.
    """
    best = None
    for path in (CHILD_LOG, TRADER_LOG):
        if not path.exists():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()[-500:]
        except Exception:
            continue
        for line in reversed(lines):
            if "elapsed=" not in line or "Cash=$" not in line or "Eq=$" not in line:
                continue
            try:
                ts_raw = line.split("]", 1)[0].lstrip("[")
                ts = int(datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                ts = int(path.stat().st_mtime)
            if best is None or ts > best:
                best = ts
            break
    return best


def trader_progress_mtime(pid: int) -> int | None:
    """Status timestamp with a launch grace floor for newly-started children."""
    best = latest_status_mtime()
    try:
        if TRADER_PID_FILE.exists() and TRADER_PID_FILE.read_text().strip() == str(pid):
            launched = int(TRADER_PID_FILE.stat().st_mtime)
            best = max(best or 0, launched)
    except Exception:
        pass
    return best


def terminate_pid(pid: int, *, reason: str) -> None:
    log(f"terminating trader pid={pid} reason={reason}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as e:
        log(f"SIGTERM failed pid={pid}: {type(e).__name__}: {e}")
        return
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
        log(f"SIGKILL trader pid={pid} after graceful timeout")
    except ProcessLookupError:
        pass
    except Exception as e:
        log(f"SIGKILL failed pid={pid}: {type(e).__name__}: {e}")


def launchctl(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl"] + args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)



def enforce_param_contract() -> None:
    try:
        if not BEST_PARAMS_FILE.exists():
            return
        before = read_params_file(BEST_PARAMS_FILE, mode="paper")
        after = locked_write_params_file(BEST_PARAMS_FILE, before, mode="paper")
        rejected = after.get("rejected_fields")
        if rejected:
            log(f"param contract stripped fields={rejected}; dynamic={{k: after[k] for k in DYNAMIC_PARAM_FIELDS}}")
    except Exception as e:
        log(f"param contract error: {type(e).__name__}: {e}")

def ensure_launchd_job(label: str, plist: Path, stale_log: Path | None = None, stale_after: int = 1200) -> None:
    if not plist.exists():
        log(f"launchd missing plist label={label} plist={plist}")
        return
    printed = launchctl(["print", f"gui/{UID}/{label}"], timeout=10)
    if printed.returncode != 0:
        boot = launchctl(["bootstrap", f"gui/{UID}", str(plist)], timeout=20)
        log(f"launchd bootstrap {label}: rc={boot.returncode} err={boot.stderr.strip()[:200]}")
        return
    if stale_log and stale_log.exists():
        age = now_s() - int(stale_log.stat().st_mtime)
        if age > stale_after:
            kick = launchctl(["kickstart", "-k", f"gui/{UID}/{label}"], timeout=20)
            log(f"launchd kick stale {label}: age={age}s rc={kick.returncode} err={kick.stderr.strip()[:200]}")


def ensure_autoimprove_stack() -> None:
    base = Path.home() / "Library" / "LaunchAgents"
    ensure_launchd_job("com.makakoo.arbitrage.autoimprove", base / "com.makakoo.arbitrage.autoimprove.plist", AUTOIMPROVE_LOG, 1200)
    ensure_launchd_job("com.makakoo.arbitrage.readiness", base / "com.makakoo.arbitrage.readiness.plist", READINESS_LOG, 5400)


def main() -> int:
    PID_FILE.write_text(str(os.getpid()))
    run_until = read_run_until()
    log(f"WATCHDOG start interval={INTERVAL}s run_until={datetime.fromtimestamp(run_until)} capital=${CAPITAL}")
    last_stack_check = 0
    last_ok_log = 0

    while True:
        now = now_s()
        remaining = run_until - now
        pid = find_trader_pid()

        if remaining <= 0:
            log("run window expired; watchdog exiting")
            return 0

        if now - last_stack_check >= max(60, INTERVAL):
            enforce_param_contract()
            ensure_autoimprove_stack()
            last_stack_check = now

        if pid and pid_alive(pid):
            status_ts = trader_progress_mtime(pid)
            if status_ts is not None:
                status_age = now - status_ts
                if status_age > STALE_STATUS_SECONDS:
                    terminate_pid(pid, reason=f"stale_status age={status_age}s>{STALE_STATUS_SECONDS}s")
                    try:
                        launch_trader(remaining)
                    except Exception as e:
                        log(f"restart after stale status failed: {type(e).__name__}: {e}")
                    time.sleep(max(5, min(INTERVAL, 15)))
                    continue
            if now - last_ok_log >= max(60, INTERVAL):
                log(f"OK trader running pid={pid} remaining={remaining}s")
                last_ok_log = now
        else:
            log("trader not running; restarting")
            try:
                launch_trader(remaining)
            except Exception as e:
                log(f"restart failed: {type(e).__name__}: {e}")

        time.sleep(max(5, INTERVAL))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("watchdog interrupted")
        raise
