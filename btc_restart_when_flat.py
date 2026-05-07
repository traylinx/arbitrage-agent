#!/usr/local/opt/python@3.11/bin/python3.11
from __future__ import annotations
import os, re, signal, subprocess, time
from datetime import datetime
from pathlib import Path

HOME=Path('/Users/sebastian/MAKAKOO')
SRC=HOME/'plugins/agent-arbitrage-agent/src'
LOG=HOME/'data/arbitrage-agent/v2/logs/btc_paper_fast_watchdog_child.log'
OUT=HOME/'data/arbitrage-agent/v2/logs/btc_restart_when_flat.log'
TRADER=str(SRC/'btc_paper_fast.py')
WATCHDOG=str(SRC/'btc_paper_fast_watchdog.py')
FREEZE=str(SRC/'btc_freeze_strategy.py')
RUN_UNTIL=HOME/'data/arbitrage-agent/v2/state/btc_paper_fast_run_until.ts'
PYBIN='/usr/local/opt/python@3.11/bin/python3.11'

def log(msg):
    line=f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('a') as f: f.write(line+'\n')

def _line_epoch(line):
    try:
        m=re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if not m:
            return None
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
    except Exception:
        return None

def latest_locked(min_epoch):
    if not LOG.exists(): return None, 'no log'
    lines=LOG.read_text(errors='replace').splitlines()[-300:]
    for line in reversed(lines):
        if 'elapsed=' in line and 'locked=$' in line and 'Eq=$' in line:
            ep=_line_epoch(line)
            if ep is not None and ep < min_epoch:
                continue
            m=re.search(r'locked=\$([0-9.]+)', line)
            return (float(m.group(1)) if m else None), line
    return None, 'no status'

def pids(pattern):
    try:
        out=subprocess.check_output(['pgrep','-f',pattern], text=True, stderr=subprocess.DEVNULL)
        me=os.getpid()
        return [int(x) for x in out.split() if x.isdigit() and int(x) != me]
    except Exception:
        return []

def main_trader_pids():
    try:
        out=subprocess.check_output(['ps','-axo','pid=,command='], text=True)
    except Exception:
        return []
    me=os.getpid()
    found=[]
    for line in out.splitlines():
        parts=line.strip().split(None,1)
        if len(parts) != 2:
            continue
        pid=int(parts[0]); cmd=parts[1]
        if pid == me:
            continue
        if TRADER in cmd and '--watchdog-main' in cmd and '--parallel-lab' not in cmd and 'btc_restart_when_flat.py' not in cmd:
            found.append(pid)
    return found

def watchdog_pids():
    try:
        out=subprocess.check_output(['ps','-axo','pid=,command='], text=True)
    except Exception:
        return []
    me=os.getpid()
    found=[]
    for line in out.splitlines():
        parts=line.strip().split(None,1)
        if len(parts) != 2:
            continue
        pid=int(parts[0]); cmd=parts[1]
        if pid == me:
            continue
        if WATCHDOG in cmd and 'btc_restart_when_flat.py' not in cmd:
            found.append(pid)
    return found

def freeze():
    env=os.environ.copy()
    env['HARVEY_HOME']=str(HOME)
    cmd=[PYBIN, FREEZE, '--label', 'auto-clob-clean-validation', '--mode', 'paper']
    res=subprocess.run(cmd, cwd=str(SRC), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    log(f'freeze rc={res.returncode} out={res.stdout.strip()[-240:]} err={res.stderr.strip()[-240:]}')
    return res.returncode == 0

log('armed: when flat, freeze CLOB clean-validation and restart watchdog/trader with BTC_FAST_GA_ENABLED=0')
min_status_epoch=time.time()-120
try:
    deadline=float(RUN_UNTIL.read_text().strip()) + 10*60
except Exception:
    deadline=time.time()+6*60*60
while time.time()<deadline:
    locked,line=latest_locked(min_status_epoch)
    log(f'check locked={locked} line={line[-180:]}')
    if locked == 0.0:
        freeze()
        trader=main_trader_pids()
        watchdog=watchdog_pids()
        log(f'flat; terminating trader pids={trader}; watchdog pids={watchdog}')
        for pid in trader + watchdog:
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
        log('done; launchd should restart watchdog, then trader, with CLOB code + FastGA disabled')
        raise SystemExit(0)
    time.sleep(30)
log('timeout; no restart')
