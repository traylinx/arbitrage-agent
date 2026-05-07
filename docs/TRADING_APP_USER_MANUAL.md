# BTC Polymarket Trading App — User Manual

Generated: 2026-05-07  
App root: `/Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src`  
Data root: `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2`  
Report root: `/Users/sebastian/MAKAKOO/data/reports/gym/btc-polymarket-trader`

## 0. Safety model

Default supported mode is paper-only:

- real market data
- fake money
- no wallet access
- no order POSTs
- no autonomous live-money orders

Live-money mode is not documented as an operator flow because the current gate is `NO_GO`. Use this manual to run validation, research, reporting, and readiness checks.

## 1. Requirements

Preferred Python:

```bash
/usr/local/opt/python@3.11/bin/python3.11 --version
```

Install Python dependencies from the source repo:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
/usr/local/opt/python@3.11/bin/python3.11 -m pip install -r requirements.txt
```

Runtime data directories are created automatically under:

```bash
/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2
```

## 2. Important files

### Core paper trader

- `btc_paper_fast.py` — main BTC 5m/15m paper trader.
- `btc_paper_fast_watchdog.py` — keeps bounded paper run alive.
- `btc_fee_model.py` — Polymarket fee/PnL model.
- `btc_param_contract.py` — trainable vs frozen parameter contract.

### Data and features

- `btc_external_metrics.py` — read-only external derivatives context.
- `btc_feature_engine.py` — minute-level feature DB builder.
- `btc_prob_dataset.py` — dataset builder from journals.

### Models

- `btc_probability_model.py` — local logistic/boosting probability model.
- `btc_prob_gate.py` — edge gate used by trader.
- `btc_model_trainer.py` — sklearn model trainer from feature DB.
- `btc_backtest_model.py` — model holdout backtest.
- `btc_model_pipeline.py` — one-shot labels/train/backtest pipeline.

### Optimization

- `btc_backtest_autoresearch.py` — parameter search over paper trades.
- `btc_parallel_backtest_orchestrator.py` — parallel optimizer with holdout.
- `autoimprove_v3.py` — periodic self-improvement loop.
- `btc_parallel_paper_lab.py` — isolated paper strategy swarm.
- `btc_live_valid_swarm_curator.py` — paper-swarm curator using CLOB-valid evidence.

### Readiness and reporting

- `btc_trading_gym.py` — full readiness/gym score.
- `btc_live_go_nogo.py` — strict canary promotion gate.
- `btc_telegram_reporter.py` — Telegram/print status report.
- `readiness_monitor.py` — periodic readiness monitoring.

## 3. Fast status commands

Run from source root:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
```

Print Telegram-style status without sending:

```bash
$PY btc_telegram_reporter.py --print
```

Run Gym readiness report:

```bash
$PY btc_trading_gym.py
cat /Users/sebastian/MAKAKOO/data/reports/gym/btc-polymarket-trader/latest.md
```

Run strict live canary gate:

```bash
$PY btc_live_go_nogo.py
```

Check running trading processes:

```bash
ps -axo pid,ppid,pgid,stat,etime,command \
  | grep -Ei 'btc_|arbitrage|polymarket|autoimprove|paper_lab|papertrader|parallelresearch|sniper' \
  | grep -v grep
```

Check launchd jobs:

```bash
launchctl list | grep 'com.makakoo.arbitrage' || true
launchctl print-disabled gui/$UID | grep 'com.makakoo.arbitrage' || true
```

## 4. Start a foreground paper-validation run

Use this when you want a controlled run in the terminal.

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

BTC_FAST_GA_ENABLED=0 \
BTC_PAPER_CAPITAL=20.0 \
BTC_PAPER_ONLY=1 \
POLYMARKET_LIVE_TRADING=0 \
$PY btc_paper_fast.py 21600 --watchdog-main
```

Meaning:

- `21600` = 6 hours.
- `BTC_FAST_GA_ENABLED=0` keeps the in-process GA off so the external optimizer is the single training authority.
- `POLYMARKET_LIVE_TRADING=0` reinforces no live trading.

Stop with `Ctrl+C`.

## 5. Start a watchdog-managed paper run

Use this when you want the watchdog to restart stale paper trader process.

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
STATE=/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state

python3 - <<'PY'
import time, pathlib
pathlib.Path('/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/btc_paper_fast_run_until.ts').write_text(str(int(time.time()) + 21600))
PY

BTC_FAST_GA_ENABLED=0 \
BTC_PAPER_CAPITAL=20.0 \
BTC_WATCH_INTERVAL=60 \
BTC_PAPER_ONLY=1 \
POLYMARKET_LIVE_TRADING=0 \
$PY btc_paper_fast_watchdog.py
```

Watch logs:

```bash
tail -f /Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/logs/btc_paper_fast_watchdog.log
```

## 6. Run parallel paper lab

Launch a controlled fake-money strategy swarm:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_parallel_paper_lab.py \
  --label manual_validation \
  --n 8 \
  --duration 21600 \
  --capital 20 \
  --profile explore
```

Report latest lab:

```bash
$PY btc_parallel_paper_lab.py --report
```

Stop lab workers:

```bash
$PY btc_parallel_paper_lab.py --stop
```

Safety: lab workers scrub live credentials and set fake-money environment markers.

## 7. Run optimizer

Run one parallel optimizer pass:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_parallel_backtest_orchestrator.py \
  --workers 8 \
  --iterations 650 \
  --boot-iters 300 \
  --min-score-gap 1.0 \
  --commit \
  --freeze-on-commit \
  --json
```

This changes only paper params under:

```bash
/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/sniper_best_params.json
```

Only dynamic params may change:

- `delta_thresh`
- `conf_thresh`
- `ens_thresh`

## 8. Run external metrics context fetch

Smoke-test external data integration:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY - <<'PY'
from btc_external_metrics import fetch_external_market_context, context_feature_subset
ctx = fetch_external_market_context(use_cache=False)
print('keys', len(ctx))
print('external_bull_score', ctx.get('external_bull_score'))
print(context_feature_subset(ctx))
PY
```

Keys live in keyring and are read through `makakoo secret get ...` where needed.

## 9. Train probability model

Collect feature data first:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_feature_engine.py
```

After enough rows exist, run pipeline:

```bash
$PY btc_model_pipeline.py
```

Alternative journal-based model path:

```bash
$PY btc_prob_dataset.py
$PY btc_probability_model.py --train
$PY btc_probability_model.py --eval
```

Model output:

```bash
/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/model/
```

## 10. Freeze strategy before promotion checks

Create freeze snapshot:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_freeze_strategy.py --label paper-validation --mode paper
```

Run paper validation after freeze, then check:

```bash
$PY btc_live_go_nogo.py
$PY btc_trading_gym.py
```

## 11. Promotion gates

`LIVE_CANARY_ALLOWED` requires all of these:

- paper uses real market data
- paper has no real order POSTs
- paper uses CLOB executable fill code
- clean CLOB fills exist and equal clean trade count
- at least 80 clean post-freeze trades
- Wilson lower WR at least 55%
- clean PnL positive
- clean drawdown <= 25%

Current state at doc generation: `NO_GO` / `KEEP_TRAINING`.

## 12. Emergency stop

Stop all paper/training/signal processes:

```bash
cd /Users/sebastian/MAKAKOO

# Disable launchd jobs.
launchctl list | awk '/com\.makakoo\.arbitrage/ {print $3}' | while read label; do
  launchctl disable "gui/$UID/$label" || true
  launchctl bootout "gui/$UID/$label" || true
done

# Kill residual matching processes.
python3 - <<'PY'
import os, signal, subprocess, time, re
parts = ['btc_', 'arbi'+'trage', 'poly'+'market', 'auto'+'improve', 'paper_'+'lab', 'paper'+'trader', 'parallel'+'research', 'sniper']
pat = re.compile('|'.join(map(re.escape, parts)), re.I)
me = os.getpid()
rows=[]
out = subprocess.check_output(['ps','-axo','pid=,pgid=,command='], text=True)
for line in out.splitlines():
    p=line.strip().split(None,2)
    if len(p)<3: continue
    pid, pgid, cmd = int(p[0]), int(p[1]), p[2]
    if pid == me: continue
    if pat.search(cmd): rows.append((pid, pgid, cmd))
for pgid in sorted({r[1] for r in rows}):
    try: os.killpg(pgid, signal.SIGTERM)
    except Exception: pass
time.sleep(2)
PY
```

Verify:

```bash
ps -axo pid,ppid,pgid,stat,etime,command \
  | grep -Ei 'btc_|arbitrage|polymarket|autoimprove|paper_lab|papertrader|parallelresearch|sniper' \
  | grep -v grep || true
```

## 13. Operator rules

- Do not run `btc_sniper_live.py` unless a separate reviewed live-canary procedure exists and gates are green.
- Do not commit runtime logs, journals, model pickles, PID files, lock files, or `.env.live`.
- Use `btc_live_go_nogo.py` and `btc_trading_gym.py` before any canary discussion.
- If a report says `KEEP_TRAINING`, continue paper validation.
- If a report says `NO_GO`, do not trade real money.
