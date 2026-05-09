# BTC Polymarket Trading App — User Manual

Generated: 2026-05-08
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

Live-money mode is blocked by the durable kill switch after the May 8 canary loss. Use this manual to run validation, research, reporting, and readiness checks. If live is restarted later, keep the circuit breakers enabled.

Emergency stop:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
$PY btc_split_live_agents.py stop
```

This does three things:

1. writes `data/arbitrage-agent/v2/state/live_trading_disabled.json`,
2. cancels current CLOB open orders best-effort,
3. stops split live-agent processes.


## 0a. Current standalone release

The deployable V2 repo of truth is now:

```text
/Users/sebastian/Projects/agent-arbitrage-agent
https://github.com/makakoo/agent-arbitrage-agent
https://github.com/makakoo/agent-arbitrage-agent/releases/tag/v0.18.0
```

Use that repo for new-server deployment, `live_shadow`, `autoimprove_shadow`, `shadow_lane_scoreboard`, `live_go_no_go`, and `live_preflight`.

This workspace plugin path remains useful for legacy BTC operator tools, audits, and historical journal-based safety checks. Do not treat it as the primary deployable release.

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
- `btc_split_paper_agents.py` — starts/stops separate `btc-5m` and `btc-15m` paper-agent processes.
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
- `btc_decision_audit.py` — read-only post-loss/decision audit over live and paper journals.
- `btc_telegram_reporter.py` — Telegram/print status report.
- `readiness_monitor.py` — periodic readiness monitoring.

### Documentation

- `docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md` — root-cause audit for the May 8 live loss and paper/lab regression.
- `CHANGELOG.md` — operator-facing change history.

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

Run read-only decision/loss audit:

```bash
$PY btc_decision_audit.py --hours 36 --limit 12
```

Check running trading processes:

```bash
ps -axo pid,ppid,pgid,stat,etime,command \
  | grep -Ei 'btc_|arbitrage|polymarket|autoimprove|paper_lab|papertrader|parallelresearch|sniper' \
  | grep -v grep
```

Check split live-agent status and strict filled journal truth:

```bash
$PY btc_split_live_agents.py status
```

Read this status carefully:

- `trades=W/L/U` in the last log line is the process-local counter.
- `journal strict filled` is the better money metric because it excludes unfilled limit orders and uses resolved filled journal rows.
- If the kill switch is active, live restart is refused unless you remove it intentionally or use the explicit override.

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

## 5a. Start the split 5m and 15m paper agents

Use this when you want 5-minute and 15-minute Polymarket BTC markets isolated into two separate paper-only processes.

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_split_paper_agents.py start --duration 21600 --capital-total 20
```

Default process layout:

| Agent | Markets | Capital | Params | Trader log |
|---|---:|---:|---|---|
| `btc-5m` | BTC 5m only | `$10` | `state/sniper_best_params_5m.json` | `logs/btc_paper_fast_5m.log` |
| `btc-15m` | BTC 15m only | `$10` | `state/sniper_best_params_15m.json` | `logs/btc_paper_fast_15m.log` |

The launcher applies process-local paper-only sizing overrides (`max_bet_pct=0.35`, `spend_ratio=0.20`) so a `$10` agent can clear Polymarket's practical 5-share minimum. Params files remain separate and frozen; this override is runtime-only.

Check both agents:

```bash
$PY btc_split_paper_agents.py status
```

Stop both agents:

```bash
$PY btc_split_paper_agents.py stop
```

Restart both agents:

```bash
$PY btc_split_paper_agents.py restart --duration 21600 --capital-total 20
```

Custom capital split:

```bash
$PY btc_split_paper_agents.py start \
  --duration 21600 \
  --capital-total 20 \
  --capital-5m 12 \
  --capital-15m 8
```

Enable per-agent in-process FastGA only when you intentionally want each process to optimize its own timeframe:

```bash
$PY btc_split_paper_agents.py start --duration 21600 --capital-total 20 --fast-ga
```

Enable paper+real learning for those paper agents:

```bash
$PY btc_split_paper_agents.py restart \
  --duration 21600 \
  --capital-total 20 \
  --fast-ga \
  --include-live-training
```

This still starts paper-only processes. They do not submit orders. The difference is optimizer input: FastGA loads resolved `mode=paper` rows plus resolved `mode=live` fills from the shared journal, filtered to the agent timeframe. Use `--live-trade-weight` if live fills should count less than paper rows.

Default recommendation: leave `--fast-ga` off and let the external optimizer remain the single training authority.

Manual one-agent start examples:

```bash
# 5m only
BTC_FAST_GA_ENABLED=0 \
$PY btc_paper_fast.py \
  --duration 21600 \
  --timeframes 5 \
  --agent-id btc-5m \
  --capital 10 \
  --best-params-file /Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/sniper_best_params_5m.json \
  --log-file /Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/logs/btc_paper_fast_5m.log

# 15m only
BTC_FAST_GA_ENABLED=0 \
$PY btc_paper_fast.py \
  --duration 21600 \
  --timeframes 15 \
  --agent-id btc-15m \
  --capital 10 \
  --best-params-file /Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/sniper_best_params_15m.json \
  --log-file /Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/logs/btc_paper_fast_15m.log
```

## 5b. Live canary controls — blocked by default

Do not use live mode until `btc_live_go_nogo.py` passes and the latest paper/shadow evidence is profitable on strict CLOB-realistic fills.

Status:

```bash
$PY btc_split_live_agents.py status
```

Stop:

```bash
$PY btc_split_live_agents.py stop
```

Live start is intentionally harder than paper start. It requires:

- funded `.env.live`,
- no active `live_trading_disabled.json`,
- passing GO/NO-GO or an explicit canary override,
- fresh external context, default max age `75s`,
- at least one premium derivatives feed active, Coinalyze or CoinGlass,
- small bankroll/order caps,
- enabled circuit breakers.

Example shape for a future tiny 5m canary **after audit only**:

```bash
BTC_LIVE_OPERATOR_OVERRIDE=AUTHOR_AUTHORIZED_CANARY_5USDC_1USDC_TICKET \
BTC_LIVE_CANARY_ACK=I_ACCEPT_CANARY_RISK_MAX_5_USDC \
BTC_LIVE_KILL_SWITCH_OVERRIDE=I_UNDERSTAND_REAL_MONEY_LOSS_RISK \
$PY btc_split_live_agents.py start \
  --only 5 \
  --duration 1800 \
  --cap-total 5 \
  --cap-5m 5 \
  --max-trade-cost 2.65 \
  --min-spend 2.50 \
  --max-filled-losses 1 \
  --max-drawdown 2.75 \
  --min-wr-trades 4 \
  --min-wr 0.55 \
  --min-seconds-left 45
```

Do **not** pass `--allow-live-param-mutation` unless the goal is an explicit live-risk experiment. Default live auto-improvement is shadow-only; paper agents train, live agents execute fixed audited params.

Default live external-data guards:

```bash
BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75
BTC_LIVE_REQUIRE_PREMIUM_DERIVATIVE_FEED=1
```

If external context is older than the max age, or both Coinalyze and CoinGlass are unavailable, live GO is rejected.

Journal behavior:

- Both agents append to the shared journal:
  `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/intraday_journal.jsonl`
- Each resolved row includes:
  - `agent_id`: `btc-5m` or `btc-15m`
  - `window_tf`: `5` or `15`

Reporting:

```bash
$PY btc_telegram_reporter.py --print
```

When split agents are running, the report includes a `Split agents:` section with per-agent PID, run-until, status line, trades, WR, and PnL.

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

For live trading, provider health matters as much as feature values. Missing data is not edge. If Coinalyze and CoinGlass are both down, live trading should stay blocked.

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

To include resolved live fills in the probability dataset too:

```bash
$PY btc_prob_dataset.py --include-live
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

- Default mode is paper validation. If Sebastian explicitly authorizes live-money agents, do not stop them automatically; keep split paper validation running beside live canaries.
- Do not run or modify `btc_sniper_live.py` casually. Treat live-money launch/stop/size changes as explicit operator actions.
- Do not commit runtime logs, journals, model pickles, PID files, lock files, or `.env.live`.
- Use `btc_live_go_nogo.py` and `btc_trading_gym.py` before any canary discussion.
- If a report says `KEEP_TRAINING`, continue paper validation.
- If a report says `NO_GO`, treat it as a hard risk warning unless Sebastian explicitly overrides it for a controlled live canary.
