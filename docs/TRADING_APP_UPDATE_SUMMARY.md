# BTC Polymarket Trading App — Update Summary

Generated: 2026-05-08
Scope: BTC-only Polymarket 5-minute and 15-minute markets.
Default mode: paper trading with real market data and no real orders.


## 2026-05-10 standalone release and documentation sync

Current deployable repo of truth:

```text
/Users/sebastian/Projects/agent-arbitrage-agent
https://github.com/makakoo/agent-arbitrage-agent
Release: https://github.com/makakoo/agent-arbitrage-agent/releases/tag/v0.18.0
```

What changed in the standalone release:

- final read-only `live_preflight` before any future live canary
- explicit `LIVE_ORDER_PLACEMENT_IMPLEMENTED=false` safety flag
- stricter live GO/NO_GO defaults
- lane scoreboard refuses to rank taker/maker candidates without taker/maker evidence
- Gamma market tradability check now fails closed when markets are not accepting orders or are closed/archived
- end-to-end `docs/USER_MANUAL.md` added

Legacy workspace repo status:

- V2 decision-engine scaffold files are now committed here too for continuity.
- New live-money deployment should still use the standalone repo, not this legacy plugin path.

## 2026-05-08 live canary postmortem and fix

Status: **all BTC live/paper trading processes stopped** and live kill switch armed.

Observed live canary outcome:

- 5m live process ran with real Polymarket CLOB orders.
- CLOB wallet after stop: `$7.48`.
- CLOB open orders after stop: `0`.
- Strict filled journal showed the canary was losing, not improving:
  - 5m live strict filled: `3W / 7L`, `30% WR`, `-$9.63`.
  - 15m historical live strict filled: `1W / 0L`, `+$1.96`.
- Raw process counters were noisy because they mixed pending/reconciled state; strict filled journal is now the status truth source.

Root causes fixed:

1. **Live circuit breaker missing.** The process kept trading after loss clustering.
   Fix: live mode now stops and arms the kill switch after configured filled-loss, drawdown, or WR-floor breaches.
2. **Duplicate reconcile accounting.** A filled order could resolve once through the normal trade path, then later be synthesized again by reconcile, inflating counters and lessons.
   Fix: pending fills are cleared on resolution, and `_resolve_from_open_order()` refuses to synthesize a second trade for an existing order id.
3. **Slow balance retry in entry path.** The pre-order path could block on CLOB balance refresh, then submit too late in the 5m window.
   Fix: entry uses non-blocking balance refresh only, then rechecks `seconds_left` immediately before order POST.
4. **Live auto-mutation on tiny/noisy sample.** GA/LLM mutation could switch params during live trading based on very small samples and unfilled rows.
   Fix: live param mutation is disabled by default; scoring ignores unfilled orders; live tuning requires a larger filled sample and explicit `--allow-live-param-mutation`.
5. **Stop command incomplete.** Manual stop did not arm the kill switch/cancel orders by itself.
   Fix: `btc_split_live_agents.py stop` arms the kill switch and best-effort cancels CLOB orders before stopping processes.

New live safety defaults:

```text
BTC_LIVE_MAX_FILLED_LOSSES=2
BTC_LIVE_MAX_DRAWDOWN_USDC=2.75
BTC_LIVE_MIN_WR_TRADES=4
BTC_LIVE_MIN_WR=0.55
BTC_LIVE_ALLOW_PARAM_MUTATION=0
BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75
BTC_LIVE_REQUIRE_PREMIUM_DERIVATIVE_FEED=1
```

Operator implication: live canary is no longer a "let it run and see" process. If it fails early, it stops itself and requires a fresh audit before restart.

## 2026-05-08 architecture audit and loss-analysis patch

Audit command:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
$PY btc_decision_audit.py --hours 36 --limit 12
```

Audit result from the May 8 loss window:

- Live strict filled: `11` trades, `4W / 7L`, `36.4% WR`, `-$7.68`.
- 5m live: `10` trades, `3W / 7L`, `30.0% WR`, `-$9.63`.
- Recent paper/lab: `1851` trades, `858W / 993L`, `46.4% WR`, `-$757.55` fake.
- `>180s` external-data-age bucket in live: `0W / 3L`, `-$7.65`.
- Filled live rows with provider summary had Coinalyze `0/8` and CoinGlass `0/8`.

Root-cause doc:

- `docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md`

Postmortem/action-plan doc:

- `docs/TRADING_APP_POSTMORTEM_ACTION_PLAN_2026-05-08.md`

Code changes:

- Added `btc_decision_audit.py`, a read-only audit tool for live/paper journals.
- Live external vote now rejects stale context by default, `BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75`.
- Live external vote now rejects GO when both Coinalyze and CoinGlass are unavailable by default.
- Added regression tests for both guards.

Hard architecture finding: the live path was not the same as the richer paper/model path. Live used a legacy BTC-delta heuristic plus hand-weighted external confirmation. Paper/lab had richer features and executable CLOB simulation. Next P0 is one shared decision engine.

## Current readiness state

The app is more capable than the original sniper, but it is **not live-money ready** after the May 8 canary. Live is blocked until paper/shadow evidence proves edge with strict filled CLOB-realistic data.

Latest verified gates:

- `btc_live_go_nogo.py`: `NO_GO`
- Latest Gym report: `KEEP_TRAINING`, score `70/100`
- Clean post-freeze evidence: too small and too weak
  - clean post-freeze trades: `2`
  - clean WR: `50.0%`
  - Wilson lower bound: `9.5%`
  - clean PnL: `$-0.83`
- Broader historical paper record looks better, but is not sufficient for promotion
  - all BTC paper trades: `453`
  - all-paper WR: `65.6%`
  - all-paper PnL: `$+310.55`
  - CLOB fills: `105`

Promotion rule: do not treat historical all-paper performance as live readiness. Promotion depends on clean post-freeze, CLOB-executable validation.

## What changed

### 1. Paper trader became live-market realistic

Main file: `btc_paper_fast.py`

Added or hardened:

- BTC-only 5m/15m trading loop.
- Real market data from Binance/Coinbase plus Polymarket Gamma/CLOB.
- CLOB executable ask-depth fill simulation.
- Gamma fallback tracking, but Gamma fallback no longer counts as live-valid evidence.
- Polymarket crypto taker-fee PnL model via `btc_fee_model.py`.
- Open-trade tracking, locked-capital reporting, equity and drawdown tracking.
- Risk halt when drawdown exceeds configured threshold.
- Max open BTC trade caps.
- Optional per-timeframe open caps.
- Optional total BTC exposure cap.
- CLOB/Gamma price gap guard.
- Executable-price edge recheck against actual CLOB ask, not stale Gamma price.
- External derivatives-flow agreement guard.
- Paper exploration mode for fake-money labs only.
- Binance rate-limit fallback path.
- Structured journal rows with strategy, CLOB price source, probability decision, external features, fees, PnL, and placement time.

### 1a. 5m and 15m paper agents can now run as separate processes

Files:

- `btc_paper_fast.py`
- `btc_split_paper_agents.py`
- `btc_telegram_reporter.py`

What changed:

- `btc_paper_fast.py` now accepts:
  - `--timeframes 5`
  - `--timeframes 15`
  - `--agent-id btc-5m`
  - `--agent-id btc-15m`
  - `--capital`
  - `--best-params-file`
  - `--log-file`
- `btc_split_paper_agents.py` starts, stops, restarts, and reports two isolated paper agents:
  - `btc-5m`: only BTC 5-minute Polymarket markets
  - `btc-15m`: only BTC 15-minute Polymarket markets
- Each split agent gets its own:
  - PID file
  - run-until file
  - params file
  - trader log
  - stdout log
  - paper bankroll allocation
- The launcher applies paper-only runtime sizing overrides (`max_bet_pct=0.35`, `spend_ratio=0.20`) so `$10` per-agent bankrolls can clear Polymarket's 5-share minimum; params files remain separate and frozen.
- Shared journal rows now include `agent_id`, so later analysis can separate 5m and 15m evidence.
- Per-agent params are seeded from the current combined params on first start, then can diverge.
- Telegram reporting detects running split agents and prints per-agent status/stats.
- `--include-live-training` lets split paper FastGA score resolved real fills (`mode=live`) beside paper rows without placing live orders.

Default split:

```text
btc-5m  -> 5m only,  $10 paper bankroll
btc-15m -> 15m only, $10 paper bankroll
```

Validated:

```text
21 tests passed
20s paper-only smoke start/status/stop worked for both agents
```

### 2. External market-data layer added

Main file: `btc_external_metrics.py`

Read-only context now pulls or supports:

- Coinalyze: funding, predicted funding, open interest, liquidations, long/short ratio.
- CoinGlass Startup: funding, OI, liquidation, taker volume, CVD, orderbook bid/ask history.
- Binance Futures: OI, top/global L/S, taker L/S, funding.
- Bybit: OI and funding.
- Bitget: 5m/15m account L/S, OI, funding, ticker spread, depth imbalance.
- Hyperliquid: L2 book, funding, premium, DEX market structure.
- Polymarket: Gamma market data and CLOB executable books.

Composite features include external flow / sentiment inputs such as `external_bull_score`, taker/CVD imbalance, liquidation imbalance, OI change, L/S ratio, funding, and orderbook imbalance.

Access audit source: `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/AUDIT-ACCESS.md`.

### 3. Probability-model pipeline added

Files:

- `btc_feature_engine.py`
- `btc_prob_dataset.py`
- `btc_probability_model.py`
- `btc_prob_gate.py`
- `btc_model_trainer.py`
- `btc_backtest_model.py`
- `btc_edge_paper.py`
- `btc_model_pipeline.py`

Capabilities:

- Build local SQLite feature rows from BTC market structure.
- Backfill 5m/15m forward labels.
- Build training datasets from paper journals and parallel lab journals.
- Optionally include resolved live fills with `btc_prob_dataset.py --include-live`.
- Normalize Up/Down probabilities into one absolute BTC-up target.
- Preserve external derivatives features captured at entry time.
- Train calibrated probability models.
- Apply a `ProbabilityGate` before trade entry.
- Block betting when the model is missing, on probation, shape-mismatched, or lacks edge.
- Backtest model edge against market breakeven after fees.

Important behavior: model unavailable means no-bet in main conservative mode. Paper labs can run explicit fake-money exploration, but that is not live evidence.

### 4. Parameter contract and freeze gate added

Files:

- `btc_param_contract.py`
- `btc_freeze_strategy.py`
- `btc_live_go_nogo.py`

Rules:

- Strategy version: `btc-window-sniper-frozen-v1`.
- Optimizers may only mutate:
  - `delta_thresh`
  - `conf_thresh`
  - `ens_thresh`
- Execution/risk constants stay code-owned.
- Freeze snapshots hash code and params.
- Go/no-go checks compare current code against freeze.
- Live canary gate fails if code or params drift after freeze.

Current gate result: `NO_GO`.

### 5. Parallel self-improvement added

Files:

- `btc_parallel_backtest_orchestrator.py`
- `btc_backtest_autoresearch.py`
- `autoimprove_v3.py`
- `btc_parallel_autoresearch.py`
- `btc_parallel_orchestrator.py`

Capabilities:

- Run multiple isolated optimization workers.
- Score candidates on real paper journal history.
- Use chronological holdout to reduce overfit.
- Commit only better normalized parameter candidates.
- Freeze on commit.
- Include extra parallel-lab journals without corrupting the main paper session.

Important behavior: this is paper-only. It writes candidate params, not real orders.

### 6. Parallel paper lab added

Files:

- `btc_parallel_paper_lab.py`
- `btc_parallel_paper_lab_watchdog.py`
- `btc_live_valid_swarm_curator.py`

Capabilities:

- Launch 1-16 isolated fake-money strategy workers.
- Scrub live wallet/order credentials from worker environments.
- Force `POLYMARKET_LIVE_TRADING=0` and fake-money markers.
- Give each strategy its own state, params, journal, and log files.
- Report live-valid CLOB-only metrics separately from Gamma fallback optimism.
- Curate workers by killing bad/halted CLOB-valid performers and launching focused waves.

Default strategy cohorts:

- quality
- balanced
- momentum
- confidence
- aggressive
- firehose
- firehose_guard
- breakout
- quarantine

### 7. Gym/readiness harness added

Files:

- `btc_trading_gym.py`
- `readiness_monitor.py`
- `btc_live_go_nogo.py`

Capabilities:

- Score current paper/live-readiness.
- Produce Markdown and JSON reports.
- Enforce safety checks:
  - paper uses live market data
  - paper code has no order POST patterns
  - paper uses CLOB executable fill code
  - trader/watchdog running
  - optimizer enabled
  - Telegram reporting enabled
  - minimum clean trades
  - Wilson lower-bound WR
  - positive clean PnL
  - drawdown cap
  - total paper trade count

Report path:

`/Users/sebastian/MAKAKOO/data/reports/gym/btc-polymarket-trader/latest.md`

### 8. Operations and reporting added

Files:

- `btc_paper_fast_watchdog.py`
- `btc_restart_when_flat.py`
- `btc_telegram_reporter.py`
- launchd plists under `~/Library/LaunchAgents/com.makakoo.arbitrage.*.plist`

Capabilities:

- Keep paper trader alive for bounded runs.
- Restart stale trader process.
- Enforce param contract on every watchdog loop.
- Print or send status reports.
- Report trader, watchdog, optimizer, run-until, clean trades, WR, PnL, drawdown, and evidence files.
- Restart when flat so param changes do not interrupt open virtual trades.

### 9. Tests added

New tests include:

- `test_btc_backtest_multijournal.py`
- `test_btc_external_metrics.py`
- `test_btc_paper_exploration.py`
- `test_btc_parallel_paper_lab.py`
- `test_btc_prob_dataset.py`
- `test_btc_telegram_reporter.py`

Coverage areas:

- Extra lab journal loading and de-duplication.
- External metrics feature defaults and bounded composite score.
- Feature DB schema migration for external columns.
- Probability gate legacy model compatibility.
- Binance rate-limit fallback behavior.
- Paper exploration constraints.
- Parallel lab paper-only credential scrubbing.
- Dataset normalization for Up/Down outcomes.
- Telegram reporter live config extraction.

## Files changed locally

Tracked modified files:

- `autoimprove_live.py`
- `btc_backtest_autoresearch.py`
- `btc_backtest_daemon.sh`
- `btc_paper_fast.py`
- `btc_sniper_live.py`
- `config.py`
- `nightly_evolve.sh`
- `nightly_scan.py`
- `nightly_strategy_research.sh`
- `polymarket_paper_trader.py`
- `real_evolve.py`
- `requirements.txt`
- `run_autoimprover.sh`
- `run_intraday_trader.sh`
- `sniper_heartbeat.py`
- `sniper_live.py`
- `strategy_genome_v3.py`

New source/test files:

- `agent.yaml`
- `autoimprove_v3.py`
- `btc_backtest_model.py`
- `btc_edge_paper.py`
- `btc_external_metrics.py`
- `btc_feature_engine.py`
- `btc_fee_model.py`
- `btc_freeze_strategy.py`
- `btc_live_go_nogo.py`
- `btc_live_valid_swarm_curator.py`
- `btc_model_pipeline.py`
- `btc_model_trainer.py`
- `btc_paper_fast_watchdog.py`
- `btc_parallel_autoresearch.py`
- `btc_parallel_backtest_orchestrator.py`
- `btc_parallel_orchestrator.py`
- `btc_parallel_paper_lab.py`
- `btc_parallel_paper_lab_watchdog.py`
- `btc_param_contract.py`
- `btc_prob_backtest.py`
- `btc_prob_dataset.py`
- `btc_prob_gate.py`
- `btc_probability_model.py`
- `btc_restart_when_flat.py`
- `btc_telegram_reporter.py`
- `btc_trading_gym.py`
- `nightly_prob_retrain.py`
- `readiness_monitor.py`
- test files listed above

## What is intentionally not included in source docs/commits

Runtime artifacts should not be committed:

- `data/arbitrage-agent/v2/logs/`
- `data/arbitrage-agent/v2/parallel_lab/`
- `data/arbitrage-agent/v2/state/*.pid`
- `data/arbitrage-agent/v2/state/*.lock`
- `data/arbitrage-agent/v2/model/*.pkl`
- large journals and generated reports unless explicitly versioned
- secrets / `.env.live`

## Bottom line

We built a serious paper-validation and research stack. We did not prove live edge yet; live fills are now first-class opt-in evidence for analysis instead of being ignored by training.

After the May 8 audit, the stricter statement is: do not restart live until the P0 shared decision engine exists and passes validation on unique, de-duplicated market windows. The old live path was not a calibrated probability engine.

The app is ready for:

- paper training
- signal-only monitoring
- model/data research
- CLOB-executable validation
- manual review of candidate trade tickets

The app is not ready by default for:

- autonomous real-money trading
- “best WR” live trade picking
- live canary unless Sebastian explicitly authorizes a controlled override
