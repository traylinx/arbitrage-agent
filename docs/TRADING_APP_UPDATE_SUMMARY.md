# BTC Polymarket Trading App — Update Summary

Generated: 2026-05-07  
Scope: BTC-only Polymarket 5-minute and 15-minute markets.  
Default mode: paper trading with real market data and no real orders.

## Current readiness state

The app is more capable than the original sniper, but it is not live-money ready yet.

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

We built a serious paper-validation and research stack. We did not prove live edge yet.

The app is ready for:

- paper training
- signal-only monitoring
- model/data research
- CLOB-executable validation
- manual review of candidate trade tickets

The app is not ready for:

- autonomous real-money trading
- “best WR” live trade picking
- live canary until gates pass
