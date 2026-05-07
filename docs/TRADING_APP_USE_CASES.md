# BTC Polymarket Trading App — Use Cases

Generated: 2026-05-07

## Use case 1 — Validate a BTC 5m/15m strategy with fake money

Actor: operator  
Goal: see whether a strategy makes money on live Polymarket/BTC data without risking funds.

Flow:

1. Start paper trader or watchdog.
2. Trader watches BTC 5m and 15m Polymarket windows.
3. Trader builds BTC, Polymarket, CLOB, and external derivatives features.
4. Probability/risk gates decide whether to enter a fake-money paper position.
5. Paper fill uses executable CLOB ask depth when available.
6. Trade resolves from market outcome.
7. Journal records result, PnL, features, params, and price source.
8. Gym reads journal and reports readiness.

Success criteria:

- clean post-freeze PnL positive
- Wilson lower WR >= 55%
- at least 80 clean CLOB-executable trades
- drawdown <= 25%

## Use case 2 — Compare many strategies in parallel

Actor: research operator  
Goal: test many parameter/strategy variants faster than one linear run.

Flow:

1. Run `btc_parallel_paper_lab.py --n 8` or `--n 16`.
2. Each worker gets isolated state, params, journal, and logs.
3. Workers use fake-money env only and scrub live credentials.
4. Lab report separates CLOB-valid evidence from Gamma fallback evidence.
5. Curator kills weak/halted workers and launches focused waves.

Success criteria:

- enough resolved CLOB-valid trades per worker
- positive CLOB-valid PnL
- no live credentials in child env
- no worker writes to main global params unless promoted by optimizer

## Use case 3 — Improve parameters without changing strategy code

Actor: optimizer  
Goal: tune thresholds while keeping strategy version frozen.

Flow:

1. Read paper journal and optional lab journals.
2. Run parallel optimizer workers.
3. Score candidates on train + chronological holdout.
4. Only allow changes to dynamic params:
   - `delta_thresh`
   - `conf_thresh`
   - `ens_thresh`
5. Commit better candidate to `sniper_best_params.json`.
6. Create freeze snapshot.
7. Start fresh clean validation.

Success criteria:

- score improves by configured gap
- holdout does not degrade
- param contract strips illegal fields
- freeze created after commit

## Use case 4 — Train a probability model from paper evidence

Actor: model researcher  
Goal: replace heuristic confidence with calibrated edge probability.

Flow:

1. Collect features through `btc_feature_engine.py` or build dataset from journals.
2. Normalize Up/Down rows into absolute BTC-up labels.
3. Preserve external derivatives features from trade entry.
4. Train model with temporal split.
5. Evaluate AUC/calibration/holdout.
6. Use `ProbabilityGate` in paper trader.
7. Put weak models on probation instead of using them in main mode.

Success criteria:

- enough labeled rows
- holdout edge survives fees/spread
- probability calibration acceptable
- model gate improves clean CLOB-valid paper results

## Use case 5 — Add external market regime awareness

Actor: data engineer / trader  
Goal: avoid trading against major derivatives flow/regime signals.

Flow:

1. Fetch external context from Coinalyze, CoinGlass, Binance, Bybit, Bitget, Hyperliquid.
2. Compute composite features like `external_bull_score`.
3. Feed context into feature engine and paper trader.
4. Skip trades where external flow disagrees with chosen direction.
5. Journal external features at entry for later model training.

Success criteria:

- context fetch succeeds or cache/stale fallback works
- missing values default safely
- composite scores remain bounded
- external-flow guard reduces losing trades in holdout

## Use case 6 — Generate operational status report

Actor: operator / Telegram monitor  
Goal: know whether system is healthy without opening logs manually.

Flow:

1. Run `btc_telegram_reporter.py --print` or launch reporter.
2. Reporter inspects trader/watchdog/optimizer state.
3. Reporter reads current params, freeze, run-until, journal stats, and Gym output.
4. Operator receives current WR, PnL, drawdown, clean trade count, and next action.

Success criteria:

- report says exactly whether system is paper-only, running, stopped, or stale
- report includes gates, not just raw WR
- no misleading live-readiness claim

## Use case 7 — Decide if live canary is allowed

Actor: risk gate  
Goal: block live testing until evidence is strong enough.

Flow:

1. Freeze code and params.
2. Run paper validation after freeze.
3. Run `btc_live_go_nogo.py`.
4. Run `btc_trading_gym.py`.
5. If any gate fails, verdict remains `NO_GO` or `KEEP_TRAINING`.
6. If all gates pass, only a separately reviewed manual live canary can be considered.

Current result at doc generation: `NO_GO`.

Success criteria:

- code hashes match freeze
- params unchanged since freeze
- all rows CLOB realistic
- minimum clean trade count met
- Wilson confidence met
- positive clean PnL
- drawdown cap met

## Use case 8 — Emergency stop

Actor: operator  
Goal: stop all trading/training/reporting loops.

Flow:

1. Disable all `com.makakoo.arbitrage.*` LaunchAgents.
2. Boot out loaded labels.
3. Kill residual process groups matching BTC/arbitrage patterns.
4. Verify process count is zero.
5. Verify launchd disabled state.

Success criteria:

- matching process count = 0
- launchctl labels absent or disabled
- no watchdog respawn

## Use case 9 — Audit data-provider value

Actor: cost-conscious operator  
Goal: decide which paid/free data sources are worth keeping.

Flow:

1. Read `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/AUDIT-ACCESS.md`.
2. Compare Coinalyze, CoinGlass, Binance, Bitget, Bybit, Hyperliquid, Polymarket.
3. Prefer free sources where features overlap.
4. Keep CoinGlass only if its unique historical/orderbook data materially improves validation.

Current recommendation:

- use Coinalyze as primary aggregated regime source
- use Bitget for unique free 5m/15m account L/S
- use Binance/Bybit/Hyperliquid for free per-venue confirmation
- cancel CoinGlass unless paid-only features prove edge before renewal

## Use case 10 — Prepare a human-reviewed manual canary ticket

Actor: human trader  
Goal: test plumbing with minimal real-money risk after gates pass.

Flow:

1. Human picks exact market, side, limit price, and size.
2. System checks ticket math and max loss.
3. Human submits manually.
4. System observes/logs result.

Non-goals:

- autonomous trade selection
- autonomous live order submission
- bypassing readiness gates
