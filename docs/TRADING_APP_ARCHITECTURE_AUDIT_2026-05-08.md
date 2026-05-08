# BTC Polymarket Trading App Architecture Audit — 2026-05-08

## Status

- Repo: `https://github.com/traylinx/arbitrage-agent.git`
- Local code folder: `/Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src`
- Runtime/data folder: `/Users/sebastian/MAKAKOO/data/arbitrage-agent/`
- Current branch/commit at audit start: `main` / `ef64b32 harden BTC live canary controls`
- Live kill switch: active via `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/state/live_trading_disabled.json`
- Active BTC processes during audit: none
- Disabled launchd labels include autoimprove, strategy sweeper, paper watchdogs, readiness, Telegram reporter, and restart-when-flat jobs.

## Hard conclusion

The loss was not bad luck alone. The system behaved worse than a coin flip because the production/live decision path was not the same as the richer paper/model path, and the auto-improvement loops amplified correlated bad bets.

The app has a lot of data wired in, but it is not being used as a calibrated probability engine in live trading. In live mode it mostly traded a legacy BTC-delta/ensemble heuristic with a hand-weighted external confirmation score. That is not a robust prediction model for 5m/15m BTC direction.

## Evidence from current journals

Command used:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
/usr/local/opt/python@3.11/bin/python3.11 btc_decision_audit.py --hours 36 --limit 6
```

### Live strict filled trades

| Slice | n | W | L | WR | PnL | Avg/trade |
|---|---:|---:|---:|---:|---:|---:|
| Live total | 11 | 4 | 7 | 36.4% | -$7.68 | -$0.70 |
| 5m live | 10 | 3 | 7 | 30.0% | -$9.63 | -$0.96 |
| 15m live | 1 | 1 | 0 | 100.0% | +$1.96 | +$1.96 |
| Down live | 8 | 2 | 6 | 25.0% | -$9.88 | -$1.24 |
| Up live | 3 | 2 | 1 | 66.7% | +$2.20 | +$0.73 |

5m Down was the money burner.

### External data age in live fills

| External context age | n | WR | PnL |
|---|---:|---:|---:|
| `>180s` | 3 | 0.0% | -$7.65 |
| no external logged | 3 | 33.3% | -$1.84 |
| `121-180s` | 2 | 50.0% | -$0.24 |
| `61-120s` | 3 | 66.7% | +$2.05 |

Critical failure: the live bot accepted trades using stale external context. One loss used `age=239s`. For 5m markets, 239 seconds is almost the whole market.

### Provider availability in live fills

On filled live rows where provider status was logged:

```text
rows_with_provider_summary=8
ok_counts={
  coinalyze: 0,
  coinglass: 0,
  binance: 8,
  bybit: 8,
  bitget: 8,
  hyperliquid: 8
}
```

So during the key live losses the expensive/aggregate sources were not actually contributing. Coinalyze and CoinGlass were both absent in those filled live decisions. The system still said GO because the public-source subset was enough to pass the old provider-count check.

### Paper/lab trades over the same 36h

| Slice | n | W | L | WR | PnL | Avg/trade |
|---|---:|---:|---:|---:|---:|---:|
| Paper/lab total | 1851 | 858 | 993 | 46.4% | -$757.55 | -$0.41 |
| 5m | 1141 | 527 | 614 | 46.2% | -$468.51 | -$0.41 |
| 15m | 710 | 331 | 379 | 46.6% | -$289.04 | -$0.41 |

This is not an edge. This is a losing classifier plus fees/slippage.

Worst strategy families:

| Family | n | WR | PnL |
|---|---:|---:|---:|
| firehose_gamma_probe | 456 | 40.8% | -$337.75 |
| aggressive_clob_floor | 424 | 40.3% | -$320.42 |
| confidence_hunter | 121 | 45.5% | -$49.50 |
| balanced_contract | 271 | 52.8% | -$17.23 |
| firehose_clob_quality | 268 | 53.4% | -$12.67 |
| quality_mid | 231 | 54.5% | +$2.50 |
| firehose_confidence_flip | 44 | 54.5% | +$21.80 |

Some tiny sub-runs had 80-87% WR, but the family-level aggregate regressed to negative. Those small winners were sampling mirages, not stable strategies.

### Auto-improvement amplified duplicate bad decisions

Top duplicated market windows:

| Market/direction | Duplicates | Wins | PnL |
|---|---:|---:|---:|
| `btc-updown-15m-1778114700 Up` | 81 | 0 | -$245.80 |
| `btc-updown-5m-1778114100 Up` | 39 | 0 | -$120.14 |
| `btc-updown-5m-1778121600 Up` | 24-26 | 0 | -$74 to -$79 |

This means the lab swarm was not 81 independent confirmations. It was 81 correlated agents taking the same wrong side in the same market. A swarm can multiply evidence only when the observations are independent; here it multiplied the same mistake.

## Data-source integration map

Source wiring is in `btc_external_metrics.py`.

| Source | Fetch function | Features | Live use | Paper/model use | Problem |
|---|---|---|---|---|---|
| Coinalyze | `fetch_coinalyze_context()` | OI, liq imbalance, funding, predicted funding, long/short | Only inside `external_bull_score`; absent in live fills inspected | Included in `prob_features` as `ca_*` | Missing values become `0.0`, so missing can look neutral. Live did not require `ca_ok=1`. |
| CoinGlass | `fetch_coinglass_context()` | OI, liqs, funding, long/short, taker, CVD, orderbook | Only inside `external_bull_score`; absent in live fills inspected | Included in `prob_features` as `cg_*` | Same missing-as-zero problem; live did not require `cg_ok=1`. |
| Binance Futures | `fetch_binance_futures_context()` | OI change, top/global L/S, taker, funding | Directly in live flow (`bn_taker_15m`, `bn_top_ls`) and composite | Included in `prob_features` | Useful, but not enough by itself for 5m edge. |
| Bybit | `fetch_bybit_context()` | OI change, funding | Mostly composite/log summary | Included in `prob_features` | Underweighted in live; not a direct gate except via composite. |
| Bitget | `fetch_bitget_context()` | 5m/15m L/S, taker, trader/position L/S, recent trades, depth, basis, funding | Directly dominates live flow (`bg_taker_5m`, `bg_recent`, `bg_depth`) | Included in `prob_features` | Over-relied on in live when Coinalyze/CoinGlass were absent. |
| Hyperliquid | `fetch_hyperliquid_context()` | L2 depth, spread, funding/premium | Directly in live flow (`hl_depth`) and composite | Included in `prob_features` | Useful microstructure proxy, but not enough alone. |
| Binance spot | `btc_sniper_live.py` / `btc_paper_fast.py` | BTC price, klines, depth, RSI/MACD/BB/momentum | Primary live signal through window delta and `SignalEngine.ensemble()` | Core model features | Live effectively uses this as primary predictor. |
| Polymarket Gamma | both live and paper | Market metadata, outcome prices | Live uses Gamma `outcomePrices` as order price | Paper uses it for initial market probability | Gamma can lag executable CLOB asks. Live did not perform the same CLOB executable-edge check as paper. |
| Polymarket CLOB | `btc_paper_fast.py`, `py_clob_client` in live | Orderbooks, asks, fills | Live places order but did not quote executable ask before deciding edge | Paper simulates executable ask/depth and stores `clob_exec_price` | Major live/paper mismatch. |

## Current decision motor — live

Code path: `btc_sniper_live.py`.

1. Track Polymarket BTC 5m/15m window and Binance BTC price.
2. Compute `delta = btc_price - window_start_price`.
3. Read Gamma `outcomePrices` and compute `poly_conviction`.
4. Run `SignalEngine.ensemble()`.
5. Apply hour adjustment.
6. Fetch external context.
7. Compute hand-weighted external vote:
   - `combo = 0.65 * external_bull_score + 0.35 * flow`
   - `flow = Bitget taker/recent/depth + Binance taker/top L/S + Hyperliquid depth`
8. If external combo is aligned, boost confidence and allow GO.
9. Use Gamma outcome price, optionally add fill premium.
10. Place live CLOB order.

The live path does **not** run `ProbabilityGate.evaluate()` and does **not** compute a calibrated `P(up)` / `P(down)` versus executable CLOB ask before placing the order.

## Current decision motor — paper/lab

Code path: `btc_paper_fast.py`.

1. Build a full feature vector: Gamma prices, BTC delta, RSI, MACD, Bollinger position, volume ratio, Binance spot depth, momentum, hour, timeframe, direction, external `ca/cg/bn/by/bg/hl` features, `external_bull_score`.
2. Ask `ProbabilityGate` for model probability and edge.
3. Fetch CLOB executable buy quote using ask-depth.
4. Replace paper price with `clob_book_ask_depth` execution price.
5. Recheck edge against executable ask.
6. Store full `prob_features`, `prob_decision`, CLOB fields, and outcome.

This is closer to a real probability/edge engine than live. But in the inspected window most paper trades were not actually model-gated.

## Probability model status

Model metadata:

| Model | Train rows | Test AUC | Test WR | Status |
|---|---:|---:|---:|---|
| shared current | 1553 | 0.318 | 69.7% | On probation / disabled |
| 5m split | 1086 | 0.249 | 63.2% | Bad AUC |
| 15m split | 480 | 0.500 | 0.0% | No edge |

`btc_prob_model_probation.json` says:

```text
Offline probability model passed backtest but failed fresh fake-money validation:
model_edge_shrunk_v1 lost 8/8 and later run hit 26 trades with -47.74 PnL / 8 halted workers.
Keep model disabled until a new model passes fresh clean holdout.
```

In the last 36h paper/lab audit:

```text
MODEL_DOWN exploration heuristic: 1597 rows
other_gate: 254 rows
```

So the lab was mostly collecting labels with a heuristic fallback, not trading from a validated probability model. That explains why “we have a model” did not translate into real prediction quality.

## Why the system predicted wrong

### 1. Live was not using the Rolls-Royce engine

The richer paper feature pipeline exists, but live used a separate heuristic engine. That is the biggest architecture break. Testing one engine and trading another is how paper can look promising while live bleeds.

### 2. External data was compressed too early

Many rich signals were collapsed into one `external_bull_score` plus a small manual live `flow` score. This throws away per-source reliability and disagreement. If Coinalyze and CoinGlass are missing, the composite still exists and can look neutral/supportive.

### 3. Missing data was treated like neutral data

`context_feature_subset()` returns `0.0` for missing keys. For model training and gates, “source missing” and “source says neutral” become the same value unless separate health features are used. That corrupts learning.

### 4. Stale data was accepted in 5m markets

For 5m markets, data older than ~60-75s is stale. The live loss set included trades with `age=137s`, `153s`, `187s`, and `239s`. The `>180s` bucket was 0W/3L.

### 5. The model is not validated

The current model has bad AUC and is explicitly on probation. AUC below 0.5 means the ranking is worse than random on the tested sample. Test WR alone is misleading because thresholds/market selection can distort it.

### 6. Auto-improve optimized noise

Small sub-runs with 10-16 trades looked excellent, then family-level aggregates collapsed. The optimizer promoted apparent winners before enough independent evidence existed.

### 7. Parallel labs counted correlated duplicates as evidence

81 agents buying the same wrong Up market is not 81 samples. It is one market mistake multiplied 81 times. Any promotion logic that treats those as independent overestimates confidence.

### 8. Execution edge was not identical between paper and live

Paper simulates CLOB ask-depth and rechecks edge. Live placed at Gamma-derived price plus premium and did not calculate `model_prob - executable_ask`. That changes the strategy.

### 9. Down-side bias hurt live

Live filled Down trades were 2W/6L, -$9.88. The current heuristic over-trusted short/down signals in a regime where short-window Down was not profitable.

## Fixes implemented in this audit pass

### 1. Added read-only trade/decision audit tool

File: `btc_decision_audit.py`

It prints:

- live WR/PnL by parameter, direction, timeframe, external-data age
- live provider availability
- paper/lab WR/PnL by strategy family and timeframe
- model-down/exploration counts
- provider nonzero rates
- duplicated market windows

### 2. Hardened live external-data gate

File: `btc_sniper_live.py`

New defaults:

```bash
BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75
BTC_LIVE_REQUIRE_PREMIUM_DERIVATIVE_FEED=1
```

Live now rejects GO when:

- external context age is older than the max age; or
- both Coinalyze and CoinGlass are unavailable.

This directly blocks the observed failure shape where live accepted 5m trades with 137-239s-old context and `ca_ok=0/cg_ok=0`.

### 3. Added tests

File: `test_btc_live_safety.py`

New coverage:

- stale external context rejects live vote
- premium derivatives feed absence rejects live vote

Validation command:

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest -q \
  test_btc_live_safety.py test_btc_external_metrics.py test_btc_paper_exploration.py \
  test_btc_parallel_paper_lab.py test_btc_backtest_multijournal.py \
  test_btc_prob_dataset.py test_btc_telegram_reporter.py
```

Result:

```text
56 passed
```

## Required architecture changes before any live restart

### P0 — one shared decision engine

Extract a single `DecisionEngine` used by both paper and live:

```text
market snapshot + BTC snapshot + external snapshot
  -> feature vector
  -> calibrated probability P(up), P(down)
  -> executable CLOB quote
  -> edge after fees/slippage/uncertainty
  -> GO/NO_GO decision with reasons
```

Live must not have its own parallel heuristic path.

### P0 — live must use executable edge

No live order unless:

```text
model_prob(direction) - executable_clob_ask >= required_edge
```

Gamma price is metadata only. It is not the execution price.

### P0 — no live while model is on probation

If `btc_prob_model_probation.json` exists, live should refuse to start unless an explicit emergency override is set. A live bot without a validated model is gambling.

### P0 — source health must be first-class features

Add features like:

```text
ca_ok, cg_ok, bn_ok, by_ok, bg_ok, hl_ok
external_age_sec
source_count_ok
missing_premium_feed
```

Missing data must not be encoded as `0.0` without a separate missingness flag.

### P0 — de-duplicate lab evidence by market window

Promotion scoring must group by `(market_slug, direction)` or `(window_start, tf, direction)` before computing confidence. 81 duplicate wrong trades should count as one wrong market idea, not 81 independent failures.

### P1 — out-of-sample gates by date/regime/timeframe

Before promotion, require:

- minimum unique markets, not just trades
- per-timeframe metrics: 5m and 15m separate
- per-regime metrics: trend, chop, high-vol, low-vol
- Wilson lower bound above breakeven
- positive PnL after fees/slippage
- stable performance on future holdout, not same-window backtest

### P1 — compare against monkey baseline

Every strategy report must include:

```text
strategy WR/PnL vs random Up/Down baseline on the same windows/prices
```

If it cannot beat random after fees and confidence intervals, it is not allowed near live money.

### P1 — use source-specific attribution

Every GO/NO_GO log should print:

- `P(up)`, `P(down)`
- market price / executable CLOB ask
- edge after fees
- source votes with age and provider health
- top positive/negative features
- final blocker if skipped

## Operating rule after this audit

No live restart until P0 is done and the new shared engine passes paper validation on unique markets. The current evidence says the old live path was not prediction. It was an uncalibrated heuristic with stale/missing external data and correlated auto-improvement loops.
