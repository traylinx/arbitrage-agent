# BTC Trading App Postmortem + Action Plan — 2026-05-08

## Current state

- Live trading: **stopped**.
- Live kill switch: **active**.
- BTC processes: **none should be running**.
- Repo: `https://github.com/traylinx/arbitrage-agent.git`
- Local folder: `/Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src`
- Root-cause audit: `docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md`
- Read-only audit command:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
$PY btc_decision_audit.py --hours 36 --limit 12
```

## What was wrong

### 1. Live and paper did not use the same decision engine

Paper/lab had the richer path:

- feature vector,
- external derivatives context,
- probability-model gate,
- CLOB executable ask simulation,
- executable-edge checks,
- detailed `prob_features` / `prob_decision` journal rows.

Live used a separate legacy path:

- BTC window delta,
- `SignalEngine.ensemble()` confidence,
- a hand-weighted external vote,
- Gamma outcome price plus optional fill premium,
- then real CLOB order placement.

This is the biggest architecture fault. Testing one engine and trading another makes paper results misleading.

### 2. Live did not trade from calibrated probability

Live did **not** compute:

```text
P(direction) - executable_clob_ask >= required_edge
```

So live was not asking the correct question: "does this bet have positive expected value after execution price, fees, slippage, and uncertainty?"

It was asking a weaker question: "did BTC move enough and do some external signals roughly agree?"

That is not enough for 5m/15m prediction markets.

### 3. External data was stale in live decisions

Observed live fills included external-data ages:

- `137s`
- `153s`
- `187s`
- `239s`

For a 5m market, `239s` is nearly the whole market. That data is too old to justify a live order.

The `>180s` bucket in the audit was:

```text
0W / 3L, -$7.65
```

### 4. Premium aggregate feeds were missing during key live fills

Filled live rows with provider summary showed:

```text
Coinalyze: 0/8
CoinGlass: 0/8
Binance: 8/8
Bybit: 8/8
Bitget: 8/8
Hyperliquid: 8/8
```

Meaning: live was not actually using the full data stack Sebastian expected. It was mostly public venue data plus a composite score.

### 5. Missing data looked like neutral data

`context_feature_subset()` converts missing keys into `0.0`.

That means:

```text
source missing
```

and

```text
source says neutral
```

can become the same numerical feature unless explicit provider-health and missingness flags are used.

This corrupts both model training and go/no-go decisions.

### 6. Probability model was not production-valid

The current probability model was already on probation:

```text
Offline probability model passed backtest but failed fresh fake-money validation.
```

Model metadata was also bad:

| Model | Train rows | Test AUC | Status |
|---|---:|---:|---|
| shared current | 1553 | 0.318 | probation / disabled |
| 5m split | 1086 | 0.249 | bad |
| 15m split | 480 | 0.500 | no edge |

AUC below `0.5` means ranking is worse than random on that sample.

### 7. Paper/lab was mostly heuristic exploration, not validated model trading

Audit showed:

```text
MODEL_DOWN exploration heuristic: 1597 rows
other_gate: 254 rows
```

So the lab was collecting labels and exploring, not proving that a validated probability model could beat the market.

### 8. Autoimprove overfit tiny samples

Small runs with 10-16 trades sometimes looked like `80%+ WR`, then collapsed at family level.

Family-level result from the same window:

| Strategy family | WR | PnL |
|---|---:|---:|
| firehose_gamma_probe | 40.8% | -$337.75 |
| aggressive_clob_floor | 40.3% | -$320.42 |
| firehose_clob_quality | 53.4% | -$12.67 |
| quality_mid | 54.5% | +$2.50 |

Tiny winners were sampling noise until proven on fresh, unique markets.

### 9. Swarm evidence was not independent

Parallel agents often bought the same market/direction.

Worst example:

```text
btc-updown-15m-1778114700 Up
81 duplicate bets
0 wins
-$245.80 fake
```

That is not 81 independent observations. It is one wrong market idea multiplied 81 times.

### 10. Down-side live bias burned the wallet

Live filled Down trades:

```text
8 trades, 2W / 6L, 25.0% WR, -$9.88
```

5m Down was the main money burner.

## What was improved already

### 1. Emergency stop / fail-closed behavior

Implemented and documented:

- `btc_split_live_agents.py stop` arms the durable kill switch.
- Stop command cancels CLOB open orders best-effort.
- Live status now reports strict filled-journal metrics.
- Live kill switch file blocks restart:

```text
data/arbitrage-agent/v2/state/live_trading_disabled.json
```

### 2. Live circuit breakers

Live now has configurable hard stops:

```bash
BTC_LIVE_MAX_FILLED_LOSSES=2
BTC_LIVE_MAX_DRAWDOWN_USDC=2.75
BTC_LIVE_MIN_WR_TRADES=4
BTC_LIVE_MIN_WR=0.55
```

If breached, live stops, cancels orders, and writes the kill switch.

### 3. Live param mutation disabled by default

Default:

```bash
BTC_LIVE_ALLOW_PARAM_MUTATION=0
```

Live will not mutate execution params based on noisy tiny samples unless explicitly allowed.

### 4. Entry path no longer waits on slow balance retry

The order-entry path now uses non-blocking balance refresh and rechecks `seconds_left` before order placement.

This reduces the late-entry failure where the bot checks too late in a 5m market.

### 5. Duplicate reconcile accounting fixed

Filled orders no longer get counted twice through normal resolution plus later reconcile synthesis.

### 6. Stale external-data guard added

Default:

```bash
BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75
```

If external context is older than `75s`, live GO is rejected.

This blocks the observed `137s` to `239s` live failure shape.

### 7. Premium-feed guard added

Default:

```bash
BTC_LIVE_REQUIRE_PREMIUM_DERIVATIVE_FEED=1
```

If both Coinalyze and CoinGlass are unavailable, live GO is rejected.

This blocks the observed case where live accepted trades while both premium aggregate feeds were absent.

### 8. Read-only decision audit tool added

New tool:

```bash
$PY btc_decision_audit.py --hours 36 --limit 12
```

It reports:

- live WR/PnL by parameter,
- live WR/PnL by direction,
- live WR/PnL by timeframe,
- live WR/PnL by external-data age,
- provider availability summary,
- paper/lab WR/PnL by strategy family,
- `MODEL_DOWN` exploration count,
- duplicate market windows.

### 9. Documentation updated

Updated/added:

- `README.md`
- `CHANGELOG.md`
- `docs/TRADING_APP_USER_MANUAL.md`
- `docs/TRADING_APP_UPDATE_SUMMARY.md`
- `docs/TRADING_APP_USE_CASES.md`
- `docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md`
- `docs/TRADING_APP_POSTMORTEM_ACTION_PLAN_2026-05-08.md`

### 10. Tests added and passing

Validation:

```text
56 passed
```

New tests cover:

- stale external context rejection,
- missing Coinalyze/CoinGlass rejection,
- live safety behavior.

## What still must be implemented

### P0 — one shared decision engine for paper and live

Build a single engine used by both modes:

```text
MarketSnapshot
+ BtcSnapshot
+ ExternalSnapshot
+ ClobQuote
=> DecisionSnapshot
=> GO/NO_GO
```

Live must not use a separate heuristic path.

Required output per decision:

```text
P(up)
P(down)
chosen direction
executable ask
edge after fees/slippage
external source health
top positive/negative features
final GO/NO_GO reason
```

### P0 — live must use executable CLOB edge

No live trade unless:

```text
P(direction) - executable_clob_ask >= required_edge
```

Gamma price may be logged as market metadata, but cannot be treated as execution truth.

### P0 — block live when probability model is on probation

If this file exists:

```text
data/arbitrage-agent/v2/model/btc_prob_model_probation.json
```

then live startup should refuse unless a named override is passed.

### P0 — add missingness/provider-health features

Add explicit model features:

```text
ca_ok
cg_ok
bn_ok
by_ok
bg_ok
hl_ok
external_age_sec
provider_count_ok
premium_feed_ok
missing_ca
missing_cg
```

Missing source data must not silently become neutral `0.0`.

### P0 — de-duplicate lab scoring by market window

Promotion scoring must group by:

```text
(market_slug, direction)
```

or:

```text
(window_start, timeframe, direction)
```

Then calculate metrics on unique market ideas, not duplicate worker bets.

### P0 — monkey baseline

Every strategy report must compare against random Up/Down on the same windows and prices.

A strategy is blocked unless it beats:

```text
random baseline after fees/slippage with confidence interval
```

### P1 — real regime classifier

Tag every market window with regime features:

- trend/chop,
- realized volatility,
- range compression/expansion,
- liquidation pressure,
- OI expansion/contraction,
- funding regime,
- orderflow agreement/disagreement,
- market time/session.

Then only allow live when regime is historically profitable out-of-sample.

### P1 — calibration and holdout report

Before promotion, require:

- out-of-sample by day,
- out-of-sample by timeframe,
- out-of-sample by regime,
- calibration curve,
- Brier score,
- AUC above random,
- Wilson lower bound above breakeven,
- positive PnL after fees/slippage.

### P1 — source attribution in logs

Every GO/NO_GO line should include:

```text
source_age
provider_ok map
P(up)/P(down)
exec_price
edge
main blockers
feature attribution
```

The operator should see why a trade is rejected or accepted without opening JSON journals.

### P1 — separate 5m and 15m model lifecycle

5m and 15m behave differently. They need independent:

- datasets,
- calibration,
- validation gates,
- regime maps,
- promotion thresholds.

Shared features are fine. Shared model weights are not safe until proven.

### P2 — better data-source value audit

Measure each provider’s incremental value:

- with source,
- without source,
- source stale,
- source missing,
- source conflicting with price action.

Then remove or downweight sources that do not improve out-of-sample edge.

### P2 — manual canary ticket generator

Before any live restart, generate a human-readable ticket:

```text
market
side
size
P(direction)
exec ask
edge
source state
risk cap
stop condition
reason for trade
reason not to trade
```

Human reviews first. Then maybe live automation later.

## Do not restart live until these are true

Minimum restart bar:

1. Shared decision engine implemented.
2. Live and paper use the same `DecisionSnapshot` path.
3. Model probation removed by fresh validation, not manually ignored.
4. Unique-market paper validation is profitable.
5. Monkey baseline beaten after fees/slippage.
6. External data is fresh and source health is logged.
7. CLOB executable-edge check is mandatory.
8. Circuit breakers remain enabled.
9. A manual canary ticket is reviewed before first live order.

Until then, live is not trading. It is gambling.
