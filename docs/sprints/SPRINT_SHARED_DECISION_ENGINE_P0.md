# Sprint: BTC Shared Decision Engine P0

Generated: 2026-05-08
Lope mode used: `lope ask`
Validator: `codex`
Status: planning output generated. No implementation run yet.

## Lope run notes

`lope negotiate` was attempted first with the configured validator team. The team run hung, and a reduced `codex` negotiate run timed out. A reduced `lope ask` prompt with the working `codex` validator completed and produced the plan below.

Live remains blocked. This sprint is the required P0 implementation before any live restart.

## Hard verdict

Live remains blocked: **YES**.

Current live path cannot restart until:

- shared `DecisionEngine` exists,
- model probation is cleared by fresh validation,
- unique-market paper evidence beats monkey baseline,
- executable CLOB ask edge is mandatory,
- source-health/missingness is first-class,
- live and paper journal identical `DecisionSnapshot` schema.

## Phase 1 — Shared decision core

Goal: one pure decision path used by paper and live.

Files:

- Add `btc_decision_engine.py`
  - `MarketSnapshot`
  - `BtcSnapshot`
  - `ExternalSnapshot`
  - `SourceHealth`
  - `ClobQuote`
  - `DecisionSnapshot`
  - `DecisionConfig`
  - `DecisionEngine.evaluate(...)`
- Add/extend `btc_model_health.py`, or keep model-health inside engine:
  - detects `/Users/sebastian/MAKAKOO/data/arbitrage-agent/v2/model/btc_prob_model_probation.json`
  - hard-blocks live if present
- Move reusable CLOB quote logic out of `btc_paper_fast.py`:
  - into `btc_decision_engine.py` or `btc_clob.py`
  - sort asks and use lowest executable ask
  - simulate depth for target spend, min shares, max spend

Rules:

- Gamma price is metadata only.
- Executable truth is CLOB ask/depth.
- GO only if:

```text
model_prob(direction) - executable_clob_ask >= required_edge
```

Tests:

- `test_btc_decision_engine.py`
  - no CLOB ask => `NO_GO`
  - Gamma favorable but CLOB ask bad => `NO_GO`
  - model edge above executable ask threshold => `GO`
  - hard poly cap blocks
  - model probation blocks live
  - model probation may allow isolated paper/lab only with explicit flag

## Phase 2 — Wire paper + live to same engine

Goal: delete decision drift.

Files:

- `btc_paper_fast.py`
  - replace local `ProbabilityGate` + `_runtime_guards_allow` + `_execution_edge_allows` decision chain with `DecisionEngine.evaluate(...)`
  - keep paper execution/journaling
  - journal full `DecisionSnapshot`
- `btc_sniper_live.py`
  - `_check_signal` becomes candidate/context builder only
  - `_place_trade` requires `sig["_decision_snapshot"].go is True`
  - order price/size/token comes from `DecisionSnapshot.clob_quote`, not Gamma + premium
  - journal `prob_features`, `prob_decision`, CLOB fields, source health
- `btc_split_live_agents.py`
  - startup preflight calls model-health/live-block check before launching agents
- `btc_live_go_nogo.py`
  - add gate: live/paper engine version hash present in rows
  - require `decision_engine_version` and `exec_edge_checked`

Tests:

- Extend `test_btc_live_safety.py`
  - live refuses startup on probation file
  - live refuses order without decision snapshot
  - live refuses stale snapshot
  - live uses executable CLOB ask, not Gamma price
  - live blocks when `exec_edge_checked != true`
- Extend `test_btc_paper_exploration.py`
  - paper journals same decision schema as live

## Phase 3 — Source health + missingness as first-class features

Goal: missing source is not neutral source.

Files:

- `btc_external_metrics.py`
  - emit provider status map:
    - `ca_ok`
    - `cg_ok`
    - `bn_ok`
    - `by_ok`
    - `bg_ok`
    - `hl_ok`
    - `external_age_sec`
    - `provider_count_ok`
    - `premium_feed_ok`
    - `missing_ca`
    - `missing_cg`
- `btc_prob_gate.py`
  - include health/missingness keys in `FEATURE_KEYS`
  - fail closed if model feature schema mismatches runtime schema
- `btc_prob_dataset.py`
  - persist health/missingness fields
- `btc_probability_model.py`
  - train/calibrate with explicit missingness
- `btc_decision_engine.py`
  - source-health gate:
    - live requires fresh external context
    - live requires premium feed unless explicit paper/lab mode
    - no silent zero imputation without missing flag

Tests:

- Extend `test_btc_external_metrics.py`
  - missing Coinalyze sets `ca_ok=0`, `missing_ca=1`
  - neutral Coinalyze sets `ca_ok=1`, `missing_ca=0`
  - stale external context blocks live
  - both Coinalyze and CoinGlass missing blocks live
- Add model schema test:
  - old bundle missing source-health keys => no live GO

## Phase 4 — Unique-market lab scoring + monkey baseline

Goal: stop counting duplicate bets as independent proof.

Files:

- Add `btc_lab_scoring.py`
  - groups by preferred key `(market_slug, direction)`
  - fallback key `(window_start, window_tf, direction)`
  - computes unique-market WR/PnL
  - reports duplicate count separately
  - computes bootstrap/Wilson confidence
  - computes deterministic monkey baseline on same windows/prices
- Update `btc_parallel_paper_lab.py`
  - status/report uses unique-market metrics
  - includes duplicate concentration
  - includes monkey baseline delta
- Update `btc_decision_audit.py`
  - add unique-market section
  - add monkey baseline comparison
- Update `btc_live_go_nogo.py`
  - require unique-market min N
  - require positive unique-market PnL
  - require beats monkey baseline after fees/slippage
  - require CLOB-only rows
  - require shared-engine rows only

Tests:

- Add `test_btc_lab_scoring.py`
  - duplicate 81-row market collapses to 1 unique idea
  - unique scoring preserves separate windows/directions
  - monkey baseline deterministic with seed
  - strategy below monkey baseline blocks promotion
- Extend `test_btc_parallel_paper_lab.py`
  - manifest/report exposes unique metrics + monkey delta
- Extend `test_btc_live_safety.py`
  - go/no-go fails if unique-market criteria absent

## Blockers before live restart

1. `live_trading_disabled.json` kill switch active.
2. Current probability model is on probation.
3. Live still needs shared `DecisionEngine` path.
4. Live must use executable CLOB ask edge, not Gamma-derived order price.
5. Source health/missingness not fully first-class in model/gates.
6. Lab evidence still needs unique-market scoring.
7. Monkey baseline gate missing.
8. Fresh model validation required after feature-schema change.

## Execution rule

No live trading in this sprint. Paper/shadow validation only.
