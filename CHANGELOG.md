# Changelog

All notable operator-facing changes for the BTC Polymarket trading app.


## 2026-05-10

### Added

- Added V2 ML decision-engine scaffold files to the legacy workspace repo:
  - `decision_engine.py`
  - `features/`
  - `scripts/backfill_*`
  - `scripts/live_shadow.py`
  - `test_decision_engine.py`
- Documented the standalone deployable repo and `v0.18.0` release link in `README.md`, `docs/TRADING_APP_USER_MANUAL.md`, and `docs/TRADING_APP_UPDATE_SUMMARY.md`.

### Current release of truth

- Standalone repo: `https://github.com/makakoo/agent-arbitrage-agent`
- Release: `https://github.com/makakoo/agent-arbitrage-agent/releases/tag/v0.18.0`
- Safety: paper-only until strict GO/NO_GO + live-preflight pass and live executor is implemented.

### Validation

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m compileall -q decision_engine.py features scripts test_decision_engine.py
/usr/local/opt/python@3.11/bin/python3.11 -m pytest -q \
  test_decision_engine.py features/test_external_metrics_loader.py scripts/test_backfill_btc_markets.py
```

Result: `45 passed`.

## 2026-05-08

### Added

- Added `btc_decision_audit.py`, a read-only audit command for live and paper journals.
  - Reports WR/PnL by live parameter set, direction, timeframe, and external-data age.
  - Reports provider availability for live fills.
  - Reports paper/lab performance by strategy family.
  - Reports `MODEL_DOWN` exploration usage.
  - Reports duplicated market windows so parallel swarms do not fake independent evidence.
- Added `docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md`, the root-cause audit for the May 8 loss.
- Added `docs/TRADING_APP_POSTMORTEM_ACTION_PLAN_2026-05-08.md`, the explicit what-was-wrong / what-improved / what-remains action plan.
- Added documentation links from `README.md` to the audit doc and this changelog.
- Added user-manual instructions for running the decision audit.

### Changed

- Live external-data confirmation now fails closed by default when external context is stale:
  - `BTC_LIVE_MAX_EXTERNAL_AGE_SEC=75`
- Live external-data confirmation now requires at least one premium aggregate derivatives feed by default:
  - `BTC_LIVE_REQUIRE_PREMIUM_DERIVATIVE_FEED=1`
  - If both Coinalyze and CoinGlass are unavailable, live GO is rejected.
- Updated user manual, update summary, and use cases to document the stale-data guard, premium-feed guard, and decision-audit workflow.

### Fixed

- Blocked the observed May 8 failure shape where 5m live trades could be accepted with external context aged `137s` to `239s`.
- Blocked the observed May 8 failure shape where live could accept trades while Coinalyze and CoinGlass were both unavailable.
- Added tests covering stale external context rejection and missing premium-feed rejection.

### Audit findings

- Live strict filled May 8 window: `11` trades, `4W / 7L`, `36.4% WR`, `-$7.68`.
- 5m live: `10` trades, `3W / 7L`, `30.0% WR`, `-$9.63`.
- Recent paper/lab: `1851` trades, `858W / 993L`, `46.4% WR`, `-$757.55` fake.
- Live `>180s` external-data-age bucket: `0W / 3L`, `-$7.65`.
- Filled live rows with provider summary had Coinalyze `0/8` and CoinGlass `0/8`.
- Key architecture finding: the live path was not the same as the richer paper/model path. Live used a legacy BTC-delta heuristic plus hand-weighted external confirmation. Paper/lab had richer features and executable CLOB simulation.

### Validation

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m py_compile btc_sniper_live.py btc_decision_audit.py
/usr/local/opt/python@3.11/bin/python3.11 -m pytest -q \
  test_btc_live_safety.py test_btc_external_metrics.py test_btc_paper_exploration.py \
  test_btc_parallel_paper_lab.py test_btc_backtest_multijournal.py \
  test_btc_prob_dataset.py test_btc_telegram_reporter.py
```

Result:

```text
56 passed
```

### Commits

- `ef64b32 harden BTC live canary controls`
- `7c8c610 audit BTC decision engine losses`

## Before 2026-05-08

- Added split 5m/15m paper-agent launcher.
- Added CLOB-realistic paper fills and live-valid CLOB-only metrics.
- Added external derivatives context from Coinalyze, CoinGlass, Binance, Bybit, Bitget, and Hyperliquid.
- Added probability-model dataset/model/gate pipeline.
- Added parallel paper lab, strategy sweeper, and readiness/gym reporting.
