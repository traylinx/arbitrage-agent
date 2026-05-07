# Arbitrage Agent — BTC Polymarket Trading App

BTC-only Polymarket 5-minute / 15-minute trading research stack.

Default supported mode: **paper trading with real market data and no real orders**.

## Docs

- [Update summary](docs/TRADING_APP_UPDATE_SUMMARY.md)
- [User manual](docs/TRADING_APP_USER_MANUAL.md)
- [Use cases](docs/TRADING_APP_USE_CASES.md)

## Fast status

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
$PY btc_live_go_nogo.py
$PY btc_trading_gym.py
$PY btc_telegram_reporter.py --print
```

## Current hard rule

If `btc_live_go_nogo.py` says `NO_GO` or Gym says `KEEP_TRAINING`, continue paper validation. Do not promote to autonomous real-money trading.
