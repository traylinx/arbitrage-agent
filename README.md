# Arbitrage Agent — BTC Polymarket Trading App

BTC-only Polymarket 5-minute / 15-minute trading research stack.

Default supported mode: **paper trading with real market data and no real orders**.

## Docs

- [Update summary](docs/TRADING_APP_UPDATE_SUMMARY.md)
- [User manual](docs/TRADING_APP_USER_MANUAL.md)
- [Use cases](docs/TRADING_APP_USE_CASES.md)
- [Architecture/root-cause audit, 2026-05-08](docs/TRADING_APP_ARCHITECTURE_AUDIT_2026-05-08.md)
- [Changelog](CHANGELOG.md)

## Fast status

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11
$PY btc_live_go_nogo.py
$PY btc_trading_gym.py
$PY btc_decision_audit.py --hours 36 --limit 12
$PY btc_telegram_reporter.py --print
```

## Split 5m / 15m paper agents

The BTC paper trader can run as two independent paper-only processes:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

# Start both agents for 6h with total $20 virtual bankroll ($10 each)
$PY btc_split_paper_agents.py start --duration 21600 --capital-total 20

# Start/restart with per-agent FastGA learning from paper + resolved live fills
$PY btc_split_paper_agents.py restart --duration 21600 --capital-total 20 --fast-ga --include-live-training

# Check both agents
$PY btc_split_paper_agents.py status

# Stop both agents
$PY btc_split_paper_agents.py stop
```

Process split:

- `btc-5m` trades only BTC 5-minute Polymarket windows.
- `btc-15m` trades only BTC 15-minute Polymarket windows.
- Each agent has its own params, PID, run-until file, and logs.
- Split agents remain paper-only: they never submit live orders.
- With `--include-live-training`, FastGA also scores resolved `mode=live` fills from the shared journal, filtered by timeframe.

## Current hard rule

Default is **paper validation**. Real-money mode is blocked by the durable kill switch until a fresh audit removes it intentionally.

The May 8 live canary lost money. The app now fails closed:

- `btc_split_live_agents.py stop` arms the live kill switch and cancels CLOB open orders.
- Live agents stop automatically after configured filled-loss, drawdown, or WR-floor breaches.
- Live external confirmation now fails closed when derivatives context is stale or both Coinalyze and CoinGlass are unavailable.
- Live agents do **not** mutate params by default. Auto-improvement is paper/shadow-only unless `--allow-live-param-mutation` is explicitly passed.
- Status prints strict filled-journal WR/PnL separately from raw in-memory counters.
- Entry path never uses the slow balance-retry loop; it rechecks seconds-left immediately before posting an order.

Safe stop/status:

```bash
cd /Users/sebastian/MAKAKOO/plugins/agent-arbitrage-agent/src
PY=/usr/local/opt/python@3.11/bin/python3.11

$PY btc_split_live_agents.py stop
$PY btc_split_live_agents.py status
```

Live restart requires all of these, on purpose:

1. no active `data/arbitrage-agent/v2/state/live_trading_disabled.json`,
2. passing `btc_live_go_nogo.py` or the explicit operator canary override,
3. small bankroll/order caps,
4. circuit breakers left enabled.
