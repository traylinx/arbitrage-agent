---
name: arbitrage-agent
description: Consolidated Polymarket trading agent — intraday momentum trading, arbitrage scanning, strategy evolution, BTC sniping, and auto-improvement. All trading code in one repo.
version: 6.0
author: Harvey OS
tags: [polymarket, trading, intraday, btc, crypto, arbitrage, sniper, evolution]
---

# Arbitrage Agent v6 — Consolidated Trading Monorepo

All Polymarket trading code consolidated from 5 locations into one repo.

---

## Directory Structure

```
agents/arbitrage-agent/
├── SKILL.md                  ← This file
├── AUDIT.md                  ← Security audit notes
├── requirements.txt          ← Python dependencies
├── .gitignore
│
│  ── Core Trading (from data/arbitrage-agent/v2/) ──
├── intraday_trader.py        ← Main intraday momentum trader
├── rtds_streamer.py          ← WebSocket streamer for BTC/ETH/SOL prices
├── scanner.py                ← Polymarket Gamma/CLOB API scanner
├── engine.py                 ← Simulation engine
├── executor.py               ← Trade executor
├── paper_trader.py           ← Paper trading mode
├── config.py                 ← Configuration
├── strategy_genome.py        ← Strategy genome + genetic evolution
├── simulate.py               ← Backtesting simulator
├── autoimprove.py            ← AI-powered strategy optimizer (MiniMax)
├── continuous_improve.py     ← Continuous improvement loop
├── __init__.py
│
│  ── Orchestration (from harvey-os/skills/arbitrage-agent/) ──
├── candle_engine.py          ← OHLCV candle aggregation from WebSocket
├── indicators.py             ← Technical indicators (RSI, MACD, Bollinger, etc.)
├── real_evolve.py            ← Real-data strategy evolution
├── log_trading_pnl.py        ← P&L logging
├── nightly_scan.py           ← Nightly opportunity scanner
├── strategy_researcher.py    ← AI-driven strategy research
├── nightly_evolve.sh         ← Cron: nightly evolution run
├── nightly_strategy_research.sh  ← Cron: nightly strategy research
├── run_autoimprover.sh       ← Cron: auto-improver launcher
├── run_intraday_trader.sh    ← Cron: intraday trader launcher
│
│  ── Blockchain / Execution (from arbitrage-research-agent/) ──
├── blockchain_check.py       ← On-chain balance verification
├── execute_arbitrage.py      ← CLOB arbitrage order execution
├── live_executor.py          ← Live order placement via py-clob-client
├── simulator.py              ← Market simulator
├── wallet_check.py           ← Wallet balance + API credential check
│
│  ── Autoresearch / Sniping (from tmp/autoresearch/) ──
├── autoresearch_loop.py      ← Automated research loop
├── btc_sniper_v5.py          ← BTC market sniper v5
├── sniper_live.py            ← Live sniper execution
├── sniper_strategy.py        ← Sniper strategy logic
├── prepare.py                ← Data preparation (training)
├── train.py                  ← Model training
│
│  ── Loose Scripts ──
└── sniper_heartbeat.py       ← Sniper process heartbeat monitor
```

---

## How to Run

### Intraday Paper Trading
```bash
cd ~/HARVEY/agents/arbitrage-agent
python3 intraday_trader.py --capital 100 --poll 5
```

### Auto-Improve (evolve strategy parameters)
```bash
python3 intraday_trader.py --improve --gens 10 --pop 30
```

### Nightly Evolution
```bash
bash nightly_evolve.sh
```

### BTC Sniper
```bash
python3 btc_sniper_v5.py
```

---

## State & Data

Runtime state lives in `~/HARVEY/data/arbitrage-agent/` (not in this repo):
- `v2/state/` — trade state, journals, best params
- `v2/logs/` — trading logs

---

## Environment

Live trading requires (in `.env`):
- `POLYMARKET_PRIVATE_KEY`
- `POLYMARKET_FUNDER_ADDRESS`

---

## IMPORTANT

- Paper trading ONLY by default. No real money without explicit config.
- Bankroll is ~$0.71 — not enough for real trades. Needs $20-50 minimum.
