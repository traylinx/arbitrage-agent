#!/usr/bin/env python3
"""
Continuous Auto-Improvement Loop v2 - doesn't break RSI bands.
"""

import json
import os
import time
import random
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state" / "intraday_trades.json"
PARAMS_FILE = Path(__file__).parent / "state" / "best_intraday_params.json"
LOG_FILE = Path(__file__).parent / "logs" / "auto_improve.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"capital": 100.0, "trades": [], "wins": 0, "losses": 0, "breakeven": 0}

def analyze_and_adjust():
    state = load_state()
    capital = state.get("capital", 100.0)
    trades = state.get("trades", [])
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)
    breakeven = state.get("breakeven", 0)
    
    total = wins + losses + breakeven
    wr = wins / max(total, 1) if total > 0 else 0
    pnl = capital - 100.0
    
    log(f"ANALYSIS: capital=${capital:.2f} PnL=${pnl:+.2f} trades={total} WR={wr:.0%}")
    
    with open(PARAMS_FILE) as f:
        data = json.load(f)
    params = data["params"]
    
    # NEVER go past RSI 40/60 - that's the trading zone
    # Only adjust stop loss and size based on performance
    
    if len(trades) < 5:
        log("Too few trades - keeping current params")
    elif pnl > 3:
        log("WINNING! Tighten stops and increase size")
        params["stop_loss_pct"] = max(params.get("stop_loss_pct", 0.3) * 0.8, 0.1)
        params["take_profit_pct"] = max(params.get("take_profit_pct", 0.6) * 0.8, 0.2)
        params["size_pct"] = min(params.get("size_pct", 0.15) * 1.3, 0.4)
    elif pnl > 1:
        log("SLIGHTLY UP - small adjustments")
        params["stop_loss_pct"] = max(params.get("stop_loss_pct", 0.3) * 0.9, 0.15)
        params["take_profit_pct"] = max(params.get("take_profit_pct", 0.6) * 0.9, 0.3)
    elif pnl < -10:
        log("LOSING BADLY - wider RSI bands for more trades")
        params["rsi_oversold"] = min(params.get("rsi_oversold", 35) + 5, 45)
        params["rsi_overbought"] = max(params.get("rsi_overbought", 65) - 5, 55)
        params["size_pct"] = max(params.get("size_pct", 0.2) * 0.7, 0.1)
    elif pnl < -3:
        log("LOSING - reduce size, keep RSI trading")
        params["size_pct"] = max(params.get("size_pct", 0.2) * 0.8, 0.1)
    else:
        log("BREAKEVEN/STABLE - minor tweaks")
        # Random small adjustment
        if random.random() < 0.3:
            params["size_pct"] = max(min(params.get("size_pct", 0.2) * random.choice([0.9, 1.1]), 0.1), 0.4)
    
    # Keep RSI in trading range 35-45 / 55-65
    params["rsi_oversold"] = max(30, min(params.get("rsi_oversold", 35), 45))
    params["rsi_overbought"] = max(55, min(params.get("rsi_overbought", 65), 70))
    
    with open(PARAMS_FILE, "w") as f:
        json.dump({"params": params, "score": 100.0, "timestamp": datetime.now().isoformat()}, f, indent=2)
    
    log(f"UPDATED: RSI={params['rsi_oversold']}/{params['rsi_overbought']} SL={params['stop_loss_pct']:.2f}% TP={params['take_profit_pct']:.2f}% size={params['size_pct']:.1%}")
    return params

if __name__ == "__main__":
    log("=== Continuous Improvement v2 Started ===")
    while True:
        analyze_and_adjust()
        time.sleep(600)
