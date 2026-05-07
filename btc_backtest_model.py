#!/usr/bin/env python3
"""
BTC Model Backtest — Simulate the edge-trader on historical feature DB.

Replays every minute where we have features + a Polymarket price, computes
the model signal vs breakeven, and records simulated trades.
Produces the same journal format as btc_edge_paper.py for direct comparison.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from btc_model_trainer import FEATURE_COLS, load_data, prepare_df
from btc_fee_model import taker_fee_usdc, resolved_buy_pnl

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
FEATURE_DB = Path(os.environ.get("BTC_FEATURE_DB", str(DATA_DIR / "state" / "btc_features.db")))
MODEL_DIR = DATA_DIR / "state" / "models"


def simulate(df: pd.DataFrame, model, horizon: str = "5m", edge_margin: float = 0.03, min_edge: float = 0.0):
    """Walk-forward simulate on a single test DataFrame."""
    if model is None:
        return []
    trades = []
    n_features = int(getattr(model, "n_features_in_", len(FEATURE_COLS)) or len(FEATURE_COLS))
    cols = FEATURE_COLS[:n_features]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[cols].values
    probs = model.predict_proba(X)[:, 1]

    for j, (_, row) in enumerate(df.iterrows()):
        prob_up = probs[j]
        yes_price = row.get("poly_5m_yes") if horizon == "5m" else row.get("poly_15m_yes")
        if yes_price is None or math.isnan(yes_price):
            continue
        no_price = row.get("poly_5m_no") if horizon == "5m" else row.get("poly_15m_no")
        if no_price is None or math.isnan(no_price):
            no_price = 1.0 - yes_price

        fee_rate = 0.072
        be_yes = yes_price + fee_rate * yes_price * (1.0 - yes_price)
        be_no = no_price + fee_rate * no_price * (1.0 - no_price)

        direction = None
        model_prob = None
        entry_price = None

        if prob_up > be_yes + edge_margin:
            direction = "Up"
            model_prob = prob_up
            entry_price = yes_price
        elif (1.0 - prob_up) > be_no + edge_margin:
            direction = "Down"
            model_prob = 1.0 - prob_up
            entry_price = no_price

        if direction is None:
            continue

        # Determine outcome from label
        label = row["y"]
        won = (direction == "Up" and label == 1) or (direction == "Down" and label == 0)
        # Assume $5 spend for sizing consistency
        spend = 5.0
        size = spend / entry_price if entry_price > 0 else 0.0
        fee = taker_fee_usdc(size, entry_price)
        pnl = resolved_buy_pnl(won, size, entry_price)

        trades.append(
            {
                "ts": int(row["ts"]),
                "dt": str(row["dt"]),
                "horizon": horizon,
                "direction": direction,
                "entry_price": round(entry_price, 4),
                "model_prob": round(model_prob, 4),
                "breakeven": round(be_yes if direction == "Up" else be_no, 4),
                "edge": round(model_prob - (be_yes if direction == "Up" else be_no), 4),
                "size": round(size, 4),
                "fee": round(fee, 4),
                "won": won,
                "pnl": round(pnl, 4),
                "label": int(label),
            }
        )
    return trades


def main():
    if not FEATURE_DB.exists():
        print(f"Feature DB not found: {FEATURE_DB}")
        return

    df = load_data(FEATURE_DB)
    print(f"Loaded {len(df)} rows from feature DB.")

    for horizon in ("5m", "15m"):
        model_path = MODEL_DIR / f"btc_model_{horizon}.pkl"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        df_h = prepare_df(df, horizon)
        # Use last 20% as holdout for backtest
        split_idx = int(len(df_h) * 0.8)
        test_df = df_h.iloc[split_idx:].copy()
        if len(test_df) < 20:
            print(f"Insufficient test data for {horizon}")
            continue

        trades = simulate(test_df, model, horizon=horizon, edge_margin=0.03)
        if not trades:
            print(f"{horizon}: No trades generated on holdout.")
            continue

        wins = sum(1 for t in trades if t["won"])
        total = len(trades)
        wr = wins / total
        pnl = sum(t["pnl"] for t in trades)
        mean_edge = sum(t["edge"] for t in trades) / total
        mean_prob = sum(t["model_prob"] for t in trades) / total

        print(f"\n=== {horizon} Backtest (holdout, n={total}) ===")
        print(f"  WR={wr:.1%}  PnL=${pnl:+.2f}  mean_edge={mean_edge:.3f}  mean_prob={mean_prob:.3f}")

        # Edge bin analysis
        trades_sorted = sorted(trades, key=lambda x: x["edge"])
        for min_e in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
            subset = [t for t in trades if t["edge"] >= min_e]
            if len(subset) < 5:
                continue
            sw = sum(1 for t in subset if t["won"])
            sp = sum(t["pnl"] for t in subset)
            print(f"  edge>={min_e:.2f}: n={len(subset):>3}  WR={sw/len(subset):.1%}  PnL=${sp:+.2f}")

        # Save backtest journal
        bt_journal = DATA_DIR / "state" / f"backtest_model_{horizon}.jsonl"
        with open(bt_journal, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")
        print(f"  Saved backtest journal to {bt_journal}")


if __name__ == "__main__":
    main()
