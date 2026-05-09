"""evaluate_edge.py — V2 §3c step 5 + §4c bridge.

Translates AUC into per-bucket realized win rate + simulated PnL,
which is what V2 actually gates on (lower CI of EV > 0, not raw AUC).

Reads the predictions parquet emitted by train_baseline.py:
    state/models/baseline_v0/tf={tf}m_offset={offset}s_predictions.parquet

Outputs:
  - Bucket curve: for each [p_lo, p_hi] bin, n / win_rate / avg_p / simulated_pnl
  - Aggregate edge stats with bootstrap 95% CI on EV per trade
  - Calibration diagnostic: avg_p should ≈ win_rate per bucket if model
    is well-calibrated

ASSUMPTION (until live PM book history is captured): all bets are made
at $0.50 (the prior, mid of the binary). Real production uses actual
yes_ask / no_ask. The 0.50 assumption is conservative for the YES side
when the model is bullish (real ask likely above 0.50, edge slightly
smaller) and aggressive on the NO side. Treat results as a SIGNAL
indicator, not a final EV claim.

Usage:
    python3.11 -m features.evaluate_edge --tf 5
    python3.11 -m features.evaluate_edge --tf 15 --threshold 0.55 --fees 0.0
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/state/models/baseline_v0"


def load_predictions(tf: int, offset_sec: int, suffix: str = "") -> pd.DataFrame:
    """Load predictions parquet. `suffix` distinguishes models:
      - "" (logreg, default from train_baseline)
      - "_xgb" / "_lgbm" (from train_gbdt)
    """
    p = DEFAULT_OUT_ROOT / f"tf={tf}m_offset={offset_sec}s{suffix}_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"predictions not found: {p} — run trainer first")
    return pd.read_parquet(p)


def bucket_curve(
    df: pd.DataFrame, market_price: float = 0.50, fees_pct: float = 0.0,
    min_n: int = 30,
) -> pd.DataFrame:
    """For each model-confidence bucket, compute realized win rate and
    simulated PnL assuming entry at `market_price`.

    Buckets cover both directions:
      p < 0.40 → bet NO at market_price (win if Down resolves)
      p ∈ (0.40, 0.55] → no edge, skip
      p > 0.55 → bet YES at market_price (win if Up resolves)
    """
    rows = []
    # YES-side buckets: model says Up, we buy YES at market_price.
    yes_buckets = [(0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in yes_buckets:
        mask = (df["p_pred"] > lo) & (df["p_pred"] <= hi)
        n = int(mask.sum())
        if n < min_n:
            continue
        wins = int(df.loc[mask, "y_true"].sum())  # Up wins → YES wins
        losses = n - wins
        wr = wins / n
        # PnL: bet 1 share at market_price; YES pays $1 if Up resolves.
        # Win:  +1 - market_price - fees
        # Loss: -market_price - fees
        win_pnl = 1.0 - market_price - fees_pct
        loss_pnl = -market_price - fees_pct
        pnl = wins * win_pnl + losses * loss_pnl
        ev = pnl / n
        rows.append({
            "side": "YES",
            "bucket": f"({lo:.2f}, {hi:.2f}]",
            "n": n,
            "win_rate": wr,
            "avg_p": float(df.loc[mask, "p_pred"].mean()),
            "edge_predicted": float(df.loc[mask, "p_pred"].mean() - market_price),
            "edge_realized": wr - market_price,
            "pnl_total_usd": pnl,
            "ev_per_trade_usd": ev,
        })
    # NO-side buckets: model says Down, we buy NO at market_price.
    no_buckets = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45)]
    for lo, hi in no_buckets:
        mask = (df["p_pred"] >= lo) & (df["p_pred"] < hi)
        n = int(mask.sum())
        if n < min_n:
            continue
        # We bet NO; win when Down resolves (y_true == 0)
        wins = int((1 - df.loc[mask, "y_true"]).sum())
        losses = n - wins
        wr = wins / n
        win_pnl = 1.0 - market_price - fees_pct
        loss_pnl = -market_price - fees_pct
        pnl = wins * win_pnl + losses * loss_pnl
        ev = pnl / n
        rows.append({
            "side": "NO",
            "bucket": f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "win_rate": wr,
            "avg_p_up": float(df.loc[mask, "p_pred"].mean()),
            "edge_predicted": float((1.0 - df.loc[mask, "p_pred"].mean()) - market_price),
            "edge_realized": wr - market_price,
            "pnl_total_usd": pnl,
            "ev_per_trade_usd": ev,
        })
    return pd.DataFrame(rows)


def bootstrap_ev_ci(
    df: pd.DataFrame, threshold_yes: float = 0.55, threshold_no: float = 0.45,
    market_price: float = 0.50, fees_pct: float = 0.0,
    n_bootstrap: int = 1000, ci_level: float = 0.95,
) -> dict:
    """Bootstrap CI on EV-per-trade for the trades the model would have placed.

    A trade is placed when:
      p_pred > threshold_yes → bet YES
      p_pred < threshold_no  → bet NO
    """
    yes_mask = df["p_pred"] > threshold_yes
    no_mask = df["p_pred"] < threshold_no
    trades_mask = yes_mask | no_mask

    if trades_mask.sum() < 30:
        return {"n_trades": int(trades_mask.sum()), "skipped": "too_few_trades"}

    # Build per-row PnL for the trades we'd have placed
    pnl_rows = np.zeros(len(df), dtype=float)
    pnl_rows[yes_mask] = np.where(
        df.loc[yes_mask, "y_true"] == 1,
        1.0 - market_price - fees_pct,
        -market_price - fees_pct,
    )
    pnl_rows[no_mask] = np.where(
        df.loc[no_mask, "y_true"] == 0,
        1.0 - market_price - fees_pct,
        -market_price - fees_pct,
    )
    trade_pnl = pnl_rows[trades_mask]

    rng = np.random.default_rng(seed=42)
    n_trades = len(trade_pnl)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(trade_pnl, size=n_trades, replace=True)
        boots[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    point = float(trade_pnl.mean())

    # Simple win rate
    wins_total = int(((yes_mask & (df["y_true"] == 1)) |
                      (no_mask & (df["y_true"] == 0))).sum())
    return {
        "n_trades": n_trades,
        "n_yes_bets": int(yes_mask.sum()),
        "n_no_bets": int(no_mask.sum()),
        "wins": wins_total,
        "win_rate": wins_total / n_trades,
        "ev_per_trade_usd": point,
        "ev_ci_low_usd": lo,
        "ev_ci_high_usd": hi,
        "total_pnl_usd": point * n_trades,
        "threshold_yes": threshold_yes,
        "threshold_no": threshold_no,
        "market_price": market_price,
        "fees_pct": fees_pct,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 §3c+§4c — translate AUC into realized edge.")
    p.add_argument("--tf", type=int, choices=[5, 15], required=True)
    p.add_argument("--offset-sec", type=int, default=60)
    p.add_argument("--threshold-yes", type=float, default=0.55)
    p.add_argument("--threshold-no", type=float, default=0.45)
    p.add_argument("--market-price", type=float, default=0.50,
                   help="Assumed entry price ($). Real prod uses live yes_ask/no_ask.")
    p.add_argument("--fees", type=float, default=0.0,
                   help="Fee fraction per trade (0.0 = maker, ~0.02 = taker)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--model", choices=["logreg", "xgb", "lgbm"], default="logreg",
                   help="Which trainer's predictions to evaluate (suffix on parquet path)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"=== EV evaluation: tf={args.tf}m offset={args.offset_sec}s ===")
    print(f"  thresholds: bet YES if p>{args.threshold_yes}, bet NO if p<{args.threshold_no}")
    print(f"  market_price=${args.market_price}, fees={args.fees * 100:.1f}%")
    print()

    suffix = "" if args.model == "logreg" else f"_{args.model}"
    df = load_predictions(args.tf, args.offset_sec, suffix=suffix)
    print(f"model: {args.model}")
    print(f"loaded {len(df)} validation predictions across {df['fold'].nunique()} folds")
    print(f"validation date range: {df['decision_ts'].min()} → {df['decision_ts'].max()}")
    print()

    print("=== bucket curve ===")
    curve = bucket_curve(df, market_price=args.market_price, fees_pct=args.fees)
    if curve.empty:
        print("  no buckets had >=30 samples")
    else:
        # Round for display
        cols = [c for c in curve.columns if c not in {"side", "bucket"}]
        for c in cols:
            if curve[c].dtype == float:
                curve[c] = curve[c].round(4)
        print(curve.to_string(index=False))
    print()

    print("=== bootstrap CI on EV per trade ===")
    ci = bootstrap_ev_ci(
        df,
        threshold_yes=args.threshold_yes,
        threshold_no=args.threshold_no,
        market_price=args.market_price,
        fees_pct=args.fees,
        n_bootstrap=args.bootstrap,
    )
    if "skipped" in ci:
        print(f"  skipped: {ci['skipped']} (n={ci['n_trades']})")
    else:
        print(f"  n_trades:           {ci['n_trades']:>8d}")
        print(f"    yes bets:         {ci['n_yes_bets']:>8d}")
        print(f"    no bets:          {ci['n_no_bets']:>8d}")
        print(f"  realized win rate:  {ci['win_rate'] * 100:>7.2f}%")
        print(f"  EV per trade:       ${ci['ev_per_trade_usd']:>+.4f}")
        print(f"  95% bootstrap CI:   [${ci['ev_ci_low_usd']:>+.4f}, ${ci['ev_ci_high_usd']:>+.4f}]")
        print(f"  total simulated PnL: ${ci['total_pnl_usd']:>+.2f}")
        print()
        if ci["ev_ci_low_usd"] > 0:
            print("  >>> V2 §3c gate met: lower 95% CI on EV > 0. Edge is real.")
        elif ci["ev_per_trade_usd"] > 0:
            print("  >>> Point estimate POSITIVE but lower CI <= 0. Need more validation data.")
        else:
            print("  >>> EV per trade <= 0. Model not actionable at this threshold/fee combination.")

    # Persist
    out = DEFAULT_OUT_ROOT / f"tf={args.tf}m_offset={args.offset_sec}s_ev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "tf_minutes": args.tf,
            "offset_sec": args.offset_sec,
            "threshold_yes": args.threshold_yes,
            "threshold_no": args.threshold_no,
            "market_price": args.market_price,
            "fees_pct": args.fees,
            "bucket_curve": curve.to_dict(orient="records") if not curve.empty else [],
            "bootstrap_ci": ci,
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\n  results → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
