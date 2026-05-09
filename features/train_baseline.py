"""train_baseline.py — V2 §3c first model: logistic regression + walk-forward CV.

This is the SANITY-CHECK baseline per V2 §3c step 1: "must beat 50% AUC by ≥0.02".
If logistic regression on V2 §2a price features can't beat 50%, neither will a
heavier model — the data probably doesn't carry the signal we hope it does, and
that itself is a critical finding.

This is NOT the production trainer. The full XGBoost+LightGBM+CatBoost stack with
Optuna and Platt calibration comes next (V2 §3b/3c). This baseline:

  1. Loads labeled_dataset/tf={tf}m_offset={offset}s.parquet
  2. Drops NaN-only feature columns (cross-ex ones waiting on live data)
  3. Drops rows missing core features
  4. Walk-forward CV: 8 folds × (4w train + 1w validate) — fits 86d at 5m and
     plenty for 211d at 15m
  5. Per fold: fit LogisticRegression, score AUC + log loss + Brier on validate
  6. Reports per-fold + aggregate
  7. Writes results to state/models/baseline_v0/{tf}m_results.json

Result interpretation:
  - aggregate AUC ≥ 0.52: signal exists, justifies heavier model
  - aggregate AUC ≈ 0.50: no clean linear signal in price features alone — try
    aggregator/PM features once live data is in
  - aggregate AUC < 0.50: features are anti-predictive (sign error in your code,
    not in the market)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/labeled_dataset"
DEFAULT_OUT_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/state/models/baseline_v0"


def load_dataset(tf: int, offset_sec: int) -> pd.DataFrame:
    p = DEFAULT_DATA_ROOT / f"tf={tf}m_offset={offset_sec}s.parquet"
    if not p.exists():
        raise FileNotFoundError(f"labeled dataset not found: {p}")
    df = pd.read_parquet(p)
    return df


def select_feature_cols(df: pd.DataFrame, min_nonnull_frac: float = 0.3) -> list[str]:
    """Keep only feature columns where at least min_nonnull_frac of rows are non-NaN.

    Drops columns that are entirely or near-entirely NaN (e.g. waiting on a
    backfill we haven't run yet). Cross-ex features that come from data which
    starts mid-horizon (e.g. Coinalyze covers ~41% of 15m markets) DO get
    selected — the trainer then drops rows with NaN, which restricts training
    to the post-coverage slice.

    Default lowered from 0.5 → 0.3 in v0.6 so the 15m trainer keeps Coinalyze
    columns that have ~41% coverage, then restricts rows accordingly.
    """
    meta_cols = {
        "tf_minutes", "window_start", "decision_ts", "decision_offset_sec",
        "binary_label_up_won", "winner", "slug", "condition_id", "volume",
        "fee_rate", "fee_taker_only", "schema_hash",
    }
    candidates = [c for c in df.columns if c not in meta_cols]
    keep = []
    for c in candidates:
        nonnull = df[c].notna().sum() / len(df)
        if nonnull >= min_nonnull_frac:
            keep.append(c)
    return keep


def walk_forward_folds(
    df: pd.DataFrame,
    train_weeks: int = 4,
    validate_weeks: int = 1,
    max_folds: int = 12,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yields (train, validate) frames. df must have `decision_ts` column.

    First fold's train begins at the earliest data + train_weeks lead-in; each
    next fold rolls forward by validate_weeks.
    """
    df = df.sort_values("decision_ts").reset_index(drop=True)
    if df.empty:
        return []
    start = df["decision_ts"].min()
    end = df["decision_ts"].max()

    folds = []
    fold_start = start
    while True:
        train_end = fold_start + timedelta(weeks=train_weeks)
        validate_end = train_end + timedelta(weeks=validate_weeks)
        if validate_end > end:
            break
        train = df[(df["decision_ts"] >= fold_start) & (df["decision_ts"] < train_end)]
        validate = df[(df["decision_ts"] >= train_end) & (df["decision_ts"] < validate_end)]
        if len(train) > 0 and len(validate) > 0:
            folds.append((train, validate))
        fold_start = fold_start + timedelta(weeks=validate_weeks)
        if len(folds) >= max_folds:
            break
    return folds


def fit_score_logreg(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Fit logistic regression on train, score on validate.

    Returns (metrics_dict, predictions_df) where predictions_df has
    decision_ts, p_pred, y_true columns for downstream EV analysis.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    X_train = train[feature_cols].fillna(0.0).to_numpy()
    y_train = train["binary_label_up_won"].astype(int).to_numpy()
    X_val = validate[feature_cols].fillna(0.0).to_numpy()
    y_val = validate["binary_label_up_won"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        return ({
            "n_train": len(train), "n_val": len(validate),
            "auc": float("nan"), "log_loss": float("nan"),
            "brier": float("nan"),
            "skip_reason": "single_class_in_train",
        }, pd.DataFrame())

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_s, y_train)
    p_val = model.predict_proba(X_val_s)[:, 1]

    metrics = {
        "n_train": len(train), "n_val": len(validate),
        "n_features": len(feature_cols),
        "auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_val, p_val, labels=[0, 1])),
        "brier": float(brier_score_loss(y_val, p_val)),
        "p_mean": float(p_val.mean()),
        "p_std": float(p_val.std()),
        "y_val_up_rate": float(y_val.mean()),
        "fold_train_start": train["decision_ts"].min().isoformat(),
        "fold_train_end": train["decision_ts"].max().isoformat(),
        "fold_val_start": validate["decision_ts"].min().isoformat(),
        "fold_val_end": validate["decision_ts"].max().isoformat(),
    }
    preds = pd.DataFrame({
        "decision_ts": validate["decision_ts"].to_numpy(),
        "p_pred": p_val,
        "y_true": y_val,
        "fold": -1,  # filled in by caller
        "tf_minutes": validate["tf_minutes"].to_numpy(),
        "window_start": validate["window_start"].to_numpy(),
    })
    return (metrics, preds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logistic-regression baseline trainer for V2 §3c sanity check.")
    p.add_argument("--tf", type=int, choices=[5, 15], required=True)
    p.add_argument("--offset-sec", type=int, default=60)
    p.add_argument("--train-weeks", type=int, default=4)
    p.add_argument("--validate-weeks", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=12)
    p.add_argument("--min-nonnull-frac", type=float, default=0.3,
                   help="A feature must have ≥ this fraction of non-NaN rows to be kept (default 0.3)")
    p.add_argument("--require-all-features", action="store_true",
                   help="Drop rows with ANY NaN feature (default: keep all rows, fillna(0.0))")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.tf, args.offset_sec)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
    print(f"loaded {len(df)} rows for tf={args.tf}m offset={args.offset_sec}s")

    feature_cols = select_feature_cols(df, min_nonnull_frac=args.min_nonnull_frac)
    print(f"using {len(feature_cols)} features (min_nonnull_frac={args.min_nonnull_frac}):")
    for c in feature_cols:
        nonnull = df[c].notna().sum()
        print(f"  {c:42s}  non_nan={nonnull}/{len(df)} ({100 * nonnull / len(df):.1f}%)")
    if not feature_cols:
        print("[error] no usable features", file=sys.stderr)
        return 2

    if args.require_all_features:
        pre_drop = len(df)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        print(f"--require-all-features: dropped {pre_drop - len(df)} rows with NaN "
              f"in selected features ({len(df)} remain)")
    else:
        # Default: keep all rows; fit_score_logreg's fillna(0.0) handles missing values.
        # Logistic regression with StandardScaler treats fillna 0.0 as the column mean
        # after fitting, so the model learns near-zero weight on absent signal.
        nan_rows = df[feature_cols].isna().any(axis=1).sum()
        print(f"keeping all {len(df)} rows; {nan_rows} have ≥1 NaN feature "
              f"(filled with 0.0 in fit/predict)")

    folds = walk_forward_folds(
        df,
        train_weeks=args.train_weeks,
        validate_weeks=args.validate_weeks,
        max_folds=args.max_folds,
    )
    if not folds:
        print(f"[error] no walk-forward folds buildable from {len(df)} rows over "
              f"{(df['decision_ts'].max() - df['decision_ts'].min()).days} days",
              file=sys.stderr)
        return 2
    print(f"\nbuilt {len(folds)} walk-forward folds "
          f"({args.train_weeks}w train + {args.validate_weeks}w validate)")

    fold_results = []
    all_preds = []
    print(f"\n{'fold':<5}{'train_n':<10}{'val_n':<8}{'AUC':<8}{'log_loss':<10}{'brier':<8}{'p_mean':<8}{'val_up%':<8}")
    print("-" * 70)
    for i, (train, validate) in enumerate(folds):
        r, preds = fit_score_logreg(train, validate, feature_cols)
        r["fold"] = i
        fold_results.append(r)
        if not preds.empty:
            preds["fold"] = i
            all_preds.append(preds)
        if "skip_reason" in r:
            print(f"{i:<5}{r['n_train']:<10}{r['n_val']:<8}SKIPPED: {r['skip_reason']}")
            continue
        print(f"{i:<5}{r['n_train']:<10}{r['n_val']:<8}"
              f"{r['auc']:.4f}  {r['log_loss']:.4f}    {r['brier']:.4f}  "
              f"{r['p_mean']:.3f}   {r['y_val_up_rate']*100:.1f}%")

    # Aggregate
    valid = [r for r in fold_results if "skip_reason" not in r and not math.isnan(r["auc"])]
    if not valid:
        print("\n[error] no valid folds")
        return 2
    agg = {
        "n_folds": len(valid),
        "auc_mean": float(np.mean([r["auc"] for r in valid])),
        "auc_std": float(np.std([r["auc"] for r in valid])),
        "auc_min": float(min(r["auc"] for r in valid)),
        "auc_max": float(max(r["auc"] for r in valid)),
        "log_loss_mean": float(np.mean([r["log_loss"] for r in valid])),
        "brier_mean": float(np.mean([r["brier"] for r in valid])),
        "log_loss_baseline_random": -math.log(0.5),  # = 0.6931
    }
    print()
    print("=== aggregate ===")
    print(f"  folds:          {agg['n_folds']}")
    print(f"  AUC mean:       {agg['auc_mean']:.4f}  (random=0.5000, V2 §3c gate=0.5500)")
    print(f"  AUC std:        {agg['auc_std']:.4f}")
    print(f"  AUC min/max:    {agg['auc_min']:.4f} / {agg['auc_max']:.4f}")
    print(f"  log_loss mean:  {agg['log_loss_mean']:.4f}  (random=0.6931)")
    print(f"  brier mean:     {agg['brier_mean']:.4f}")
    print()
    if agg["auc_mean"] >= 0.55:
        print(f"  >>> AUC ≥ 0.55: V2 §3c gate met by BASELINE. Pursue heavier models.")
    elif agg["auc_mean"] >= 0.52:
        print(f"  >>> AUC ≥ 0.52: weak but real signal. Heavier models worth trying.")
    elif agg["auc_mean"] >= 0.50:
        print(f"  >>> AUC ≈ 0.50: no clean linear signal — need order flow / aggregator features.")
    else:
        print(f"  >>> AUC < 0.50: ANTI-predictive. Check feature signs / labels.")

    # Persist results
    out_path = args.out / f"tf={args.tf}m_offset={args.offset_sec}s_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "tf_minutes": args.tf,
            "offset_sec": args.offset_sec,
            "train_weeks": args.train_weeks,
            "validate_weeks": args.validate_weeks,
            "feature_cols": feature_cols,
            "n_dataset_rows": len(df),
            "fold_results": fold_results,
            "aggregate": agg,
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"  results → {out_path}")

    # Persist per-fold validation predictions for downstream EV analysis
    if all_preds:
        import pyarrow as pa, pyarrow.parquet as pq
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_path = args.out / f"tf={args.tf}m_offset={args.offset_sec}s_predictions.parquet"
        pq.write_table(pa.Table.from_pandas(preds_df, preserve_index=False), preds_path,
                       compression="snappy")
        print(f"  predictions → {preds_path} ({len(preds_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
