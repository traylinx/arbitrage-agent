#!/usr/bin/env python3
"""
BTC Probability Model Trainer — Walk-forward calibrated classifier.

Trains on local SQLite feature DB, outputs a calibrated model pickle.
Target: binary UP/DOWN after 5m or 15m.
Uses LogisticRegression + isotonic calibration for interpretability.
Validation: walk-forward by calendar day to avoid lookahead.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
FEATURE_DB = Path(os.environ.get("BTC_FEATURE_DB", str(DATA_DIR / "state" / "btc_features.db")))
MODEL_DIR = DATA_DIR / "state" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    from btc_external_metrics import EXTERNAL_MODEL_FEATURE_KEYS
except Exception:
    EXTERNAL_MODEL_FEATURE_KEYS = []

# Feature columns used by the model
FEATURE_COLS = [
    "return_1m",
    "return_5m",
    "return_15m",
    "volatility_5m",
    "volatility_15m",
    "rsi_14",
    "macd_hist",
    "bb_position",
    "atr_14",
    "ob_imbalance_5",
    "spread_bps",
    "basis_bps",
    "funding_rate",
    "volume_delta_5m",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "vwap_deviation",
] + list(EXTERNAL_MODEL_FEATURE_KEYS)


def load_data(db_path: Path, horizon: str = "5m") -> pd.DataFrame:
    """Load features + labels from SQLite."""
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT f.*, l.label_5m, l.label_15m, l.return_fwd_5m, l.return_fwd_15m
        FROM features f
        LEFT JOIN labels l ON f.ts = l.ts
        WHERE f.price IS NOT NULL
        ORDER BY f.ts
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date
    return df


def prepare_df(df: pd.DataFrame, horizon: str = "5m") -> pd.DataFrame:
    label_col = f"label_{horizon}"
    ret_col = f"return_fwd_{horizon}"
    # Drop rows without labels
    out = df.dropna(subset=[label_col]).copy()
    out["y"] = out[label_col].astype(int)
    # Ensure all feature cols exist
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)
    return out


def walk_forward_splits(df: pd.DataFrame, n_train_days: int = 7, n_test_days: int = 1):
    """Generate (train_df, test_df) splits by calendar day."""
    dates = sorted(df["date"].unique())
    for i in range(n_train_days, len(dates), n_test_days):
        train_dates = dates[i - n_train_days : i]
        test_dates = dates[i : i + n_test_days]
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        yield train_df, test_df


def train_model(df: pd.DataFrame, horizon: str = "5m") -> dict:
    """Walk-forward train + evaluate. Returns metrics and saves best model."""
    df = prepare_df(df, horizon)
    if len(df) < 200:
        raise ValueError(f"Need >=200 labeled rows, got {len(df)}")

    splits = list(walk_forward_splits(df, n_train_days=7, n_test_days=1))
    if not splits:
        # Fallback: single train/test split 80/20 by time
        split_idx = int(len(df) * 0.8)
        splits = [(df.iloc[:split_idx], df.iloc[split_idx:])]

    aucs = []
    briers = []
    accs = []
    loglosses = []
    fold_results = []

    best_brier = float("inf")
    best_model = None
    best_fold = 0

    for fold, (train_df, test_df) in enumerate(splits, 1):
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["y"].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["y"].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Fold {fold}: skip (only one class)")
            continue

        base = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            solver="lbfgs",
        )
        model = CalibratedClassifierCV(base, method="isotonic", cv=5)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)
        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, prob)

        aucs.append(auc)
        briers.append(brier)
        accs.append(acc)
        loglosses.append(ll)

        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "auc": round(auc, 4),
                "brier": round(brier, 4),
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4),
                "base_rate": round(float(y_test.mean()), 4),
            }
        )

        if brier < best_brier:
            best_brier = brier
            best_model = model
            best_fold = fold

        print(
            f"  Fold {fold}: AUC={auc:.3f}  Brier={brier:.4f}  ACC={acc:.3f}  base={y_test.mean():.3f}"
        )

    summary = {
        "horizon": horizon,
        "n_total": len(df),
        "n_folds": len(aucs),
        "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "auc_std": round(float(np.std(aucs)), 4) if aucs else None,
        "brier_mean": round(float(np.mean(briers)), 4) if briers else None,
        "acc_mean": round(float(np.mean(accs)), 4) if accs else None,
        "logloss_mean": round(float(np.mean(loglosses)), 4) if loglosses else None,
        "folds": fold_results,
        "feature_cols": FEATURE_COLS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save best model
    if best_model is not None:
        model_path = MODEL_DIR / f"btc_model_{horizon}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        summary["model_path"] = str(model_path)
        summary["best_fold"] = best_fold
        print(f"  → Best model saved to {model_path} (fold {best_fold}, brier={best_brier:.4f})")

        # Coefficients from the base estimator (last calibrated clone)
        try:
            lr = best_model.calibrated_classifiers_[0].estimator
            coefs = dict(zip(FEATURE_COLS, lr.coef_[0].tolist()))
            summary["coefficients"] = {k: round(v, 6) for k, v in coefs.items()}
            print("  Coefficients:")
            for k, v in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
                print(f"    {k:20s}: {v:+.4f}")
        except Exception:
            pass

    # Threshold analysis: find prob threshold that gives highest accuracy above 55%
    all_test = pd.concat([s[1] for s in splits])
    if best_model is not None and len(all_test) > 0:
        X_all = all_test[FEATURE_COLS].values
        y_all = all_test["y"].values
        probs = best_model.predict_proba(X_all)[:, 1]
        threshold_analysis = []
        for thr in np.arange(0.50, 0.80, 0.02):
            mask = probs >= thr
            if mask.sum() < 10:
                continue
            acc_t = accuracy_score(y_all[mask], (probs[mask] >= thr).astype(int))
            n = int(mask.sum())
            threshold_analysis.append(
                {"threshold": round(thr, 2), "n": n, "accuracy": round(acc_t, 4)}
            )
        summary["threshold_analysis"] = threshold_analysis

    return summary


def main():
    if not FEATURE_DB.exists():
        print(f"Feature DB not found: {FEATURE_DB}")
        print("Run btc_feature_engine.py first to collect data.")
        return

    df = load_data(FEATURE_DB)
    print(f"Loaded {len(df)} rows from feature DB.")
    labeled = df.dropna(subset=["label_5m"])
    print(f"Labeled rows: {len(labeled)} (5m) / {len(df.dropna(subset=['label_15m']))} (15m)")

    if len(labeled) < 200:
        print("Insufficient labeled data. Need >=200 rows. Collect more features.")
        return

    print("\n=== Training 5m model ===")
    summary_5m = train_model(df, horizon="5m")

    print("\n=== Training 15m model ===")
    summary_15m = train_model(df, horizon="15m")

    report = {
        "model_5m": summary_5m,
        "model_15m": summary_15m,
    }
    report_path = MODEL_DIR / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
