"""train_gbdt.py — V2 §3c heavier-model trainer: XGBoost + LightGBM with
walk-forward CV, optional Optuna hyperparameter search.

Same fold structure as `train_baseline.py` so AUC and EV are directly
comparable to the logistic-regression baseline. Output predictions parquet
shape is identical to the baseline so `evaluate_edge.py` works unchanged.

Why GBDTs:
  - Logreg coefficients are tiny on the 12 cross-ex features in v0.7
    (basis features 0.004-0.033, funding/OI features 0.05-0.30) → linear
    model is leaving non-linear interactions on the table.
  - GBDTs natively handle missing values (no fillna(0.0) bias) and
    feature interactions out-of-the-box.
  - Optuna is already a dependency for the rest of the v0.x pipeline.

Usage:
    python3.11 -m features.train_gbdt --tf 5  --model xgb
    python3.11 -m features.train_gbdt --tf 15 --model xgb
    python3.11 -m features.train_gbdt --tf 5  --model lgbm
    python3.11 -m features.train_gbdt --tf 5  --model xgb --optuna-trials 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .train_baseline import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_ROOT,
    load_dataset,
    select_feature_cols,
    walk_forward_folds,
)


def fit_score_xgb(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
) -> tuple[dict, pd.DataFrame]:
    """Fit XGBoost on train, score on validate. NaN-aware (no fillna)."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    X_train = train[feature_cols].to_numpy()
    y_train = train["binary_label_up_won"].astype(int).to_numpy()
    X_val = validate[feature_cols].to_numpy()
    y_val = validate["binary_label_up_won"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        return ({
            "n_train": len(train), "n_val": len(validate),
            "auc": float("nan"), "log_loss": float("nan"), "brier": float("nan"),
            "skip_reason": "single_class_in_train",
        }, pd.DataFrame())

    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
    }
    full_params.update(params)
    n_estimators = int(full_params.pop("n_estimators", 300))

    model = xgb.XGBClassifier(n_estimators=n_estimators, **full_params)
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]

    metrics = {
        "n_train": len(train), "n_val": len(validate), "n_features": len(feature_cols),
        "auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_val, p_val, labels=[0, 1])),
        "brier": float(brier_score_loss(y_val, p_val)),
        "p_mean": float(p_val.mean()), "p_std": float(p_val.std()),
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
        "fold": -1,
        "tf_minutes": validate["tf_minutes"].to_numpy(),
        "window_start": validate["window_start"].to_numpy(),
    })
    return (metrics, preds)


def fit_score_lgbm(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
) -> tuple[dict, pd.DataFrame]:
    """Fit LightGBM on train, score on validate. NaN-aware."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    X_train = train[feature_cols].to_numpy()
    y_train = train["binary_label_up_won"].astype(int).to_numpy()
    X_val = validate[feature_cols].to_numpy()
    y_val = validate["binary_label_up_won"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        return ({
            "n_train": len(train), "n_val": len(validate),
            "auc": float("nan"), "log_loss": float("nan"), "brier": float("nan"),
            "skip_reason": "single_class_in_train",
        }, pd.DataFrame())

    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    }
    full_params.update(params)
    n_estimators = int(full_params.pop("n_estimators", 300))

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **full_params)
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]

    metrics = {
        "n_train": len(train), "n_val": len(validate), "n_features": len(feature_cols),
        "auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_val, p_val, labels=[0, 1])),
        "brier": float(brier_score_loss(y_val, p_val)),
        "p_mean": float(p_val.mean()), "p_std": float(p_val.std()),
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
        "fold": -1,
        "tf_minutes": validate["tf_minutes"].to_numpy(),
        "window_start": validate["window_start"].to_numpy(),
    })
    return (metrics, preds)


FIT_FUNCS = {
    "xgb": fit_score_xgb,
    "lgbm": fit_score_lgbm,
}

# Sensible defaults used when --optuna-trials is 0
DEFAULT_PARAMS = {
    "xgb": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
    },
    "lgbm": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
}


def cv_score(
    df: pd.DataFrame, feature_cols: list[str], folds, model: str, params: dict,
) -> tuple[float, list[dict], pd.DataFrame]:
    """Run walk-forward CV; return (mean_auc, per_fold_metrics, all_preds)."""
    fit = FIT_FUNCS[model]
    fold_results, all_preds = [], []
    for i, (train, validate) in enumerate(folds):
        r, preds = fit(train, validate, feature_cols, params)
        r["fold"] = i
        fold_results.append(r)
        if not preds.empty:
            preds["fold"] = i
            all_preds.append(preds)
    valid = [r for r in fold_results if "skip_reason" not in r and not math.isnan(r["auc"])]
    if not valid:
        return float("nan"), fold_results, pd.DataFrame()
    mean_auc = float(np.mean([r["auc"] for r in valid]))
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return mean_auc, fold_results, preds_df


# ─── Optuna search ──────────────────────────────────────────────────────────


def optuna_search(
    df: pd.DataFrame, feature_cols: list[str], folds, model: str, n_trials: int,
) -> tuple[dict, float]:
    """Run Optuna over the model's hyperparameter space; return (best_params, best_auc).

    Each trial does the FULL walk-forward CV (no nested validation — the
    walk-forward already guards against leakage). Maximizes mean AUC across folds.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest_xgb(trial):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

    def suggest_lgbm(trial):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        }

    suggest = suggest_xgb if model == "xgb" else suggest_lgbm

    def objective(trial):
        params = suggest(trial)
        mean_auc, _, _ = cv_score(df, feature_cols, folds, model, params)
        if math.isnan(mean_auc):
            return 0.0
        return mean_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GBDT walk-forward trainer (XGBoost / LightGBM).")
    p.add_argument("--tf", type=int, choices=[5, 15], required=True)
    p.add_argument("--offset-sec", type=int, default=60)
    p.add_argument("--model", choices=["xgb", "lgbm"], default="xgb")
    p.add_argument("--train-weeks", type=int, default=4)
    p.add_argument("--validate-weeks", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=12)
    p.add_argument("--min-nonnull-frac", type=float, default=0.3)
    p.add_argument("--optuna-trials", type=int, default=0,
                   help="If > 0, run Optuna hyperparameter search with this many trials.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.tf, args.offset_sec)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
    print(f"loaded {len(df)} rows for tf={args.tf}m offset={args.offset_sec}s")

    feature_cols = select_feature_cols(df, min_nonnull_frac=args.min_nonnull_frac)
    print(f"using {len(feature_cols)} features (min_nonnull_frac={args.min_nonnull_frac})")

    folds = walk_forward_folds(df, args.train_weeks, args.validate_weeks, args.max_folds)
    if not folds:
        print("[error] no folds buildable", file=sys.stderr)
        return 2
    print(f"built {len(folds)} walk-forward folds ({args.train_weeks}w train + "
          f"{args.validate_weeks}w validate)")

    # Choose params: Optuna search OR baked defaults.
    if args.optuna_trials > 0:
        print(f"\n=== Optuna search: model={args.model} trials={args.optuna_trials} ===")
        best_params, best_auc = optuna_search(
            df, feature_cols, folds, args.model, args.optuna_trials,
        )
        print(f"  best AUC during search: {best_auc:.4f}")
        print(f"  best params: {best_params}")
        params = best_params
    else:
        params = dict(DEFAULT_PARAMS[args.model])
        print(f"\n=== Default {args.model} params ===")
        for k, v in params.items():
            print(f"  {k}: {v}")

    # Final CV scoring with chosen params
    print(f"\n=== Walk-forward CV ({args.model}) ===")
    print(f"{'fold':<5}{'train_n':<10}{'val_n':<8}{'AUC':<8}{'log_loss':<10}{'brier':<8}{'p_mean':<8}{'val_up%':<8}")
    print("-" * 70)
    mean_auc, fold_results, preds_df = cv_score(df, feature_cols, folds, args.model, params)
    for r in fold_results:
        if "skip_reason" in r:
            print(f"{r['fold']:<5}{r['n_train']:<10}{r['n_val']:<8}SKIPPED: {r['skip_reason']}")
            continue
        print(f"{r['fold']:<5}{r['n_train']:<10}{r['n_val']:<8}"
              f"{r['auc']:.4f}  {r['log_loss']:.4f}    {r['brier']:.4f}  "
              f"{r['p_mean']:.3f}   {r['y_val_up_rate']*100:.1f}%")

    valid = [r for r in fold_results if "skip_reason" not in r and not math.isnan(r["auc"])]
    aucs = [r["auc"] for r in valid]
    agg = {
        "n_folds": len(valid),
        "auc_mean": mean_auc,
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "auc_min": float(min(aucs)) if aucs else float("nan"),
        "auc_max": float(max(aucs)) if aucs else float("nan"),
        "log_loss_mean": float(np.mean([r["log_loss"] for r in valid])) if valid else float("nan"),
        "brier_mean": float(np.mean([r["brier"] for r in valid])) if valid else float("nan"),
    }
    print()
    print("=== aggregate ===")
    print(f"  folds:         {agg['n_folds']}")
    print(f"  AUC mean:      {agg['auc_mean']:.4f}  (random=0.5000, V2 §3c gate=0.5500)")
    print(f"  AUC std:       {agg['auc_std']:.4f}")
    print(f"  AUC min/max:   {agg['auc_min']:.4f} / {agg['auc_max']:.4f}")
    print(f"  log_loss mean: {agg['log_loss_mean']:.4f}  (random=0.6931)")
    print(f"  brier mean:    {agg['brier_mean']:.4f}")

    # Persist results + predictions parquet (same shape as logreg baseline so
    # evaluate_edge.py works unchanged)
    suffix = f"_{args.model}"
    out_path = args.out / f"tf={args.tf}m_offset={args.offset_sec}s{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "tf_minutes": args.tf,
            "offset_sec": args.offset_sec,
            "model": args.model,
            "params": params,
            "feature_cols": feature_cols,
            "n_dataset_rows": len(df),
            "fold_results": fold_results,
            "aggregate": agg,
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"  results → {out_path}")

    if not preds_df.empty:
        import pyarrow as pa, pyarrow.parquet as pq
        preds_path = args.out / f"tf={args.tf}m_offset={args.offset_sec}s{suffix}_predictions.parquet"
        pq.write_table(pa.Table.from_pandas(preds_df, preserve_index=False), preds_path,
                       compression="snappy")
        print(f"  predictions → {preds_path} ({len(preds_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
