"""save_model_gbdt.py — Train an XGBoost or LightGBM model on the FULL
labeled dataset and persist artifacts compatible with `predictor_adapter`.

GBDTs differ from the logreg baseline path in two ways:
  - No StandardScaler — they're scale-invariant.
  - NaN-aware natively — no fillna(0.0).

Output artifacts mirror save_model.py with two changes:
  - No `scaler.pkl` (predictor_adapter skips scaling if file is missing).
  - metadata.json records the GBDT params + feature_importance instead of
    `coefficients`/`intercept`.

Usage:
    python3.11 -m features.save_model_gbdt --tf 5  --model xgb  --params-from-trainer
    python3.11 -m features.save_model_gbdt --tf 15 --model xgb  --params-from-trainer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_base import all_features, feature_schema_hash  # noqa: F401
from .train_baseline import load_dataset, select_feature_cols
from .train_gbdt import DEFAULT_PARAMS

DEFAULT_OUT_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/state/models/baseline_v0"


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def load_trainer_params(out_root: Path, tf: int, offset_sec: int, model: str) -> dict:
    """Read the params Optuna found from the trainer's results JSON."""
    p = out_root / f"tf={tf}m_offset={offset_sec}s_{model}_results.json"
    if not p.exists():
        return dict(DEFAULT_PARAMS[model])
    data = json.loads(p.read_text())
    params = data.get("params", {})
    return params or dict(DEFAULT_PARAMS[model])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persist GBDT (XGBoost/LightGBM) model artifacts.")
    p.add_argument("--tf", type=int, choices=[5, 15], required=True)
    p.add_argument("--offset-sec", type=int, default=60)
    p.add_argument("--model", choices=["xgb", "lgbm"], default="xgb")
    p.add_argument("--params-from-trainer", action="store_true",
                   help="Load params from the trainer's results JSON (Optuna best). "
                        "Otherwise uses DEFAULT_PARAMS.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args()


def main() -> int:
    # Register feature implementations so the schema hash is right
    import features.price  # noqa: F401
    import features.crossex  # noqa: F401

    args = parse_args()
    out_dir = args.out / f"{args.tf}m_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.tf, args.offset_sec)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
    feat_cols = select_feature_cols(df)
    print(f"loaded {len(df)} rows, {len(feat_cols)} features")
    print(f"  features: {feat_cols}")

    if args.params_from_trainer:
        params = load_trainer_params(args.out, args.tf, args.offset_sec, args.model)
        print(f"  params (from trainer Optuna result): {params}")
    else:
        params = dict(DEFAULT_PARAMS[args.model])
        print(f"  params (defaults): {params}")

    X = df[feat_cols].to_numpy()  # GBDTs handle NaN natively
    y = df["binary_label_up_won"].astype(int).to_numpy()

    full_params = dict(params)
    n_estimators = int(full_params.pop("n_estimators", 300))

    if args.model == "xgb":
        import xgboost as xgb_pkg
        full_params.setdefault("objective", "binary:logistic")
        full_params.setdefault("eval_metric", "logloss")
        full_params.setdefault("tree_method", "hist")
        full_params.setdefault("verbosity", 0)
        model = xgb_pkg.XGBClassifier(n_estimators=n_estimators, **full_params)
        lib_version = xgb_pkg.__version__
    else:
        import lightgbm as lgb_pkg
        full_params.setdefault("objective", "binary")
        full_params.setdefault("metric", "binary_logloss")
        full_params.setdefault("verbosity", -1)
        model = lgb_pkg.LGBMClassifier(n_estimators=n_estimators, **full_params)
        lib_version = lgb_pkg.__version__

    model.fit(X, y)
    train_acc = float((model.predict(X) == y).mean())
    print(f"train accuracy: {train_acc:.4f}")

    # Feature importance (consistent across both libs via .feature_importances_)
    importances = list(zip(feat_cols, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("feature importance (top to bottom):")
    for fc, imp in importances:
        print(f"  {fc:42s}  {float(imp):.4f}")

    schema = {
        "feature_names": feat_cols,
        "schema_hash": feature_schema_hash(),
        "ordered": True,
        # GBDTs handle NaN natively; predictor_adapter must skip its fillna AND
        # skip scaling if scaler.pkl is missing.
        "fillna_value": None,
        "use_scaler": False,
        "model_kind": args.model,
    }
    schema_bytes = json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    schema["payload_sha256"] = hashlib.sha256(schema_bytes).hexdigest()

    metadata = {
        "tf_minutes": args.tf,
        "offset_sec": args.offset_sec,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(df)),
        "decision_ts_range": [
            df["decision_ts"].min().isoformat(),
            df["decision_ts"].max().isoformat(),
        ],
        "feature_count": len(feat_cols),
        "schema_hash": feature_schema_hash(),
        "train_accuracy": train_acc,
        "feature_importance": {fc: float(imp) for fc, imp in importances},
        "lib_versions": {
            "python": sys.version.split()[0],
            args.model: lib_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git_commit": git_commit_sha(),
        "model_class": "XGBClassifier" if args.model == "xgb" else "LGBMClassifier",
        "model_params": params,
    }

    (out_dir / "model.pkl").write_bytes(pickle.dumps(model))
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Note: deliberately no scaler.pkl — predictor_adapter detects its absence.

    print()
    print(f"=== persisted to {out_dir} ===")
    for f in out_dir.iterdir():
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
