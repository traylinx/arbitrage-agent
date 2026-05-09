"""save_model.py — Train on ALL labeled data and persist artifacts.

The walk-forward CV trainer is for *evaluation*. For production we
retrain on the full dataset (no holdout) and persist:

    state/models/baseline_v0/{tf}m/
      model.pkl            sklearn LogisticRegression
      scaler.pkl           StandardScaler fit on full training data
      feature_schema.json  ordered list of feature names + sha256 hash
      metadata.json        train timestamps, lib versions, git commit

V2 §4b loader is what reads these. Schema hash gates the fail-closed
check in ModelDecisionEngine.

Usage:
    python3.11 -m features.save_model --tf 5
    python3.11 -m features.save_model --tf 15
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
import sklearn

from .feature_base import all_features, feature_schema_hash
from .train_baseline import load_dataset, select_feature_cols

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and persist baseline model artifacts.")
    p.add_argument("--tf", type=int, choices=[5, 15], required=True)
    p.add_argument("--offset-sec", type=int, default=60)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args()


def main() -> int:
    # Register feature implementations so the schema hash is right
    import features.price  # noqa: F401
    import features.crossex  # noqa: F401

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    out_dir = args.out / f"{args.tf}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.tf, args.offset_sec)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
    feat_cols = select_feature_cols(df)
    pre = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"loaded {pre} rows, {len(df)} after NaN drop, {len(feat_cols)} features")

    X = df[feat_cols].fillna(0.0).to_numpy()
    y = df["binary_label_up_won"].astype(int).to_numpy()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_s, y)

    train_acc = (model.predict(X_s) == y).mean()
    print(f"train accuracy: {train_acc:.4f}")
    print(f"intercept: {model.intercept_[0]:+.4f}")
    for fc, coef in sorted(zip(feat_cols, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {fc:35s}  coef={coef:+.4f}")

    # Persist artifacts
    schema = {
        "feature_names": feat_cols,
        "schema_hash": feature_schema_hash(),
        "ordered": True,
        "fillna_value": 0.0,
    }
    schema_bytes = json.dumps(schema, sort_keys=True).encode("utf-8")
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
        "train_accuracy": float(train_acc),
        "intercept": float(model.intercept_[0]),
        "coefficients": {fc: float(c) for fc, c in zip(feat_cols, model.coef_[0])},
        "lib_versions": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git_commit": git_commit_sha(),
        "model_class": "LogisticRegression",
        "model_params": model.get_params(),
    }

    (out_dir / "model.pkl").write_bytes(pickle.dumps(model))
    (out_dir / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print(f"=== persisted to {out_dir} ===")
    for f in out_dir.iterdir():
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
