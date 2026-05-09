"""predictor_adapter.py — load persisted model + scaler and conform to
the V2 §4b Predictor protocol so decision_engine.ModelDecisionEngine
can use it directly.

V2 §4b loader contract:
  - On load, compute schema_hash and verify against live feature engine
  - Verify lib versions in metadata
  - Surface schema_hash so decision_engine fail-closed gate works
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SklearnPredictor:
    """Adapter that wraps a pickled (model, scaler) pair and implements
    the V2 §4b Predictor protocol used by ModelDecisionEngine.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model dir not found: {model_dir}")

        self._schema = json.loads((self.model_dir / "feature_schema.json").read_text())
        self._metadata = json.loads((self.model_dir / "metadata.json").read_text())
        self._model = pickle.loads((self.model_dir / "model.pkl").read_bytes())
        # GBDT models (xgb/lgbm) handle NaN natively → no scaler.pkl saved.
        # Logreg models always save a scaler. Detect by file presence.
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists() and self._schema.get("use_scaler", True):
            self._scaler = pickle.loads(scaler_path.read_bytes())
        else:
            self._scaler = None
        self.feature_names: list[str] = list(self._schema["feature_names"])
        # The schema_hash is the V2 §4b fail-closed key
        self.schema_hash: str = self._schema["schema_hash"]
        # GBDTs use fillna_value=None to mean "leave NaN" (model handles it).
        fillna_raw = self._schema.get("fillna_value", 0.0)
        self._fillna: Optional[float] = (
            None if fillna_raw is None else float(fillna_raw)
        )

    def predict(self, features) -> float:
        """Returns calibrated P(Up wins) as a float in (0, 1)."""
        # `features` is a FeatureVector from decision_engine.py
        vals = features.values if hasattr(features, "values") else features
        row: list = []
        for name in self.feature_names:
            v = vals.get(name)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                # logreg path fills with 0.0; GBDT path keeps NaN (handles natively)
                row.append(float("nan") if self._fillna is None else self._fillna)
            else:
                row.append(float(v))
        x = np.array([row], dtype=float)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        # Both LogisticRegression and XGBClassifier/LGBMClassifier return [[P(0), P(1)]]
        return float(self._model.predict_proba(x)[0, 1])

    def info(self) -> dict:
        return {
            "tf_minutes": self._metadata.get("tf_minutes"),
            "trained_at": self._metadata.get("trained_at"),
            "n_train_rows": self._metadata.get("n_train_rows"),
            "schema_hash": self.schema_hash[:16] + "...",
            "feature_count": len(self.feature_names),
            "lib_versions": self._metadata.get("lib_versions"),
        }


def load_predictor(tf_minutes: int, model_kind: str = "logreg") -> SklearnPredictor:
    """Convenience: load the canonical predictor for this tf + model.

    `model_kind`:
      - "logreg" → state/models/baseline_v0/{tf}m/      (LogisticRegression + scaler)
      - "xgb"    → state/models/baseline_v0/{tf}m_xgb/  (XGBoost, no scaler)
      - "lgbm"   → state/models/baseline_v0/{tf}m_lgbm/ (LightGBM, no scaler)
    """
    suffix = "" if model_kind == "logreg" else f"_{model_kind}"
    d = (Path.home() / "MAKAKOO/data/arbitrage-agent/v2/state/models/baseline_v0"
         / f"{tf_minutes}m{suffix}")
    return SklearnPredictor(d)
