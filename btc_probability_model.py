#!/usr/bin/env python3
"""
Phase 1: BTC Polymarket Probability Model
==========================================
Logistic Regression + LightGBM dual ensemble with isotonic calibration.

Train: python btc_probability_model.py --train
Predict: python btc_probability_model.py --predict '{"btc_delta": 25, "poly_price": 0.55, ...}'
Eval:   python btc_probability_model.py --eval

Output: data/arbitrage-agent/v2/model/btc_prob_model_current.{pkl,json}
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR    = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
MODEL_DIR   = DATA_DIR / "model"
DATASET_FILE = MODEL_DIR / "train_dataset.jsonl"
MODEL_OUT    = MODEL_DIR / "btc_prob_model_current"
MODEL_PKL    = MODEL_DIR / "btc_prob_model_current.pkl"
MODEL_META   = MODEL_DIR / "btc_prob_model_current_meta.json"
MODEL_PROBATION = MODEL_DIR / "btc_prob_model_probation.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    from btc_external_metrics import EXTERNAL_PROB_FEATURE_KEYS
except Exception:
    EXTERNAL_PROB_FEATURE_KEYS = []


def stable_sigmoid(raw):
    """Numerically stable sigmoid for scalar or numpy-array logits."""
    if isinstance(raw, np.ndarray):
        x = np.clip(raw.astype(float), -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-x))
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(x):
        return 1.0 if x > 0 else 0.0
    if x >= 0:
        z = math.exp(-min(x, 709.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -709.0))
    return z / (1.0 + z)

# ── Feature list (must match dataset.py) ────────────────────────────────────
FEATURE_KEYS = [
    # Market-implied
    "poly_price_enter",
    "poly_vs_50",

    # Delta
    "btc_delta",
    "btc_delta_pct",

    # RSI
    "rsi_5m",
    "rsi_15m",

    # MACD
    "macd_hist_5m",
    "macd_hist_15m",

    # Bollinger
    "bb_pos_5m",
    "bb_pos_15m",

    # Volume
    "vol_ratio_5m",
    "vol_ratio_15m",

    # Orderbook
    "ob_imbalance",

    # Momentum
    "momentum_5m_pct",
    "momentum_15m_pct",

    # Volatility
    "realized_vol",

    # Time
    "hour_utc",
    "tf_minutes",
    "direction_sign",

    # Poly buckets (interaction with direction)
    "poly_bucket_high",
    "poly_bucket_very_high",
] + list(EXTERNAL_PROB_FEATURE_KEYS)

FEATURE_KEYS_SET = set(FEATURE_KEYS)


def json_safe(obj):
    """Convert numpy scalars/arrays to JSON-native values for CLI output."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(min_samples: int = 100) -> tuple[list, list, list]:
    """Load dataset. Returns (X_list, y_list, meta_list)."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}. Run btc_prob_dataset.py first.")

    rows = []
    with DATASET_FILE.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue

    if len(rows) < min_samples:
        raise ValueError(f"Only {len(rows)} rows, need {min_samples}. Run dataset builder first.")

    X, y, meta = [], [], []
    for row in rows:
        feats = row.get("features", {})
        # Filter to known features
        filtered = {k: feats.get(k, 0.0) for k in FEATURE_KEYS}
        # Check for NaN/Inf
        if any(math.isnan(v) or math.isinf(v) for v in filtered.values()):
            continue
        X.append(filtered)
        y.append(row.get("outcome", 0))
        meta.append(row.get("meta", {}))

    print(f"[model] Loaded {len(X)} samples, {sum(y)} Up / {len(y)-sum(y)} Down")
    return X, y, meta


def X_to_array(X: list[dict]) -> np.ndarray:
    """Convert feature dicts to numpy array."""
    return np.array([[x.get(k, 0.0) for k in FEATURE_KEYS] for x in X], dtype=np.float32)


# ── Train/test split ─────────────────────────────────────────────────────────
def temporal_split(X, y, meta, test_frac: float = 0.2):
    """Time-ordered split: last test_frac windows = test set."""
    n = len(y)
    split = int(n * (1 - test_frac))
    # Sort by window_start to ensure temporal order
    order = sorted(range(n), key=lambda i: meta[i].get("window_start", 0))
    train_idx = order[:split]
    test_idx  = order[split:]
    X_train = X[train_idx]
    y_train = np.array(y, dtype=np.int32)[train_idx]
    X_test  = X[test_idx]
    y_test  = np.array(y, dtype=np.int32)[test_idx]
    print(f"[model] Temporal split: {len(X_train)} train, {len(X_test)} test")
    return X_train, y_train, X_test, y_test


# ── Model: Logistic Regression ───────────────────────────────────────────────
class LogisticModel:
    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.classes_: tuple = (0, 1)
        self.fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        # Add intercept
        Xb = np.hstack([np.ones((n, 1)), X])

        # Sigmoid + log-loss gradient descent with L2
        lr = 0.01
        reg = 0.01
        self.classes_ = (0, 1)
        self.coef_ = np.zeros(d + 1)
        for _ in range(5000):
            probs = stable_sigmoid(Xb @ self.coef_)
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            grad = Xb.T @ (probs - y) / n + reg * np.append(0, self.coef_[1:]) / n
            self.coef_ -= lr * grad
            if np.linalg.norm(grad) < 1e-5:
                break
        self.intercept_ = self.coef_[0]
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])
        raw = Xb @ self.coef_
        prob_up = stable_sigmoid(raw)
        return np.column_stack([1 - prob_up, prob_up])

    def feature_importance(self) -> dict:
        """Return feature → coefficient magnitude (unsigned)."""
        if not self.fitted:
            return {}
        coefs = self.coef_[1:]  # skip intercept
        return {k: float(abs(coefs[i])) for i, k in enumerate(FEATURE_KEYS)}


# ── Model: Gradient Boosting (sklearn-compatible, no C dep) ─────────────────
class SimpleGradientBoosting:
    """Minimal gradient boosting from scratch — no sklearn required."""

    def __init__(self, n_estimators=50, max_depth=3, lr=0.1, min_leaf=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.min_leaf = min_leaf
        self.trees: list = []
        self.fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        # Initial prediction: log-odds of mean
        p = max(0.001, min(0.999, y.mean()))
        self.base_score = math.log(p / (1 - p))
        self.trees = []
        raw = np.full(n, self.base_score, dtype=float)

        for t in range(self.n_estimators):
            # Compute pseudo-residuals
            probs = stable_sigmoid(raw)
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            residuals = y - probs

            # Fit a simple regression tree to residuals
            # Use sklearn if available, else fallback to manual tree
            try:
                from sklearn.tree import DecisionTreeRegressor
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_leaf,
                    random_state=t
                )
                tree.fit(X, residuals)
                self.trees.append(tree)
            except ImportError:
                # Manual stump
                tree = self._fit_stump(X, residuals)
                self.trees.append(tree)
            raw += self.lr * self._tree_predict(self.trees[-1], X)

        self.fitted = True
        return self

    def _fit_stump(self, X: np.ndarray, y: np.ndarray):
        """Fit a single depth-1 tree (stump)."""
        n, d = X.shape
        best_mse = float('inf')
        best = None
        for feat_idx in range(min(d, 10)):  # sample features
            vals = X[:, feat_idx]
            for threshold in np.percentile(vals, [25, 50, 75]):
                left  = y[vals <= threshold]
                right = y[vals > threshold]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                pred_l = left.mean()
                pred_r = right.mean()
                mse = (sum((left - pred_l) ** 2) + sum((right - pred_r) ** 2)) / n
                if mse < best_mse:
                    best_mse = mse
                    best = (feat_idx, threshold, float(pred_l), float(pred_r))
        return {"type": "stump", "feat": best[0], "thresh": best[1],
                "pred_l": best[2], "pred_r": best[3]} if best else None

    def _tree_predict(self, tree, X: np.ndarray) -> np.ndarray:
        if tree is None:
            return np.zeros(len(X), dtype=float)
        if isinstance(tree, dict) and tree.get("type") == "stump":
            vals = X[:, tree["feat"]]
            return np.where(vals > tree["thresh"], tree["pred_r"], tree["pred_l"]).astype(float)
        return np.asarray(tree.predict(X), dtype=float)

    def _predict_single(self, x, max_trees: int) -> float:
        if not self.fitted and not hasattr(self, "base_score"):
            raise ValueError("Model not fitted")
        p = self.base_score
        for i, tree in enumerate(self.trees[:max_trees]):
            if tree is None:
                continue
            p += self.lr * self._tree_predict(tree, x.reshape(1, -1))[0]
        return stable_sigmoid(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        raw = np.full(len(X), self.base_score, dtype=float)
        for tree in self.trees:
            raw += self.lr * self._tree_predict(tree, X)
        prob_up = stable_sigmoid(raw)
        prob_up = np.clip(prob_up, 1e-9, 1 - 1e-9)
        return np.column_stack([1 - prob_up, prob_up])


# ── Calibration ──────────────────────────────────────────────────────────────
def isotonic_calibrate(probs: np.ndarray, y: np.ndarray,
                       n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Isotonic regression calibration.
    Returns (bin_edges, bin_values) for applying to new predictions.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_edges = bins
    bin_values = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:  # include right edge of last bin
            mask = (probs >= bins[i])
        if mask.sum() >= 2:
            bin_values[i] = y[mask].mean()
        else:
            bin_values[i] = (bins[i] + bins[i + 1]) / 2 if i < n_bins - 1 else 0.75

    # Monotonic adjustment (ensure non-decreasing)
    for i in range(1, n_bins):
        if bin_values[i] < bin_values[i - 1]:
            bin_values[i] = bin_values[i - 1]

    return bin_edges, bin_values


def apply_calibration(probs: np.ndarray,
                      bin_edges: np.ndarray,
                      bin_values: np.ndarray) -> np.ndarray:
    """Apply isotonic calibration to raw probabilities."""
    calibrated = np.zeros_like(probs)
    for i in range(len(probs)):
        idx = np.searchsorted(bin_edges[1:], probs[i], side='right')
        idx = min(idx, len(bin_values) - 1)
        calibrated[i] = bin_values[idx]
    return np.clip(calibrated, 0.001, 0.999)


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-9, precision + recall)

    # AUC (approximate via rank)
    auc = _auc(y_true, y_prob)

    # Calibration error per decile
    cal_errors = _calibration_error(y_true, y_prob, n_bins=10)

    # Mean decile calibration error
    mean_cal_error = float(np.mean(np.abs(cal_errors)))

    return {
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "auc":        auc,
        "mean_cal_error": mean_cal_error,
        "cal_errors_per_decile": cal_errors,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "wr_at_threshold": tp / max(1, tp + fn),
    }


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Approximate AUC using Mann-Whitney U statistic."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    auc_sum = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                auc_sum += 1
            elif p == n:
                auc_sum += 0.5
    return auc_sum / (len(pos) * len(neg))


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> np.ndarray:
    bins = np.linspace(0, 1, n_bins + 1)
    errors = np.zeros(n_bins)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i])
        if mask.sum() >= 2:
            pred_mean = y_prob[mask].mean()
            actual    = y_true[mask].mean()
            errors[i] = abs(pred_mean - actual)
        else:
            errors[i] = 0.0
    return errors


# ── Main training ─────────────────────────────────────────────────────────────
def train_model():
    print(f"\n{'='*60}")
    print("BTC PROBABILITY MODEL — TRAINING")
    print(f"{'='*60}\n")

    # Load
    X_raw, y_raw, meta_raw = load_dataset(min_samples=100)
    X_all = X_to_array(X_raw)
    y_all = np.array(y_raw, dtype=np.int32)

    # Temporal split
    X_train, y_train, X_test, y_test = temporal_split(X_all, y_all, meta_raw, test_frac=0.2)

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\n[LR] Training logistic regression...")
    lr_model = LogisticModel()
    lr_model.fit(X_train, y_train)

    lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
    lr_test_probs  = lr_model.predict_proba(X_test)[:, 1]

    lr_train_eval = evaluate(y_train, lr_train_probs)
    lr_test_eval  = evaluate(y_test,  lr_test_probs)
    print(f"[LR] Train AUC={lr_train_eval['auc']:.3f}, "
          f"CalErr={lr_train_eval['mean_cal_error']:.3f}")
    print(f"[LR] Test  AUC={lr_test_eval['auc']:.3f}, "
          f"CalErr={lr_test_eval['mean_cal_error']:.3f}, "
          f"WR={lr_test_eval['wr_at_threshold']:.1%}")

    # LR feature importance
    fi = lr_model.feature_importance()
    top_fi = sorted(fi.items(), key=lambda x: -x[1])[:8]
    print(f"[LR] Top features: {top_fi}")

    # ── Gradient Boosting ────────────────────────────────────────────────────
    print("\n[GB] Training gradient boosting...")
    try:
        from sklearn.tree import DecisionTreeRegressor
        use_sklearn = True
    except ImportError:
        use_sklearn = False

    gb_model = SimpleGradientBoosting(n_estimators=50, max_depth=3, lr=0.1)
    gb_model.fit(X_train, y_train)

    gb_train_probs = gb_model.predict_proba(X_train)[:, 1]
    gb_test_probs  = gb_model.predict_proba(X_test)[:, 1]

    gb_train_eval = evaluate(y_train, gb_train_probs)
    gb_test_eval  = evaluate(y_test,  gb_test_probs)
    print(f"[GB] Train AUC={gb_train_eval['auc']:.3f}, "
          f"CalErr={gb_train_eval['mean_cal_error']:.3f}")
    print(f"[GB] Test  AUC={gb_test_eval['auc']:.3f}, "
          f"CalErr={gb_test_eval['mean_cal_error']:.3f}, "
          f"WR={gb_test_eval['wr_at_threshold']:.1%}")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    print("\n[ENS] Blending LR + GB (50/50)...")

    # Blend before calibration
    ensemble_train = 0.5 * lr_train_probs + 0.5 * gb_train_probs
    ensemble_test  = 0.5 * lr_test_probs  + 0.5 * gb_test_probs

    # Isotonic calibration on training set (OOF for fair eval)
    cal_edges, cal_values = isotonic_calibrate(ensemble_train, y_train)

    # Apply calibration to test
    ens_test_cal = apply_calibration(ensemble_test, cal_edges, cal_values)

    ens_train_eval = evaluate(y_train, ensemble_train)
    ens_test_eval  = evaluate(y_test,  ens_test_cal)

    print(f"[ENS] Train AUC={ens_train_eval['auc']:.3f}, "
          f"CalErr={ens_train_eval['mean_cal_error']:.3f}")
    print(f"[ENS] Test  AUC={ens_test_eval['auc']:.3f}, "
          f"CalErr={ens_test_eval['mean_cal_error']:.3f}, "
          f"WR={ens_test_eval['wr_at_threshold']:.1%}")

    # ── Calibration curve print ──────────────────────────────────────────────
    print("\n[CAL] Calibration curve (test set):")
    print("  Predicted  Actual  Error")
    for i, err in enumerate(ens_test_eval['cal_errors_per_decile']):
        lo = i / 10
        hi = (i + 1) / 10
        # Get actual WR in this bin
        mask = (ens_test_cal >= lo) & (ens_test_cal < hi)
        if i == 9: mask = (ens_test_cal >= lo)
        actual = y_test[mask].mean() if mask.sum() > 0 else 0
        mid = (lo + hi) / 2
        print(f"  {mid:.1f}       {actual:.1%}    {err:.1%}")

    # ── Final model: retrain on all data ─────────────────────────────────────
    print("\n[FIT] Retraining on full dataset for production model...")
    lr_full = LogisticModel()
    lr_full.fit(X_all, y_all)

    gb_full = SimpleGradientBoosting(n_estimators=50, max_depth=3, lr=0.1)
    gb_full.fit(X_all, y_all)

    full_probs = 0.5 * lr_full.predict_proba(X_all)[:, 1] + \
                 0.5 * gb_full.predict_proba(X_all)[:, 1]
    cal_edges_full, cal_values_full = isotonic_calibrate(full_probs, y_all)

    # ── Save model ───────────────────────────────────────────────────────────
    # Pickle keeps the actual GB trees. Metadata stays JSON-only. Previously the
    # trainer reported LR+GB metrics but inference silently used LR as a proxy,
    # so live/backtest behavior did not match the training report.
    model_bundle = {
        "lr_coef": lr_full.coef_.tolist(),
        "lr_intercept": float(lr_full.intercept_),
        "gb_n_estimators": gb_full.n_estimators,
        "gb_max_depth": gb_full.max_depth,
        "gb_lr": gb_full.lr,
        "gb_base_score": gb_full.base_score,
        "gb_trees": gb_full.trees,
        "gb_inference": "stored_trees",
        "cal_edges": cal_edges_full.tolist(),
        "cal_values": cal_values_full.tolist(),
        "feature_keys": FEATURE_KEYS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_all),
        "test_auc": ens_test_eval['auc'],
        "test_cal_err": ens_test_eval['mean_cal_error'],
        "test_wr": ens_test_eval['wr_at_threshold'],
    }

    # Save pickle
    with MODEL_PKL.open("wb") as f:
        pickle.dump(model_bundle, f)

    # Save metadata
    meta_bundle = {k: v for k, v in model_bundle.items() if k != "gb_trees"}
    with MODEL_META.open("w") as f:
        json.dump(meta_bundle, f, indent=2)

    print(f"\n[MODEL] Saved to {MODEL_PKL}")
    print(f"[MODEL] Test AUC: {ens_test_eval['auc']:.3f}")
    print(f"[MODEL] Test Cal Err: {ens_test_eval['mean_cal_error']:.3f}")
    print(f"[MODEL] Test WR@50%: {ens_test_eval['wr_at_threshold']:.1%}")

    return model_bundle


# ── Inference ─────────────────────────────────────────────────────────────────
class ProbabilityModel:
    """Load and run the trained probability model with safe paper-only fallback."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        fallback_confidence: float = 0.50,
        edge_threshold: float = 0.05,
    ):
        self.model_path = model_path or MODEL_PKL
        self.fallback_confidence = fallback_confidence
        self.edge_threshold = edge_threshold
        self.bundle: Optional[dict] = None
        self.cal_edges: Optional[np.ndarray] = None
        self.cal_values: Optional[np.ndarray] = None
        self.loaded = False
        self.load_error: Optional[str] = None
        self.prob_shrink = float(os.environ.get("BTC_PROB_SHRINK", "0.65"))
        self._load()

    def _load(self):
        if (
            MODEL_PROBATION.exists()
            and os.environ.get("BTC_IGNORE_MODEL_PROBATION", "0") != "1"
            and self.model_path.resolve() == MODEL_PKL.resolve()
        ):
            self.load_error = f"model on probation: {MODEL_PROBATION}"
            return
        if not self.model_path.exists():
            self.load_error = f"model not found: {self.model_path}"
            return
        try:
            with self.model_path.open("rb") as f:
                self.bundle = pickle.load(f)
            if not isinstance(self.bundle, dict):
                raise ValueError("model bundle is not a dict")
            if "lr_coef" not in self.bundle:
                raise KeyError("lr_coef")
            cal_edges = self.bundle.get("cal_edges")
            cal_values = self.bundle.get("cal_values")
            self.cal_edges = np.array(cal_edges, dtype=float) if cal_edges is not None else None
            self.cal_values = np.array(cal_values, dtype=float) if cal_values is not None else None
            self.loaded = True
        except Exception as e:
            self.bundle = None
            self.cal_edges = None
            self.cal_values = None
            self.loaded = False
            self.load_error = f"model load failed: {e}"

    def _coef(self) -> np.ndarray:
        """Flatten legacy/new coefficient shapes to a single vector."""
        return np.array(self.bundle.get("lr_coef", []), dtype=float).reshape(-1)

    def _model_feature_count(self) -> int:
        """Infer feature count for bundles with embedded or separate intercepts."""
        if not self.bundle:
            return len(FEATURE_KEYS)
        coef = self._coef()
        raw_keys = list(self.bundle.get("feature_keys") or FEATURE_KEYS)
        if coef.size == 0:
            return len(raw_keys)
        if "lr_intercept" in self.bundle and coef.size == len(raw_keys):
            return int(coef.size)
        return max(0, int(coef.size) - 1)

    def _active_feature_keys(self) -> list[str]:
        """Feature order from model bundle, with legacy shape fallback."""
        if not self.bundle:
            return FEATURE_KEYS
        keys = list(self.bundle.get("feature_keys") or FEATURE_KEYS)
        feature_count = self._model_feature_count()
        if feature_count <= 0:
            return keys
        if len(keys) >= feature_count:
            return keys[:feature_count]
        padding = [k for k in FEATURE_KEYS if k not in keys]
        return (keys + padding)[:feature_count]

    def _fallback_prediction(self, features: dict, reason: Optional[str] = None) -> dict:
        """Return a no-bet prediction when model inference is unavailable."""
        fallback_prob = float(self.fallback_confidence)
        poly_up, poly_down = self._market_prices(features)
        load_reason = reason or self.load_error or "unknown"
        return {
            "prob_up":       fallback_prob,
            "prob_down":     1.0 - fallback_prob,
            "model_conf":    fallback_prob,
            "edge":          0.0,
            "edge_required": self.edge_threshold,
            "should_bet":    False,
            "bet_on":        None,
            "model_prob":    fallback_prob,
            "market_prob":   max(poly_up, poly_down),
            "reasoning":     [
                f"model unavailable: {load_reason}",
                "paper-only fallback: no bet",
            ],
        }

    @staticmethod
    def _finite_prob(value, default: float = 0.5) -> float:
        try:
            out = float(value)
            if not math.isfinite(out):
                out = default
        except (TypeError, ValueError):
            out = default
        return min(0.999999, max(0.000001, out))

    def _market_prices(self, features: dict) -> tuple[float, float]:
        poly_up = self._finite_prob(features.get("poly_price_enter", 0.5))
        poly_down = self._finite_prob(features.get("poly_price_down", 1.0 - poly_up))
        return poly_up, poly_down

    def _shrink_probability(self, prob: float) -> float:
        try:
            shrink = max(0.0, min(1.0, float(self.prob_shrink)))
        except Exception:
            shrink = 0.65
        return max(0.001, min(0.999, 0.5 + (float(prob) - 0.5) * shrink))

    def predict(self, features: dict) -> dict:
        """Predict probability of Up outcome."""
        if not self.loaded:
            return self._fallback_prediction(features)

        try:
            feature_keys = self._active_feature_keys()
            x = np.array([[features.get(k, 0.0) for k in feature_keys]], dtype=np.float32)

            # LR prediction. Support both saved bundle shapes:
            #   - [intercept, coef_0, ...] (current native trainer)
            #   - [coef_0, ...] + lr_intercept (legacy/sklearn-style)
            coef = self._coef()
            if coef.size == x.shape[1] + 1:
                raw_lr = float(coef[0] + (x[0] @ coef[1:]))
            elif coef.size == x.shape[1]:
                raw_lr = float(self.bundle.get("lr_intercept", 0.0) + (x[0] @ coef))
            else:
                raise ValueError(f"coef/features shape mismatch: coef={coef.size} features={x.shape[1]}")
            prob_lr = stable_sigmoid(raw_lr)

            # GB prediction. New bundles store actual trained trees; old bundles
            # fall back to LR proxy so existing models remain readable.
            gb_trees = self.bundle.get("gb_trees")
            if isinstance(gb_trees, list) and gb_trees:
                gb_model = SimpleGradientBoosting(
                    n_estimators=int(self.bundle.get("gb_n_estimators", len(gb_trees))),
                    max_depth=int(self.bundle.get("gb_max_depth", 3)),
                    lr=float(self.bundle.get("gb_lr", 0.1)),
                )
                gb_model.base_score = float(self.bundle.get("gb_base_score", 0.0))
                gb_model.trees = gb_trees
                gb_model.fitted = True
                prob_gb = float(gb_model.predict_proba(x)[:, 1][0])
            else:
                prob_gb = prob_lr

            # Ensemble
            prob_raw = 0.5 * prob_lr + 0.5 * prob_gb

            # Calibrate when calibration payload exists; old bundles may not have it.
            has_calibration = (
                self.cal_edges is not None
                and self.cal_values is not None
                and len(self.cal_edges) >= 2
                and len(self.cal_values) > 0
            )
            if has_calibration:
                idx = np.searchsorted(self.cal_edges[1:], prob_raw)
                idx = min(idx, len(self.cal_values) - 1)
                prob_up = float(np.clip(self.cal_values[idx], 0.001, 0.999))
            else:
                prob_up = float(np.clip(prob_raw, 0.001, 0.999))
            prob_up = self._shrink_probability(prob_up)
            prob_down = 1.0 - prob_up
        except Exception as e:
            return self._fallback_prediction(features, f"prediction failed: {e}")

        poly_up, poly_down = self._market_prices(features)
        edge_up    = prob_up  - poly_up
        edge_down  = prob_down - poly_down

        # Determine bet direction and edge
        if edge_up > edge_down and edge_up > 0:
            bet_on = "Up"
            edge = edge_up
            model_prob = prob_up
            market_prob = poly_up
        elif edge_down > 0:
            bet_on = "Down"
            edge = edge_down
            model_prob = prob_down
            market_prob = poly_down
        else:
            bet_on = None
            edge = 0.0
            model_prob = max(prob_up, prob_down)
            market_prob = poly_up if prob_up > prob_down else poly_down

        return {
            "prob_up":       prob_up,
            "prob_down":     prob_down,
            "model_conf":    model_prob,
            "edge":          edge,
            "edge_required": self.edge_threshold,
            "should_bet":    edge > self.edge_threshold,
            "bet_on":        bet_on,
            "model_prob":    model_prob,
            "market_prob":   market_prob,
            "reasoning":     [
                f"model_prob_up={prob_up:.1%}",
                f"market_prob={market_prob:.1%}",
                f"edge={edge:+.1%}",
                f"should_bet={edge > self.edge_threshold}",
            ],
        }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Train model")
    p.add_argument("--eval",  action="store_true",  help="Evaluate saved model on dataset")
    p.add_argument("--predict", type=str, help='JSON features string, e.g. \'{"poly_price_enter":0.55,"btc_delta":25}\'')
    p.add_argument("--model", type=str, default=str(MODEL_PKL))
    args = p.parse_args()

    if args.train:
        train_model()

    elif args.predict:
        feats = json.loads(args.predict)
        model = ProbabilityModel(Path(args.model))
        result = model.predict(feats)
        print(json.dumps(result, indent=2))

    elif args.eval:
        # Evaluation/backtest validates the candidate model itself. Production
        # runtime may still keep current model on probation, but eval must not
        # silently score the paper-only fallback as if it were the model.
        os.environ.setdefault("BTC_IGNORE_MODEL_PROBATION", "1")
        X_raw, y_raw, _ = load_dataset()
        model = ProbabilityModel(Path(args.model))
        probs = []
        for feats in X_raw:
            r = model.predict(feats)
            probs.append(r["prob_up"])
        probs = np.array(probs)
        y = np.array(y_raw)
        eval_result = evaluate(y, probs)
        print(json.dumps(eval_result, indent=2, default=json_safe))
