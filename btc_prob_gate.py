#!/usr/bin/env python3
"""
Phase 3: ProbabilityGate — Drop-in replacement for broken ensemble confidence gate
==================================================================================
Replaces the additive ensemble confidence check in btc_paper_fast.py.

Key changes:
  OLD (broken): if sig["conf"] >= params.conf_thresh: place_trade(...)
  NEW (correct): if prob_gate.should_bet(features) -> edge > threshold: place_trade(...)

The ProbabilityGate wraps the trained model and applies:
  1. Poly price edge filter: model_prob > market_prob + edge_threshold
  2. Hard poly price ceiling: never bet on poly > 0.72 without edge > 0.12
  3. Edge minimum: edge > 0.04 (slippage buffer after fees)

Usage in btc_paper_fast.py:
  from btc_prob_gate import ProbabilityGate

  # In PaperTrader.__init__:
  self.prob_gate = ProbabilityGate(
      model_path=MODEL_DIR / "btc_prob_model_current.pkl",
      edge_threshold=0.05,
      poly_price_ceiling=0.72,
      poly_ceiling_edge_buffer=0.12,
  )

  # In _place_trade(), replace:
  #   if sig["conf"] >= self.params.conf_thresh:
  # With:
  #   prob_decision = self.prob_gate.evaluate(features_dict)
  #   if not prob_decision.should_bet:
  #       self._log_skip("prob_gate", f"⏸️ NO EDGE: {prob_decision.summary}")
  #       return None
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR    = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
MODEL_DIR   = DATA_DIR / "model"
MODEL_PKL   = MODEL_DIR / "btc_prob_model_current.pkl"
MODEL_PROBATION = MODEL_DIR / "btc_prob_model_probation.json"

try:
    from btc_external_metrics import EXTERNAL_PROB_FEATURE_KEYS
except Exception:
    EXTERNAL_PROB_FEATURE_KEYS = []

# ── Feature keys (must match btc_prob_dataset.py and btc_probability_model.py) ──
FEATURE_KEYS = [
    "poly_price_enter", "poly_vs_50",
    "btc_delta", "btc_delta_pct",
    "rsi_5m", "rsi_15m",
    "macd_hist_5m", "macd_hist_15m",
    "bb_pos_5m", "bb_pos_15m",
    "vol_ratio_5m", "vol_ratio_15m",
    "ob_imbalance",
    "momentum_5m_pct", "momentum_15m_pct",
    "realized_vol",
    "hour_utc", "tf_minutes", "direction_sign",
    "poly_bucket_high", "poly_bucket_very_high",
] + list(EXTERNAL_PROB_FEATURE_KEYS)


def stable_sigmoid(raw: float) -> float:
    """Numerically stable sigmoid for extreme model logits."""
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


@dataclass
class BetDecision:
    """Structured output from the probability gate."""
    should_bet: bool
    direction: str          # "Up" | "Down" | None
    model_prob: float       # model's probability for chosen direction
    market_prob: float      # Polymarket-implied probability
    edge: float             # model_prob - market_prob
    prob_up: float
    prob_down: float
    gate_reason: str        # human-readable gate decision reason
    summary: str             # one-line log string


class ProbabilityGate:
    """
    Evaluates whether a trade has edge over the market.

    Gate logic:
      1. Get model probability from trained model
      2. Compute edge = model_prob - market_prob
      3. Apply poly_price_ceiling: hard cap on market-implied probability
      4. edge must exceed threshold AND poly ceiling rule

    Invariants enforced:
      - NEVER bet poly > 0.72 without edge > 0.12
      - NEVER bet poly > 0.80 at all (market already priced in)
      - Edge must be positive AND exceed edge_threshold
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        edge_threshold: float = 0.05,
        poly_price_ceiling: float = 0.72,
        poly_ceiling_edge_buffer: float = 0.12,
        hard_poly_cap: float = 0.82,
        fallback_confidence: float = 0.50,  # if model unavailable, default to no-bet
    ):
        self.model_path = model_path or MODEL_PKL
        self.edge_threshold = edge_threshold
        self.poly_price_ceiling = poly_price_ceiling
        self.poly_ceiling_edge_buffer = poly_ceiling_edge_buffer
        self.hard_poly_cap = hard_poly_cap
        self.fallback_confidence = fallback_confidence
        self._bundle: Optional[dict] = None
        self._cal_edges: Optional[list] = None
        self._cal_values: Optional[list] = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self.prob_shrink = float(os.environ.get("BTC_PROB_SHRINK", "0.65"))
        self._load()

    def _load(self):
        """Load model bundle lazily."""
        if os.environ.get("BTC_PROB_GATE_DISABLED", "0") == "1":
            self._load_error = "probability gate disabled by BTC_PROB_GATE_DISABLED=1"
            return
        if (
            MODEL_PROBATION.exists()
            and os.environ.get("BTC_IGNORE_MODEL_PROBATION", "0") != "1"
            and self.model_path.resolve() == MODEL_PKL.resolve()
        ):
            self._load_error = f"model on probation: {MODEL_PROBATION}"
            return
        if not self.model_path.exists():
            self._load_error = f"model not found: {self.model_path}"
            return
        try:
            with self.model_path.open("rb") as f:
                self._bundle = pickle.load(f)
            if not isinstance(self._bundle, dict):
                raise ValueError("model bundle is not a dict")
            if "lr_coef" not in self._bundle:
                raise KeyError("lr_coef")
            self._cal_edges = self._bundle.get("cal_edges")
            self._cal_values = self._bundle.get("cal_values")
            self._loaded = True
        except Exception as e:
            self._load_error = f"model load failed: {e}"

    def _fallback_decision(
        self,
        features: dict,
        direction_hint: Optional[str],
        reason: Optional[str] = None,
    ) -> BetDecision:
        """Safe paper-only no-bet fallback when model inference is unavailable."""
        poly_up, poly_down = self._market_prices(features)
        poly_price = self._market_prob_for_direction(direction_hint, poly_up, poly_down)
        load_reason = reason or self._load_error or "unknown"
        fallback_prob = float(self.fallback_confidence)
        return BetDecision(
            should_bet=False,
            direction=direction_hint,
            model_prob=fallback_prob,
            market_prob=poly_price,
            edge=0.0,
            prob_up=fallback_prob,
            prob_down=1.0 - fallback_prob,
            gate_reason=f"model unavailable: {load_reason}",
            summary=f"MODEL_DOWN: {load_reason}",
        )

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
        """Return absolute Up/Down market probabilities from feature dict."""
        poly_up = self._finite_prob(features.get("poly_price_enter", 0.5))
        poly_down = self._finite_prob(features.get("poly_price_down", 1.0 - poly_up))
        return poly_up, poly_down

    def _market_prob_for_direction(self, direction: Optional[str], poly_up: float, poly_down: float) -> float:
        if direction == "Down":
            return poly_down
        if direction == "Up":
            return poly_up
        return max(poly_up, poly_down)

    def _coef(self):
        """Flatten legacy/new coefficient shapes to a single vector."""
        import numpy as np
        return np.array(self._bundle.get("lr_coef", []), dtype=float).reshape(-1)

    def _model_feature_count(self) -> int:
        """Infer feature count for bundles with embedded or separate intercepts."""
        if not self._bundle:
            return len(FEATURE_KEYS)
        coef = self._coef()
        raw_keys = list(self._bundle.get("feature_keys") or FEATURE_KEYS)
        if coef.size == 0:
            return len(raw_keys)
        if "lr_intercept" in self._bundle and coef.size == len(raw_keys):
            return int(coef.size)
        return max(0, int(coef.size) - 1)

    def _lr_predict(self, x: list[float]) -> float:
        """Logistic regression prediction (no sklearn needed)."""
        import numpy as np
        coef = self._coef()
        xv = np.array(x, dtype=float)
        if coef.size == xv.size + 1:
            raw = float(coef[0] + (xv @ coef[1:]))
        elif coef.size == xv.size:
            raw = float(self._bundle.get("lr_intercept", 0.0) + (xv @ coef))
        else:
            raise ValueError(f"coef/features shape mismatch: coef={coef.size} features={xv.size}")
        return stable_sigmoid(raw)

    def _gb_predict(self, x: list[float]) -> Optional[float]:
        """Gradient-boosting prediction for bundles that persist actual trees."""
        trees = self._bundle.get("gb_trees") if self._bundle else None
        if not isinstance(trees, list) or not trees:
            return None
        import numpy as np
        X = np.array([x], dtype=float)
        raw = float(self._bundle.get("gb_base_score", 0.0))
        lr = float(self._bundle.get("gb_lr", 0.1))
        for tree in trees:
            if tree is None:
                continue
            if isinstance(tree, dict) and tree.get("type") == "stump":
                feat = int(tree.get("feat", 0))
                thresh = float(tree.get("thresh", 0.0))
                pred = float(tree.get("pred_r" if X[0, feat] > thresh else "pred_l", 0.0))
            else:
                pred = float(tree.predict(X)[0])
            raw += lr * pred
        return stable_sigmoid(raw)

    def _calibrate(self, raw_prob: float) -> float:
        """Apply isotonic calibration."""
        import numpy as np
        if self._cal_edges is None or self._cal_values is None:
            return raw_prob
        edges = np.array(self._cal_edges)
        values = np.array(self._cal_values)
        if len(edges) < 2 or len(values) == 0:
            return raw_prob
        idx = int(np.searchsorted(edges[1:], raw_prob))
        idx = min(idx, len(values) - 1)
        return float(np.clip(values[idx], 0.001, 0.999))

    def _shrink_probability(self, prob: float) -> float:
        """Conservative small-sample calibration guard.

        Lab-trained models with <500 rows can be directionally useful but wildly
        overconfident. Shrink toward 50% unless explicitly disabled by
        BTC_PROB_SHRINK=1.
        """
        try:
            shrink = max(0.0, min(1.0, float(self.prob_shrink)))
        except Exception:
            shrink = 0.65
        return max(0.001, min(0.999, 0.5 + (float(prob) - 0.5) * shrink))

    def _active_feature_keys(self) -> list[str]:
        """Feature order from model bundle, with legacy shape fallback."""
        if not self._bundle:
            return FEATURE_KEYS
        keys = list(self._bundle.get("feature_keys") or FEATURE_KEYS)
        feature_count = self._model_feature_count()
        if feature_count <= 0:
            return keys
        if len(keys) >= feature_count:
            return keys[:feature_count]
        padding = [k for k in FEATURE_KEYS if k not in keys]
        return (keys + padding)[:feature_count]

    def _feats_to_array(self, features: dict) -> list[float]:
        """Convert feature dict to ordered array matching the loaded model."""
        return [float(features.get(k, 0.0)) for k in self._active_feature_keys()]

    def evaluate(self, features: dict, direction_hint: Optional[str] = None) -> BetDecision:
        """
        Evaluate whether to bet on a window.

        Args:
            features: dict of feature values (see FEATURE_KEYS)
            direction_hint: "Up" or "Down" (from delta check), or None

        Returns:
            BetDecision with should_bet, probabilities, edge, and reason
        """
        poly_up, poly_down = self._market_prices(features)

        # ── Hard cap: market already priced it in ─────────────────────────────
        hinted_market_prob = self._market_prob_for_direction(direction_hint, poly_up, poly_down)
        if direction_hint in ("Up", "Down") and hinted_market_prob >= self.hard_poly_cap:
            return BetDecision(
                should_bet=False,
                direction=direction_hint,
                model_prob=hinted_market_prob,
                market_prob=hinted_market_prob,
                edge=0.0,
                prob_up=poly_up,
                prob_down=poly_down,
                gate_reason=f"poly={hinted_market_prob:.2f} >= hard_cap={self.hard_poly_cap:.2f}",
                summary=f"POLY_CAP: poly={hinted_market_prob:.2f} >= {self.hard_poly_cap:.2f}",
            )

        # ── Model unavailable fallback ────────────────────────────────────────
        if not self._loaded:
            return self._fallback_decision(features, direction_hint)

        # ── Model prediction ──────────────────────────────────────────────────
        try:
            x = self._feats_to_array(features)
            raw_prob_up = self._lr_predict(x)
            gb_prob_up = self._gb_predict(x)
            if gb_prob_up is not None:
                raw_prob_up = 0.5 * raw_prob_up + 0.5 * gb_prob_up
            prob_up_cal = self._shrink_probability(self._calibrate(raw_prob_up))
            prob_down_cal = 1.0 - prob_up_cal
        except Exception as e:
            return self._fallback_decision(features, direction_hint, f"prediction failed: {e}")

        # ── Edge calculation ──────────────────────────────────────────────────
        edge_up   = prob_up_cal   - poly_up
        edge_down = prob_down_cal - poly_down

        # ── Direction selection ────────────────────────────────────────────────
        if direction_hint in ("Up", "Down"):
            # Respect the delta signal direction
            if direction_hint == "Up":
                edge = edge_up
                prob = prob_up_cal
                market_prob = poly_up
            else:
                edge = edge_down
                prob = prob_down_cal
                market_prob = poly_down
        elif edge_up >= edge_down:
            edge = edge_up
            prob = prob_up_cal
            direction_hint = "Up"
            market_prob = poly_up
        else:
            edge = edge_down
            prob = prob_down_cal
            direction_hint = "Down"
            market_prob = poly_down

        if market_prob >= self.hard_poly_cap:
            return BetDecision(
                should_bet=False,
                direction=direction_hint,
                model_prob=prob,
                market_prob=market_prob,
                edge=0.0,
                prob_up=prob_up_cal,
                prob_down=prob_down_cal,
                gate_reason=f"poly={market_prob:.2f} >= hard_cap={self.hard_poly_cap:.2f}",
                summary=f"POLY_CAP: poly={market_prob:.2f} >= {self.hard_poly_cap:.2f}",
            )

        # ── Poly ceiling soft rule ─────────────────────────────────────────────
        poly_ceiling_edge = self.poly_ceiling_edge_buffer if market_prob >= self.poly_price_ceiling else 0.0

        # ── Final gate: edge > threshold + poly ceiling buffer ─────────────────
        effective_thresh = self.edge_threshold + poly_ceiling_edge
        should_bet = edge > effective_thresh

        # Build reason
        if not should_bet:
            if market_prob >= self.poly_price_ceiling and edge <= self.poly_ceiling_edge_buffer:
                reason = (f"poly={market_prob:.2f}>=ceiling={self.poly_price_ceiling:.2f} "
                          f"but edge={edge:.1%}<buffer={self.poly_ceiling_edge_buffer:.1%}")
            elif edge <= self.edge_threshold:
                reason = f"edge={edge:.1%} <= threshold={self.edge_threshold:.1%}"
            elif market_prob >= self.hard_poly_cap:
                reason = f"poly={market_prob:.2f} >= hard_cap={self.hard_poly_cap:.2f}"
            else:
                reason = f"no edge (prob_up={prob_up_cal:.1%}, poly={market_prob:.1%})"
        else:
            reason = "EDGE_OK"

        summary = (
            f"{'BET' if should_bet else 'SKIP'} "
            f"poly={market_prob:.2f} model_prob={prob:.1%} "
            f"edge={edge:+.1%} eff_thresh={effective_thresh:.1%}"
        )

        return BetDecision(
            should_bet=should_bet,
            direction=direction_hint,
            model_prob=prob,
            market_prob=market_prob,
            edge=edge,
            prob_up=prob_up_cal,
            prob_down=prob_down_cal,
            gate_reason=reason,
            summary=summary,
        )


# ── Convenience: quick eval from CLI ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--feats", type=str, required=True,
                   help='JSON features dict')
    p.add_argument("--direction", type=str, default=None,
                   help='"Up" or "Down" direction hint')
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Edge threshold (default 0.05)")
    p.add_argument("--model", type=str, default=str(MODEL_PKL))
    args = p.parse_args()

    feats = json.loads(args.feats)
    gate = ProbabilityGate(
        model_path=Path(args.model),
        edge_threshold=args.threshold,
    )
    decision = gate.evaluate(feats, direction_hint=args.direction)
    print(f"Direction: {decision.direction}")
    print(f"Model Prob Up: {decision.prob_up:.1%}")
    print(f"Model Prob Down: {decision.prob_down:.1%}")
    print(f"Market Prob: {decision.market_prob:.1%}")
    print(f"Edge: {decision.edge:+.1%}")
    print(f"Should Bet: {decision.should_bet}")
    print(f"Reason: {decision.gate_reason}")
    print(f"Log: {decision.summary}")
