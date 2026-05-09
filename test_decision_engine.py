#!/usr/bin/env python3.11
"""Tests for decision_engine.py — the V2 §4 core.

Coverage focuses on:
  1. Maker-price direction (V1 had `bid - 1 tick` = behind bid; codex `[HIGH] SPEC.md:250-252`)
  2. Two-sided executable EV (V1 single-sided Kelly; codex `[HIGH] SPEC.md:248-249`)
  3. Fail-closed gates (codex `[MED] SPEC.md:317`)
  4. Kelly fractional sizing bounds

Run: python3.11 test_decision_engine.py
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decision_engine import (
    Decision,
    FeatureVector,
    HeuristicDecisionEngine,
    ModelDecisionEngine,
    PMOrderbook,
    Wallet,
    compute_executable_ev,
    kelly_fraction,
    maker_price,
)


# ─── Fakes for the predictor protocol ────────────────────────────────────────


class FakePredictor:
    """Returns whatever p_up is configured. Mirrors the Predictor protocol."""

    def __init__(self, p_up: float, schema_hash: str = "abc123def456"):
        self._p = p_up
        self.schema_hash = schema_hash

    def predict(self, features: FeatureVector) -> float:
        return self._p


class RaisingPredictor:
    schema_hash = "abc123def456"

    def predict(self, features: FeatureVector) -> float:
        raise RuntimeError("simulated model crash")


def fv(decision_ts: datetime | None = None, schema_hash: str = "abc123def456",
       **values) -> FeatureVector:
    """Build a FeatureVector for tests."""
    return FeatureVector(
        decision_ts=decision_ts or datetime.now(timezone.utc),
        schema_hash=schema_hash,
        values=values,
    )


def book(yes_bid: float, yes_ask: float,
         no_bid: float | None = None, no_ask: float | None = None) -> PMOrderbook:
    """Build a well-formed binary book; defaults to inverse pricing."""
    nb = no_bid if no_bid is not None else (1.0 - yes_ask)
    na = no_ask if no_ask is not None else (1.0 - yes_bid)
    return PMOrderbook(
        yes_token_id="YES_TOK", no_token_id="NO_TOK",
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=nb, no_ask=na,
    )


def good_wallet(balance: float = 100.0) -> Wallet:
    return Wallet(usdc_balance=balance, open_orders=0, last_refreshed_age_sec=5.0)


# ─── 1. Maker price (the V1 bug) ─────────────────────────────────────────────


class TestMakerPrice(unittest.TestCase):

    def test_buy_uses_inside_spread_NOT_behind_bid(self):
        """V1 said `bid - 1 tick` for a buy = $0.49 when bid=$0.50.
        That's behind the queue. Correct is $0.51 (inside spread).
        """
        # bid=$0.50, ask=$0.55. V1 wrong = $0.49. Correct = $0.51.
        result = maker_price("BUY", best_bid=0.50, best_ask=0.55, tick=0.01)
        self.assertEqual(result, 0.51, "BUY must post INSIDE the spread")
        self.assertGreater(result, 0.50, "BUY must NOT post behind bid (V1 bug)")

    def test_buy_caps_below_ask(self):
        """If best_bid + tick would meet or cross ask, cap at ask - tick."""
        # Tight spread: bid=$0.49, ask=$0.50, tick=$0.01 → would compute $0.50 == ask
        # Cap to ask - tick = $0.49.
        result = maker_price("BUY", best_bid=0.49, best_ask=0.50, tick=0.01)
        self.assertEqual(result, 0.49)
        self.assertLess(result, 0.50, "BUY must NEVER post at or above ask")

    def test_sell_uses_inside_spread(self):
        result = maker_price("SELL", best_bid=0.50, best_ask=0.55, tick=0.01)
        self.assertEqual(result, 0.54)
        self.assertLess(result, 0.55)
        self.assertGreater(result, 0.50)

    def test_sell_caps_above_bid(self):
        # Tight spread: bid=$0.49, ask=$0.50.  ask - tick = $0.49 == bid.
        # Cap to max(0.49, 0.49+tick) = 0.50? No — max(ask-tick, bid+tick) = max(0.49, 0.50) = 0.50
        # But that's at the ask. The semantics for SELL on a tight market is murky;
        # callers should detect zero-spread separately.
        result = maker_price("SELL", best_bid=0.49, best_ask=0.50, tick=0.01)
        self.assertGreater(result, 0.49, "SELL must not post at or below bid")

    def test_invalid_book_raises(self):
        with self.assertRaises(ValueError):
            maker_price("BUY", best_bid=0.50, best_ask=0.50, tick=0.01)
        with self.assertRaises(ValueError):
            maker_price("BUY", best_bid=0.0, best_ask=0.50)
        with self.assertRaises(ValueError):
            maker_price("BUY", best_bid=0.51, best_ask=0.50)
        with self.assertRaises(ValueError):
            maker_price("BUY", 0.50, 0.55, tick=0.0)

    def test_unknown_side_raises(self):
        with self.assertRaises(ValueError):
            maker_price("HOLD", 0.50, 0.55)


# ─── 2. Executable EV math ───────────────────────────────────────────────────


class TestExecutableEV(unittest.TestCase):

    def test_yes_edge_when_p_high(self):
        """Model says 70% Up; YES costs 50¢. Edge = 0.70 - 0.50 = 0.20."""
        yes_edge, no_edge = compute_executable_ev(p_up=0.70, yes_ask=0.50, no_ask=0.51)
        self.assertAlmostEqual(yes_edge, 0.20, places=4)
        self.assertAlmostEqual(no_edge, 0.30 - 0.51, places=4)
        self.assertGreater(yes_edge, no_edge)

    def test_no_edge_when_p_low(self):
        """Model says 30% Up; NO costs 50¢. Edge = (1-0.30) - 0.50 = 0.20 on NO side."""
        yes_edge, no_edge = compute_executable_ev(p_up=0.30, yes_ask=0.51, no_ask=0.50)
        self.assertAlmostEqual(no_edge, 0.20, places=4)
        self.assertGreater(no_edge, yes_edge)

    def test_no_edge_when_market_efficient(self):
        """Model agrees with market — no edge."""
        yes_edge, no_edge = compute_executable_ev(p_up=0.50, yes_ask=0.50, no_ask=0.51)
        self.assertAlmostEqual(yes_edge, 0.0, places=4)
        self.assertLess(no_edge, 0)  # NO costs more than 1-p

    def test_fees_eat_edge(self):
        yes_edge, _ = compute_executable_ev(p_up=0.70, yes_ask=0.50, no_ask=0.51, fees_pct=0.02)
        self.assertAlmostEqual(yes_edge, 0.18, places=4)


# ─── 3. Kelly fraction ───────────────────────────────────────────────────────


class TestKellyFraction(unittest.TestCase):

    def test_kelly_zero_when_p_below_ask(self):
        self.assertEqual(kelly_fraction(p_win=0.40, ask=0.50), 0.0)
        self.assertEqual(kelly_fraction(p_win=0.50, ask=0.50), 0.0)

    def test_kelly_positive_when_edge_exists(self):
        # p=0.70 ask=0.50 → (0.70-0.50)/(1-0.50) = 0.40 → × 0.25 = 0.10
        f = kelly_fraction(p_win=0.70, ask=0.50, multiplier=0.25)
        self.assertAlmostEqual(f, 0.10, places=4)

    def test_kelly_bounded_by_multiplier(self):
        # p=0.99 ask=0.10 → (0.99-0.10)/0.90 = 0.989 → × 0.25 = 0.247
        f = kelly_fraction(p_win=0.99, ask=0.10, multiplier=0.25)
        self.assertLessEqual(f, 0.25)
        self.assertGreater(f, 0.20)

    def test_kelly_handles_degenerate_ask(self):
        self.assertEqual(kelly_fraction(p_win=0.7, ask=0.0), 0.0)
        self.assertEqual(kelly_fraction(p_win=0.7, ask=1.0), 0.0)


# ─── 4. ModelDecisionEngine — happy paths ────────────────────────────────────


class TestModelDecisionEngineHappy(unittest.TestCase):

    def test_picks_yes_when_p_high(self):
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=0.70))
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "bet")
        self.assertEqual(d.side, "YES")
        self.assertAlmostEqual(d.edge, 0.20, places=4)
        self.assertAlmostEqual(d.p_model, 0.70, places=4)
        self.assertAlmostEqual(d.p_market, 0.50, places=4)
        self.assertGreater(d.fraction, 0)
        self.assertLessEqual(d.fraction, 0.25)

    def test_picks_no_when_p_low(self):
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=0.30))
        # book helper computes NO from inverse: yes_bid=0.49 yes_ask=0.50
        # → no_bid = 1 - yes_ask = 0.50, no_ask = 1 - yes_bid = 0.51
        # NO edge = (1 - 0.30) - 0.51 = 0.19 (above 0.04 threshold)
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "bet")
        self.assertEqual(d.side, "NO")
        self.assertAlmostEqual(d.p_model, 0.70, places=4)  # 1 - p_up
        self.assertAlmostEqual(d.p_market, 0.51, places=4)  # no_ask, the executable price
        self.assertAlmostEqual(d.edge, 0.19, places=4)

    def test_skips_at_market_efficient(self):
        """Market efficient — no edge after threshold."""
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=0.51))
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("no_edge", d.reason)

    def test_threshold_gate(self):
        """Edge below threshold = skip even if positive."""
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=0.52), edge_threshold=0.04)
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        # yes_edge = 0.52 - 0.50 = 0.02 < 0.04 → skip
        self.assertEqual(d.action, "skip")


# ─── 5. ModelDecisionEngine — fail-closed gates ──────────────────────────────


class TestModelDecisionEngineFailClosed(unittest.TestCase):

    def setUp(self):
        self.eng = ModelDecisionEngine(predictor=FakePredictor(p_up=0.70))

    def test_skip_on_features_missing(self):
        d = self.eng.decide(None, book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertEqual(d.reason, "features_missing")

    def test_skip_on_schema_hash_mismatch(self):
        bad = fv(schema_hash="0000")
        d = self.eng.decide(bad, book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("schema_mismatch", d.reason)

    def test_skip_on_required_feature_nan(self):
        eng = ModelDecisionEngine(
            predictor=FakePredictor(p_up=0.70),
            required_features=["ret_5m", "spread_bps"],
        )
        d = eng.decide(fv(ret_5m=float("nan"), spread_bps=2.0), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("required_feature_nan", d.reason)

    def test_skip_on_book_missing(self):
        d = self.eng.decide(fv(), None, good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertEqual(d.reason, "book_missing")

    def test_skip_on_inverted_book(self):
        # yes_ask + no_ask = 0.6 + 0.6 = 1.2 → inverted
        bad = PMOrderbook(
            yes_token_id="Y", no_token_id="N",
            yes_bid=0.55, yes_ask=0.60, no_bid=0.55, no_ask=0.60,
        )
        d = self.eng.decide(fv(), bad, good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("book_inverted", d.reason)

    def test_skip_on_bad_book(self):
        bad = PMOrderbook(
            yes_token_id="Y", no_token_id="N",
            yes_bid=0.0, yes_ask=0.0, no_bid=1.0, no_ask=1.0,
        )
        d = self.eng.decide(fv(), bad, good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("bad_book", d.reason)

    def test_skip_on_wallet_missing(self):
        d = self.eng.decide(fv(), book(0.49, 0.50), None)
        self.assertEqual(d.action, "skip")
        self.assertEqual(d.reason, "wallet_missing")

    def test_skip_on_low_balance(self):
        w = Wallet(usdc_balance=0.50, open_orders=0, last_refreshed_age_sec=5.0)
        d = self.eng.decide(fv(), book(0.49, 0.50), w)
        self.assertEqual(d.action, "skip")
        self.assertIn("insufficient_balance", d.reason)

    def test_skip_on_open_orders(self):
        w = Wallet(usdc_balance=100.0, open_orders=1, last_refreshed_age_sec=5.0)
        d = self.eng.decide(fv(), book(0.49, 0.50), w)
        self.assertEqual(d.action, "skip")
        self.assertIn("open_orders_present", d.reason)

    def test_skip_on_stale_wallet(self):
        w = Wallet(usdc_balance=100.0, open_orders=0, last_refreshed_age_sec=120.0)
        d = self.eng.decide(fv(), book(0.49, 0.50), w)
        self.assertEqual(d.action, "skip")
        self.assertIn("wallet_stale", d.reason)

    def test_skip_on_predictor_exception(self):
        eng = ModelDecisionEngine(predictor=RaisingPredictor())
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("predictor_error", d.reason)
        self.assertIn("RuntimeError", d.reason)

    def test_skip_on_nan_prediction(self):
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=float("nan")))
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("bad_prediction", d.reason)

    def test_skip_on_out_of_range_prediction(self):
        eng = ModelDecisionEngine(predictor=FakePredictor(p_up=1.5))
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("out_of_range", d.reason)


# ─── 6. Heuristic engine is disabled by default ──────────────────────────────


class TestHeuristicEngineDisabled(unittest.TestCase):

    def test_default_returns_skip(self):
        eng = HeuristicDecisionEngine()
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")
        self.assertIn("heuristic_disabled", d.reason)

    def test_explicitly_enabled_still_skips_v2_path(self):
        """Even if enabled, heuristic is not implemented in V2 — fails closed."""
        eng = HeuristicDecisionEngine(enabled=True)
        d = eng.decide(fv(), book(0.49, 0.50), good_wallet())
        self.assertEqual(d.action, "skip")


# ─── 7. End-to-end smoke ─────────────────────────────────────────────────────


class TestEndToEndSmoke(unittest.TestCase):
    """One full path: features → model → executable EV → bet decision."""

    def test_full_path(self):
        # Setup: model is highly confident in Up; YES costs 50¢
        eng = ModelDecisionEngine(
            predictor=FakePredictor(p_up=0.70),
            edge_threshold=0.04,
            kelly_multiplier=0.25,
            assume_maker_execution=True,  # zero fees
            required_features=["ret_5m", "microprice_imbalance"],
        )
        features = fv(ret_5m=15.0, microprice_imbalance=0.3, spread_bps=2.5)
        b = book(yes_bid=0.49, yes_ask=0.50)
        w = good_wallet(balance=10.0)

        d = eng.decide(features, b, w)

        self.assertEqual(d.action, "bet")
        self.assertEqual(d.side, "YES")
        self.assertAlmostEqual(d.edge, 0.20, places=4)
        # Kelly: (0.70 - 0.50)/(1 - 0.50) × 0.25 = 0.10
        self.assertAlmostEqual(d.fraction, 0.10, places=4)
        # Implied stake = 0.10 × $10 = $1.00 (above $1 min ticket per V2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
