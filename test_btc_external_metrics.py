import json
import fcntl
import numpy as np
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

import btc_external_metrics as em
from btc_external_metrics import (
    EXTERNAL_MODEL_FEATURE_KEYS,
    add_composite_features,
    context_feature_subset,
)
from btc_feature_engine import _init_db
from btc_prob_gate import ProbabilityGate
from btc_probability_model import ProbabilityModel, SimpleGradientBoosting


class ExternalMetricsTests(unittest.TestCase):
    def test_feature_subset_defaults_missing_values_to_zero(self):
        subset = context_feature_subset({"external_bull_score": 0.25})
        self.assertEqual(set(subset), set(EXTERNAL_MODEL_FEATURE_KEYS))
        self.assertEqual(subset["external_bull_score"], 0.25)
        self.assertEqual(subset["cg_taker_30m_imbalance"], 0.0)

    def test_composite_score_is_bounded(self):
        ctx = add_composite_features(
            {
                "cg_taker_30m_imbalance": 9,
                "cg_cvd_30m_imbalance": 9,
                "cg_orderbook_30m_imbalance": 9,
                "cg_liq_30m_imbalance": 9,
                "cg_oi_30m_chg_pct": 9,
                "bg_ls_5m_imbalance": 9,
                "bg_depth_imbalance": 9,
                "bg_ls_5m_chg": 9,
            }
        )
        self.assertLessEqual(ctx["external_bull_score"], 1.0)
        self.assertGreater(ctx["external_bull_score"], 0.0)

    def test_feature_db_schema_gets_external_columns(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "features.db"
            _init_db(db)
            conn = sqlite3.connect(db)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(features)").fetchall()}
            conn.close()
        self.assertTrue(set(EXTERNAL_MODEL_FEATURE_KEYS).issubset(cols))

    def test_probability_gate_accepts_legacy_model_feature_shape(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "legacy.pkl"
            # Old tiny model: 2 features + intercept. New external keys must not
            # create a shape mismatch.
            bundle = {
                "lr_coef": [0.0, 0.0, 0.0],
                "feature_keys": ["poly_price_enter", "poly_vs_50"],
                "cal_edges": [0.0, 1.0],
                "cal_values": [0.5, 0.5],
            }
            model_path.write_bytes(pickle.dumps(bundle))
            gate = ProbabilityGate(model_path=model_path)
            decision = gate.evaluate(
                {
                    "poly_price_enter": 0.49,
                    "poly_vs_50": -0.02,
                    "external_bull_score": 0.8,
                },
                direction_hint="Up",
            )
        self.assertFalse(decision.should_bet)
        self.assertEqual(decision.prob_up, 0.5)

    def test_probability_gate_missing_model_is_paper_only_no_bet(self):
        with tempfile.TemporaryDirectory() as td:
            missing_model = Path(td) / "missing.pkl"
            gate = ProbabilityGate(model_path=missing_model)
            decision = gate.evaluate(
                {"poly_price_enter": 0.35, "external_bull_score": 1.0},
                direction_hint="Up",
            )
        self.assertFalse(decision.should_bet)
        self.assertEqual(decision.edge, 0.0)
        self.assertEqual(decision.prob_up, 0.5)
        self.assertIn("model unavailable", decision.gate_reason)

    def test_probability_model_missing_model_is_paper_only_no_bet(self):
        with tempfile.TemporaryDirectory() as td:
            missing_model = Path(td) / "missing.pkl"
            model = ProbabilityModel(model_path=missing_model)
            prediction = model.predict({"poly_price_enter": 0.35, "external_bull_score": 1.0})
        self.assertFalse(prediction["should_bet"])
        self.assertIsNone(prediction["bet_on"])
        self.assertEqual(prediction["edge"], 0.0)
        self.assertEqual(prediction["prob_up"], 0.5)
        self.assertIn("paper-only fallback", " ".join(prediction["reasoning"]))

    def test_probability_model_accepts_legacy_model_feature_shape(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "legacy.pkl"
            bundle = {
                "lr_coef": [0.0, 0.0, 0.0],
                "feature_keys": ["poly_price_enter", "poly_vs_50"],
                "cal_edges": [0.0, 1.0],
                "cal_values": [0.5, 0.5],
            }
            model_path.write_bytes(pickle.dumps(bundle))
            model = ProbabilityModel(model_path=model_path)
            prediction = model.predict(
                {
                    "poly_price_enter": 0.49,
                    "poly_vs_50": -0.02,
                    "external_bull_score": 0.8,
                }
            )
        self.assertFalse(prediction["should_bet"])
        self.assertEqual(prediction["prob_up"], 0.5)

    def test_probability_gate_uses_stored_gradient_boosting_trees(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "gb.pkl"
            feature_keys = ["poly_price_enter", "poly_vs_50"]
            X = np.array(
                [[0.25, -0.50], [0.30, -0.40], [0.70, 0.40], [0.80, 0.60]] * 8,
                dtype=np.float32,
            )
            y = np.array([0, 0, 1, 1] * 8)
            gb = SimpleGradientBoosting(n_estimators=8, max_depth=2, lr=0.2).fit(X, y)
            bundle = {
                "lr_coef": [0.0, 0.0, 0.0],
                "feature_keys": feature_keys,
                "gb_n_estimators": gb.n_estimators,
                "gb_max_depth": gb.max_depth,
                "gb_lr": gb.lr,
                "gb_base_score": gb.base_score,
                "gb_trees": gb.trees,
                "gb_inference": "stored_trees",
            }
            model_path.write_bytes(pickle.dumps(bundle))
            gate = ProbabilityGate(model_path=model_path, edge_threshold=0.01)
            decision = gate.evaluate({"poly_price_enter": 0.40, "poly_vs_50": 0.60}, direction_hint="Up")
        self.assertGreater(decision.prob_up, 0.5)
        self.assertEqual(decision.gate_reason, "EDGE_OK")

    def test_refresh_lock_returns_stale_context_in_parallel_worker(self):
        old_file = em.CONTEXT_FILE
        old_lock = em.CONTEXT_LOCK_FILE
        old_ttl = em.CONTEXT_TTL_SECONDS
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                em.CONTEXT_FILE = root / "context.json"
                em.CONTEXT_LOCK_FILE = root / "context.lock"
                em.CONTEXT_TTL_SECONDS = -1
                em.CONTEXT_FILE.write_text(json.dumps({"fetched_at": 1, "ok": 1, "cg_ok": 1}) + "\n")
                with em.CONTEXT_LOCK_FILE.open("a+") as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    ctx = em.fetch_external_market_context(use_cache=True)
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                self.assertEqual(ctx["ok"], 1)
                self.assertEqual(ctx["stale_due_to_refresh_lock"], 1)
        finally:
            em.CONTEXT_FILE = old_file
            em.CONTEXT_LOCK_FILE = old_lock
            em.CONTEXT_TTL_SECONDS = old_ttl

    def test_coinglass_429_is_reported(self):
        old_secret = em._secret
        old_get_json = em._get_json
        try:
            em._secret = lambda name: "x" if name == "COINGLASS_API_KEY" else ""
            em._get_json = lambda *a, **k: {"code": "429", "msg": "Too Many Requests"}
            ctx = em.fetch_coinglass_context()
        finally:
            em._secret = old_secret
            em._get_json = old_get_json
        self.assertEqual(ctx["cg_ok"], 0)
        self.assertEqual(ctx["cg_rate_limited"], 1)
        self.assertEqual(ctx["cg_last_error_code"], "429")

    def test_coinalyze_context_parses_free_regime_feed(self):
        old_secret = em._secret
        old_coinalyze = em._coinalyze
        now = 1_700_000_000

        def fake_coinalyze(path, params):
            if path == "/v1/open-interest-history":
                return [{"symbol": "BTCUSDT_PERP.A", "history": [{"t": now - 1800, "c": 100.0}, {"t": now, "c": 110.0}]}]
            if path == "/v1/liquidation-history":
                return [{"symbol": "BTCUSDT_PERP.A", "history": [{"t": now, "l": 2.0, "s": 6.0}]}]
            if path == "/v1/funding-rate-history":
                return [{"symbol": "BTCUSDT_PERP.A", "history": [{"t": now, "c": -0.002}]}]
            if path == "/v1/predicted-funding-rate-history":
                return [{"symbol": "BTCUSDT_PERP.A", "history": [{"t": now, "c": 0.003}]}]
            if path == "/v1/long-short-ratio-history":
                return [{"symbol": "BTCUSDT_PERP.A", "history": [{"t": now - 1800, "l": 40.0, "s": 60.0, "r": 0.67}, {"t": now, "l": 45.0, "s": 55.0, "r": 0.82}]}]
            if path == "/v1/open-interest":
                return [{"symbol": "BTCUSDT_PERP.A", "value": 123.0}]
            if path == "/v1/funding-rate":
                return [{"symbol": "BTCUSDT_PERP.A", "value": -0.004}]
            return []

        try:
            em._secret = lambda name: "x" if name == "COINALYZE_API_KEY" else ""
            em._coinalyze = fake_coinalyze
            ctx = em.fetch_coinalyze_context()
        finally:
            em._secret = old_secret
            em._coinalyze = old_coinalyze

        self.assertEqual(ctx["ca_ok"], 1)
        self.assertEqual(ctx["ca_endpoint_count"], 7)
        self.assertAlmostEqual(ctx["ca_oi_30m_chg_pct"], 0.10)
        self.assertAlmostEqual(ctx["ca_liq_1h_imbalance"], 0.50)
        self.assertAlmostEqual(ctx["ca_ls_30m_imbalance"], -0.10)
        self.assertAlmostEqual(ctx["ca_ls_30m_chg"], 0.05)
        self.assertAlmostEqual(ctx["ca_funding_now"], -0.004)

    def test_free_venue_context_parsers(self):
        old_binance = em._binance_futures
        old_bybit = em._bybit
        old_post = em._post_json
        try:
            em._binance_futures = lambda path, params: {
                "/futures/data/openInterestHist": [{"timestamp": "1", "sumOpenInterest": "100"}, {"timestamp": "2", "sumOpenInterest": "120"}],
                "/futures/data/topLongShortPositionRatio": [{"timestamp": "2", "longAccount": "0.55", "shortAccount": "0.45"}],
                "/futures/data/takerlongshortRatio": [{"timestamp": "2", "buyVol": "70", "sellVol": "30"}],
                "/futures/data/globalLongShortAccountRatio": [{"timestamp": "2", "longAccount": "0.52", "shortAccount": "0.48"}],
                "/fapi/v1/fundingRate": [{"fundingTime": "2", "fundingRate": "0.0001"}],
            }[path]
            em._bybit = lambda path, params: {
                "/v5/market/open-interest": {"list": [{"timestamp": "1", "openInterest": "100"}, {"timestamp": "2", "openInterest": "90"}]},
                "/v5/market/funding/history": {"list": [{"fundingRateTimestamp": "2", "fundingRate": "-0.0002"}]},
            }[path]
            em._post_json = lambda url, payload, timeout=10: {
                "l2Book": {"levels": [[{"px": "100", "sz": "6"}], [{"px": "101", "sz": "2"}]]},
                "fundingHistory": [{"time": 1, "fundingRate": "0.001", "premium": "-0.002"}],
                "predictedFundings": [["BTC", [["HlPerp", {"fundingRate": "0.003"}], ["BinPerp", {"fundingRate": "0.004"}], ["BybitPerp", {"fundingRate": "0.005"}]]]],
            }[payload["type"]]
            bn = em.fetch_binance_futures_context()
            by = em.fetch_bybit_context()
            hl = em.fetch_hyperliquid_context()
        finally:
            em._binance_futures = old_binance
            em._bybit = old_bybit
            em._post_json = old_post

        self.assertEqual(bn["bn_ok"], 1)
        self.assertAlmostEqual(bn["bn_oi_30m_chg_pct"], 0.20)
        self.assertAlmostEqual(bn["bn_taker_15m_imbalance"], 0.40)
        self.assertEqual(by["by_ok"], 1)
        self.assertAlmostEqual(by["by_oi_30m_chg_pct"], -0.10)
        self.assertEqual(hl["hl_ok"], 1)
        self.assertAlmostEqual(hl["hl_depth_imbalance"], 0.50)
        self.assertAlmostEqual(hl["hl_pred_funding_hl"], 0.003)


if __name__ == "__main__":
    unittest.main()
