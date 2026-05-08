#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import btc_prob_dataset as ds
from btc_external_metrics import EXTERNAL_MODEL_FEATURE_KEYS


class ProbabilityDatasetTests(unittest.TestCase):
    def test_actual_up_outcome_converts_relative_journal_win(self):
        self.assertEqual(ds.actual_up_outcome("Up", True), 1)
        self.assertEqual(ds.actual_up_outcome("Up", False), 0)
        self.assertEqual(ds.actual_up_outcome("Down", True), 0)
        self.assertEqual(ds.actual_up_outcome("Down", False), 1)

    def test_normalize_poly_prices_inverts_down_rows_to_up_probability(self):
        up = ds.normalize_poly_prices("Up", 0.62)
        self.assertAlmostEqual(up["poly_price_up"], 0.62)
        self.assertAlmostEqual(up["poly_price_down"], 0.38)
        self.assertAlmostEqual(up["chosen_poly_price"], 0.62)

        down = ds.normalize_poly_prices("Down", 0.62)
        self.assertAlmostEqual(down["poly_price_up"], 0.38)
        self.assertAlmostEqual(down["poly_price_down"], 0.62)
        self.assertAlmostEqual(down["chosen_poly_price"], 0.62)

    def test_journal_external_features_preserve_entry_context(self):
        feats = ds.journal_external_features(
            {
                "external_bull_score": 0.25,
                "prob_features": {"cg_taker_30m_imbalance": 0.4},
            }
        )
        self.assertEqual(set(feats), set(EXTERNAL_MODEL_FEATURE_KEYS))
        self.assertEqual(feats["external_bull_score"], 0.25)
        self.assertEqual(feats["cg_taker_30m_imbalance"], 0.4)

    def test_load_journal_trades_from_multiple_lab_sources_keeps_source_identity(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            p1 = base / "run" / "01_alpha" / "state" / "intraday_journal.jsonl"
            p2 = base / "run" / "02_beta" / "state" / "intraday_journal.jsonl"
            p1.parent.mkdir(parents=True)
            p2.parent.mkdir(parents=True)
            row = {
                "mode": "paper",
                "won": True,
                "direction": "Up",
                "window_start": 1,
                "window_tf": 5,
                "market_id": "m",
                "placed_at": "t",
            }
            p1.write_text(json.dumps({**row, "strategy": "alpha"}) + "\n")
            p2.write_text(json.dumps({**row, "strategy": "beta"}) + "\n")

            trades = ds.load_journal_trades_from_sources([p1, p2])
            self.assertEqual(len(trades), 2)
            self.assertNotEqual(ds.dataset_key_from_trade(trades[0]), ds.dataset_key_from_trade(trades[1]))
            self.assertTrue(trades[0]["_source_journal"].endswith("intraday_journal.jsonl"))
            self.assertEqual(trades[0]["_source_line"], 1)

    def test_load_journal_trades_can_opt_into_live_fills(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intraday_journal.jsonl"
            paper = {
                "mode": "paper",
                "won": True,
                "direction": "Up",
                "window_start": 1,
                "window_tf": 5,
                "placed_at": "paper-t",
            }
            live = {
                "mode": "live",
                "won": False,
                "direction": "Down",
                "window_start": 2,
                "window_tf": 15,
                "placed_at": "live-t",
            }
            p.write_text(json.dumps(paper) + "\n" + json.dumps(live) + "\n")

            default_trades = ds.load_journal_trades_from_sources([p], include_live=False)
            live_trades = ds.load_journal_trades_from_sources([p], include_live=True)

            self.assertEqual([t["mode"] for t in default_trades], ["paper"])
            self.assertEqual([t["mode"] for t in live_trades], ["paper", "live"])
            self.assertEqual(ds.dataset_key_from_trade(live_trades[1])[0], "source")
            self.assertEqual(ds.dataset_key_from_trade({**live, "market_id": None})[0], "live")

    def test_journal_poly_price_prefers_pretrade_prob_features(self):
        trade = {"poly_price": 0.35, "prob_features": {"poly_price_enter": 0.61}}
        self.assertAlmostEqual(ds.journal_poly_price_for_features(trade), 0.61)

    def test_journal_feature_snapshot_normalizes_down_price_without_network(self):
        feats = ds.journal_feature_snapshot(
            {
                "direction": "Down",
                "window_tf": 15,
                "btc_delta": -42,
                "prob_features": {
                    "poly_price_enter": 0.60,  # chosen Down price in old journals
                    "rsi_5m": 40.0,
                    "macd_hist_5m": -1.0,
                },
            }
        )
        self.assertIsNotNone(feats)
        self.assertAlmostEqual(feats["poly_price_enter"], 0.40)
        self.assertAlmostEqual(feats["poly_price_down"], 0.60)
        self.assertAlmostEqual(feats["chosen_poly_price"], 0.60)
        self.assertEqual(feats["direction_sign"], -1.0)
        self.assertEqual(feats["tf_minutes"], 15.0)

    def test_require_feature_snapshot_skips_network_rehydration(self):
        old_out = ds.OUT_FILE
        old_load = ds.load_journal_trades
        old_extract = ds.extract_features_for_window
        calls = []
        with tempfile.TemporaryDirectory() as td:
            try:
                ds.OUT_FILE = Path(td) / "train.jsonl"
                ds.load_journal_trades = lambda: [
                    {
                        "mode": "paper",
                        "won": True,
                        "direction": "Up",
                        "window_start": 1000,
                        "window_tf": 5,
                        "market_id": "m",
                        "placed_at": "t",
                        "poly_price": 0.55,
                        "prob_features": {"poly_price_enter": 0.55},
                    }
                ]
                def fake_extract(*args, **kwargs):
                    calls.append((args, kwargs))
                    return {"rsi_5m": 50.0}

                ds.extract_features_for_window = fake_extract
                with mock.patch.dict(
                    "os.environ",
                    {"BTC_PROB_DATASET_REQUIRE_FEATURE_SNAPSHOT": "1"},
                    clear=False,
                ):
                    rows = ds.build_dataset(append=False)
                self.assertEqual(rows, [])
                self.assertEqual(calls, [])
            finally:
                ds.OUT_FILE = old_out
                ds.load_journal_trades = old_load
                ds.extract_features_for_window = old_extract


if __name__ == "__main__":
    unittest.main()
