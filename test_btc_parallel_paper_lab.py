#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import btc_parallel_paper_lab as lab


class ParallelPaperLabTests(unittest.TestCase):
    def test_strategy_specs_are_unique_and_paper_safe(self):
        names = [s.name for s in lab.STRATEGIES]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 10)
        self.assertIn("firehose", {s.cohort for s in lab.STRATEGIES})
        self.assertTrue(any(s.cohort == "firehose" for s in lab.STRATEGIES[:8]))
        for spec in lab.STRATEGIES:
            self.assertLessEqual(spec.max_open, 1)
            self.assertLessEqual(spec.max_bet_pct, 0.20)
            self.assertLessEqual(spec.risk_halt_drawdown_pct, 0.20)
            self.assertGreater(spec.status_seconds, 0)
            overrides = spec.overrides()
            self.assertIn("delta_thresh", overrides)
            self.assertIn("conf_thresh", overrides)
            self.assertIn("ens_thresh", overrides)

    def test_worker_env_is_paper_only_and_scrubs_live_credentials(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = next(s for s in lab.STRATEGIES if s.cohort == "firehose")
            args = Namespace(capital=20.0, loop_sleep=5.0, market_check=15.0)
            env = lab.build_worker_env(
                {
                    "POLYMARKET_PRIVATE_KEY": "secret",
                    "POLYMARKET_API_SECRET": "secret",
                    "POLY_KEY": "secret",
                    "KEEP_ME": "yes",
                },
                spec,
                args,
                root / "logs",
                root / "logs" / "paper.log",
                root / "state" / "journal.jsonl",
                root / "state" / "params.json",
            )
            self.assertEqual(env["KEEP_ME"], "yes")
            self.assertNotIn("POLYMARKET_PRIVATE_KEY", env)
            self.assertNotIn("POLYMARKET_API_SECRET", env)
            self.assertNotIn("POLY_KEY", env)
            self.assertEqual(env["BTC_PAPER_ONLY"], "1")
            self.assertEqual(env["BTC_FAKE_MONEY_ONLY"], "1")
            self.assertEqual(env["BTC_LAB_FAKE_MONEY_ONLY"], "1")
            self.assertEqual(env["BTC_PAPER_EXPLORATION"], "1")
            self.assertEqual(env["BTC_ALLOW_PAPER_EXPLORATION"], "1")
            self.assertEqual(env["BTC_EXPLORATION_MIN_EDGE"], "0.08")
            self.assertEqual(env["BTC_EXPLORATION_MIN_CONF"], "0.45")
            self.assertEqual(env["BTC_EXPLORATION_MIN_POLY"], "0.25")
            self.assertEqual(env["BTC_EXPLORATION_MAX_POLY"], "0.70")
            self.assertEqual(env["BTC_EXPLORATION_MAX_SLIPPAGE_BPS"], "250")
            self.assertEqual(env["BTC_EXPLORATION_DELTA_FLOOR"], "1.00")
            self.assertEqual(env["BTC_PROB_MIN_POLY_PRICE"], "0.20")
            self.assertEqual(env["BTC_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE"], "0")
            self.assertEqual(env["BTC_CLOB_GAMMA_GAP_OVERRIDE_PROB"], "0.90")
            self.assertEqual(env["BTC_FAST_GA_ENABLED"], "0")
            self.assertEqual(env["BTC_DISABLE_PARAM_RELOAD"], "1")
            self.assertEqual(env["BTC_PROB_GATE_DISABLED"], "1")
            self.assertEqual(env["BTC_LIVE_CANARY_ACK"], "PAPER_LAB_NO_LIVE")
            self.assertEqual(env["BTC_REQUIRE_CLOB_QUOTE"], "0")
            self.assertEqual(env["BTC_LAB_COHORT"], "firehose")
            self.assertEqual(env["BTC_LOOP_SLEEP_SECONDS"], str(spec.loop_sleep))
            self.assertEqual(env["BTC_MARKET_CHECK_SECONDS"], str(spec.market_check))

    def test_worker_env_accepts_lab_scoped_cross_tf_risk_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = next(s for s in lab.STRATEGIES if s.cohort == "firehose")
            args = Namespace(capital=20.0, loop_sleep=5.0, market_check=15.0)
            env = lab.build_worker_env(
                {
                    "BTC_LAB_MAX_OPEN_BTC_TRADES": "2",
                    "BTC_LAB_MAX_OPEN_BTC_TRADES_PER_TF": "1",
                    "BTC_LAB_MAX_TOTAL_BTC_EXPOSURE_PCT": "0.32",
                    "BTC_LAB_ALLOW_CROSS_TF_CORRELATED_OPEN": "1",
                    "BTC_LAB_PROB_MIN_POLY_PRICE": "0.35",
                    "BTC_LAB_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE": "0",
                    "BTC_LAB_CLOB_GAMMA_GAP_OVERRIDE_PROB": "0.95",
                    "BTC_LAB_REQUIRE_CLOB_QUOTE": "1",
                    "BTC_LAB_REQUIRE_EXTERNAL_FLOW_AGREEMENT": "0",
                    "BTC_LAB_EXTERNAL_BULL_SCORE_MIN": "-0.20",
                    "BTC_LAB_EXTERNAL_BULL_SCORE_MAX": "-0.05",
                    "BTC_LAB_PROB_HARD_POLY_CAP": "0.56",
                    "BTC_LAB_PROB_POLY_PRICE_CEILING": "0.55",
                },
                spec,
                args,
                root / "logs",
                root / "logs" / "paper.log",
                root / "state" / "journal.jsonl",
                root / "state" / "params.json",
            )
            self.assertEqual(env["BTC_MAX_OPEN_BTC_TRADES"], "2")
            self.assertEqual(env["BTC_MAX_OPEN_BTC_TRADES_PER_TF"], "1")
            self.assertEqual(env["BTC_MAX_TOTAL_BTC_EXPOSURE_PCT"], "0.32")
            self.assertEqual(env["BTC_ALLOW_CROSS_TF_CORRELATED_OPEN"], "1")
            self.assertEqual(env["BTC_PROB_MIN_POLY_PRICE"], "0.35")
            self.assertEqual(env["BTC_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE"], "0")
            self.assertEqual(env["BTC_CLOB_GAMMA_GAP_OVERRIDE_PROB"], "0.95")
            self.assertEqual(env["BTC_REQUIRE_CLOB_QUOTE"], "1")
            self.assertEqual(env["BTC_REQUIRE_EXTERNAL_FLOW_AGREEMENT"], "0")
            self.assertEqual(env["BTC_EXTERNAL_BULL_SCORE_MIN"], "-0.20")
            self.assertEqual(env["BTC_EXTERNAL_BULL_SCORE_MAX"], "-0.05")
            self.assertEqual(env["BTC_PROB_HARD_POLY_CAP"], "0.56")
            self.assertEqual(env["BTC_PROB_POLY_PRICE_CEILING"], "0.55")

    def test_model_gate_profile_is_paper_only_and_uses_candidate_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_path = root / "candidate.pkl"
            model_path.write_bytes(b"fake")
            spec = next(s for s in lab.STRATEGIES if s.name == "firehose_gamma_probe")
            args = Namespace(
                capital=20.0,
                loop_sleep=5.0,
                market_check=15.0,
                profile="model_gate",
                model_path=str(model_path),
                model_edge=0.02,
                model_delta=1.0,
                prob_shrink=0.65,
            )
            env = lab.build_worker_env(
                {"POLYMARKET_PRIVATE_KEY": "secret"},
                spec,
                args,
                root / "logs",
                root / "logs" / "paper.log",
                root / "state" / "journal.jsonl",
                root / "state" / "params.json",
            )
            self.assertNotIn("POLYMARKET_PRIVATE_KEY", env)
            self.assertEqual(env["BTC_PAPER_ONLY"], "1")
            self.assertEqual(env["BTC_FAKE_MONEY_ONLY"], "1")
            self.assertEqual(env["POLYMARKET_LIVE_TRADING"], "0")
            self.assertEqual(env["BTC_LIVE_CANARY_ACK"], "PAPER_LAB_NO_LIVE")
            self.assertEqual(env["BTC_PAPER_EXPLORATION"], "0")
            self.assertEqual(env["BTC_ALLOW_PAPER_EXPLORATION"], "0")
            self.assertEqual(env["BTC_PROB_GATE_DISABLED"], "0")
            self.assertEqual(env["BTC_IGNORE_MODEL_PROBATION"], "1")
            self.assertEqual(env["BTC_PROB_MODEL_PATH"], str(model_path))
            self.assertEqual(env["BTC_PROB_EDGE_THRESHOLD"], "0.02")
            self.assertEqual(env["BTC_PROB_SHRINK"], "0.65")
            self.assertEqual(env["BTC_FORCE_DELTA_THRESHOLD"], "1.0")
            self.assertEqual(env["BTC_DELTA_HINT_SIGNAL"], "1")
            self.assertEqual(env["BTC_DELTA_HINT_MIN_CONF"], "0.45")

    def test_report_summarizes_isolated_journal_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            strat = root / "01_balanced"
            strat.mkdir()
            journal = strat / "journal.jsonl"
            log = strat / "paper.log"
            journal.write_text(
                json.dumps(
                    {
                        "won": True,
                        "pnl": 1.5,
                        "price_source": "gamma_outcomePrices_fallback",
                        "prob_decision": {"edge": 0.02, "gate_reason": "edge_ok"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "won": False,
                        "pnl": -2.0,
                        "price_source": "clob_book_ask_depth",
                        "clob_token_id": "token-1",
                        "clob_best_ask": 0.55,
                        "clob_slippage_bps": 12.5,
                        "external_bull_score": 0.41,
                        "exploration_mode": True,
                        "exploration_reason": "PAPER_EXPLORATION_MODEL_DOWN",
                        "prob_decision": {
                            "edge": 0.12,
                            "model_prob": 0.70,
                            "market_prob": 0.58,
                            "prob_up": 0.70,
                            "prob_down": 0.30,
                            "gate_reason": "edge_ok",
                        },
                    }
                )
                + "\n"
            )
            log.write_text("[x] elapsed=0.1h trades=2(W:1 L:1) HALT=drawdown\n")
            manifest = {
                "strategies": [
                    {
                        "name": "balanced",
                        "pid": 999999,
                        "journal_file": str(journal),
                        "log_file": str(log),
                        "capital": 20.0,
                    }
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            with redirect_stdout(StringIO()):
                rows = lab.report(root, json_out=False)
            self.assertEqual(rows[0]["trades"], 2)
            self.assertEqual(rows[0]["wins"], 1)
            self.assertEqual(rows[0]["losses"], 1)
            self.assertAlmostEqual(rows[0]["pnl"], -0.5)
            self.assertTrue(rows[0]["halted"])
            self.assertEqual(rows[0]["last_price_source"], "clob_book_ask_depth")
            self.assertEqual(rows[0]["price_sources"], ["clob_book_ask_depth", "gamma_outcomePrices_fallback"])
            self.assertEqual(rows[0]["clob_trades"], 1)
            self.assertEqual(rows[0]["live_valid_trades"], 1)
            self.assertEqual(rows[0]["live_valid_wins"], 0)
            self.assertEqual(rows[0]["live_valid_losses"], 1)
            self.assertAlmostEqual(rows[0]["live_valid_pnl"], -2.0)
            self.assertAlmostEqual(rows[0]["live_valid_wr"], 0.0)
            self.assertAlmostEqual(rows[0]["clob_ratio"], 0.5)
            self.assertEqual(rows[0]["gamma_fallback_trades"], 1)
            self.assertAlmostEqual(rows[0]["gamma_fallback_pnl"], 1.5)
            self.assertAlmostEqual(rows[0]["avg_edge"], 0.07)
            self.assertAlmostEqual(rows[0]["last_edge"], 0.12)
            self.assertAlmostEqual(rows[0]["last_model_prob"], 0.70)
            self.assertEqual(rows[0]["last_gate_reason"], "edge_ok")
            self.assertAlmostEqual(rows[0]["last_external_bull_score"], 0.41)
            self.assertTrue(rows[0]["last_exploration_mode"])
            self.assertEqual(rows[0]["last_exploration_reason"], "PAPER_EXPLORATION_MODEL_DOWN")
            self.assertAlmostEqual(rows[0]["last_clob_best_ask"], 0.55)

    def test_live_valid_metrics_do_not_count_gamma_fallback_token_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            strat = root / "01_firehose"
            strat.mkdir()
            journal = strat / "journal.jsonl"
            log = strat / "paper.log"
            journal.write_text(
                json.dumps(
                    {
                        "won": True,
                        "pnl": 2.5,
                        "price_source": "gamma_outcomePrices_fallback",
                        "clob_token_id": "token-present-but-not-executable-fill",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "won": False,
                        "pnl": -1.0,
                        "price_source": "clob_book_ask_depth",
                        "clob_token_id": "token-executable",
                    }
                )
                + "\n"
            )
            log.write_text("[x] elapsed=0.1h trades=2(W:1 L:1)\n")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "strategies": [
                            {
                                "name": "firehose",
                                "pid": 999999,
                                "journal_file": str(journal),
                                "log_file": str(log),
                                "capital": 20.0,
                            }
                        ]
                    }
                )
            )
            with redirect_stdout(StringIO()):
                rows = lab.report(root, json_out=False)
            self.assertEqual(rows[0]["trades"], 2)
            self.assertAlmostEqual(rows[0]["pnl"], 1.5)
            self.assertEqual(rows[0]["live_valid_trades"], 1)
            self.assertEqual(rows[0]["live_valid_wins"], 0)
            self.assertEqual(rows[0]["live_valid_losses"], 1)
            self.assertAlmostEqual(rows[0]["live_valid_pnl"], -1.0)
            self.assertAlmostEqual(rows[0]["clob_ratio"], 0.5)

    def test_btc_paper_fast_accepts_isolated_env_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env.update(
                {
                    "HARVEY_HOME": td,
                    "BTC_STRATEGY_NAME": "unit_env",
                    "BTC_LOG_DIR": str(Path(td) / "logs"),
                    "BTC_JOURNAL_FILE": str(Path(td) / "state" / "journal.jsonl"),
                    "BTC_BEST_PARAMS_FILE": str(Path(td) / "state" / "params.json"),
                    "BTC_PAPER_LOG_FILE": str(Path(td) / "logs" / "paper.log"),
                    "BTC_PROB_MODEL_PATH": str(Path(td) / "model.pkl"),
                    "BTC_PARAM_OVERRIDES_JSON": json.dumps(
                        {"delta_thresh": 8.2, "conf_thresh": 0.51, "ens_thresh": 0.33, "spend_ratio": 0.11}
                    ),
                }
            )
            code = (
                "import btc_paper_fast as m; "
                "p=m.apply_param_overrides(m.SniperParams()); "
                "print(p.name, p.delta_thresh, p.conf_thresh, p.ens_thresh, p.spend_ratio); "
                "print(m.JOURNAL_FILE); print(m.PAPER_LOG_FILE); print(m.PROB_MODEL_PATH)"
            )
            out = subprocess.check_output([str(lab.PYTHON), "-c", code], cwd=str(lab.SRC_DIR), env=env, text=True)
            self.assertIn("8.2", out)
            self.assertIn("0.51", out)
            self.assertIn("0.33", out)
            self.assertIn("0.11", out)
            self.assertIn(str(Path(td) / "state" / "journal.jsonl"), out)
            self.assertIn(str(Path(td) / "logs" / "paper.log"), out)
            self.assertIn(str(Path(td) / "model.pkl"), out)


if __name__ == "__main__":
    unittest.main()
