#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import btc_telegram_reporter as reporter


class TelegramReporterConfigTests(unittest.TestCase):
    def test_env_from_command_extracts_live_btc_assignments(self):
        cmd = (
            "/usr/local/bin/python btc_paper_fast.py 21600 --watchdog-main "
            'BTC_STRATEGY_NAME=main_confidence_flip_clob_4pct '
            'BTC_PARAM_OVERRIDES_JSON={"delta_thresh":9.0,"conf_thresh":0.45,"ens_thresh":0.85} '
            "BTC_PROB_EDGE_THRESHOLD=0.04 BTC_REQUIRE_CLOB_QUOTE=1 "
            "POLYMARKET_LIVE_TRADING=0 USER=sebastian"
        )
        env = reporter.env_from_command(cmd)
        self.assertEqual(env["BTC_STRATEGY_NAME"], "main_confidence_flip_clob_4pct")
        self.assertEqual(env["BTC_PROB_EDGE_THRESHOLD"], "0.04")
        self.assertEqual(env["BTC_REQUIRE_CLOB_QUOTE"], "1")
        self.assertEqual(env["POLYMARKET_LIVE_TRADING"], "0")
        self.assertNotIn("USER", env)

    def test_live_trader_config_prefers_main_live_env_over_frozen_params(self):
        rows = [
            (
                10,
                "/usr/local/bin/python btc_paper_fast.py 21600 --parallel-lab "
                "BTC_STRATEGY_NAME=lab BTC_PARAM_OVERRIDES_JSON={\"delta_thresh\":1}",
            ),
            (
                20,
                "/usr/local/bin/python btc_paper_fast.py 21600 --watchdog-main "
                'BTC_STRATEGY_NAME=main_confidence_flip_clob_4pct '
                'BTC_PARAM_OVERRIDES_JSON={"delta_thresh":9.0,"conf_thresh":0.45,"ens_thresh":0.85} '
                "BTC_PROB_EDGE_THRESHOLD=0.04 BTC_REQUIRE_CLOB_QUOTE=1 "
                "BTC_RISK_HALT_DRAWDOWN_PCT=0.20 BTC_MAX_OPEN_BTC_TRADES=2 "
                "BTC_MAX_OPEN_BTC_TRADES_PER_TF=1 BTC_MAX_TOTAL_BTC_EXPOSURE_PCT=0.32 "
                "BTC_ALLOW_CROSS_TF_CORRELATED_OPEN=1 "
                "BTC_PROB_SHRINK=0.65 BTC_FORCE_DELTA_THRESHOLD=1.0",
            ),
        ]
        cfg = reporter.live_trader_config(
            rows,
            {"delta_thresh": 99.0, "conf_thresh": 0.99, "ens_thresh": 0.99, "strategy_version": "frozen"},
        )
        self.assertEqual(cfg["pid"], 20)
        self.assertEqual(cfg["source"], "live-env")
        self.assertEqual(cfg["strategy"], "main_confidence_flip_clob_4pct")
        self.assertEqual(cfg["delta_thresh"], 9.0)
        self.assertEqual(cfg["conf_thresh"], 0.45)
        self.assertEqual(cfg["ens_thresh"], 0.85)
        self.assertEqual(cfg["edge_threshold"], "0.04")
        self.assertEqual(cfg["clob_required"], "1")
        self.assertEqual(cfg["max_open"], "2")
        self.assertEqual(cfg["max_open_tf"], "1")
        self.assertEqual(cfg["max_total_exposure"], "0.32")
        self.assertEqual(cfg["cross_tf_corr"], "1")

    def test_live_agent_report_lines_use_live_journal_rows(self):
        old_journal = reporter.JOURNAL_FILE
        with tempfile.TemporaryDirectory() as td:
            try:
                reporter.JOURNAL_FILE = Path(td) / "journal.jsonl"
                reporter.JOURNAL_FILE.write_text(
                    json.dumps({
                        "mode": "live",
                        "window_tf": 5,
                        "placed_at": "2026-05-07T12:00:00",
                        "won": True,
                        "pnl": 1.25,
                        "btc_delta": 10.0,
                    })
                    + "\n"
                    + json.dumps({
                        "mode": "paper",
                        "window_tf": 5,
                        "placed_at": "2026-05-07T12:01:00",
                        "won": False,
                        "pnl": -1.0,
                        "btc_delta": 10.0,
                    })
                    + "\n"
                )
                lines = reporter.live_agent_report_lines(
                    [(99, "/usr/local/bin/python btc_sniper_live.py --live --timeframes 5 --duration 21600")],
                    None,
                )
                self.assertEqual(lines[0], "Live agents:")
                self.assertIn("pid=99", lines[1])
                self.assertIn("1 fills", lines[1])
                self.assertIn("PnL=$+1.25", lines[1])
            finally:
                reporter.JOURNAL_FILE = old_journal


if __name__ == "__main__":
    unittest.main()
