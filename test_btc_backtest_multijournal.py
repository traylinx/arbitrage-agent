#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BacktestMultiJournalTests(unittest.TestCase):
    def test_loads_extra_lab_journals_and_dedupes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            main = root / "main.jsonl"
            extra = root / "lab" / "journal.jsonl"
            extra.parent.mkdir()
            row1 = {"mode": "paper", "strategy": "main", "placed_at": "2026-01-01T00:00:00", "window_start": 1, "window_tf": 5, "direction": "Up", "btc_delta": 12, "conf": 0.6, "won": True, "pnl": 1.0, "price_source": "clob_book_ask_depth"}
            row2 = {"mode": "paper", "strategy": "lab", "placed_at": "2026-01-01T00:05:00", "window_start": 2, "window_tf": 5, "direction": "Down", "btc_delta": -13, "conf": 0.7, "won": False, "pnl": -2.0, "price_source": "clob_book_ask_depth"}
            main.write_text(json.dumps(row1) + "\n")
            extra.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")
            env = os.environ.copy()
            env.update({
                "HARVEY_HOME": td,
                "BTC_JOURNAL_FILE": str(main),
                "BTC_EXTRA_JOURNAL_FILES": str(extra),
                "BTC_REPLAY_REQUIRE_CLOB": "1",
            })
            code = "import btc_backtest_autoresearch as b; print(len(b.journal_paths())); print(len(b.load_btc_trades()))"
            out = subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent, env=env, text=True)
            self.assertTrue(out.strip().endswith("2"), out)


if __name__ == "__main__":
    unittest.main()
