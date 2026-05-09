#!/usr/bin/env python3.11
"""Tests for backfill_btc_markets.decode_outcome.

Run: python3.11 -m pytest test_backfill_btc_markets.py -v
Or:  python3.11 test_backfill_btc_markets.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backfill_btc_markets import decode_outcome, slug_for, date_for_window


FIXTURE_PATH = (
    Path.home()
    / "MAKAKOO/development/sprints/queued/SPRINT-ARBITRAGE-ML-V1"
    / "oracle_sample_btc_updown_5m_1778154600.json"
)


class TestSlugAndDate(unittest.TestCase):
    def test_slug_format(self):
        self.assertEqual(slug_for(5, 1778154600), "btc-updown-5m-1778154600")
        self.assertEqual(slug_for(15, 1778154000), "btc-updown-15m-1778154000")

    def test_date_for_window(self):
        # 1778154600 = 2026-05-07T11:50:00 UTC
        self.assertEqual(date_for_window(1778154600), "2026-05-07")


class TestDecodeOutcome(unittest.TestCase):
    """Verify the label decoder on the persisted PM fixture and synthetic edge cases."""

    def test_real_fixture_down_won(self):
        """Sample market btc-updown-5m-1778154600 resolved Down.
        outcomePrices=["0","1"] outcomes=["Up","Down"] → Down won → binary_label_up_won=0
        """
        if not FIXTURE_PATH.exists():
            self.skipTest(f"fixture not present at {FIXTURE_PATH}")
        markets = json.loads(FIXTURE_PATH.read_text())
        m = markets[0]
        decoded = decode_outcome(m)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["winner"], "Down")
        self.assertEqual(decoded["binary_label_up_won"], 0)
        self.assertEqual(decoded["resolution_source"], "https://data.chain.link/streams/btc-usd")
        self.assertEqual(decoded["slug"], "btc-updown-5m-1778154600")
        # fee schedule decoded
        self.assertAlmostEqual(decoded["fee_rate"], 0.07)
        self.assertAlmostEqual(decoded["fee_rebate_rate"], 0.2)
        self.assertTrue(decoded["fee_taker_only"])

    def test_synthetic_up_won(self):
        m = {
            "umaResolutionStatus": "resolved",
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["1", "0"],
            "slug": "btc-updown-5m-1778100000",
            "conditionId": "0xabc",
            "clobTokenIds": ["111", "222"],
            "feeSchedule": {"rate": 0.07, "rebateRate": 0.2, "takerOnly": True},
            "resolutionSource": "https://data.chain.link/streams/btc-usd",
        }
        d = decode_outcome(m)
        self.assertIsNotNone(d)
        self.assertEqual(d["winner"], "Up")
        self.assertEqual(d["binary_label_up_won"], 1)
        self.assertEqual(d["clob_token_id_up"], "111")
        self.assertEqual(d["clob_token_id_down"], "222")

    def test_unresolved_returns_none(self):
        m = {
            "umaResolutionStatus": "open",
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0.5", "0.5"],
        }
        self.assertIsNone(decode_outcome(m))

    def test_disputed_returns_none(self):
        m = {
            "umaResolutionStatus": "resolved",
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0.5", "0.5"],  # no clear winner
        }
        self.assertIsNone(decode_outcome(m))

    def test_string_encoded_outcomes(self):
        # gamma-api sometimes returns outcomes/prices as JSON-encoded strings
        m = {
            "umaResolutionStatus": "resolved",
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0", "1"]',
            "slug": "x",
            "conditionId": "0xabc",
            "clobTokenIds": '["111", "222"]',
            "feeSchedule": '{"rate": 0.07}',
        }
        d = decode_outcome(m)
        self.assertIsNotNone(d)
        self.assertEqual(d["winner"], "Down")
        self.assertEqual(d["binary_label_up_won"], 0)
        self.assertAlmostEqual(d["fee_rate"], 0.07)

    def test_missing_outcome_arrays(self):
        m = {"umaResolutionStatus": "resolved"}
        self.assertIsNone(decode_outcome(m))


if __name__ == "__main__":
    unittest.main(verbosity=2)
