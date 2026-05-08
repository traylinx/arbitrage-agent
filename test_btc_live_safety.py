#!/usr/bin/env python3
import time
import unittest
from unittest import mock

import btc_sniper_live as m


class LiveSafetyTests(unittest.TestCase):
    def test_score_trades_ignores_unfilled_live_orders(self):
        sniper = m.LiveSniper(m.SniperParams(), live=False)
        rows = [
            {"mode": "live", "filled": False, "won": False, "pnl": 0, "exit_reason": "unfilled"},
            {"mode": "live", "filled": True, "won": True, "pnl": 2.0},
            {"mode": "live", "filled": True, "won": False, "pnl": -2.5},
        ]
        scored = sniper._scored_trades(rows)
        self.assertEqual(len(scored), 2)
        self.assertEqual(sum(1 for r in scored if r["won"]), 1)

    def test_resolving_trade_clears_pending_fill_to_prevent_duplicate_loss(self):
        sniper = m.LiveSniper(m.SniperParams(), live=False)
        t = m.Trade(
            window_start=1000,
            direction="Down",
            spend=2.5,
            poly_price=0.5,
            btc_delta=-20,
            btc_price_enter=100,
            conf=0.9,
            reasons=[],
            placed_at=1010,
            order_id="oid-1",
            filled=True,
            filled_size=5,
            entry_elapsed_sec=10,
            seconds_left_at_entry=290,
        )
        sniper.trades.append(t)
        sniper._pending_fills["oid-1"] = {"window_start": 1000, "window_tf": 5}

        with mock.patch.object(sniper, "_analyse_trade"):
            sniper._resolve_trade(t, "Up")

        self.assertTrue(t.resolved)
        self.assertEqual(sniper.losses, 1)
        self.assertNotIn("oid-1", sniper._pending_fills)
        sniper._resolve_from_open_order("oid-1", {"window_start": 1000, "window_tf": 5})
        self.assertEqual(sniper.losses, 1)
        self.assertEqual(len(sniper.trades), 1)

    def test_place_trade_uses_non_blocking_balance_refresh_and_rechecks_time_left(self):
        sniper = m.LiveSniper(m.SniperParams(conf_thresh=0.8), live=False)
        sniper.live = True
        sniper._client = mock.Mock()
        sniper._client.get_open_orders.return_value = []
        sniper._client.place_order.return_value = "oid"
        sniper._balance = 5.0
        sniper._balance_cache = 5.0
        sniper._balance_cache_time = 0
        sniper.btc_price = 100.0
        win = m.WindowState(5)
        now = time.time()
        win.window_start = int(now - 10)
        win.market_id = "m"
        win._up_token_id = "up"
        win._down_token_id = "down"
        win._outcome_prices = [0.50, 0.50]
        sig = {"direction": "Up", "delta": 20.0, "conf": 0.9, "tier": "NORMAL", "reasons": []}

        with mock.patch.object(sniper, "_refresh_balance") as fast_refresh, mock.patch.object(
            sniper, "_refresh_balance_with_retry", side_effect=AssertionError("slow refresh forbidden")
        ):
            trade = sniper._place_trade(sig, win, 5)

        self.assertIsNotNone(trade)
        fast_refresh.assert_called()
        sniper._client.place_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
