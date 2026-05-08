#!/usr/bin/env python3
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import btc_paper_fast as m
from btc_prob_gate import BetDecision


class PaperExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._old_paper_log_file = m.PAPER_LOG_FILE
        m.PAPER_LOG_FILE = Path(cls._tmp.name) / "btc_sniper_paper_fast.test.log"

    @classmethod
    def tearDownClass(cls):
        m.PAPER_LOG_FILE = cls._old_paper_log_file
        cls._tmp.cleanup()

    def test_btc_price_falls_back_when_binance_rate_limited(self):
        class Resp:
            def __init__(self, status, payload):
                self.status_code = status
                self._payload = payload
            def json(self):
                return self._payload

        calls = [
            Resp(418, {"code": -1003}),
            Resp(200, {"price": "82534.5"}),
        ]
        with patch.object(m.requests, "get", side_effect=calls):
            self.assertEqual(m.get_btc_price(), 82534.5)

    def test_klines_fall_back_to_coinbase_when_binance_rate_limited(self):
        class Resp:
            def __init__(self, status, payload):
                self.status_code = status
                self._payload = payload
            def json(self):
                return self._payload

        calls = [
            Resp(418, {"code": -1003}),
            Resp(200, [
                [2000, 99.0, 103.0, 100.0, 102.0, 2.0],
                [1000, 98.0, 102.0, 99.0, 101.0, 1.0],
            ]),
        ]
        with patch.object(m.requests, "get", side_effect=calls):
            closes, highs, lows, vols = m.get_klines("5m", 2)
        self.assertEqual(closes, [101.0, 102.0])
        self.assertEqual(highs, [102.0, 103.0])
        self.assertEqual(lows, [98.0, 99.0])
        self.assertEqual(vols, [1.0, 2.0])

    def test_orderbook_depth_falls_back_when_binance_rate_limited(self):
        class Resp:
            def __init__(self, status, payload):
                self.status_code = status
                self._payload = payload
            def json(self):
                return self._payload

        calls = [
            Resp(418, {"code": -1003}),
            Resp(200, {"bids": [["100", "2.0", 1]], "asks": [["101", "1.0", 1]]}),
        ]
        with patch.object(m.requests, "get", side_effect=calls):
            ob = m.get_orderbook_depth(1)
        self.assertEqual(ob["bids"], [(100.0, 2.0)])
        self.assertEqual(ob["asks"], [(101.0, 1.0)])
        self.assertGreater(ob["imbalance"], 0.0)

    def test_vol_ratio_handles_exact_n_closes_without_index_error(self):
        trader = m.PaperTrader(m.SniperParams())
        closes = [100.0 + i for i in range(30)]
        self.assertGreater(trader._vol_ratio(closes, 30), 0.0)

    def test_numeric_series_drops_bad_values(self):
        trader = m.PaperTrader(m.SniperParams())
        self.assertEqual(trader._numeric_series([1, "2", None, "bad", float("nan")]), [1.0, 2.0])

    def test_parse_timeframes_accepts_only_btc_market_windows(self):
        self.assertEqual(m.parse_timeframes("5"), [5])
        self.assertEqual(m.parse_timeframes("5m,15m"), [5, 15])
        self.assertEqual(m.parse_timeframes([15]), [15])
        with self.assertRaises(ValueError):
            m.parse_timeframes("1m")

    def test_paper_trader_can_be_isolated_to_single_timeframe_agent(self):
        trader = m.PaperTrader(m.SniperParams(), timeframes=[5], agent_id="btc-5m")
        self.assertEqual(trader.agent_id, "btc-5m")
        self.assertEqual(trader.timeframes, [5])
        self.assertEqual(set(trader.windows), {5})

    def test_fast_ga_can_opt_into_resolved_live_rows(self):
        with tempfile.TemporaryDirectory() as td:
            journal = Path(td) / "journal.jsonl"
            rows = [
                {
                    "mode": "paper",
                    "window_tf": 5,
                    "won": True,
                    "pnl": 1.0,
                    "btc_delta": 12.0,
                    "conf": 0.8,
                    "poly_price": 0.50,
                    "direction": "Up",
                },
                {
                    "mode": "live",
                    "window_tf": 5,
                    "won": False,
                    "pnl": -0.5,
                    "btc_delta": 14.0,
                    "conf": 0.9,
                    "poly_price": 0.51,
                    "direction": "Up",
                },
                {
                    "mode": "live",
                    "window_tf": 15,
                    "won": True,
                    "pnl": 0.4,
                    "btc_delta": 10.0,
                    "conf": 0.7,
                    "poly_price": 0.49,
                    "direction": "Down",
                },
            ]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))

            paper_only = m.FastGA(timeframes=[5], include_live=False)._load_trades(journal)
            with_live = m.FastGA(timeframes=[5], include_live=True)._load_trades(journal)

            self.assertEqual([t["mode"] for t in paper_only], ["paper"])
            self.assertEqual([t["mode"] for t in with_live], ["paper", "live"])

    def test_exploration_decision_allows_model_down_paper_label_collection(self):
        old = {
            "PAPER_EXPLORATION_ENABLED": m.PAPER_EXPLORATION_ENABLED,
            "PAPER_EXPLORATION_ALLOW": m.PAPER_EXPLORATION_ALLOW,
            "PAPER_EXPLORATION_FORCE": m.PAPER_EXPLORATION_FORCE,
            "PAPER_EXPLORATION_MIN_EDGE": m.PAPER_EXPLORATION_MIN_EDGE,
            "PAPER_EXPLORATION_MIN_CONF": m.PAPER_EXPLORATION_MIN_CONF,
            "PAPER_EXPLORATION_MIN_POLY": m.PAPER_EXPLORATION_MIN_POLY,
            "PAPER_EXPLORATION_MAX_POLY": m.PAPER_EXPLORATION_MAX_POLY,
            "PAPER_EXPLORATION_DELTA_FLOOR": m.PAPER_EXPLORATION_DELTA_FLOOR,
        }
        try:
            m.PAPER_EXPLORATION_ENABLED = True
            m.PAPER_EXPLORATION_ALLOW = True
            m.PAPER_EXPLORATION_FORCE = True
            m.PAPER_EXPLORATION_MIN_EDGE = 0.0
            m.PAPER_EXPLORATION_MIN_CONF = 0.1
            m.PAPER_EXPLORATION_MIN_POLY = 0.25
            m.PAPER_EXPLORATION_MAX_POLY = 0.74
            m.PAPER_EXPLORATION_DELTA_FLOOR = 1.0
            trader = m.PaperTrader(m.SniperParams(delta_thresh=5.0))
            self.assertEqual(trader._active_delta_threshold(), 1.0)
            blocked = BetDecision(
                should_bet=False,
                direction="Up",
                model_prob=0.5,
                market_prob=0.52,
                edge=0.0,
                prob_up=0.5,
                prob_down=0.5,
                gate_reason="model unavailable",
                summary="MODEL_DOWN: missing model",
            )
            dec = trader._exploration_decision(
                {"direction": "Up", "conf": 0.60},
                {
                    "poly_price_enter": 0.52,
                    "btc_delta": 12.0,
                    "external_bull_score": 0.45,
                    "cg_taker_30m_imbalance": 0.30,
                    "cg_cvd_30m_imbalance": 0.20,
                    "bg_taker_5m_imbalance": 0.25,
                    "bg_depth_imbalance": 0.10,
                },
                blocked,
            )
            self.assertIsNotNone(dec)
            self.assertTrue(dec.should_bet)
            self.assertEqual(dec.direction, "Up")
            self.assertIn("PAPER_EXPLORATION_MODEL_DOWN", dec.gate_reason)
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_delta_hint_signal_can_feed_model_gate_without_exploration(self):
        old = {
            "DELTA_HINT_SIGNAL_ENABLED": m.DELTA_HINT_SIGNAL_ENABLED,
            "FORCE_DELTA_THRESHOLD": m.FORCE_DELTA_THRESHOLD,
            "PAPER_EXPLORATION_ENABLED": m.PAPER_EXPLORATION_ENABLED,
            "PAPER_EXPLORATION_ALLOW": m.PAPER_EXPLORATION_ALLOW,
        }
        try:
            m.DELTA_HINT_SIGNAL_ENABLED = True
            m.FORCE_DELTA_THRESHOLD = 1.0
            m.PAPER_EXPLORATION_ENABLED = False
            m.PAPER_EXPLORATION_ALLOW = False
            trader = m.PaperTrader(m.SniperParams(delta_thresh=8.0))
            self.assertEqual(trader._active_delta_threshold(), 1.0)
            sig = trader._delta_hint_signal(-2.5)
            self.assertIsNotNone(sig)
            self.assertEqual(sig["direction"], "Down")
            self.assertFalse(trader._exploration_allowed())
            self.assertIn("model_gate_delta_hint", sig["reasons"])
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_execution_edge_rechecks_actual_clob_price(self):
        old = {
            "PROB_EDGE_THRESHOLD": m.PROB_EDGE_THRESHOLD,
            "PROB_POLY_PRICE_CEILING": m.PROB_POLY_PRICE_CEILING,
            "PROB_POLY_CEILING_EDGE_BUFFER": m.PROB_POLY_CEILING_EDGE_BUFFER,
            "PROB_HARD_POLY_CAP": m.PROB_HARD_POLY_CAP,
        }
        try:
            m.PROB_EDGE_THRESHOLD = 0.02
            m.PROB_POLY_PRICE_CEILING = 0.70
            m.PROB_POLY_CEILING_EDGE_BUFFER = 0.12
            m.PROB_HARD_POLY_CAP = 0.80
            trader = m.PaperTrader(m.SniperParams())
            sig = {
                "direction": "Up",
                "_prob_decision": {
                    "model_prob": 0.768,
                    "market_prob": 0.505,
                    "edge": 0.263,
                    "gate_reason": "EDGE_OK",
                },
            }
            self.assertFalse(trader._execution_edge_allows(sig, 5, "Up", 0.710))
            self.assertTrue(trader._execution_edge_allows(sig, 5, "Up", 0.600))
            self.assertAlmostEqual(sig["_prob_decision"]["market_prob"], 0.600)
            self.assertTrue(sig["_prob_decision"]["exec_edge_checked"])
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_place_trade_uses_cached_window_market_not_refetched_market(self):
        trader = m.PaperTrader(m.SniperParams(delta_thresh=8.0, spend_ratio=0.15, max_bet_pct=0.20))
        trader._risk_allows_new_trade = lambda direction, tf: True
        trader._execution_edge_allows = lambda sig, tf, direction, executable_price: True
        cached_market = {
            "id": "cached-market",
            "slug": "btc-up-or-down-cached",
            "conditionId": "cond-cached",
            "outcomes": '["Up","Down"]',
            "outcomePrices": '["0.69","0.31"]',
            "clobTokenIds": '["up-cached","down-cached"]',
        }
        refetched_market = {
            "id": "wrong-market",
            "outcomes": '["Up","Down"]',
            "outcomePrices": '["0.51","0.49"]',
            "clobTokenIds": '["up-wrong","down-wrong"]',
        }
        quote_calls = []

        def fake_quote(token_id, target_spend, min_size=5.0, max_spend=None):
            quote_calls.append(token_id)
            return {
                "price": 0.31,
                "size": 10.0,
                "cost": 3.10,
                "best_ask": 0.31,
                "levels_used": 1,
                "slippage_bps": 0.0,
            }

        win = {"start": 12345, "price": 81000.0, "traded": False, "market_id": "cached-market", "market": cached_market}
        sig = {"direction": "Down", "conf": 0.8, "_prob_features": {}, "_prob_decision": {"model_prob": 0.65}}
        with patch.object(m, "fetch_btc_markets", return_value=refetched_market) as fetch_mock, patch.object(
            m, "fetch_clob_buy_quote", side_effect=fake_quote
        ):
            trade = trader._place_trade(sig, win, 15, 80980.0)

        self.assertIsNotNone(trade)
        fetch_mock.assert_not_called()
        self.assertEqual(quote_calls, ["down-cached"])
        self.assertEqual(trade["market_id"], "cached-market")
        self.assertEqual(trade["market_slug"], "btc-up-or-down-cached")
        self.assertEqual(trade["condition_id"], "cond-cached")
        self.assertEqual(trade["clob_token_id"], "down-cached")
        self.assertAlmostEqual(trade["gamma_price"], 0.31)
        self.assertAlmostEqual(trade["prob_features"]["clob_exec_price"], 0.31)

    def test_exploration_decision_uses_down_market_price_for_down_trades(self):
        old = {
            "PAPER_EXPLORATION_ENABLED": m.PAPER_EXPLORATION_ENABLED,
            "PAPER_EXPLORATION_ALLOW": m.PAPER_EXPLORATION_ALLOW,
            "PAPER_EXPLORATION_FORCE": m.PAPER_EXPLORATION_FORCE,
            "PAPER_EXPLORATION_MIN_EDGE": m.PAPER_EXPLORATION_MIN_EDGE,
            "PAPER_EXPLORATION_MIN_CONF": m.PAPER_EXPLORATION_MIN_CONF,
            "PAPER_EXPLORATION_MIN_POLY": m.PAPER_EXPLORATION_MIN_POLY,
            "PAPER_EXPLORATION_MAX_POLY": m.PAPER_EXPLORATION_MAX_POLY,
        }
        try:
            m.PAPER_EXPLORATION_ENABLED = True
            m.PAPER_EXPLORATION_ALLOW = True
            m.PAPER_EXPLORATION_FORCE = True
            m.PAPER_EXPLORATION_MIN_EDGE = 0.0
            m.PAPER_EXPLORATION_MIN_CONF = 0.1
            m.PAPER_EXPLORATION_MIN_POLY = 0.25
            m.PAPER_EXPLORATION_MAX_POLY = 0.60
            trader = m.PaperTrader(m.SniperParams(delta_thresh=5.0))
            blocked = BetDecision(
                should_bet=False,
                direction="Down",
                model_prob=0.5,
                market_prob=0.55,
                edge=0.0,
                prob_up=0.5,
                prob_down=0.5,
                gate_reason="model unavailable",
                summary="MODEL_DOWN: missing model",
            )
            dec = trader._exploration_decision(
                {"direction": "Down", "conf": 0.8},
                {
                    "poly_price_enter": 0.80,  # Up price; should not cap Down
                    "poly_price_down": 0.40,
                    "btc_delta": -20.0,
                    "external_bull_score": -0.40,
                    "bg_taker_5m_imbalance": -0.25,
                },
                blocked,
            )
            self.assertIsNotNone(dec)
            self.assertEqual(dec.direction, "Down")
            self.assertAlmostEqual(dec.market_prob, 0.40)
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_exploration_decision_rejects_lottery_tail_prices(self):
        old = {
            "PAPER_EXPLORATION_ENABLED": m.PAPER_EXPLORATION_ENABLED,
            "PAPER_EXPLORATION_ALLOW": m.PAPER_EXPLORATION_ALLOW,
            "PAPER_EXPLORATION_FORCE": m.PAPER_EXPLORATION_FORCE,
            "PAPER_EXPLORATION_MIN_EDGE": m.PAPER_EXPLORATION_MIN_EDGE,
            "PAPER_EXPLORATION_MIN_CONF": m.PAPER_EXPLORATION_MIN_CONF,
            "PAPER_EXPLORATION_MIN_POLY": m.PAPER_EXPLORATION_MIN_POLY,
            "PAPER_EXPLORATION_MAX_POLY": m.PAPER_EXPLORATION_MAX_POLY,
        }
        try:
            m.PAPER_EXPLORATION_ENABLED = True
            m.PAPER_EXPLORATION_ALLOW = True
            m.PAPER_EXPLORATION_FORCE = True
            m.PAPER_EXPLORATION_MIN_EDGE = 0.0
            m.PAPER_EXPLORATION_MIN_CONF = 0.1
            m.PAPER_EXPLORATION_MIN_POLY = 0.25
            m.PAPER_EXPLORATION_MAX_POLY = 0.70
            trader = m.PaperTrader(m.SniperParams(delta_thresh=5.0))
            blocked = BetDecision(
                should_bet=False,
                direction="Up",
                model_prob=0.5,
                market_prob=0.02,
                edge=0.0,
                prob_up=0.5,
                prob_down=0.5,
                gate_reason="model unavailable",
                summary="MODEL_DOWN: missing model",
            )
            dec = trader._exploration_decision(
                {"direction": "Up", "conf": 0.8},
                {
                    "poly_price_enter": 0.02,
                    "btc_delta": 20.0,
                    "external_bull_score": 0.40,
                    "bg_taker_5m_imbalance": 0.25,
                },
                blocked,
            )
            self.assertIsNone(dec)
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_runtime_guards_skip_old_low_prob_clob_outlier_loss(self):
        old = {
            "MAX_CLOB_GAMMA_GAP": m.MAX_CLOB_GAMMA_GAP,
            "REQUIRE_EXEC_PRICE_WITHIN_GAMMA": m.REQUIRE_EXEC_PRICE_WITHIN_GAMMA,
            "CLOB_GAMMA_GAP_OVERRIDE_PROB": m.CLOB_GAMMA_GAP_OVERRIDE_PROB,
            "PROB_MIN_DIRECTION_PROB": m.PROB_MIN_DIRECTION_PROB,
            "PROB_MIN_POLY_PRICE": m.PROB_MIN_POLY_PRICE,
            "REQUIRE_PRE_EXEC_EDGE": m.REQUIRE_PRE_EXEC_EDGE,
            "PRE_EXEC_EDGE_THRESHOLD": m.PRE_EXEC_EDGE_THRESHOLD,
            "REQUIRE_EXTERNAL_FLOW_AGREEMENT": m.REQUIRE_EXTERNAL_FLOW_AGREEMENT,
            "EXTERNAL_FLOW_SOFT_THRESHOLD": m.EXTERNAL_FLOW_SOFT_THRESHOLD,
        }
        try:
            m.MAX_CLOB_GAMMA_GAP = 0.12
            m.REQUIRE_EXEC_PRICE_WITHIN_GAMMA = True
            m.CLOB_GAMMA_GAP_OVERRIDE_PROB = 0.75
            m.PROB_MIN_DIRECTION_PROB = 0.55
            m.PROB_MIN_POLY_PRICE = 0.15
            m.REQUIRE_PRE_EXEC_EDGE = True
            m.PRE_EXEC_EDGE_THRESHOLD = 0.03
            m.REQUIRE_EXTERNAL_FLOW_AGREEMENT = True
            m.EXTERNAL_FLOW_SOFT_THRESHOLD = 0.05

            trader = m.PaperTrader(m.SniperParams(spend_ratio=0.15, max_bet_pct=0.20))
            trader._risk_allows_new_trade = lambda direction, tf: True
            cached_market = {
                "id": "loss-market",
                "slug": "btc-up-or-down-loss",
                "conditionId": "cond-loss",
                "outcomes": '["Up","Down"]',
                "outcomePrices": '["0.445","0.555"]',
                "clobTokenIds": '["up-loss","down-loss"]',
            }

            def fake_quote(token_id, target_spend, min_size=5.0, max_spend=None):
                return {
                    "price": 0.08,
                    "size": 31.25,
                    "cost": 2.50,
                    "best_ask": 0.08,
                    "levels_used": 1,
                    "slippage_bps": 0.0,
                }

            win = {"start": 12345, "price": 81000.0, "traded": False, "market_id": "loss-market", "market": cached_market}
            sig = {
                "direction": "Up",
                "conf": 0.7,
                "_prob_features": {
                    "external_bull_score": -0.20,
                    "bg_taker_5m_imbalance": -0.12,
                    "bn_taker_15m_imbalance": -0.08,
                    "cg_taker_30m_imbalance": -0.10,
                },
                "_prob_decision": {
                    "model_prob": 0.466,
                    "market_prob": 0.445,
                    "edge": 0.0214,
                    "prob_up": 0.466,
                    "prob_down": 0.534,
                    "gate_reason": "EDGE_OK",
                },
            }
            with patch.object(m, "fetch_clob_buy_quote", side_effect=fake_quote):
                trade = trader._place_trade(sig, win, 5, 81020.0)

            self.assertIsNone(trade)
            self.assertEqual(trader.trades, [])
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_runtime_guards_allow_clean_high_prob_model_gated_winner(self):
        old = {
            "MAX_CLOB_GAMMA_GAP": m.MAX_CLOB_GAMMA_GAP,
            "REQUIRE_EXEC_PRICE_WITHIN_GAMMA": m.REQUIRE_EXEC_PRICE_WITHIN_GAMMA,
            "CLOB_GAMMA_GAP_OVERRIDE_PROB": m.CLOB_GAMMA_GAP_OVERRIDE_PROB,
            "PROB_MIN_DIRECTION_PROB": m.PROB_MIN_DIRECTION_PROB,
            "PROB_MIN_POLY_PRICE": m.PROB_MIN_POLY_PRICE,
            "REQUIRE_PRE_EXEC_EDGE": m.REQUIRE_PRE_EXEC_EDGE,
            "PRE_EXEC_EDGE_THRESHOLD": m.PRE_EXEC_EDGE_THRESHOLD,
            "REQUIRE_EXTERNAL_FLOW_AGREEMENT": m.REQUIRE_EXTERNAL_FLOW_AGREEMENT,
            "EXTERNAL_FLOW_SOFT_THRESHOLD": m.EXTERNAL_FLOW_SOFT_THRESHOLD,
        }
        try:
            m.MAX_CLOB_GAMMA_GAP = 0.12
            m.REQUIRE_EXEC_PRICE_WITHIN_GAMMA = True
            m.CLOB_GAMMA_GAP_OVERRIDE_PROB = 0.75
            m.PROB_MIN_DIRECTION_PROB = 0.55
            m.PROB_MIN_POLY_PRICE = 0.15
            m.REQUIRE_PRE_EXEC_EDGE = True
            m.PRE_EXEC_EDGE_THRESHOLD = 0.03
            m.REQUIRE_EXTERNAL_FLOW_AGREEMENT = True
            m.EXTERNAL_FLOW_SOFT_THRESHOLD = 0.05

            trader = m.PaperTrader(m.SniperParams(spend_ratio=0.15, max_bet_pct=0.20))
            trader._risk_allows_new_trade = lambda direction, tf: True
            cached_market = {
                "id": "winner-market",
                "slug": "btc-up-or-down-winner",
                "conditionId": "cond-winner",
                "outcomes": '["Up","Down"]',
                "outcomePrices": '["0.445","0.555"]',
                "clobTokenIds": '["up-winner","down-winner"]',
            }

            def fake_quote(token_id, target_spend, min_size=5.0, max_spend=None):
                return {
                    "price": 0.42,
                    "size": 5.9523809524,
                    "cost": 2.50,
                    "best_ask": 0.42,
                    "levels_used": 1,
                    "slippage_bps": 0.0,
                }

            win = {"start": 12345, "price": 81000.0, "traded": False, "market_id": "winner-market", "market": cached_market}
            sig = {
                "direction": "Up",
                "conf": 0.8,
                "_prob_features": {
                    "external_bull_score": 0.20,
                    "bg_taker_5m_imbalance": 0.12,
                    "bn_taker_15m_imbalance": 0.08,
                    "cg_taker_30m_imbalance": 0.10,
                },
                "_prob_decision": {
                    "model_prob": 0.78,
                    "market_prob": 0.445,
                    "edge": 0.335,
                    "prob_up": 0.78,
                    "prob_down": 0.22,
                    "gate_reason": "EDGE_OK",
                },
            }
            with patch.object(m, "fetch_clob_buy_quote", side_effect=fake_quote):
                trade = trader._place_trade(sig, win, 5, 81020.0)

            self.assertIsNotNone(trade)
            self.assertEqual(trade["direction"], "Up")
            self.assertEqual(trade["market_id"], "winner-market")
            self.assertAlmostEqual(trade["poly_price"], 0.42)
            self.assertTrue(trade["prob_decision"]["exec_edge_checked"])
        finally:
            for key, value in old.items():
                setattr(m, key, value)


    def test_risk_cap_allows_one_open_per_timeframe_when_total_cap_two(self):
        old = {
            "MAX_OPEN_BTC_TRADES": m.MAX_OPEN_BTC_TRADES,
            "MAX_OPEN_BTC_TRADES_PER_TF": m.MAX_OPEN_BTC_TRADES_PER_TF,
        }
        try:
            m.MAX_OPEN_BTC_TRADES = 2
            m.MAX_OPEN_BTC_TRADES_PER_TF = 1
            trader = m.PaperTrader(m.SniperParams())
            trader.trades.append({"direction": "Up", "window_tf": 5, "spend": 2.8, "resolved": False})

            self.assertTrue(trader._risk_allows_new_trade("Down", 15))
            self.assertFalse(trader._risk_allows_new_trade("Up", 15))
            self.assertFalse(trader._risk_allows_new_trade("Down", 5))
            self.assertGreaterEqual(trader.telemetry["correlated_open"], 1)
            self.assertGreaterEqual(trader.telemetry["open_cap_tf"], 1)
        finally:
            for key, value in old.items():
                setattr(m, key, value)


    def test_paper_can_allow_cross_timeframe_correlated_open_when_enabled(self):
        old = {
            "MAX_OPEN_BTC_TRADES": m.MAX_OPEN_BTC_TRADES,
            "MAX_OPEN_BTC_TRADES_PER_TF": m.MAX_OPEN_BTC_TRADES_PER_TF,
            "ALLOW_CROSS_TF_CORRELATED_OPEN": m.ALLOW_CROSS_TF_CORRELATED_OPEN,
        }
        try:
            m.MAX_OPEN_BTC_TRADES = 2
            m.MAX_OPEN_BTC_TRADES_PER_TF = 1
            m.ALLOW_CROSS_TF_CORRELATED_OPEN = True
            trader = m.PaperTrader(m.SniperParams())
            trader.trades.append({"direction": "Up", "window_tf": 5, "spend": 2.8, "resolved": False})

            self.assertTrue(trader._risk_allows_new_trade("Up", 15))
            self.assertFalse(trader._risk_allows_new_trade("Up", 5))
            self.assertGreaterEqual(trader.telemetry["correlated_cross_tf_allowed"], 1)
            self.assertGreaterEqual(trader.telemetry["open_cap_tf"], 1)
        finally:
            for key, value in old.items():
                setattr(m, key, value)


    def test_total_exposure_pct_separate_from_per_trade_bet_pct(self):
        old = {"MAX_TOTAL_BTC_EXPOSURE_PCT": m.MAX_TOTAL_BTC_EXPOSURE_PCT}
        try:
            cached_market = {
                "id": "exposure-market",
                "slug": "btc-up-or-down-exposure",
                "conditionId": "cond-exposure",
                "outcomes": '["Up","Down"]',
                "outcomePrices": '["0.50","0.50"]',
                "clobTokenIds": '["up-exposure","down-exposure"]',
            }
            win = {"start": 12345, "price": 81000.0, "traded": False, "market_id": "exposure-market", "market": cached_market}
            sig = {"direction": "Down", "conf": 0.8, "_prob_features": {}, "_prob_decision": {"model_prob": 0.80}}

            def fake_quote(token_id, target_spend, min_size=5.0, max_spend=None):
                return {"price": 0.50, "size": 5.0, "cost": 2.50, "best_ask": 0.50, "levels_used": 1, "slippage_bps": 0.0}

            # Legacy/default: params.max_bet_pct also acts as total exposure cap,
            # so a first $2.80 open trade leaves <$2.50 available on a $20 book.
            m.MAX_TOTAL_BTC_EXPOSURE_PCT = None
            trader = m.PaperTrader(m.SniperParams(spend_ratio=0.14, max_bet_pct=0.16))
            trader.starting = 20.0
            trader.peak_equity = 20.0
            trader._balance = 17.20
            trader.trades.append({"direction": "Up", "window_tf": 5, "spend": 2.80, "resolved": False})
            trader._risk_allows_new_trade = lambda direction, tf: True
            trader._runtime_guards_allow = lambda sig, tf, direction, executable_price, gamma_price: True
            trader._execution_edge_allows = lambda sig, tf, direction, executable_price: True
            with patch.object(m, "fetch_clob_buy_quote", side_effect=fake_quote):
                self.assertIsNone(trader._place_trade(sig, win, 15, 81000.0))
            self.assertGreaterEqual(trader.telemetry["exposure_cap"], 1)

            # Paper validation can widen total exposure while preserving per-trade max_bet_pct.
            m.MAX_TOTAL_BTC_EXPOSURE_PCT = 0.32
            trader = m.PaperTrader(m.SniperParams(spend_ratio=0.14, max_bet_pct=0.16))
            trader.starting = 20.0
            trader.peak_equity = 20.0
            trader._balance = 17.20
            trader.trades.append({"direction": "Up", "window_tf": 5, "spend": 2.80, "resolved": False})
            trader._risk_allows_new_trade = lambda direction, tf: True
            trader._runtime_guards_allow = lambda sig, tf, direction, executable_price, gamma_price: True
            trader._execution_edge_allows = lambda sig, tf, direction, executable_price: True
            with patch.object(m, "fetch_clob_buy_quote", side_effect=fake_quote):
                trade = trader._place_trade(sig, win, 15, 81000.0)
            self.assertIsNotNone(trade)
            self.assertAlmostEqual(trade["spend"], 2.50)
        finally:
            for key, value in old.items():
                setattr(m, key, value)

    def test_low_executable_price_is_hard_block_even_with_high_model_prob(self):
        old = {
            "PROB_MIN_POLY_PRICE": m.PROB_MIN_POLY_PRICE,
            "ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE": m.ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE,
            "CLOB_GAMMA_GAP_OVERRIDE_PROB": m.CLOB_GAMMA_GAP_OVERRIDE_PROB,
        }
        try:
            m.PROB_MIN_POLY_PRICE = 0.20
            m.ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE = False
            m.CLOB_GAMMA_GAP_OVERRIDE_PROB = 0.75
            trader = m.PaperTrader(m.SniperParams())
            sig = {
                "_prob_decision": {
                    "model_prob": 0.90,
                    "prob_up": 0.90,
                    "prob_down": 0.10,
                    "market_prob": 0.09,
                    "edge": 0.81,
                }
            }
            self.assertFalse(trader._runtime_guards_allow(sig, 15, "Up", 0.09, 0.09))
        finally:
            for key, value in old.items():
                setattr(m, key, value)



if __name__ == "__main__":
    unittest.main()
