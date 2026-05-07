#!/usr/bin/env python3
"""
BTC Edge Paper Trader — Probability-model-driven paper trading.

Replaces the 3-param heuristic (delta + conf + hour biases) with a calibrated
probability model. Core idea:

  1. Every minute, build feature vector from Binance spot/futures/order book.
  2. Load calibrated sklearn model (LogisticRegression + isotonic calibration).
  3. Predict P(BTC up in 5m) and P(BTC up in 15m).
  4. Get current Polymarket YES price p for the active window.
  5. Compute breakeven probability: p + fee_rate * p * (1-p).
  6. Only enter when model_prob > breakeven + edge_margin.

This turns the problem from "predict direction" into "find mispriced contracts",
which is structurally easier than predicting a random walk.

Local Python only. No live orders. Logs to intraday_journal_model.jsonl.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from btc_feature_engine import (
    FEATURE_DB,
    build_feature_vector,
    fetch_fut_klines,
    fetch_fut_premium,
    fetch_spot_depth,
    fetch_spot_klines,
    fetch_poly_btc_markets,
    save_features,
    compute_labels,
)
from btc_fee_model import resolved_buy_pnl, taker_fee_usdc

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
MODEL_DIR = STATE_DIR / "models"

JOURNAL_FILE = Path(os.environ.get("BTC_MODEL_JOURNAL", str(STATE_DIR / "intraday_journal_model.jsonl")))
LOG_FILE = Path(os.environ.get("BTC_MODEL_LOG", str(DATA_DIR / "logs" / "btc_edge_paper.log")))
BEST_PARAMS_FILE = STATE_DIR / "sniper_best_params.json"

PAPER_CAPITAL = float(os.environ.get("BTC_PAPER_CAPITAL", "100.0"))
EDGE_MARGIN = float(os.environ.get("BTC_EDGE_MARGIN", "0.03"))  # require 3% edge above breakeven
MIN_SPEND = 2.50
MAX_OPEN_TRADES = int(os.environ.get("BTC_MAX_OPEN_TRADES", "2"))
RISK_HALT_DRAWDOWN_PCT = float(os.environ.get("BTC_RISK_HALT_DRAWDOWN_PCT", "0.20"))
LOOP_SLEEP_SECONDS = float(os.environ.get("BTC_LOOP_SLEEP_SECONDS", "1.0"))
MARKET_CHECK_SECONDS = float(os.environ.get("BTC_MARKET_CHECK_SECONDS", "5.0"))
STATUS_SECONDS = float(os.environ.get("BTC_STATUS_SECONDS", "30.0"))
MIN_TRADES_FOR_ACCURACY_CHECK = int(os.environ.get("BTC_MIN_TRADES_ACCURACY", "20"))
MIN_RECENT_ACCURACY = float(os.environ.get("BTC_MIN_RECENT_ACCURACY", "0.52"))

STATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5_000_000:
            os.replace(LOG_FILE, Path(str(LOG_FILE) + ".1"))
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_model(horizon: str) -> Optional[object]:
    path = MODEL_DIR / f"btc_model_{horizon}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log(f"[model] Failed to load {path}: {e}")
        return None


def fetch_clob_ask(token_id: str, target_spend: float, min_size: float = 5.0) -> Optional[dict]:
    if not token_id:
        return None
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": str(token_id)}, timeout=8)
        if r.status_code != 200:
            return None
        book = r.json()
        asks = []
        for level in book.get("asks") or []:
            try:
                price = float(level.get("price"))
                size = float(level.get("size"))
                if 0 < price < 1 and size > 0:
                    asks.append((price, size))
            except Exception:
                continue
        if not asks:
            return None
        asks.sort(key=lambda x: x[0])
        best_ask = asks[0][0]
        target_size = max(float(min_size), target_spend / best_ask)
        filled = 0.0
        cost = 0.0
        for price, available in asks:
            if filled >= target_size:
                break
            take = min(available, target_size - filled)
            if cost + take * price > target_spend:
                take = max(0.0, (target_spend - cost) / price)
            if take <= 0:
                break
            filled += take
            cost += take * price
        if filled <= 0 or cost <= 0:
            return None
        return {"price": cost / filled, "size": filled, "cost": cost, "best_ask": best_ask}
    except Exception:
        return None


def clob_token_id_for_direction(market: dict, direction: str) -> Optional[str]:
    outcomes = [str(x).lower() for x in market.get("outcomes", ["Up", "Down"])]
    token_ids = [str(x) for x in market.get("clobTokenIds", [])]
    if not token_ids:
        return None
    wanted = str(direction).lower()
    for idx, outcome in enumerate(outcomes):
        if outcome == wanted and idx < len(token_ids):
            return token_ids[idx]
    idx = 0 if wanted == "up" else 1
    return token_ids[idx] if idx < len(token_ids) else None


def fetch_btc_market(tf_minutes: int):
    now_ts = int(time.time())
    window_sec = tf_minutes * 60
    current_window = (now_ts // window_sec) * window_sec
    slug_prefix = f"btc-updown-{tf_minutes}m"
    for offset in (0, 1):
        window = current_window + offset * window_sec
        slug = f"{slug_prefix}-{window}"
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                continue
            m = markets[0]
            if not m.get("acceptingOrders", False):
                continue
            if m.get("closed", True):
                continue
            return m
        except Exception:
            continue
    return None


def fetch_market_resolution(market_id: str) -> Optional[str]:
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=8)
        if r.status_code != 200:
            return None
        m = r.json()
        prices_raw = m.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        labels_raw = m.get("outcomes", '["Up","Down"]')
        labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
        for i, p in enumerate(prices):
            try:
                if float(p) >= 0.99:
                    lbl = labels[i] if i < len(labels) else ""
                    if str(lbl).lower() in ("up", "yes"):
                        return "Up"
                    elif str(lbl).lower() in ("down", "no"):
                        return "Down"
            except Exception:
                pass
        return None
    except Exception:
        return None


def breakeven_prob(price: float, fee_rate: float = 0.072) -> float:
    """Minimum P(win) required for zero EV when buying at `price`."""
    price = min(0.9999, max(0.0001, price))
    fee = fee_rate * price * (1.0 - price)
    return price + fee


def model_features_for_prediction(fv, model=None) -> list[float]:
    from btc_model_trainer import FEATURE_COLS
    n_features = int(getattr(model, "n_features_in_", len(FEATURE_COLS)) or len(FEATURE_COLS))
    cols = FEATURE_COLS[:n_features]
    return [
        float(getattr(fv, c, 0.0) or 0.0)
        for c in cols
    ]


@dataclass
class OpenTrade:
    trade_id: str
    market_id: str
    direction: str  # "Up" or "Down"
    side_token: str
    tf_minutes: int
    entry_price: float
    size: float
    spend: float
    fee: float
    model_prob: float
    breakeven_prob: float
    edge: float
    window_start: int
    placed_at: str
    resolved: bool = False
    won: Optional[bool] = None
    pnl: float = 0.0


def journal_trade(trade: OpenTrade) -> None:
    row = asdict(trade)
    row["mode"] = "paper_model"
    try:
        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        log(f"[journal] error: {e}")


def load_recent_trades(n: int = 100) -> list[dict]:
    if not JOURNAL_FILE.exists():
        return []
    try:
        lines = JOURNAL_FILE.read_text().strip().splitlines()
        trades = [json.loads(line) for line in lines if line.strip()]
        return trades[-n:]
    except Exception:
        return []


def recent_accuracy(trades: list[dict]) -> float:
    resolved = [t for t in trades if t.get("resolved") is True]
    if len(resolved) < MIN_TRADES_FOR_ACCURACY_CHECK:
        return 1.0  # not enough data; assume OK
    wins = sum(1 for t in resolved if t.get("won"))
    return wins / len(resolved)


class EdgePaperTrader:
    def __init__(self):
        self.bankroll = PAPER_CAPITAL
        self.starting = PAPER_CAPITAL
        self.peak_equity = PAPER_CAPITAL
        self.risk_halted = False
        self.risk_halt_reason = None
        self.open_trades: list[OpenTrade] = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.trade_counter = 0
        self.running = True
        self.models: dict[str, Optional[object]] = {"5m": None, "15m": None}

    def _load_models(self):
        for horizon in ("5m", "15m"):
            m = load_model(horizon)
            if m is not None:
                self.models[horizon] = m
                log(f"[model] Loaded {horizon} model")
            else:
                log(f"[model] No {horizon} model found; heuristic mode only")

    def _calc_pnl(self, won: bool, size: float, entry_price: float) -> float:
        return resolved_buy_pnl(won, size, entry_price)

    def _risk_check(self) -> bool:
        if self.risk_halted:
            return False
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.bankroll) / self.peak_equity
            if dd >= RISK_HALT_DRAWDOWN_PCT:
                self.risk_halted = True
                self.risk_halt_reason = f"drawdown {dd:.1%}"
                log(f"🛑 RISK HALT: {self.risk_halt_reason}")
                return False
        # Model accuracy check
        recent = load_recent_trades(50)
        acc = recent_accuracy(recent)
        if acc < MIN_RECENT_ACCURACY and len([t for t in recent if t.get("resolved")]) >= MIN_TRADES_FOR_ACCURACY_CHECK:
            self.risk_halted = True
            self.risk_halt_reason = f"model accuracy {acc:.1%} below {MIN_RECENT_ACCURACY:.0%}"
            log(f"🛑 RISK HALT: {self.risk_halt_reason}")
            return False
        return True

    def _enter(self, fv, tf_minutes: int, market: dict, model_prob: float, poly_price: float) -> Optional[OpenTrade]:
        be = breakeven_prob(poly_price)
        edge = model_prob - be
        if edge < EDGE_MARGIN:
            return None

        direction = "Up" if model_prob >= 0.5 else "Down"
        spend = max(MIN_SPEND, self.bankroll * 0.15)
        spend = min(spend, self.bankroll * 0.25, 10.0)

        token_id = clob_token_id_for_direction(market, direction)
        if not token_id:
            return None

        quote = fetch_clob_ask(token_id, spend)
        if not quote:
            return None

        entry_price = quote["price"]
        size = quote["size"]
        fee = taker_fee_usdc(size, entry_price)
        cost = entry_price * size + fee
        if cost > self.bankroll:
            return None

        self.trade_counter += 1
        trade = OpenTrade(
            trade_id=f"mdl_{int(time.time())}_{self.trade_counter}",
            market_id=market.get("id", ""),
            direction=direction,
            side_token=token_id,
            tf_minutes=tf_minutes,
            entry_price=entry_price,
            size=size,
            spend=cost,
            fee=fee,
            model_prob=model_prob,
            breakeven_prob=be,
            edge=edge,
            window_start=(int(time.time()) // (tf_minutes * 60)) * (tf_minutes * 60),
            placed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.bankroll -= cost
        self.open_trades.append(trade)
        journal_trade(trade)
        log(
            f"🟡 BET {tf_minutes}m {direction} | model={model_prob:.3f} be={be:.3f} edge={edge:.3f} "
            f"px={entry_price:.3f} size={size:.2f} cost=${cost:.2f} fee=${fee:.3f}"
        )
        return trade

    def _resolve(self):
        now = time.time()
        resolved = []
        for t in self.open_trades:
            if t.resolved:
                continue
            tf_sec = t.tf_minutes * 60
            if now < t.window_start + tf_sec + 10:
                continue
            actual = fetch_market_resolution(t.market_id)
            if not actual:
                continue
            won = (actual == t.direction)
            pnl = self._calc_pnl(won, t.size, t.entry_price)
            t.resolved = True
            t.won = won
            t.pnl = pnl
            self.bankroll += (t.size if won else 0.0)
            self.total_pnl += pnl
            if won:
                self.wins += 1
            else:
                self.losses += 1
            journal_trade(t)
            log(
                f"{'✅' if won else '❌'} RESOLVED {t.tf_minutes}m {t.direction} "
                f"pnl=${pnl:+.3f} bankroll=${self.bankroll:.2f} model={t.model_prob:.3f}"
            )
            resolved.append(t)
        self.open_trades = [t for t in self.open_trades if not t.resolved]
        self.peak_equity = max(self.peak_equity, self.bankroll)

    def run(self, duration: int = 28800):
        log("=== BTC Edge Paper Trader Starting ===")
        log(f"Capital=${PAPER_CAPITAL:.2f} edge_margin={EDGE_MARGIN:.2f} max_dd={RISK_HALT_DRAWDOWN_PCT:.0%}")
        self._load_models()

        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "running", False))
        last_market_check = {5: 0, 15: 0}
        last_status = 0
        last_feature_save = 0
        deadline = time.time() + duration

        while self.running and time.time() < deadline:
            now = time.time()

            if not self._risk_check():
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            # Fetch data
            try:
                spot_k = fetch_spot_klines(120)
                fut_k = fetch_fut_klines(120)
                fut_p = fetch_fut_premium()
                depth = fetch_spot_depth()
                poly = fetch_poly_btc_markets()
                fv = build_feature_vector(spot_k, fut_k, fut_p, depth, poly)
                if fv:
                    save_features(fv)
                    if now - last_feature_save >= 60:
                        compute_labels(FEATURE_DB)
                        last_feature_save = now
            except Exception as e:
                log(f"[data] error: {e}")
                fv = None

            for tf in (5, 15):
                if now - last_market_check.get(tf, 0) < MARKET_CHECK_SECONDS:
                    continue
                last_market_check[tf] = now

                # Max open trades guard
                if len([t for t in self.open_trades if not t.resolved]) >= MAX_OPEN_TRADES:
                    continue

                market = fetch_btc_market(tf)
                if not market:
                    continue

                # Skip if too close to window end
                ts_str = market.get("endDate", "") or market.get("endDate_iso", "")
                try:
                    from datetime import datetime as dt_cls
                    dt_end = dt_cls.fromisoformat(ts_str.replace("Z", "+00:00"))
                    sec_to_end = int(dt_end.timestamp()) - now
                    if sec_to_end < 60:
                        continue
                except Exception:
                    pass

                # Get Polymarket price
                prices_raw = market.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if not prices:
                    continue
                yes_price = float(prices[0])

                # Model prediction
                horizon = f"{tf}m"
                model = self.models.get(horizon)
                model_prob = None
                if model is not None and fv is not None:
                    X = [model_features_for_prediction(fv, model)]
                    prob_up = model.predict_proba(X)[0][1]
                    model_prob = prob_up
                else:
                    # Fallback: skip if no model
                    continue

                # Only trade if model is confident enough vs price
                # For Up/YES: prob_up vs yes_price
                # For Down/NO: prob_down vs no_price
                no_price = float(prices[1]) if len(prices) > 1 else 1.0 - yes_price
                be_yes = breakeven_prob(yes_price)
                be_no = breakeven_prob(no_price)

                if model_prob > be_yes + EDGE_MARGIN:
                    self._enter(fv, tf, market, model_prob, yes_price)
                elif (1.0 - model_prob) > be_no + EDGE_MARGIN:
                    self._enter(fv, tf, market, 1.0 - model_prob, no_price)

            self._resolve()

            if now - last_status >= STATUS_SECONDS:
                total = self.wins + self.losses
                wr = self.wins / total if total else 0.0
                dd = (self.peak_equity - self.bankroll) / self.peak_equity if self.peak_equity else 0.0
                log(
                    f"STATUS trades={total} W/L={self.wins}/{self.losses} "
                    f"WR={wr:.1%} PnL=${self.total_pnl:+.2f} bankroll=${self.bankroll:.2f} "
                    f"dd={dd:.1%} open={len(self.open_trades)}"
                )
                last_status = now

            time.sleep(LOOP_SLEEP_SECONDS)

        # Final resolve
        log("=== Shutting down, resolving remaining trades ===")
        for _ in range(30):
            self._resolve()
            if not self.open_trades:
                break
            time.sleep(10)

        total = self.wins + self.losses
        wr = self.wins / total if total else 0.0
        log(f"FINAL: trades={total} WR={wr:.1%} PnL=${self.total_pnl:+.2f} bankroll=${self.bankroll:.2f}")


def main():
    trader = EdgePaperTrader()
    trader.run()


if __name__ == "__main__":
    main()
