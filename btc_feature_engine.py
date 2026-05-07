#!/usr/bin/env python3
"""
BTC Feature Engine — Local microstructure + TA feature pipeline.

Data sources (all free REST, no auth):
  1. Binance Spot   : 1m klines, order book depth
  2. Binance Futures: 1m klines, mark price, funding rate
  3. Polymarket     : YES/NO mid prices for current 5m/15m BTC markets

Stores 1-minute feature vectors in SQLite for model training + live inference.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
FEATURE_DB = Path(os.environ.get("BTC_FEATURE_DB", str(DATA_DIR / "state" / "btc_features.db")))

# Binance endpoints
SPOT_KLINES = "https://api.binance.com/api/v3/klines"
SPOT_DEPTH = "https://api.binance.com/api/v3/depth"
SPOT_TICKER = "https://api.binance.com/api/v3/ticker/price"
FUT_KLINES = "https://fapi.binance.com/fapi/v1/klines"
FUT_PREMIUM = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Polymarket
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

try:
    from btc_external_metrics import (
        EXTERNAL_MODEL_FEATURE_KEYS,
        context_feature_subset,
        fetch_external_market_context,
    )
except Exception:  # keep feature engine alive if optional module is unavailable
    EXTERNAL_MODEL_FEATURE_KEYS = []

    def fetch_external_market_context(*args, **kwargs):
        return {}

    def context_feature_subset(ctx, keys=None):
        return {}

DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DB.parent.mkdir(parents=True, exist_ok=True)


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            ts INTEGER PRIMARY KEY,
            dt TEXT,
            price REAL,
            return_1m REAL,
            return_5m REAL,
            return_15m REAL,
            volatility_5m REAL,
            volatility_15m REAL,
            rsi_14 REAL,
            macd_hist REAL,
            bb_position REAL,
            atr_14 REAL,
            ob_imbalance_5 REAL,
            spread_bps REAL,
            basis_bps REAL,
            funding_rate REAL,
            volume_delta_5m REAL,
            hour_sin REAL,
            hour_cos REAL,
            minute_sin REAL,
            minute_cos REAL,
            vwap_deviation REAL,
            poly_5m_yes REAL,
            poly_5m_no REAL,
            poly_15m_yes REAL,
            poly_15m_no REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            ts INTEGER PRIMARY KEY,
            label_5m INTEGER,
            label_15m INTEGER,
            return_fwd_5m REAL,
            return_fwd_15m REAL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_features_dt ON features(dt)"
    )
    existing = {r[1] for r in conn.execute("PRAGMA table_info(features)").fetchall()}
    for col in EXTERNAL_MODEL_FEATURE_KEYS:
        if col not in existing:
            # Column names are controlled constants: [a-z0-9_].
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} REAL")
    conn.commit()
    conn.close()


def fetch_spot_klines(limit: int = 120) -> list[list]:
    try:
        r = requests.get(
            SPOT_KLINES,
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_fut_klines(limit: int = 120) -> list[list]:
    try:
        r = requests.get(
            FUT_KLINES,
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_fut_premium() -> dict:
    try:
        r = requests.get(FUT_PREMIUM, params={"symbol": "BTCUSDT"}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_spot_depth() -> dict:
    try:
        r = requests.get(SPOT_DEPTH, params={"symbol": "BTCUSDT", "limit": 20}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_spot_price() -> Optional[float]:
    try:
        r = requests.get(SPOT_TICKER, params={"symbol": "BTCUSDT"}, timeout=10)
        r.raise_for_status()
        return float(r.json().get("price", 0))
    except Exception:
        return None


def fetch_poly_btc_markets() -> dict[str, float]:
    """Return {slug: yes_price} for active BTC up/down 5m and 15m markets."""
    out: dict[str, float] = {}
    now_ts = int(time.time())
    for tf_minutes in (5, 15):
        window_sec = tf_minutes * 60
        current_window = (now_ts // window_sec) * window_sec
        for offset in (0, 1):
            window = current_window + offset * window_sec
            slug = f"btc-updown-{tf_minutes}m-{window}"
            try:
                r = requests.get(
                    f"{GAMMA_API}/markets",
                    params={"slug": slug},
                    timeout=8,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                markets = data if isinstance(data, list) else data.get("data", [])
                if not markets:
                    continue
                m = markets[0]
                prices_raw = m.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if prices:
                    out[f"{tf_minutes}m_yes"] = float(prices[0])
                    if len(prices) > 1:
                        out[f"{tf_minutes}m_no"] = float(prices[1])
                    else:
                        out[f"{tf_minutes}m_no"] = 1.0 - float(prices[0])
            except Exception:
                continue
    return out


def _to_candles(klines: list[list]) -> list[dict]:
    """Binance kline → dict with open, high, low, close, volume."""
    out = []
    for k in klines:
        out.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "ts": int(k[0]) // 1000,
            }
        )
    return out


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(c, 0) for c in changes]
    losses = [-min(c, 0) for c in changes]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(closes: list[float]) -> float:
    if len(closes) < 35:
        return 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    n = min(len(ema12), len(ema26))
    if n < 9:
        return 0.0
    macd = [ema12[-n + i] - ema26[-n + i] for i in range(n)]
    signal = _ema(macd, 9)
    if signal:
        return macd[-1] - signal[-1]
    return 0.0


def _ema(data: list[float], n: int) -> list[float]:
    if len(data) < n:
        return []
    k = 2.0 / (n + 1)
    result = [sum(data[:n]) / n]
    for v in data[n:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _bb_position(closes: list[float], period: int = 20, std: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((x - mean) ** 2 for x in window) / period
    sd = math.sqrt(variance)
    upper = mean + std * sd
    lower = mean - std * sd
    band = upper - lower
    if band <= 0:
        return 0.5
    return (closes[-1] - lower) / band


def _atr(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs[-period:]) / period


def _ob_imbalance(depth: dict, levels: int = 5) -> float:
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    bid_vol = sum(float(b[1]) for b in bids[:levels] if len(b) >= 2)
    ask_vol = sum(float(a[1]) for a in asks[:levels] if len(a) >= 2)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _spread_bps(depth: dict) -> float:
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if not bids or not asks:
        return 0.0
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    if mid == 0:
        return 0.0
    return (best_ask - best_bid) / mid * 10_000.0


def _volume_delta(candles: list[dict], n: int = 5) -> float:
    """Approximate buy-sell volume delta using close position in candle."""
    if len(candles) < n:
        return 0.0
    delta = 0.0
    for c in candles[-n:]:
        range_ = c["high"] - c["low"]
        if range_ == 0:
            delta += 0.0
        else:
            pos = (c["close"] - c["low"]) / range_  # 0=bearish, 1=bullish
            delta += (2 * pos - 1) * c["volume"]
    return delta


def _vwap_deviation(candles: list[dict]) -> float:
    if len(candles) < 2:
        return 0.0
    total_pv = sum(c["close"] * c["volume"] for c in candles[-20:])
    total_v = sum(c["volume"] for c in candles[-20:])
    vwap = total_pv / total_v if total_v else candles[-1]["close"]
    price = candles[-1]["close"]
    return (price - vwap) / vwap if vwap else 0.0


def _log_return(closes: list[float], n: int) -> float:
    if len(closes) < n + 1:
        return 0.0
    prev = closes[-(n + 1)]
    cur = closes[-1]
    if prev <= 0:
        return 0.0
    return math.log(cur / prev)


def _volatility(closes: list[float], n: int) -> float:
    if len(closes) < n + 1:
        return 0.0
    rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)


def _cyclical(value: float, period: float) -> tuple[float, float]:
    rad = 2.0 * math.pi * value / period
    return math.sin(rad), math.cos(rad)


@dataclass
class FeatureVector:
    ts: int
    dt: str
    price: float
    return_1m: float
    return_5m: float
    return_15m: float
    volatility_5m: float
    volatility_15m: float
    rsi_14: float
    macd_hist: float
    bb_position: float
    atr_14: float
    ob_imbalance_5: float
    spread_bps: float
    basis_bps: float
    funding_rate: float
    volume_delta_5m: float
    hour_sin: float
    hour_cos: float
    minute_sin: float
    minute_cos: float
    vwap_deviation: float
    poly_5m_yes: Optional[float]
    poly_5m_no: Optional[float]
    poly_15m_yes: Optional[float]
    poly_15m_no: Optional[float]
    ca_oi_30m_chg_pct: float = 0.0
    ca_liq_1h_imbalance: float = 0.0
    ca_funding_1h: float = 0.0
    ca_pred_funding_1h: float = 0.0
    ca_ls_30m_imbalance: float = 0.0
    ca_ls_30m_chg: float = 0.0
    ca_oi_now: float = 0.0
    ca_funding_now: float = 0.0
    cg_oi_30m_chg_pct: float = 0.0
    cg_oi_1h_chg_pct: float = 0.0
    cg_liq_30m_imbalance: float = 0.0
    cg_liq_1h_imbalance: float = 0.0
    cg_funding_30m: float = 0.0
    cg_funding_1h: float = 0.0
    cg_longs_30m_imbalance: float = 0.0
    cg_taker_30m_imbalance: float = 0.0
    cg_cvd_30m_imbalance: float = 0.0
    cg_orderbook_30m_imbalance: float = 0.0
    cg_oi_binance_share: float = 0.0
    bn_oi_30m_chg_pct: float = 0.0
    bn_top_ls_15m_imbalance: float = 0.0
    bn_taker_15m_imbalance: float = 0.0
    bn_global_ls_15m_imbalance: float = 0.0
    bn_funding_latest: float = 0.0
    by_oi_30m_chg_pct: float = 0.0
    by_funding_latest: float = 0.0
    bg_ls_5m_imbalance: float = 0.0
    bg_ls_5m_chg: float = 0.0
    bg_ls_15m_imbalance: float = 0.0
    bg_taker_5m_imbalance: float = 0.0
    bg_taker_15m_imbalance: float = 0.0
    bg_trader_ls_5m_imbalance: float = 0.0
    bg_position_ls_5m_imbalance: float = 0.0
    bg_recent_trade_imbalance: float = 0.0
    bg_depth_imbalance: float = 0.0
    bg_spread_bps: float = 0.0
    bg_mark_basis_bps: float = 0.0
    bg_index_basis_bps: float = 0.0
    bg_funding_hours_to_next: float = 0.0
    hl_depth_imbalance: float = 0.0
    hl_spread_bps: float = 0.0
    hl_funding_latest: float = 0.0
    hl_premium_latest: float = 0.0
    hl_pred_funding_hl: float = 0.0
    hl_pred_funding_binance: float = 0.0
    hl_pred_funding_bybit: float = 0.0
    external_bull_score: float = 0.0

    def to_tuple(self) -> tuple:
        return (
            self.ts,
            self.dt,
            self.price,
            self.return_1m,
            self.return_5m,
            self.return_15m,
            self.volatility_5m,
            self.volatility_15m,
            self.rsi_14,
            self.macd_hist,
            self.bb_position,
            self.atr_14,
            self.ob_imbalance_5,
            self.spread_bps,
            self.basis_bps,
            self.funding_rate,
            self.volume_delta_5m,
            self.hour_sin,
            self.hour_cos,
            self.minute_sin,
            self.minute_cos,
            self.vwap_deviation,
            self.poly_5m_yes,
            self.poly_5m_no,
            self.poly_15m_yes,
            self.poly_15m_no,
        )


def build_feature_vector(
    spot_klines: list[list],
    fut_klines: list[list],
    fut_premium: dict,
    spot_depth: dict,
    poly_prices: dict,
) -> Optional[FeatureVector]:
    spot = _to_candles(spot_klines)
    fut = _to_candles(fut_klines)
    if len(spot) < 20:
        return None

    closes = [c["close"] for c in spot]
    price = closes[-1]
    now = datetime.now(timezone.utc)
    ts = int(now.timestamp())

    # Futures / basis
    mark_price = float(fut_premium.get("markPrice", price) or price)
    funding_rate = float(fut_premium.get("lastFundingRate", 0.0) or 0.0)
    basis_bps = (mark_price - price) / price * 10_000.0 if price else 0.0

    # Time encodings
    hour_sin, hour_cos = _cyclical(now.hour, 24.0)
    minute_sin, minute_cos = _cyclical(now.minute, 60.0)

    external = context_feature_subset(fetch_external_market_context())

    return FeatureVector(
        ts=ts,
        dt=now.strftime("%Y-%m-%d %H:%M:%S"),
        price=price,
        return_1m=_log_return(closes, 1),
        return_5m=_log_return(closes, 5),
        return_15m=_log_return(closes, 15),
        volatility_5m=_volatility(closes, 5),
        volatility_15m=_volatility(closes, 15),
        rsi_14=_rsi(closes, 14),
        macd_hist=_macd_hist(closes),
        bb_position=_bb_position(closes, 20, 2.0),
        atr_14=_atr(spot, 14),
        ob_imbalance_5=_ob_imbalance(spot_depth, 5),
        spread_bps=_spread_bps(spot_depth),
        basis_bps=basis_bps,
        funding_rate=funding_rate,
        volume_delta_5m=_volume_delta(spot, 5),
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        minute_sin=minute_sin,
        minute_cos=minute_cos,
        vwap_deviation=_vwap_deviation(spot),
        poly_5m_yes=poly_prices.get("5m_yes"),
        poly_5m_no=poly_prices.get("5m_no"),
        poly_15m_yes=poly_prices.get("15m_yes"),
        poly_15m_no=poly_prices.get("15m_no"),
        **external,
    )


def save_features(fv: FeatureVector, db_path: Path = FEATURE_DB) -> None:
    _init_db(db_path)
    row = asdict(fv)
    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join(cols)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute(
        f"INSERT OR REPLACE INTO features ({col_sql}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    conn.close()


def compute_labels(db_path: Path = FEATURE_DB) -> None:
    """Back-fill forward-return labels for every row that lacks them."""
    _init_db(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    rows = conn.execute(
        "SELECT ts, price FROM features WHERE ts NOT IN (SELECT ts FROM labels) ORDER BY ts"
    ).fetchall()
    if not rows:
        conn.close()
        return

    # Load all future prices for fast lookup
    all_prices = {
        r[0]: r[1]
        for r in conn.execute("SELECT ts, price FROM features ORDER BY ts").fetchall()
    }
    inserted = 0
    for ts, price in rows:
        if price is None or price <= 0:
            continue
        # Find nearest future ts for 5m (300s) and 15m (900s)
        tgt_5 = ts + 300
        tgt_15 = ts + 900
        # Exact or next available
        future_5 = None
        future_15 = None
        for ft, fp in sorted(all_prices.items()):
            if ft >= tgt_5 and future_5 is None:
                future_5 = fp
            if ft >= tgt_15 and future_15 is None:
                future_15 = fp
            if future_5 is not None and future_15 is not None:
                break
        if future_5 is not None and future_5 > 0:
            ret_5 = math.log(future_5 / price)
            label_5 = 1 if future_5 > price else 0
        else:
            ret_5 = None
            label_5 = None
        if future_15 is not None and future_15 > 0:
            ret_15 = math.log(future_15 / price)
            label_15 = 1 if future_15 > price else 0
        else:
            ret_15 = None
            label_15 = None
        conn.execute(
            "INSERT OR REPLACE INTO labels (ts, label_5m, label_15m, return_fwd_5m, return_fwd_15m) VALUES (?, ?, ?, ?, ?)",
            (ts, label_5, label_15, ret_5, ret_15),
        )
        inserted += 1
    conn.commit()
    conn.close()
    print(f"[labels] Back-filled {inserted} rows.")


def run_loop(interval: int = 60) -> None:
    _init_db(FEATURE_DB)
    print(f"[feature_engine] DB={FEATURE_DB}  interval={interval}s")
    while True:
        t0 = time.time()
        try:
            spot_k = fetch_spot_klines(120)
            fut_k = fetch_fut_klines(120)
            fut_p = fetch_fut_premium()
            depth = fetch_spot_depth()
            poly = fetch_poly_btc_markets()
            fv = build_feature_vector(spot_k, fut_k, fut_p, depth, poly)
            if fv:
                save_features(fv)
                print(
                    f"[{fv.dt}] price={fv.price:,.0f} rsi={fv.rsi_14:.1f} "
                    f"basis={fv.basis_bps:.2f}bps ob_imb={fv.ob_imbalance_5:+.3f} "
                    f"poly5m={fv.poly_5m_yes} poly15m={fv.poly_15m_yes}"
                )
            else:
                print("[feature_engine] insufficient data for feature vector")
        except Exception as e:
            print(f"[feature_engine] ERROR: {e}")
        compute_labels(FEATURE_DB)
        elapsed = time.time() - t0
        sleep_for = max(1, interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    run_loop()
