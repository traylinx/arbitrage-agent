#!/usr/local/opt/python@3.11/bin/python3.11
"""
BTC Sniper Pro — Production Live Trading
========================================
ONE system: monitors Polymarket for BTC prediction markets,
places real CLOB orders, auto-improves from wins AND losses.

Signal logic:
  - Watch for BTC 5-min prediction markets on CLOB
  - Track BTC/USD momentum on Binance (1m, 5m, 15m)
  - When BTC moves > delta_thresh in a 5-min window → place trade
  - Take profit / stop loss based on BTC price movement
  - GA optimizes params over time

Live trading via py_clob_client with .env.live credentials.
"""

import copy
import json
import os
import random
import signal
import subprocess
import sys
import time
import requests
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
LIVE_KILL_SWITCH_FILE = Path(os.path.expanduser(os.environ.get(
    "BTC_LIVE_KILL_SWITCH_FILE",
    str(STATE_DIR / "live_trading_disabled.json"),
)))
LOG_DIR = DATA_DIR / "logs"
JOURNAL_FILE = Path(
    os.environ.get("BTC_JOURNAL_FILE", str(STATE_DIR / "intraday_journal.jsonl"))
)
# Launcher pins this to a tf-specific file (sniper_best_params_5m.json or 15m.json)
# so each agent uses its own paper-trained optimum. Without the override live ran
# with ens_thresh=0.30 (loose) while paper's per-tf optima were 0.50 — half the
# signals it took were noise the model would normally skip.
BEST_PARAMS_FILE = Path(
    os.environ.get("BTC_BEST_PARAMS_FILE", str(STATE_DIR / "sniper_best_params.json"))
)
FITNESS_HISTORY = DATA_DIR / "fitness_history.jsonl"
PAPER_BALANCE_FILE = STATE_DIR / "sniper_paper_balance.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PAPER_CAPITAL = 100.0
TAKER_FEE_BPS = 200  # 2%
POLYFEE = 0.01
# Caps are env-overridable so a $5 canary can run with $1 ticket sizes.
# Defaults preserve original behavior; explicit env vars opt in.
MIN_SPEND = float(os.environ.get("BTC_MIN_SPEND", "2.50"))
MAX_TRADE_COST = float(os.environ.get("BTC_MAX_TRADE_COST", "3.00"))
STOP_AFTER_FIRST_LOSS = os.environ.get("BTC_STOP_AFTER_FIRST_LOSS", "0") == "1"
MIN_SECONDS_LEFT = float(os.environ.get("BTC_MIN_SECONDS_LEFT", "90"))
LIVE_REQUIRE_EXTERNAL_CONTEXT = os.environ.get("BTC_LIVE_REQUIRE_EXTERNAL_CONTEXT", "1") == "1"
LIVE_MAX_FILLED_LOSSES = int(os.environ.get("BTC_LIVE_MAX_FILLED_LOSSES", "2"))
LIVE_MAX_DRAWDOWN_USDC = float(os.environ.get("BTC_LIVE_MAX_DRAWDOWN_USDC", "2.75"))
LIVE_MIN_WR_TRADES = int(os.environ.get("BTC_LIVE_MIN_WR_TRADES", "4"))
LIVE_MIN_WR = float(os.environ.get("BTC_LIVE_MIN_WR", "0.55"))
LIVE_ALLOW_PARAM_MUTATION = os.environ.get("BTC_LIVE_ALLOW_PARAM_MUTATION", "0") == "1"
PARAM_MIN_SAMPLE = int(os.environ.get("BTC_PARAM_MIN_SAMPLE", "20"))
# When set, clamps live bankroll for sizing purposes regardless of CLOB balance.
_max_bankroll_env = os.environ.get("BTC_MAX_BANKROLL_USDC", "")
MAX_BANKROLL_USDC = float(_max_bankroll_env) if _max_bankroll_env else None
BINANCE_REST = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
POLYMARKET_CLOB = "https://clob.polymarket.com"
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# ── Binance Free Endpoints ──────────────────────────────────────────────────────
BINANCE_DEPTH = "https://api.binance.com/api/v3/depth"
BINANCE_TRADES = "https://api.binance.com/api/v3/trades"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker"
BINANCE_UIKLINES = "https://api.binance.com/api/v3/uiKlines"
_WARMUP_DONE = False


# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_PATH = Path(
    os.environ.get("BTC_LOG_FILE", str(LOG_DIR / "btc_sniper_live.log"))
)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except:
        pass


# ── AI Helper ─────────────────────────────────────────────────────────────────
AI_URL = os.environ.get("SWITCHAI_URL", "http://localhost:18080/v1/chat/completions")
AI_KEY = os.environ.get("SWITCHAI_KEY", "sk-test-123")
AI_MODEL = os.environ.get("LLM_MODEL", "minimax:MiniMax-M2.7")


def ai_complete(prompt: str, max_tokens: int = 1200) -> str:
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "thinking": {"type": "disable"},
        "tools": [],
        "tool_choice": "auto",
    }
    try:
        r = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                AI_URL,
                "-H",
                "Content-Type: application/json",
                "-H",
                f"Authorization: Bearer {AI_KEY}",
                "-d",
                json.dumps(payload),
                "--max-time",
                "30",
            ],
            capture_output=True,
            text=True,
            timeout=35,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log(f"AI error: {e}")
    return ""


# ── Sniper Params ─────────────────────────────────────────────────────────────
@dataclass
class SniperParams:
    version: str = "pro1.0"
    name: str = ""

    # Signal thresholds
    delta_thresh: float = 50.0  # min BTC price move in window ($)
    conf_thresh: float = 0.08  # min ensemble confidence
    ens_thresh: float = 0.08  # min ensemble agreement

    # Sizing
    spend_ratio: float = 0.20  # fraction of bankroll per trade
    max_bet_pct: float = 0.50  # max bet as % of bankroll

    # Exits
    profit_target_bps: int = 300  # profit target in basis points
    stop_loss_bps: int = 150  # stop loss in basis points
    max_hold_seconds: int = 600  # force close after N seconds

    # Markets
    min_market_volume: float = 5000  # min 24h volume to trade
    max_spread_bps: int = 500  # max spread to trade (50%)

    # GA
    session_minutes: int = 60
    pop_size: int = 12

    def __post_init__(self):
        if not self.name:
            self.name = f"sniper_{datetime.now().strftime('%H%M%S')}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SniperParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def mutate(self, rate: float = 0.25) -> "SniperParams":
        new = copy.deepcopy(self)
        new.name = (
            f"mut_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
        )
        mutations = [
            # Ranges tuned to include known-good values: delta=12, conf=0.80, ens=0.60
            ("delta_thresh", 8.0, 50.0, 2.0),
            ("conf_thresh", 0.60, 0.95, 0.05),
            ("ens_thresh", 0.40, 0.90, 0.05),
            ("spend_ratio", 0.05, 0.40, 0.02),
            ("max_bet_pct", 0.20, 0.80, 0.05),
            ("profit_target_bps", 100, 800, 50),
            ("stop_loss_bps", 50, 400, 25),
            ("max_hold_seconds", 120, 900, 60),
            ("min_market_volume", 2000, 50000, 1000),
            ("max_spread_bps", 200, 1000, 50),
        ]
        for attr, lo, hi, step in mutations:
            if random.random() < rate:
                val = round(random.uniform(lo, hi) / step) * step
                setattr(new, attr, val)
        return new

    def crossover(self, a: "SniperParams", b: "SniperParams") -> "SniperParams":
        da, db = a.to_dict(), b.to_dict()
        keys = list(da.keys())
        pt = random.randint(1, len(keys) - 1)
        child = {}
        for i, k in enumerate(keys):
            child[k] = da[k] if i < pt else db[k]
        child["name"] = f"x_{datetime.now().strftime('%H%M%S')}"
        return SniperParams.from_dict(child)

    @classmethod
    def random_population(cls, size: int) -> list["SniperParams"]:
        pop = []
        for _ in range(size):
            p = cls()
            p.name = (
                f"init_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
            )
            p = p.mutate(rate=1.0)
            pop.append(p)
        return pop


# ── Signal Engine ─────────────────────────────────────────────────────────────
class SignalEngine:
    def __init__(self):
        self.ph: list[tuple[float, float]] = []  # (timestamp, price)

    def update(self, price: float, ts: float):
        self.ph.append((ts, price))
        if len(self.ph) > 600:
            self.ph.pop(0)

    def _ema(self, data: list, n: int) -> list:
        if len(data) < n:
            return data[:]
        k = 2 / (n + 1)
        result = [sum(data[:n]) / n]
        for p in data[n:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    def rsi(self, n: int = 14) -> tuple[float, float]:
        if len(self.ph) < n + 2:
            return 50.0, 0.0
        ds = [self.ph[i][1] - self.ph[i - 1][1] for i in range(1, len(self.ph))]
        g = [d for d in ds[-n:] if d > 0]
        l = [-d for d in ds[-n:] if d < 0]
        ag = sum(g) / n if g else 0
        al = sum(l) / n if l else 1e-9
        rs = ag / al
        rsi_val = 100 - (100 / (1 + rs))
        # Second value = RSI slope (momentum)
        slope = rsi_val - (50 if len(ds) > n else rsi_val)
        return rsi_val, slope

    def momentum(self, window_secs: int = 300) -> float:
        if len(self.ph) < 2:
            return 0.0
        now = self.ph[-1][0]
        cutoff = now - window_secs
        for i, (ts, price) in enumerate(self.ph):
            if ts >= cutoff:
                if i == 0:
                    return 0.0
                return self.ph[-1][1] - self.ph[i][1]
        return 0.0

    def acceleration(self, window_secs: int = 120) -> float:
        """Rate of change of momentum — positive = speeding up, negative = slowing down."""
        if len(self.ph) < window_secs * 2:
            return 0.0
        now = self.ph[-1][0]
        old_cutoff = now - window_secs * 2
        new_cutoff = now - window_secs

        old_mom = 0.0
        new_mom = 0.0
        for i, (ts, price) in enumerate(self.ph):
            if ts >= old_cutoff and ts < new_cutoff:
                if i > 0:
                    old_mom = self.ph[-1][1] - self.ph[i][1]
            if ts >= new_cutoff:
                if i > 0:
                    new_mom = self.ph[-1][1] - self.ph[i][1]
                break
        return new_mom - old_mom

    def ensemble(self, ens_thresh: float, window_delta: float = 0.0) -> dict:
        """Multi-indicator ensemble signal.
        Integrates: window delta, momentum, RSI, VWAP, Fib, Smart Money (OBI+CVD).
        Only fires when delta >= 12 (79% WR in live data)."""
        if len(self.ph) < 30:
            return {
                "direction": "Neutral",
                "conf": 0.0,
                "reasons": [],
                "conditions": {},
            }

        mom_1m = self.momentum(60)
        mom_5m = self.momentum(300)
        mom_30m = self.momentum(1800)  # 30-minute trend context
        rsi_val, rsi_slope = self.rsi(14)

        current_price = self.ph[-1][1]
        vwap = self.vwap(lookback=60)
        vwap_dist_bps = (current_price - vwap) / vwap * 10000 if vwap else 0
        cvd = self.cvd(lookback=50)
        obi = self.obi()
        fib_dist = self.fib_levels().get("distance_bps", 9999)
        sm = self.smart_money_signal()

        reasons = []
        conf_ups = []
        conf_downs = []
        conditions = {
            "vwap_dist_bps": round(vwap_dist_bps, 1),
            "fib_dist_bps": round(fib_dist, 1),
            "obi": round(obi, 3),
            "cvd": round(cvd, 4),
            "sm_signal": sm["signal"],
        }

        # ── WINDOW DELTA: Primary signal ──
        if window_delta >= 20:
            conf_ups.append(0.90)
            reasons.append(f"BIG+{window_delta:.0f}")
        elif window_delta >= 15:
            conf_ups.append(0.80)
            reasons.append(f"win_d+{window_delta:.0f}")
        elif window_delta >= 12:
            conf_ups.append(0.70)
            reasons.append(f"win_d+{window_delta:.0f}")
        elif window_delta > 0:
            conf_ups.append(0.25)
        elif window_delta <= -20:
            conf_downs.append(0.90)
            reasons.append(f"BIG-{abs(window_delta):.0f}")
        elif window_delta <= -15:
            conf_downs.append(0.80)
            reasons.append(f"win_d-{abs(window_delta):.0f}")
        elif window_delta <= -12:
            conf_downs.append(0.70)
            reasons.append(f"win_d-{abs(window_delta):.0f}")
        else:
            conf_downs.append(0.25)

        # ── MOMENTUM (1m + 5m) ──
        if mom_1m > 5 and conf_ups:
            conf_ups.append(0.08)
            reasons.append(f"1m+{mom_1m:.0f}")
        elif mom_1m < -5 and conf_downs:
            conf_downs.append(0.08)
            reasons.append(f"1m-{abs(mom_1m):.0f}")
        if mom_5m > 10 and conf_ups:
            conf_ups.append(0.06)
            reasons.append(f"5m+{mom_5m:.0f}")
        elif mom_5m < -10 and conf_downs:
            conf_downs.append(0.06)
            reasons.append(f"5m-{abs(mom_5m):.0f}")

        # ── RSI ──
        if rsi_val < 35 and rsi_slope > 0 and conf_ups:
            conf_ups.append(0.07)
            reasons.append(f"RSI_OV+{rsi_val:.0f}")
        elif rsi_val > 65 and rsi_slope < 0 and conf_downs:
            conf_downs.append(0.07)
            reasons.append(f"RSI_OB-{rsi_val:.0f}")

        # ── VWAP: price near VWAP = weak. Far above/below = trend confirmation ──
        if abs(vwap_dist_bps) < 10 and conf_ups:
            conf_ups.append(0.10)
            reasons.append(f"NearVWAP({vwap_dist_bps:+.0f}bps)")
        elif abs(vwap_dist_bps) < 10 and conf_downs:
            conf_downs.append(0.10)
            reasons.append(f"NearVWAP({vwap_dist_bps:+.0f}bps)")
        elif vwap_dist_bps > 15 and conf_ups:
            conf_ups.append(0.12)
            reasons.append(f"AboveVWAP({vwap_dist_bps:+.0f}bps)")
        elif vwap_dist_bps < -15 and conf_downs:
            conf_downs.append(0.12)
            reasons.append(f"BelowVWAP({vwap_dist_bps:+.0f}bps)")

        # ── FIBONACCI CONFLOUENCE: price at key Fib level = stronger signal ──
        if fib_dist < 15 and conf_ups:
            conf_ups.append(0.12)
            reasons.append(f"FibConfluence({fib_dist:.0f}bps)")
        elif fib_dist < 15 and conf_downs:
            conf_downs.append(0.12)
            reasons.append(f"FibConfluence({fib_dist:.0f}bps)")

        # ── 30-MINUTE TREND FILTER: SOFT penalty when delta < 15 and momentum contradicts ──
        strong_move = abs(window_delta) >= 15
        if not strong_move:
            if conf_downs and mom_30m > 150:
                conf_downs.append(-0.15)
                reasons.append("trend_filter_up")
            if conf_ups and mom_30m < -150:
                conf_ups.append(-0.15)
                reasons.append("trend_filter_down")

        # ── SMART MONEY: OBI + CVD alignment ──
        if sm["signal"] == "bull" and conf_ups:
            conf_ups.append(sm["conf"])
            reasons.append(f"SM_bull(OBI={obi:.2f})")
        elif sm["signal"] == "bear" and conf_downs:
            conf_downs.append(sm["conf"])
            reasons.append(f"SM_bear(OBI={obi:.2f})")
        elif sm["signal"] != "neutral" and (conf_ups or conf_downs):
            # SM strongly disagrees — SOFT penalty, not hard block (delta overrides)
            if conf_ups and sm["signal"] == "bear":
                conf_ups.append(-0.15)
                reasons.append(f"SM_diverge({sm['signal']})")
            elif conf_downs and sm["signal"] == "bull":
                conf_downs.append(-0.15)
                reasons.append(f"SM_diverge({sm['signal']})")

        # ── DIVERGENCE CHECK: momentum vs delta (tighter: 2 not 5) — SOFT penalty ──
        if conf_ups and mom_1m < -2:
            conf_ups.append(-0.10)
            reasons.append("mom_div")
        if conf_downs and mom_1m > 2:
            conf_downs.append(-0.10)
            reasons.append("mom_div")

        if not conf_ups and not conf_downs:
            return {
                "direction": "Neutral",
                "conf": 0.0,
                "reasons": reasons,
                "conditions": conditions,
            }

        up_conf = max(0, max(conf_ups)) if conf_ups else 0.0
        down_conf = max(0, max(conf_downs)) if conf_downs else 0.0
        if conf_downs and any(c < 0 for c in conf_ups):
            up_conf += -0.10
        if conf_ups and any(c < 0 for c in conf_downs):
            down_conf += -0.10
        up_conf = max(0, up_conf)
        down_conf = max(0, down_conf)

        if up_conf > ens_thresh and up_conf > down_conf:
            return {
                "direction": "Up",
                "conf": min(up_conf, 0.98),
                "reasons": reasons,
                "conditions": conditions,
            }
        elif down_conf > ens_thresh and down_conf > up_conf:
            return {
                "direction": "Down",
                "conf": min(down_conf, 0.98),
                "reasons": reasons,
                "conditions": conditions,
            }

        return {
            "direction": "Neutral",
            "conf": 0.0,
            "reasons": reasons,
            "conditions": conditions,
        }

    # ── FREE INDICATORS (no API key needed) ──────────────────────────────────

    def vwap(self, interval: str = "1m", lookback: int = 60) -> float:
        """Volume-Weighted Average Price from Binance klines.
        Above VWAP = bullish intraday trend. Below = bearish."""
        klines = get_binance_kline(interval=interval, limit=lookback)
        if not klines:
            return self.ph[-1][1] if self.ph else 0.0
        cum_vp = 0.0
        cum_vol = 0.0
        for k in klines:
            try:
                high = float(k[2])
                low = float(k[3])
                close = float(k[4])
                vol = float(k[5])
                typical = (high + low + close) / 3.0
                cum_vp += typical * vol
                cum_vol += vol
            except (ValueError, IndexError):
                continue
        if cum_vol <= 0:
            return self.ph[-1][1] if self.ph else 0.0
        return cum_vp / cum_vol

    def cvd(self, lookback: int = 50) -> float:
        """Cumulative Volume Delta — net buyer volume from Binance trades.
        Positive CVD = aggressive buying. Negative = selling pressure.
        Used for divergence detection."""
        trades = get_binance_trades(limit=lookback)
        if not trades:
            return 0.0
        cvd = 0.0
        for t in trades:
            try:
                qty = float(t.get("qty", 0))
                is_buy = not t.get("isBuyerMaker", True)
                cvd += qty if is_buy else -qty
            except (ValueError, KeyError):
                continue
        return cvd

    def obi(self) -> float:
        """Order Book Imbalance from Binance depth.
        Returns: bid_qty / (bid_qty + ask_qty) — 0.5 = balanced.
        >0.6 = buy wall dominant (bullish pressure).
        <0.4 = sell wall dominant (bearish pressure)."""
        depth = get_binance_depth(limit=50)
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.5
        return bid_vol / total

    def fib_levels(self) -> dict:
        """Fibonacci retracement levels from 4H swing high/low.
        Key levels: 0.236, 0.382, 0.5, 0.618, 0.786.
        Returns {level: price} dict. Price near level = potential reversal."""
        klines = get_binance_kline(interval="4h", limit=100)
        if not klines or len(klines) < 20:
            return {}
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        swing_high = max(highs[-20:])
        swing_low = min(lows[-20:])
        diff = swing_high - swing_low
        if diff < 50:
            return {}
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        current = self.ph[-1][1] if self.ph else (swing_high + swing_low) / 2
        levels = {}
        for r in fib_ratios:
            levels[r] = swing_high - r * diff
        nearest = min(levels.items(), key=lambda x: abs(x[1] - current))
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "levels": levels,
            "nearest_fib": nearest[0],
            "nearest_price": nearest[1],
            "distance_bps": abs(nearest[1] - current) / current * 10000,
        }

    def fib_confluence(self) -> float:
        """Returns confidence boost (0.0-0.2) when price is AT a Fib level.
        Strongest signal: price within 5bps of a key Fib level."""
        fib = self.fib_levels()
        if not fib or fib["distance_bps"] > 50:
            return 0.0
        d = fib["distance_bps"]
        if d <= 5:
            return 0.20
        elif d <= 15:
            return 0.12
        elif d <= 30:
            return 0.07
        return 0.0

    def volume_profile(self, interval: str = "5m", lookback: int = 20) -> dict:
        """Volume profile: which price levels have had the most volume.
        Returns {price_bucket: volume}. High volume nodes = support/resistance."""
        klines = get_binance_kline(interval=interval, limit=lookback)
        if not klines:
            return {}
        profile = {}
        for k in klines:
            try:
                vol = float(k[5])
                close = float(k[4])
                bucket = round(close, -2)
                profile[bucket] = profile.get(bucket, 0.0) + vol
            except (ValueError, IndexError):
                continue
        return profile

    def smart_money_signal(self) -> dict:
        """Composite signal from order flow + CVD + OBI.
        Returns: {signal: 'bull'|'bear'|'neutral', conf: 0.0-0.3}"""
        cvd = self.cvd()
        obi = self.obi()
        obi_score = 0.0
        if obi > 0.65:
            obi_score = 0.15
        elif obi > 0.58:
            obi_score = 0.08
        elif obi < 0.35:
            obi_score = 0.15
        elif obi < 0.42:
            obi_score = 0.08
        cvd_score = 0.0
        cvd_norm = cvd / max(abs(cvd), 1.0) if cvd != 0 else 0.0
        if cvd_norm > 0.7:
            cvd_score = 0.10
        elif cvd_norm < -0.7:
            cvd_score = 0.10
        total = obi_score + cvd_score
        if total < 0.05:
            return {"signal": "neutral", "conf": 0.0}
        direction = "bull" if obi > 0.5 else "bear"
        return {"signal": direction, "conf": min(total, 0.25)}


# ── Lessons Learned ────────────────────────────────────────────────────────────
class LessonsLearned:
    """
    Records conditions at entry and outcome for every trade.
    After each resolution, updates per-condition win rates.
    Slowly adapts param adjustments based on accumulated evidence.
    """

    def __init__(self):
        self.file = DATA_DIR / "lessons_learned.json"
        self.conditions: dict[str, dict] = {}
        self.load()

    def load(self):
        if self.file.exists():
            try:
                with open(self.file) as f:
                    self.conditions = json.load(f)
                log(f"[LESSONS] Loaded {len(self.conditions)} condition records")
            except:
                pass

    def save(self):
        try:
            with open(self.file, "w") as f:
                json.dump(self.conditions, f, indent=2)
        except:
            pass

    def record(self, direction: str, won: bool, conditions: dict):
        """Record outcome for a set of conditions. conditions = {key: value}."""
        for key, val in conditions.items():
            if key == "window_tf":
                val = str(val)
            else:
                try:
                    val = round(float(val), 4)
                except (ValueError, TypeError):
                    val = str(val)
            k = f"{key}={val}"
            if k not in self.conditions:
                self.conditions[k] = {"wins": 0, "losses": 0, "total": 0}
            c = self.conditions[k]
            c["total"] += 1
            if won:
                c["wins"] += 1
            else:
                c["losses"] += 1

    def wr(self, key: str, val) -> float:
        k = f"{key}={round(float(val), 4) if isinstance(val, float) else str(val)}"
        if k not in self.conditions:
            return 0.5
        c = self.conditions[k]
        return c["wins"] / max(c["total"], 1)

    def best_val(self, key: str, values: list) -> float:
        """Return the value in `values` that had the highest historical WR."""
        best, best_wr = values[0], 0.0
        for v in values:
            wr = self.wr(key, v)
            if wr > best_wr:
                best, best_wr = v, wr
        return best

    def analyse(self, trade: dict) -> str:
        """Return a human-readable analysis of what this trade teaches us."""
        lessons = []
        c = trade.get("conditions", {})
        direction = trade.get("direction", "?")
        won = trade.get("won", False)
        delta = abs(trade.get("btc_delta", 0))
        conf = trade.get("conf", 0)
        entry_age = trade.get("entry_age_secs", 999)
        vwap_dist = abs(c.get("vwap_dist_bps", 9999))
        fib_dist = c.get("fib_dist_bps", 9999)
        obi = c.get("obi", 0.5)
        cvd = c.get("cvd", 0)
        sm_signal = c.get("sm_signal", "neutral")

        if won:
            if delta >= 15:
                lessons.append(f"BIG delta(${delta:.0f}) → WIN ✅")
            if entry_age < 90:
                lessons.append(f"Early entry ({entry_age:.0f}s) → WIN ✅")
            if vwap_dist < 10:
                lessons.append(f"Near VWAP → WIN ✅")
            if fib_dist < 20:
                lessons.append(f"Fib confluence → WIN ✅")
            if conf > 0.85:
                lessons.append(f"High conf({conf:.2f}) → WIN ✅")
            if sm_signal == direction.lower():
                lessons.append(f"SmartMoney({sm_signal}) aligned → WIN ✅")
        else:
            if delta < 12:
                lessons.append(f"Small delta(${delta:.0f}) → LOSS ❌")
            if entry_age > 150:
                lessons.append(f"Late entry ({entry_age:.0f}s) → LOSS ❌")
            if conf < 0.8:
                lessons.append(f"Low conf({conf:.2f}) → LOSS ❌")
            if sm_signal != direction.lower() and sm_signal != "neutral":
                lessons.append(f"SmartMoney divergence({sm_signal}) → LOSS ❌")

        return " | ".join(lessons) if lessons else "Marginal trade"

    def suggest_params(self) -> dict:
        """Analyze lessons and return param adjustment suggestions."""
        suggestions = {}
        for key in list(self.conditions.keys()):
            if isinstance(key, str):
                if not key.startswith("delta=") and not key.startswith("conf="):
                    continue
                parts = key.split("=")
                if len(parts) != 2:
                    continue
                k, raw_val = parts
            elif isinstance(key, tuple) and len(key) >= 2:
                cond_type = key[0]
                cond_val = key[1]
                k = cond_type
                raw_val = str(cond_val)
            else:
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            c = self.conditions[key]
            if c["total"] < 3:
                continue
            wr = c["wins"] / c["total"]
            if wr < 0.45 and c["total"] >= 5:
                suggestions[k] = {
                    "val": val,
                    "wr": wr,
                    "action": "avoid",
                    "n": c["total"],
                }
            elif wr > 0.75 and c["total"] >= 3:
                suggestions[k] = {
                    "val": val,
                    "wr": wr,
                    "action": "prefer",
                    "n": c["total"],
                }
        return suggestions


def fetch_btc_markets(tf_minutes: int = 5) -> Optional[dict]:
    """
    Fetch active BTC prediction market for the given timeframe.
    Slug patterns:
      5m:  btc-updown-5m-{window_start_unix}
      15m: btc-updown-15m-{window_start_unix}
    Windows start at XX:00, XX:05, XX:10 UTC (5m) or XX:00, XX:15, XX:30 (15m).
    Checks current window first, then next two.
    """
    now_ts = int(time.time())
    window_sec = tf_minutes * 60
    current_window = (now_ts // window_sec) * window_sec
    slug_prefix = f"btc-updown-{tf_minutes}m"
    slugs_to_try = [
        f"{slug_prefix}-{current_window}",
        f"{slug_prefix}-{current_window + window_sec}",
        f"{slug_prefix}-{current_window + window_sec * 2}",
    ]
    GAMMA_API = "https://gamma-api.polymarket.com"

    for slug in slugs_to_try:
        try:
            r = requests.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                continue
            m = markets[0]
            if not isinstance(m, dict):
                continue
            if not m.get("acceptingOrders", False):
                continue
            if m.get("closed", True):
                continue
            end_date = m.get("endDate", "")
            if end_date:
                from datetime import datetime

                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    if end_dt.timestamp() < now_ts:
                        continue
                except:
                    pass
            return m
        except Exception as e:
            continue
    return None


def fetch_btc_market_for_window(tf_minutes: int, window_start: int) -> Optional[dict]:
    """Fetch the exact BTC up/down market for a specific window start.

    The live runner used to call ``fetch_btc_markets`` only after a new 5m
    window had already started. That function checks current+future slugs, but
    because we only invoked it after rollover the bot often discovered the
    market 30-70s late. For 5m markets that is fatal: the signal forms, then
    the min-left guard skips the trade.

    This exact-window fetch is used for prefetching the next market before the
    window starts, then arming it immediately at rollover.
    """
    slug = f"btc-updown-{tf_minutes}m-{int(window_start)}"
    GAMMA_API = "https://gamma-api.polymarket.com"
    now_ts = int(time.time())
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"slug": slug},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        if not markets:
            return None
        m = markets[0]
        if not isinstance(m, dict):
            return None
        if not m.get("acceptingOrders", False):
            return None
        if m.get("closed", True):
            return None
        end_date = m.get("endDate", "")
        if end_date:
            from datetime import datetime

            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if end_dt.timestamp() < now_ts:
                    return None
            except Exception:
                pass
        return m
    except Exception:
        return None


def fetch_market_resolution(
    market_id: str, direction: str, up_idx: int = 0, down_idx: int = 1
) -> Optional[str]:
    """Query Polymarket Gamma API for market resolution.
    Returns 'Up', 'Down', or None if not yet resolved."""
    GAMMA_API = "https://gamma-api.polymarket.com"
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=8)
        if r.status_code != 200:
            return None
        m = r.json()
        prices_raw = m.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        labels_raw = m.get("outcomes", '["Up","Down"]')
        labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
        # Resolve: winning outcome goes to $1.00
        for i, price in enumerate(prices):
            try:
                p = float(price)
                if p >= 0.99:
                    label = labels[i] if i < len(labels) else ""
                    if str(label).lower() in ("up", "yes"):
                        return "Up"
                    elif str(label).lower() in ("down", "no"):
                        return "Down"
            except:
                pass
        return None  # not yet resolved
    except Exception:
        return None


def get_orderbook(market_id: str) -> dict:
    try:
        r = requests.get(f"{POLYMARKET_CLOB}/orderbook/{market_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return {"bids": [], "asks": []}


def get_polio_price(market_id: str, side: str = "Yes") -> float:
    ob = get_orderbook(market_id)
    if side == "Yes":
        asks = ob.get("asks", [])
        if asks:
            return float(asks[0][0])
    else:
        bids = ob.get("bids", [])
        if bids:
            return float(bids[0][0])
    return 0.50


def get_binance_btc() -> Optional[float]:
    try:
        r = requests.get(BINANCE_REST, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except:
        return None


def get_binance_kline(
    symbol: str = "BTCUSDT", interval: str = "1m", start_ms: int = None, limit: int = 10
) -> Optional[list]:
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms:
            params["startTime"] = start_ms
        r = requests.get(BINANCE_KLINES, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def get_binance_depth(symbol: str = "BTCUSDT", limit: int = 50) -> dict:
    """Return orderbook bids/asks from Binance."""
    try:
        r = requests.get(
            BINANCE_DEPTH, params={"symbol": symbol, "limit": limit}, timeout=5
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"bids": [], "asks": []}


def get_binance_trades(symbol: str = "BTCUSDT", limit: int = 100) -> list:
    """Return recent trades from Binance. Each trade: {price, qty, quoteQty, isBuyerMaker}."""
    try:
        r = requests.get(
            BINANCE_TRADES, params={"symbol": symbol, "limit": limit}, timeout=5
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []


def get_binance_24h_ticker(symbol: str = "BTCUSDT") -> dict:
    """Return 24h ticker stats from Binance."""
    try:
        r = requests.get(BINANCE_TICKER, params={"symbol": symbol}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}


# ── CLOB Client Wrapper ─────────────────────────────────────────────────────────
# Migrated 2026-05-07 from py_clob_client (v1) to py_clob_client_v2 because
# Polymarket upgraded the CLOB schema and v1 orders now fail with
# 'order_version_mismatch'. Same external API, different SDK underneath.
class CLOBClient:
    def __init__(self):
        from dotenv import load_dotenv
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.clob_types import (
            AssetType,
            BalanceAllowanceParams,
        )

        ENV_PATH = HARVEY_HOME / "data" / "arbitrage-agent" / ".env.live"
        load_dotenv(ENV_PATH)

        pk = os.environ.get("POLYMARKET_PRIVATE_KEY")
        funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS")
        sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", 2))

        self._client = ClobClient(
            POLYMARKET_CLOB,
            key=pk,
            chain_id=137,
            signature_type=sig_type,
            funder=funder,
        )
        # Always derive fresh creds; static .env.live creds are stale post-migration
        derived = self._client.create_or_derive_api_key()
        self._client.set_api_creds(derived)

        try:
            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL, signature_type=sig_type
            )
        except TypeError:
            # v2 may not take signature_type
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        try:
            self._client.update_balance_allowance(params)
        except Exception:
            pass
        bal_resp = self._client.get_balance_allowance(params)
        self._balance = float(bal_resp.get("balance", 0)) / 1e6
        log(f"[CLOB] Connected. Balance: ${self._balance:.2f}")

    @property
    def balance(self) -> float:
        return self._balance

    def place_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> Optional[str]:
        from py_clob_client_v2.order_builder.constants import BUY
        from py_clob_client_v2.clob_types import OrderArgs, OrderType

        try:
            order_args = OrderArgs(
                price=min(price, 0.99),
                size=size,
                side=BUY,
                token_id=token_id,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed, OrderType.GTC)
            if resp and resp.get("success"):
                oid = resp.get("orderID", "unknown")
                log(f"[CLOB] Order placed: {side} {size}@{price:.4f} oid={oid}")
                return oid
            else:
                log(f"[CLOB] Order rejected: {resp}")
                recovered = self._recover_live_order(token_id)
                if recovered:
                    return recovered
                return None
        except Exception as e:
            log(f"[CLOB] Order error: {e}")
            recovered = self._recover_live_order(token_id)
            if recovered:
                return recovered
            return None

    def _recover_live_order(self, token_id: str) -> Optional[str]:
        """CLOB may post the order then throw a request exception.

        Recover the newest live order for this asset so caller tracks it instead
        of posting duplicates on the next loop.
        """
        try:
            orders = [
                o for o in self.get_open_orders()
                if isinstance(o, dict)
                and str(o.get("asset_id") or o.get("token_id") or "") == str(token_id)
                and str(o.get("status", "")).upper() == "LIVE"
            ]
            if not orders:
                return None
            orders.sort(key=lambda o: float(o.get("created_at") or 0), reverse=True)
            oid = orders[0].get("id")
            if oid:
                log(f"[CLOB] Recovered posted order after API error: oid={oid}")
                return str(oid)
        except Exception as e:
            log(f"[CLOB] recover_live_order failed: {e}")
        return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = self._client.cancel_orders([order_id])
            return order_id in (r or {}).get("canceled", [])
        except Exception as e:
            log(f"[CLOB] Cancel error: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        try:
            r = self._client.cancel_all()
            log(f"[CLOB] Cancel-all response: {r}")
            return True
        except Exception as e:
            log(f"[CLOB] Cancel-all error: {e}")
            return False

    def get_open_orders(self) -> list:
        try:
            return self._client.get_open_orders() or []
        except Exception as e:
            log(f"[CLOB] get_open_orders error: {e}")
            return []

    def get_order_status(self, order_id: str) -> Optional[dict]:
        try:
            return self._client.get_order(order_id=order_id)
        except Exception:
            return None


# ── Trade ─────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    window_start: int
    direction: str
    spend: float
    poly_price: float
    btc_delta: float
    btc_price_enter: float
    conf: float
    reasons: list
    placed_at: float
    order_id: Optional[str] = None
    token_id: Optional[str] = None
    market_id: Optional[str] = None
    resolved: bool = False
    won: bool = False
    pnl: float = 0.0
    exit_reason: str = ""
    journaled: bool = False
    resolved_at: float = 0.0
    window_tf: int = 5
    # On-chain fill verification — flips True only when CLOB get_order
    # reports size_matched > 0. Without this, _resolve_trade fabricates
    # WIN/LOSS for limit orders that never crossed the spread.
    filled: bool = False
    filled_size: float = 0.0
    fill_verified_at: float = 0.0
    entry_elapsed_sec: float = 0.0
    seconds_left_at_entry: float = 0.0


# ── Live Sniper ───────────────────────────────────────────────────────────────
class WindowState:
    """Tracks state for a single timeframe (5m or 15m)."""

    def __init__(self, tf_minutes: int):
        self.tf = tf_minutes
        self.se = SignalEngine()
        self.market_id: Optional[str] = None
        self.market_question: str = ""
        self.window_start: Optional[int] = None
        self.window_price: Optional[float] = None
        self.traded_this_window = False
        self.pause_until: float = 0.0
        self._outcome_prices: list[float] = [0.50, 0.50]
        self._up_token_id: Optional[str] = None
        self._down_token_id: Optional[str] = None
        self.deployed: float = 0.0

    def reset_window(self, window_start: int, window_price: float, market: dict):
        self.window_start = window_start
        self.window_price = window_price
        self.traded_this_window = False
        self.pause_until = 0.0
        self.market_id = market.get("id")
        self.market_question = market.get("question", "?")
        try:
            tokens = json.loads(market.get("clobTokenIds", "[]"))
            outcomes_raw = json.loads(market.get("outcomePrices", "[]"))
            self._outcome_prices = (
                [float(outcomes_raw[0]), float(outcomes_raw[1])]
                if len(outcomes_raw) >= 2
                else [0.50, 0.50]
            )
            self._up_token_id = (
                tokens[1] if len(tokens) > 1 else (tokens[0] if tokens else None)
            )
            self._down_token_id = tokens[0] if tokens else None
        except Exception:
            self._outcome_prices = [0.50, 0.50]
            self._up_token_id = None
            self._down_token_id = None


class LiveSniper:
    """
    Production BTC sniper: watches Polymarket BTC markets (5m + 15m simultaneously),
    places real CLOB orders, analyses every trade and auto-improves.
    """

    def __init__(self, params: SniperParams, live: bool = True):
        self.params = params
        self.live = live
        self.bankroll = PAPER_CAPITAL
        self.starting = PAPER_CAPITAL
        self.trades: list[Trade] = []
        self.wins = self.losses = self.blocks = 0
        self.unfilled = 0
        self.total_pnl = 0.0
        self.running = True

        self.btc_price: Optional[float] = None
        self.t0 = time.time()

        # Timeframes can be filtered via BTC_TIMEFRAMES env (e.g. "5" or "15")
        # so we can run two isolated live processes — one per timeframe.
        _tf_env = os.environ.get("BTC_TIMEFRAMES", "5,15").strip()
        try:
            tfs = sorted({int(x) for x in _tf_env.split(",") if x.strip()})
        except Exception:
            tfs = [5, 15]
        if not tfs:
            tfs = [5, 15]
        self._timeframes = tfs
        self.windows: dict[int, WindowState] = {tf: WindowState(tf) for tf in tfs}
        # Default 75/25 split favors 5m WR. For tiny canary bankroll the 15m
        # 25% slot is too small for 5-share min at $0.50 ($2.50). Override via env.
        _split_5 = float(os.environ.get("BTC_CAPITAL_SPLIT_5M", "0.75"))
        _split_15 = float(os.environ.get("BTC_CAPITAL_SPLIT_15M", "0.25"))
        if len(tfs) == 1:
            self._capital_split = {tfs[0]: 1.0}
        else:
            self._capital_split = {5: _split_5, 15: _split_15}

        self._open_orders: dict[str, dict] = {}
        self._reconciled_orders: set[str] = set()
        self._pending_fills: dict[str, dict] = {}
        self._client: Optional[CLOBClient] = None
        # Load persisted paper balance so restarts don't wipe gains
        if PAPER_BALANCE_FILE.exists():
            try:
                saved = json.loads(PAPER_BALANCE_FILE.read_text())
                saved_bal = float(saved.get("balance", PAPER_CAPITAL))
            except Exception:
                saved_bal = PAPER_CAPITAL
        else:
            saved_bal = PAPER_CAPITAL
        self._balance: float = saved_bal
        self.bankroll = saved_bal
        self.starting = PAPER_CAPITAL
        self._last_ga_evolve: float = 0.0
        self._balance_cache_time: float = 0.0
        self._balance_cache: float = saved_bal

        self.lessons = LessonsLearned()
        self._total_deployed: float = 0.0
        self._last_analysis: float = 0.0
        self._adapt_cooldown: float = 0.0
        self._external_ctx: dict = {}
        self._external_ctx_loaded_at: float = 0.0

        if os.environ.get("BTC_MAX_BET_PCT_OVERRIDE"):
            self.params.max_bet_pct = float(os.environ["BTC_MAX_BET_PCT_OVERRIDE"])
            log(f"[CANARY] max_bet_pct override={self.params.max_bet_pct:.2f}")
        if os.environ.get("BTC_SPEND_RATIO_OVERRIDE"):
            self.params.spend_ratio = float(os.environ["BTC_SPEND_RATIO_OVERRIDE"])
            log(f"[CANARY] spend_ratio override={self.params.spend_ratio:.2f}")

        if live:
            try:
                self._client = CLOBClient()
                self.bankroll = self._client.balance
                if MAX_BANKROLL_USDC is not None:
                    capped = min(self.bankroll, MAX_BANKROLL_USDC)
                    if capped < self.bankroll:
                        log(f"[CANARY] Bankroll clamped from ${self.bankroll:.2f} to ${capped:.2f} via BTC_MAX_BANKROLL_USDC")
                    self.bankroll = capped
                self._balance = self.bankroll
                self.starting = self.bankroll
                log(f"[CLOB] Connected. Bankroll for sizing: ${self.bankroll:.2f}")
            except Exception as e:
                log(f"[CLOB] Failed to connect: {e}. Running in SIM mode.")
                self.live = False

    def _external_context(self) -> dict:
        """Cached derivatives/flow context used by live decisions."""
        now = self._now()
        if self._external_ctx and now - self._external_ctx_loaded_at < 45:
            return self._external_ctx
        try:
            from btc_external_metrics import fetch_external_market_context

            ctx = fetch_external_market_context(use_cache=True) or {}
        except Exception as e:
            ctx = {"ok": 0, "error": f"{type(e).__name__}: {e}"}
        self._external_ctx = ctx
        self._external_ctx_loaded_at = now
        return ctx

    @staticmethod
    def _f(ctx: dict, key: str, default: float = 0.0) -> float:
        try:
            v = ctx.get(key, default)
            return float(v if v is not None else default)
        except Exception:
            return default

    def _external_summary(self, ctx: dict) -> str:
        age = self._f(ctx, "cache_age_sec", self._now() - self._f(ctx, "fetched_at", self._now()))
        return (
            f"ext_score={self._f(ctx,'external_bull_score'):+.3f} "
            f"flow5m={self._f(ctx,'bg_taker_5m_imbalance'):+.3f} "
            f"bg_recent={self._f(ctx,'bg_recent_trade_imbalance'):+.3f} "
            f"bn_taker15={self._f(ctx,'bn_taker_15m_imbalance'):+.3f} "
            f"bn_topLS={self._f(ctx,'bn_top_ls_15m_imbalance'):+.3f} "
            f"bg_depth={self._f(ctx,'bg_depth_imbalance'):+.3f} "
            f"hl_depth={self._f(ctx,'hl_depth_imbalance'):+.3f} "
            f"oi_bn30={self._f(ctx,'bn_oi_30m_chg_pct'):+.3f} "
            f"oi_by30={self._f(ctx,'by_oi_30m_chg_pct'):+.3f} "
            f"ok ca/cg/bn/by/bg/hl={int(self._f(ctx,'ca_ok'))}/{int(self._f(ctx,'cg_ok'))}/{int(self._f(ctx,'bn_ok'))}/{int(self._f(ctx,'by_ok'))}/{int(self._f(ctx,'bg_ok'))}/{int(self._f(ctx,'hl_ok'))} "
            f"age={age:.0f}s"
        )

    def _external_vote(self, direction: str, base_conf: float) -> tuple[bool, float, str, dict]:
        ctx = self._external_context()
        providers_ok = sum(
            1 for k in ("ca_ok", "cg_ok", "bn_ok", "by_ok", "bg_ok", "hl_ok")
            if self._f(ctx, k) > 0
        )
        if not ctx or self._f(ctx, "ok") <= 0 or providers_ok < 2:
            reason = f"NO_GO external context missing/weak providers_ok={providers_ok} {ctx.get('error','') if isinstance(ctx, dict) else ''}"
            return (not LIVE_REQUIRE_EXTERNAL_CONTEXT), base_conf, reason, ctx

        ext_score = self._f(ctx, "external_bull_score")
        flow = (
            0.35 * self._f(ctx, "bg_taker_5m_imbalance")
            + 0.20 * self._f(ctx, "bg_recent_trade_imbalance")
            + 0.15 * self._f(ctx, "bn_taker_15m_imbalance")
            + 0.10 * self._f(ctx, "bn_top_ls_15m_imbalance")
            + 0.10 * self._f(ctx, "bg_depth_imbalance")
            + 0.10 * self._f(ctx, "hl_depth_imbalance")
        )
        combo = max(-1.0, min(1.0, 0.65 * ext_score + 0.35 * flow))
        want = 1.0 if direction == "Up" else -1.0
        aligned = combo * want
        conf = base_conf
        if aligned >= 0.15:
            conf = min(0.98, conf + 0.08)
            verdict = "GO strong external alignment"
        elif aligned >= 0.03:
            conf = min(0.98, conf + 0.03)
            verdict = "GO mild external alignment"
        elif aligned <= -0.12:
            return False, conf, f"NO_GO external contra combo={combo:+.3f} aligned={aligned:+.3f}", ctx
        elif aligned <= -0.03:
            conf = max(0.0, conf - 0.12)
            verdict = "CAUTION external mildly contra"
        else:
            verdict = "GO external neutral"
        return True, conf, f"{verdict} combo={combo:+.3f} aligned={aligned:+.3f}", ctx

    def _halt_after_first_loss_if_needed(self, source: str):
        if not (STOP_AFTER_FIRST_LOSS and self.live and self.losses >= 1):
            return
        self._stop_live(f"STOP_AFTER_FIRST_LOSS after {source}")

    def _stop_live(self, reason: str):
        """Stop live loop, cancel open orders, and arm the durable kill switch."""
        log(f"[CANARY] LIVE STOP: {reason}; canceling open orders and arming kill switch")
        self.running = False
        if self._client:
            try:
                self._client.cancel_all_orders()
            except Exception as e:
                log(f"[CANARY] cancel-all after loss failed: {e}")
        try:
            LIVE_KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
            LIVE_KILL_SWITCH_FILE.write_text(json.dumps({
                "disabled": True,
                "reason": reason,
                "ts": datetime.utcnow().isoformat() + "Z",
            }, indent=2))
        except Exception as e:
            log(f"[CANARY] kill-switch write failed: {e}")

    def _live_circuit_breaker(self, source: str):
        """Fail closed on live canaries that are not proving edge."""
        if not self.live:
            return
        filled = self.wins + self.losses
        pnl = self.total_pnl
        if LIVE_MAX_FILLED_LOSSES > 0 and self.losses >= LIVE_MAX_FILLED_LOSSES:
            self._stop_live(
                f"max filled losses reached ({self.losses}/{LIVE_MAX_FILLED_LOSSES}) after {source}"
            )
            return
        if LIVE_MAX_DRAWDOWN_USDC > 0 and pnl <= -abs(LIVE_MAX_DRAWDOWN_USDC):
            self._stop_live(
                f"max live drawdown reached (${pnl:.2f} <= -${abs(LIVE_MAX_DRAWDOWN_USDC):.2f}) after {source}"
            )
            return
        if filled >= LIVE_MIN_WR_TRADES and LIVE_MIN_WR > 0:
            wr = self.wins / max(filled, 1)
            if wr < LIVE_MIN_WR:
                self._stop_live(
                    f"live WR below floor ({wr:.0%} < {LIVE_MIN_WR:.0%} over {filled} filled) after {source}"
                )

    def _window_for_tf(self, tf: int) -> WindowState:
        return self.windows[tf]

    def _cap_for_tf(self, tf: int) -> float:
        available = self._balance - self._total_deployed
        if available < MIN_SPEND * 2:
            return 0.0
        cap = self._balance * self._capital_split[tf]
        return min(cap, available)

    def _now(self) -> float:
        return time.time()

    def _attach_market(self, win: WindowState, tf: int, mkt: dict, source: str = "fetch"):
        """Attach a Polymarket market to a WindowState and log tradable prices."""
        tokens = []
        outcomes_raw = []
        outcomes_labels = []
        try:
            tokens = json.loads(mkt.get("clobTokenIds", "[]"))
            outcomes_raw = json.loads(mkt.get("outcomePrices", "[]"))
            outcomes_labels = json.loads(mkt.get("outcomes", "[]"))
        except Exception:
            pass
        win.market_id = mkt.get("id")
        win.market_question = mkt.get("question", "?")
        # Dynamically match UP/DOWN by outcome label (Polymarket ordering can vary)
        up_idx, down_idx = 0, 1
        for i, label in enumerate(outcomes_labels):
            if str(label).lower() in ("up", "yes"):
                up_idx = i
            elif str(label).lower() in ("down", "no"):
                down_idx = i
        win._outcome_prices = [0.50, 0.50]
        if len(outcomes_raw) >= 2:
            # Preserve historical convention: _outcome_prices[0]=DOWN, [1]=UP
            win._outcome_prices = [
                float(outcomes_raw[down_idx]),
                float(outcomes_raw[up_idx]),
            ]
        win._up_token_id = tokens[up_idx] if up_idx < len(tokens) else None
        win._down_token_id = tokens[down_idx] if down_idx < len(tokens) else None
        log(f"[PM {tf}m] Market({source}): {win.market_question}")
        log(
            f"[PM {tf}m] labels={outcomes_labels} up_idx={up_idx} down_idx={down_idx} | DOWN=${win._outcome_prices[0]:.3f} UP=${win._outcome_prices[1]:.3f}"
        )

    def _journal_trade(self, t: Trade):
        if t.journaled:
            return
        t.journaled = True
        d = {
            "mode": "live" if self.live else "paper",
            "params": self.params.to_dict(),
            "window_start": t.window_start,
            "window_tf": getattr(t, "window_tf", 5),
            "direction": t.direction,
            "spend": t.spend,
            "poly_price": t.poly_price,
            "btc_delta": t.btc_delta,
            "btc_price_enter": t.btc_price_enter,
            "conf": t.conf,
            "reasons": t.reasons,
            "filled": getattr(t, "filled", False),
            "filled_size": round(getattr(t, "filled_size", 0.0), 4),
            "entry_elapsed_sec": round(getattr(t, "entry_elapsed_sec", 0.0), 3),
            "seconds_left_at_entry": round(getattr(t, "seconds_left_at_entry", 0.0), 3),
            "won": t.won,
            "pnl": round(t.pnl, 4),
            "exit_reason": t.exit_reason,
            "agent_id": os.environ.get("BTC_AGENT_ID", ""),
            "timeframes": ",".join(str(tf) for tf in self._timeframes),
            "placed_at": datetime.fromtimestamp(t.placed_at).isoformat(),
            "resolved_at": datetime.fromtimestamp(t.resolved_at).isoformat()
            if t.resolved_at
            else None,
        }
        try:
            with open(JOURNAL_FILE, "a") as f:
                f.write(json.dumps(d) + "\n")
        except:
            pass

    # ── Signal check (per window) ────────────────────────────────────────────
    def _check_signal(self, win: WindowState, tf: int) -> Optional[dict]:
        if not self.btc_price or not win.window_price:
            return None
        if self._now() < win.pause_until:
            return None
        if win.traded_this_window:
            return None
        cap = self._cap_for_tf(tf)
        if cap < MIN_SPEND:
            return None

        delta = self.btc_price - win.window_price
        up_price = win._outcome_prices[1]
        down_price = win._outcome_prices[0]
        poly_conviction = abs(up_price - 0.5) * 2
        ext_ctx = self._external_context()
        ext_summary = self._external_summary(ext_ctx)

        # Verbose diagnostic log (every check, so we can see why it's NOT firing)
        if abs(delta) >= 3:  # only log meaningful deltas to avoid spam
            sig_probe = win.se.ensemble(self.params.ens_thresh, window_delta=delta)
            log(
                f"[EVAL {tf}m] delta=${delta:+.1f} poly_conv={poly_conviction:.2f} "
                f"→ dir={sig_probe['direction']} conf={sig_probe['conf']:.2f} "
                f"(need: |d|>={self.params.delta_thresh:.0f} & conf>={self.params.conf_thresh:.2f}) | {ext_summary}"
            )

        if abs(delta) < self.params.delta_thresh and poly_conviction < 0.90:
            if abs(delta) >= 3:
                log(f"[NO_GO {tf}m] base delta/poly too weak")
            return None

        sig = win.se.ensemble(self.params.ens_thresh, window_delta=delta)

        if sig["direction"] == "Neutral":
            log(f"[NO_GO {tf}m] ensemble neutral | {ext_summary}")
            return None
        if (
            sig["conf"] < self.params.conf_thresh
            and abs(delta) < self.params.delta_thresh + 5
            and poly_conviction < 0.90
        ):
            log(
                f"[NO_GO {tf}m] base_conf={sig['conf']:.2f} below conf_thresh={self.params.conf_thresh:.2f}"
            )
            return None

        direction = sig["direction"]
        hour = int(datetime.utcnow().strftime("%H"))
        conf = self._hour_adjust_conf(sig["conf"], direction, hour)
        if conf < self.params.conf_thresh:
            log(f"[NO_GO {tf}m] hour-adjusted conf={conf:.2f} below threshold")
            return None

        ext_ok, conf, ext_reason, ext_ctx = self._external_vote(direction, conf)
        ext_summary = self._external_summary(ext_ctx)
        if not ext_ok:
            log(f"[NO_GO {tf}m] {direction} rejected by external stack: {ext_reason} | {ext_summary}")
            return None
        log(f"[GO_CHECK {tf}m] {direction} base_conf={sig['conf']:.2f} adj_conf={conf:.2f} {ext_reason} | {ext_summary}")
        if conf < self.params.conf_thresh:
            log(f"[NO_GO {tf}m] external-adjusted conf={conf:.2f} below threshold")
            return None

        trade_price = up_price if direction == "Up" else down_price
        if trade_price > 0.72 and poly_conviction < 0.90:
            log(f"[NO_GO {tf}m] trade_price={trade_price:.3f} too expensive for weak PM conviction")
            return None
        conviction = abs(trade_price - 0.50)
        if conviction < 0.04:
            price_lo, price_hi = 0.40, 0.60
        elif conviction < 0.08:
            price_lo, price_hi = 0.42, 0.58
        elif conviction < 0.15:
            price_lo, price_hi = 0.45, 0.55
        else:
            price_lo, price_hi = 0.40, 0.60
        if not (price_lo <= trade_price <= price_hi):
            log(f"[NO_GO {tf}m] trade_price={trade_price:.3f} outside band {price_lo:.2f}-{price_hi:.2f}")
            return None

        PREFERRED_UP = {1, 11, 13, 15, 16, 17, 18, 19, 22}
        PREFERRED_DOWN = {0, 9, 12, 14, 20, 23, 1, 11, 13}
        tier = "NORMAL"
        if (
            abs(delta) >= 25
            and 0.48 <= trade_price <= 0.52
            and (
                (direction == "Up" and hour in PREFERRED_UP)
                or (direction == "Down" and hour in PREFERRED_DOWN)
            )
        ):
            tier = "ULTIMATE"

        return {
            "direction": sig["direction"],
            "delta": delta,
            "conf": conf,
            "tier": tier,
            "reasons": [*sig["reasons"], ext_reason, ext_summary],
            "conditions": {
                **sig.get("conditions", {}),
                "external_bull_score": self._f(ext_ctx, "external_bull_score"),
                "external_combo_reason": ext_reason,
                "bg_taker_5m_imbalance": self._f(ext_ctx, "bg_taker_5m_imbalance"),
                "bn_taker_15m_imbalance": self._f(ext_ctx, "bn_taker_15m_imbalance"),
                "bg_recent_trade_imbalance": self._f(ext_ctx, "bg_recent_trade_imbalance"),
            },
        }

    def _refresh_balance(self):
        """Best-effort balance refresh without blocking the trading loop.

        This used to retry 6 times with sleeps up to ~30s. In live 5m trading
        that blocked window rollover and made the agent attach prefetched
        markets 60s late. Balance freshness must not outrank entry timing.
        """
        if not self._client:
            return
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL, signature_type=2
            )
            try:
                self._client.update_balance_allowance(params)
            except Exception:
                pass
            bal_resp = self._client.get_balance_allowance(params)
            raw = bal_resp.get("balance", "0")
            new_balance = float(raw) / 1e6
            if new_balance > 0:
                self._balance = new_balance
                self.bankroll = new_balance
                self._balance_cache = new_balance
                self._balance_cache_time = self._now()
                return
        except Exception:
            pass

    def _verify_fill_size(self, order_id: str) -> float:
        """Return on-chain matched size (shares) for a CLOB order.

        Truth source for whether a limit ever crossed the spread. Used by the
        reconciliation + resolution paths to suppress phantom WIN/LOSS logging
        for orders that never filled. Returns 0.0 on any error or unknown
        state — caller treats 0 as 'unfilled'.
        """
        if not self.live or not self._client or not order_id:
            return 0.0
        try:
            o = self._client.get_order_status(order_id) or {}
        except Exception:
            return 0.0
        if not isinstance(o, dict):
            return 0.0
        for key in ("size_matched", "sizeMatched", "matched_size", "matchedSize"):
            v = o.get(key)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        try:
            orig = float(o.get("original_size", o.get("originalSize", 0)) or 0)
            rem = float(o.get("size_remaining", o.get("sizeRemaining", orig)) or orig)
            return max(0.0, orig - rem)
        except Exception:
            return 0.0

    def _refresh_balance_with_retry(self):
        """Call update_balance_allowance then get_balance_allowance in a retry loop.
        Called after detecting a fill — the on-chain state may take a few seconds.
        """
        if not self._client:
            return
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL, signature_type=2
            )
            for attempt in range(8):
                try:
                    self._client.update_balance_allowance(params)
                except Exception:
                    pass
                try:
                    bal_resp = self._client.get_balance_allowance(params)
                    raw = bal_resp.get("balance", "0")
                    new_balance = float(raw) / 1e6
                    if new_balance > 0:
                        old = self._balance
                        self._balance = new_balance
                        self.bankroll = new_balance
                        if abs(new_balance - old) > 0.01:
                            log(
                                f"[CLOB] Balance updated: ${old:.2f} → ${new_balance:.2f}"
                            )
                        return
                except Exception:
                    pass
                import time

                time.sleep(2.0 * (attempt + 1))
        except Exception:
            pass

    # ── Order execution ───────────────────────────────────────────────────────
    def _place_trade(self, sig: dict, win: WindowState, tf: int) -> Optional[Trade]:
        if not win.market_id:
            return None
        # Don't place orders too late in window — orderbook may vanish.
        win_sec = tf * 60
        time_left = (win.window_start + win_sec) - self._now()
        if time_left < MIN_SECONDS_LEFT:
            log(
                f"[SKIP {tf}m] Only {time_left:.0f}s left in window — min={MIN_SECONDS_LEFT:.0f}s"
            )
            return None

        now = self._now()
        if now - self._balance_cache_time > 10:
            # Never call the slow retry loop on the entry path. It can block
            # for >60s, then place a stale order in the final seconds of a 5m
            # market. Slow retry is only safe after fills/resolution.
            self._refresh_balance()
            self._balance_cache = self._balance
            self._balance_cache_time = now
        else:
            self._balance = self._balance_cache

        now = self._now()
        time_left = (win.window_start + win_sec) - now
        if time_left < MIN_SECONDS_LEFT:
            log(
                f"[SKIP {tf}m] Entry path delayed; only {time_left:.0f}s left "
                f"in window — min={MIN_SECONDS_LEFT:.0f}s"
            )
            return None

        cap = self._cap_for_tf(tf)
        if cap < MIN_SPEND:
            log(f"[WARN] Cap ${cap:.2f} < min ${MIN_SPEND:.2f}, skipping {tf}m trade")
            self.blocks += 1
            return None

        outcome = "Yes" if sig["direction"] == "Up" else "No"
        poly_price = (
            win._outcome_prices[1] if outcome == "Yes" else win._outcome_prices[0]
        )
        if poly_price <= 0:
            poly_price = 0.50
        # Cross-the-spread premium so resting limit orders actually fill.
        _fill_premium_bps = float(os.environ.get("BTC_FILL_PREMIUM_BPS", "0"))
        if _fill_premium_bps > 0:
            poly_price = min(poly_price * (1.0 + _fill_premium_bps / 10000.0), 0.99)

        tier = sig.get("tier", "NORMAL")
        tier_max_bet_pct = 0.60 if tier == "ULTIMATE" else self.params.max_bet_pct
        spend = min(
            max(MIN_SPEND, cap * self.params.spend_ratio),
            cap * tier_max_bet_pct,
        )
        spend = min(spend, cap * 0.95, MAX_TRADE_COST)
        log(
            f"[DEBUG {tf}m] [{tier}] cap={cap:.2f} spend={spend:.2f} price={poly_price:.4f} "
            f"size={spend / poly_price:.2f} balance=${self._balance:.2f}"
        )

        size = spend / poly_price
        # Polymarket 5/15-min markets enforce orderMinSize=5; SDK rejects below.
        min_shares = float(os.environ.get("BTC_MIN_SHARES", "5.0"))
        if size < min_shares:
            min_cost = min_shares * poly_price
            if min_cost > MAX_TRADE_COST:
                log(
                    f"[BLOCK {tf}m] min_shares={min_shares:.2f} costs=${min_cost:.2f} "
                    f"> max_order=${MAX_TRADE_COST:.2f}; no executable order"
                )
                self.blocks += 1
                return None
            if min_cost > cap * 0.95:
                log(
                    f"[BLOCK {tf}m] min_shares={min_shares:.2f} costs=${min_cost:.2f} "
                    f"> cap95=${cap * 0.95:.2f}; bankroll cap too small"
                )
                self.blocks += 1
                return None
            size = min_shares
            cost = min_cost
        else:
            cost = size * poly_price
        if cost > MAX_TRADE_COST:
            size = MAX_TRADE_COST / poly_price
            size = float(int(size * 100)) / 100
            cost = size * poly_price
        if cost > cap * 0.95:
            size = cap * 0.95 / poly_price
            size = float(int(size * 100)) / 100
            cost = size * poly_price
        # Absolute share cap — no math path can blow past this.
        _abs_size_cap = float(os.environ.get("BTC_ABS_SIZE_CAP", "0"))
        if _abs_size_cap > 0 and size > _abs_size_cap:
            size = _abs_size_cap
            cost = size * poly_price
        if cost < MIN_SPEND or size < 1.0:
            log(
                f"[BLOCK {tf}m] cost=${cost:.2f} size={size:.2f} below min_spend=${MIN_SPEND:.2f}"
            )
            self.blocks += 1
            return None

        down_token_id = win._down_token_id
        up_token_id = win._up_token_id
        trade_token_id = up_token_id if outcome == "Yes" else down_token_id
        order_id = None

        if self.live and self._client:
            if os.environ.get("BTC_SINGLE_OPEN_ORDER", "1") == "1":
                live_open = self._client.get_open_orders()
                if live_open:
                    ids = ",".join(str(o.get("id", ""))[:12] for o in live_open[:3] if isinstance(o, dict))
                    log(
                        f"[NO_GO {tf}m] existing CLOB open_orders={len(live_open)} ids={ids}; "
                        "single-open-order guard blocks duplicate"
                    )
                    self.blocks += 1
                    return None
            now = self._now()
            time_left = (win.window_start + win_sec) - now
            if time_left < MIN_SECONDS_LEFT:
                log(
                    f"[SKIP {tf}m] Pre-order check delayed; only {time_left:.0f}s left "
                    f"in window — min={MIN_SECONDS_LEFT:.0f}s"
                )
                return None
            # Use cached balance — don't block with slow retry refresh
            if self._balance < MIN_SPEND:
                self.blocks += 1
                return None
            if self._balance < spend:
                spend = self._balance * 0.90
                size = spend / poly_price
                size = float(int(size * 100)) / 100
                cost = size * poly_price
            order_id = self._client.place_order(
                token_id=trade_token_id or "",
                side=outcome,
                price=poly_price,
                size=size,
            )
            if not order_id:
                # Clear market state so next iteration fetches fresh market
                # (handles stale orderbook errors, resolved markets, etc)
                log(
                    f"[RECOVER {tf}m] Order failed — clearing market_id to force refetch"
                )
                win.market_id = None
                win._up_token_id = None
                win._down_token_id = None
                return None
            self._open_orders[order_id] = {
                "placed_at": now,
                "direction": sig["direction"],
                "spend": cost,
                "poly_price": poly_price,
                "window_start": win.window_start,
                "window_tf": tf,
                "token_id": trade_token_id,
                "market_id": win.market_id,
            }
            self._total_deployed += cost

        win.traded_this_window = True

        trade = None
        try:
            trade = Trade(
                window_start=win.window_start or int(now // (tf * 60) * (tf * 60)),
                direction=sig["direction"],
                spend=spend,
                poly_price=poly_price,
                btc_delta=sig["delta"],
                btc_price_enter=self.btc_price or 0.0,
                conf=sig["conf"],
                reasons=sig["reasons"],
                placed_at=now,
                order_id=order_id,
                token_id=trade_token_id,
                market_id=win.market_id,
                entry_elapsed_sec=now - (win.window_start or int(now // (tf * 60) * (tf * 60))),
                seconds_left_at_entry=(win.window_start or int(now // (tf * 60) * (tf * 60))) + win_sec - now,
            )
            self.trades.append(trade)
        except Exception as e:
            log(f"[WARN] Could not create Trade object: {e}")

        emoji = "🟢" if self.live else "🟡"
        log(
            f"  {emoji} BET {tf}m: {sig['direction']} | ΔBTC=${sig['delta']:+8.2f} | "
            f"poly={poly_price:.4f} | cost=${cost:.2f} | conf={sig['conf']:.2f} | "
            f"Bk=${self._balance:.2f}"
        )
        return trade

    # ── Reconcile open orders from CLOB ───────────────────────────────────────
    def _reconcile_open_orders(self):
        """Poll CLOB for open orders and resolve any that have been filled or market closed."""
        if not self.live or not self._client:
            return

        try:
            # v2 CLOB exposes live GTC orders via get_open_orders() with
            # status="LIVE" and id=<order hash>. The older v1 get_orders()
            # shape used status="open"/orderID, which made every live order
            # look absent and broke fill/cancel reconciliation.
            open_orders = self._client.get_open_orders() or []
        except Exception:
            return

        live_order_ids = {
            str(o.get("id") or o.get("orderID") or o.get("order_id") or "")
            for o in open_orders
            if isinstance(o, dict)
        }
        our_open_ids = set(self._open_orders.keys())

        # An order leaving the open list can mean filled OR cancelled.
        # Distinguish via on-chain matched size; only treat fills as positions.
        for oid in our_open_ids - live_order_ids:
            if oid in self._reconciled_orders:
                continue
            self._reconciled_orders.add(oid)
            info = self._open_orders.pop(oid, {})
            matched = self._verify_fill_size(oid)
            for t in self.trades:
                if t.order_id == oid:
                    t.fill_verified_at = self._now()
                    if matched > 0:
                        t.filled = True
                        t.filled_size = matched
                    break
            if matched <= 0:
                log(f"[RECONCILE] Order {oid[:16]}... CANCELLED (size_matched=0)")
                # Refund our local deployment counter — money never left.
                self._total_deployed -= info.get("spend", 0)
                continue
            log(f"[RECONCILE] Order {oid[:16]}... FILLED size={matched:.2f}")
            self._resolve_from_open_order(oid, info)
            self._refresh_balance_with_retry()

        # Recompute _total_deployed from ground truth to prevent accounting drift
        self._total_deployed = sum(
            i.get("spend", 0) for i in self._open_orders.values()
        )

    def _resolve_from_open_order(self, order_id: str, info: dict):
        """Handle a CLOB fill for an order we tracking.
        If the market window has closed, resolve it now with actual direction.
        Otherwise mark reconciled and let the window-expiry logic handle it.
        """
        if not info:
            return
        placed_at = info.get("placed_at", self._now())
        direction = info.get("direction", "Up")
        window_start = info.get("window_start")
        spend = info.get("spend", 0)
        poly_price = info.get("poly_price", 0.50)

        for t in self.trades:
            if t.order_id == order_id and not t.resolved:
                tf_sec = getattr(t, "window_tf", 5) * 60
                w_end = (window_start or t.window_start) + tf_sec
                if self._now() >= w_end + 5:
                    # ONLY resolve from Polymarket oracle — Binance diverges from PM resolution
                    market_id_for_res = t.market_id or info.get("market_id")
                    actual_dir = None
                    if market_id_for_res:
                        actual_dir = fetch_market_resolution(
                            market_id_for_res, t.direction
                        )
                    if actual_dir:
                        log(f"[RESOLVE] Polymarket oracle: {actual_dir}")
                        self._resolve_trade(t, actual_dir)
                    elif self._now() >= w_end + 600:
                        # Safety: if PM still hasn't resolved 10min after window, force loss
                        log(f"[RESOLVE] PM oracle timeout — forcing LOSS")
                        anti_dir = "Down" if t.direction == "Up" else "Up"
                        self._resolve_trade(t, anti_dir)
                    else:
                        log(
                            f"[RESOLVE] Waiting for PM oracle (market={market_id_for_res})"
                        )
                        self._pending_fills[order_id] = info
                        return
                else:
                    self._pending_fills[order_id] = info
                    log(f"[RECONCILE] Fill pending window close: {order_id[:16]}...")
                return

        # If the original Trade already exists and resolved, do not synthesize a
        # second trade from the same CLOB order. This was inflating live losses
        # in status after the pending-fill path resolved first and reconcile
        # later saw the same order again.
        if any(t.order_id == order_id for t in self.trades):
            self._pending_fills.pop(order_id, None)
            return

        if not window_start:
            return
        tf_sec = info.get("window_tf", 5) * 60
        if self._now() >= window_start + tf_sec + 5:
            # ONLY resolve from Polymarket oracle
            market_id_for_res = info.get("market_id")
            actual_dir = None
            if market_id_for_res:
                actual_dir = fetch_market_resolution(market_id_for_res, direction)
            if actual_dir:
                log(f"[RESOLVE] Polymarket oracle: {actual_dir}")
            elif self._now() >= window_start + tf_sec + 600:
                log(f"[RESOLVE] PM oracle timeout — forcing LOSS")
                actual_dir = "Down" if direction == "Up" else "Up"
            else:
                log(f"[RESOLVE] Waiting for PM oracle (market={market_id_for_res})")
                return
            won = direction == actual_dir
            pnl = (
                (spend * (1.0 / poly_price - 1) * (1 - TAKER_FEE_BPS / 10000))
                if won
                else -spend
            )
            trade = Trade(
                window_start=window_start,
                direction=direction,
                spend=spend,
                poly_price=poly_price,
                btc_delta=0,
                btc_price_enter=self.btc_price or 0,
                conf=0.5,
                reasons=["reconciled_fill"],
                placed_at=placed_at,
                order_id=order_id,
                token_id=info.get("token_id", ""),
                market_id=info.get("market_id"),
                resolved=True,
                won=won,
                pnl=pnl,
                resolved_at=self._now(),
                exit_reason="reconciled_fill",
                filled=True,
                filled_size=(spend / poly_price) if poly_price > 0 else 0.0,
                fill_verified_at=self._now(),
                entry_elapsed_sec=placed_at - window_start,
                seconds_left_at_entry=(window_start + tf_sec) - placed_at,
                journaled=True,
                window_tf=info.get("window_tf", 5),
            )
            self.trades.append(trade)
            self._total_deployed -= spend
            self.total_pnl += pnl
            if won:
                self.wins += 1
            else:
                self.losses += 1
            self._journal_trade(trade)
            self._analyse_trade(trade)
            if self.live:
                self._refresh_balance_with_retry()
                self.bankroll = self._balance
                self._halt_after_first_loss_if_needed("reconciled_fill")
                self._live_circuit_breaker("reconciled_fill")
            else:
                self._balance += pnl
                self.bankroll = self._balance
                try:
                    PAPER_BALANCE_FILE.write_text(json.dumps({"balance": self._balance}))
                except Exception:
                    pass
            log(
                f"[RECONCILE] Created+resolved trade {order_id[:16]}... won={won} pnl=${pnl:+.4f}"
            )

    # ── Resolve trade ─────────────────────────────────────────────────────────
    def _resolve_trade(self, t: Trade, actual_dir: str):
        # Live mode: an unfilled limit must NOT be booked as WIN/LOSS.
        # Final on-chain check before settling — prior session lost ~$11
        # while bot logged "4W 3L +$0.57" because it skipped this gate.
        if self.live and not t.filled and t.order_id:
            matched = self._verify_fill_size(t.order_id)
            if matched > 0:
                t.filled = True
                t.filled_size = matched
                t.fill_verified_at = self._now()

        if self.live and not t.filled:
            t.resolved = True
            t.won = False
            t.pnl = 0.0
            t.resolved_at = self._now()
            t.exit_reason = "unfilled"
            if self._client and t.order_id:
                try:
                    self._client.cancel_order(t.order_id)
                except Exception:
                    pass
            if t.order_id:
                self._pending_fills.pop(t.order_id, None)
            for oid, info in list(self._open_orders.items()):
                if info.get("order_id") == t.order_id or oid == t.order_id:
                    self._open_orders.pop(oid, None)
                    self._total_deployed -= info.get("spend", 0)
                    break
            self.unfilled += 1
            self._journal_trade(t)
            log(
                f"       ⚪ UNFILLED {getattr(t, 'window_tf', 5)}m: {t.direction} | "
                f"price={t.poly_price:.4f} | order never crossed — $0 booked"
            )
            return

        won = t.direction == actual_dir
        if won:
            pnl = t.spend * (1.0 / t.poly_price - 1) * (1 - TAKER_FEE_BPS / 10000)
        else:
            pnl = -t.spend

        t.resolved = True
        t.won = won
        t.pnl = pnl
        t.resolved_at = self._now()
        t.exit_reason = f"{'WIN' if won else 'LOSS'}"

        if self.live and self._client and t.order_id:
            self._client.cancel_order(t.order_id)
        if t.order_id:
            self._pending_fills.pop(t.order_id, None)

        for oid, info in list(self._open_orders.items()):
            if info.get("order_id") == t.order_id or oid == t.order_id:
                self._open_orders.pop(oid, None)
                self._total_deployed -= info.get("spend", 0)
                break

        self.total_pnl += pnl
        if won:
            self.wins += 1
        else:
            self.losses += 1

        self._journal_trade(t)
        self._analyse_trade(t)

        if self.live:
            self._refresh_balance_with_retry()
            self.bankroll = self._balance
        else:
            # Sim mode: track virtual bankroll
            self._balance += pnl
            self.bankroll = self._balance
            try:
                PAPER_BALANCE_FILE.write_text(json.dumps({"balance": self._balance}))
            except Exception:
                pass

        result_emoji = "🟢" if won else "🔴"
        log(
            f"       {result_emoji} RESOLVED {getattr(t, 'window_tf', 5)}m: {actual_dir} | "
            f"{'WIN' if won else 'LOSS'} ${pnl:+7.2f} | Bk=${self._balance:.2f}"
        )
        self._halt_after_first_loss_if_needed("_resolve_trade")
        self._live_circuit_breaker("_resolve_trade")

    # ── Per-trade analysis + adaptive param tuning ──────────────────────────
    def _analyse_trade(self, t: Trade):
        """Analyze trade outcome and record lessons. Slowly adapt params."""
        try:
            now = self._now()
            entry_elapsed = getattr(t, "entry_elapsed_sec", 0.0) or (
                (t.placed_at - t.window_start) if t.placed_at and t.window_start else 999
            )
            seconds_left = getattr(t, "seconds_left_at_entry", 0.0) or (
                (t.window_start + getattr(t, "window_tf", 5) * 60 - t.placed_at)
                if t.placed_at and t.window_start
                else 0.0
            )
            conditions = {
                "window_tf": getattr(t, "window_tf", 5),
                "delta": abs(t.btc_delta),
                "conf": t.conf,
                "direction": t.direction,
            }
            for item in t.reasons or []:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    k, v = item
                    if isinstance(v, (int, float)):
                        conditions[f"reason_{k}"] = v
            conditions["entry_elapsed_sec"] = round(entry_elapsed, 0)
            conditions["seconds_left_at_entry"] = round(seconds_left, 0)
            analysis = self.lessons.analyse(
                {
                    "direction": t.direction,
                    "won": t.won,
                    "btc_delta": t.btc_delta,
                    "conf": t.conf,
                    "conditions": conditions,
                }
            )
            self.lessons.record(t.direction, t.won, conditions)
            self.lessons.save()
            log(f"       📊 LESSON: {analysis}")

            trades_since_adapt = (self.wins + self.losses) - getattr(
                self, "_trades_at_last_adapt", 0
            )
            if trades_since_adapt >= 3 or now - self._adapt_cooldown > 180:
                self._adapt_cooldown = now
                self._trades_at_last_adapt = self.wins + self.losses
                self._adapt_params()
        except Exception as e:
            log(f"       [WARN] _analyse_trade failed: {e}")

    def _adapt_params(self):
        """Meta-harness style improvement:
        baseline → LLM propose → simulate → apply only if delta > 0.
        Uses journal + lessons as ground truth for simulation."""
        if self.live and not LIVE_ALLOW_PARAM_MUTATION:
            log("       🧬 SHADOW ONLY: live param mutation disabled; paper training must promote params")
            return
        try:
            improvement = self._meta_harness_improve()
            if improvement:
                log(f"       🧬 ADAPTED: {improvement}")
        except Exception as e:
            log(f"       [WARN] _adapt_params failed: {e}")

    def _meta_harness_improve(self) -> Optional[str]:
        """Run one meta-harness cycle. Returns description of what changed, or None."""
        recent = self._load_recent_trades(n=50)
        scored_recent = self._scored_trades(recent)
        if len(scored_recent) < max(5, PARAM_MIN_SAMPLE if self.live else 5):
            log(
                f"       🧬 Waiting for scored filled sample "
                f"({len(scored_recent)}/{max(5, PARAM_MIN_SAMPLE if self.live else 5)})"
            )
            return None

        baseline_score = self._score_trades(scored_recent)
        log(
            f"       🧬 META-HARNESS: baseline_score={baseline_score:.4f} from {len(scored_recent)} scored filled trades"
        )

        sugg = self.lessons.suggest_params()
        if not sugg and len(scored_recent) < 10:
            return None

        prompt = self._build_improvement_prompt(scored_recent, baseline_score, sugg)
        proposal = ai_complete(prompt, max_tokens=800)

        if not proposal:
            return None

        try:
            import re

            match = re.search(r"\{.*\}", proposal, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
        except Exception:
            log(f"       [WARN] Could not parse LLM improvement proposal")
            return None

        changes = data.get("changes", [])
        if not changes:
            return None

        change_desc = ", ".join(
            f"{c['param']}:{c['current']}→{c['suggested']}" for c in changes
        )
        proposed_params = copy.deepcopy(self.params)
        for c in changes:
            if hasattr(proposed_params, c["param"]):
                try:
                    val = float(c["suggested"])
                    setattr(proposed_params, c["param"], val)
                except (ValueError, TypeError):
                    pass

        simulated_trades = self._simulate_trades(scored_recent, proposed_params)
        proposed_score = self._score_trades(simulated_trades)
        delta = proposed_score - baseline_score

        log(f"       🧬 PROPOSED [{change_desc}]")
        log(
            f"       🧬 SIMULATED: baseline={baseline_score:.4f} proposed={proposed_score:.4f} delta={delta:+.4f}"
        )

        if delta > 0:
            old_vals = {c["param"]: getattr(self.params, c["param"]) for c in changes}
            for c in changes:
                if hasattr(self.params, c["param"]):
                    try:
                        setattr(self.params, c["param"], float(c["suggested"]))
                    except (ValueError, TypeError):
                        pass
            new_vals = {c["param"]: getattr(self.params, c["param"]) for c in changes}
            desc = ", ".join(f"{k}={old_vals[k]}→{new_vals[k]}" for k in old_vals)
            self._save_improvement(change_desc, baseline_score, proposed_score, delta)
            return f"{desc} (score {baseline_score:.3f}→{proposed_score:.3f} Δ+{delta:.4f})"
        else:
            log(f"       🧬 REJECTED: delta={delta:+.4f} ≤ 0 — keeping current params")
            return None

    def _load_recent_trades(self, n: int = 100) -> list[dict]:
        trades = []
        try:
            with open(JOURNAL_FILE) as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    trades.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
        return trades

    def _score_trades(self, trades: list[dict]) -> float:
        """Score = win rate * avg_pnl - avg_loss_rate * loss_ratio.
        Higher = better strategy."""
        trades = self._scored_trades(trades)
        if not trades:
            return 0.0
        wins = [t for t in trades if t.get("won")]
        losses = [t for t in trades if not t.get("won")]
        if not wins and not losses:
            return 0.0
        wr = len(wins) / max(len(wins) + len(losses), 1)
        avg_win = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
        avg_loss = abs(sum(t.get("pnl", 0) for t in losses)) / max(len(losses), 1)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        score = total_pnl * 10 + wr * 5 + min(len(trades), 50) * 0.1
        return score

    def _scored_trades(self, trades: list[dict]) -> list[dict]:
        """Return only economically meaningful outcomes for tuning/scoring.

        Live limit orders that never filled are execution labels, not strategy
        wins/losses. Counting them as losses made GA prefer nonsense mutations
        after tiny samples. Paper rows often lack a filled flag; treat those as
        filled unless they explicitly say unfilled.
        """
        out = []
        for t in trades:
            if not isinstance(t, dict):
                continue
            if t.get("exit_reason") == "unfilled":
                continue
            if t.get("filled") is False:
                continue
            if t.get("won") is True:
                out.append(t)
                continue
            try:
                pnl = float(t.get("pnl", 0) or 0)
            except Exception:
                pnl = 0.0
            if t.get("won") is False and pnl < 0:
                out.append(t)
        return out

    def _simulate_trades(self, trades: list[dict], params: SniperParams) -> list[dict]:
        """Apply new params to historical trades and compute outcomes.
        Re-evaluates whether each trade would have fired given the new params."""
        simulated = []
        for t in self._scored_trades(trades):
            delta = abs(t.get("btc_delta", 0))
            conf = t.get("conf", 0)
            if delta < params.delta_thresh:
                continue
            if conf < params.conf_thresh:
                continue
            sim = dict(t)
            simulated.append(sim)
        return simulated

    def _build_improvement_prompt(
        self, trades: list[dict], baseline_score: float, sugg: dict
    ) -> str:
        wins = [t for t in trades if t.get("won")]
        losses = [t for t in trades if not t.get("won")]
        avg_win = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
        avg_loss = abs(sum(t.get("pnl", 0) for t in losses)) / max(len(losses), 1)
        wr = len(wins) / max(len(wins) + len(losses), 1)
        current = self.params.to_dict()
        lesson_str = json.dumps(sugg, indent=2) if sugg else "No strong patterns yet"

        recent_sample = "\n".join(
            f"  {t.get('direction', '?')} {'WIN' if t.get('won') else 'LOSS'} "
            f"delta=${abs(t.get('btc_delta', 0)):.0f} conf={t.get('conf', 0):.2f} "
            f"pnl=${t.get('pnl', 0):+.2f} {t.get('exit_reason', '')}"
            for t in trades[-15:]
        )

        prompt = f"""You are a BTC Polymarket trading strategist. Analyse this trade journal and propose parameter improvements.

## Current Strategy Score
baseline_score = {baseline_score:.4f} (win_rate * pnl_weight + volume_bonus)

## Current Params
{json.dumps(current, indent=2)}

## Lesson Ledger (win rates by condition)
{lesson_str}

## Recent Trades (last 15)
{recent_sample}

## Summary
Total: {len(trades)} trades | WR: {wr:.0%} | avg_win: ${avg_win:.2f} | avg_loss: ${avg_loss:.2f}

## Your Task
Propose 1-2 specific param changes that would improve the score.
Consider: delta_thresh, conf_thresh, ens_thresh, spend_ratio.
Also consider: new filter rules (e.g., "only trade when OBI > 0.6", "skip when RSI > 70").

Return JSON:
{{
  "changes": [
    {{
      "param": "delta_thresh",
      "current": 12.0,
      "suggested": 15.0,
      "reason": "Historical trades with delta>=15 have 81% WR vs 79% for delta>=12"
    }}
  ]
}}"""
        return prompt

    def _save_improvement(
        self, desc: str, baseline: float, proposed: float, delta: float
    ):
        """Log improvement to fitness_history for tracking."""
        try:
            entry = {
                "ts": datetime.now().isoformat(),
                "change": desc,
                "baseline_score": baseline,
                "proposed_score": proposed,
                "delta": delta,
                "params": self.params.to_dict(),
            }
            with open(FITNESS_HISTORY, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self, duration: int = None):
        deadline = self._now() + (duration or 86400 * 30)  # 30 days default

        log(f"\n{'=' * 60}")
        log(f"BTC SNIPER PRO — {'LIVE TRADING' if self.live else 'PAPER SIM'}")
        log(f"{'=' * 60}")
        log(f"Bankroll: ${self.starting:.2f}")
        log(
            f"Delta>: ${self.params.delta_thresh:.0f} | Conf>=: {self.params.conf_thresh:.2f}"
        )
        log(f"Ens>:     {self.params.ens_thresh:.2f}")
        log(
            f"Spend:   {self.params.spend_ratio:.0%} | Max bet: {self.params.max_bet_pct:.0%}"
        )
        log(
            f"TP:      {self.params.profit_target_bps}bps | SL: {self.params.stop_loss_bps}bps"
        )
        log(f"{'=' * 60}\n")

        last_check = {5: 0, 15: 0}
        last_market = {5: 0, 15: 0}
        last_status = 0
        last_reconcile = 0
        last_window = {5: 0, 15: 0}
        last_prefetch = {5: 0, 15: 0}
        prefetched_markets: dict[int, dict[int, dict]] = defaultdict(dict)
        prefetch_seconds = float(os.environ.get("BTC_MARKET_PREFETCH_SECONDS", "150"))
        checks = 0
        sig_counts = defaultdict(int)

        # Pre-warm: load some BTC price history into each SignalEngine
        warmup_klines = get_binance_kline(interval="1m", limit=60)
        if warmup_klines:
            for k in warmup_klines:
                try:
                    price = float(k[4])
                    ts = float(k[0]) / 1000
                    for win in self.windows.values():
                        win.se.update(price, ts)
                except (ValueError, IndexError):
                    continue
            log(f"[WARM] Pre-loaded {len(warmup_klines)} candles into signal engines")

        while self.running and self._now() < deadline:
            now = self._now()
            try:
                btc_price = get_binance_btc()
                if btc_price:
                    self.btc_price = btc_price
                    for win in self.windows.values():
                        win.se.update(btc_price, now)

                # ── Per-timeframe window + market detection ──
                for tf in self._timeframes:
                    win = self.windows[tf]
                    win_sec = tf * 60
                    window_ts = int(now / win_sec) * win_sec
                    next_window_ts = window_ts + win_sec
                    time_to_next = next_window_ts - now
                    if (
                        0 < time_to_next <= prefetch_seconds
                        and prefetched_markets[tf].get(next_window_ts) is None
                        and now - last_prefetch.get(tf, 0) > 10
                    ):
                        m_next = fetch_btc_market_for_window(tf, next_window_ts)
                        last_prefetch[tf] = now
                        if m_next:
                            prefetched_markets[tf][next_window_ts] = m_next
                            log(
                                f"[PM {tf}m] Prefetched next market @{datetime.fromtimestamp(next_window_ts)} "
                                f"({time_to_next:.0f}s before start)"
                            )
                    if window_ts != last_window.get(tf, 0):
                        last_window[tf] = window_ts
                        if self.btc_price:
                            win.reset_window(window_ts, self.btc_price, {})
                            log(
                                f"\n[WIN {tf}m] @{datetime.fromtimestamp(window_ts)} price=${self.btc_price:.2f}"
                            )
                            if prefetched_markets[tf].get(window_ts):
                                self._attach_market(
                                    win,
                                    tf,
                                    prefetched_markets[tf].pop(window_ts),
                                    source="prefetch",
                                )
                            last_market[tf] = 0

                # ── Market fetching (per timeframe, staggered) ──
                for tf in self._timeframes:
                    win = self.windows[tf]
                    if win.market_id is None and now - last_market.get(tf, 0) > 10:
                        current_window = int(now / (tf * 60)) * (tf * 60)
                        mkt = fetch_btc_market_for_window(tf, current_window)
                        if mkt:
                            self._attach_market(win, tf, mkt, source="current")
                            last_market[tf] = int(now)

                # ── Resolve trades (per-window duration) ──
                for t in self.trades:
                    if t.resolved:
                        continue
                    tf_sec = getattr(t, "window_tf", 5) * 60
                    w_end = t.window_start + tf_sec
                    if now >= w_end + 5:
                        # ONLY use Polymarket oracle — never Binance
                        market_id_for_res = getattr(t, "market_id", None)
                        actual_dir = None
                        if market_id_for_res:
                            actual_dir = fetch_market_resolution(
                                market_id_for_res, t.direction
                            )
                        if actual_dir:
                            log(f"[RESOLVE] Polymarket oracle: {actual_dir}")
                            self._resolve_trade(t, actual_dir)
                            self.pause_until = now + 5
                        elif now >= w_end + 600:
                            # Safety timeout: if PM hasn't resolved 10min after window, force loss
                            log(f"[RESOLVE] PM oracle timeout — forcing LOSS")
                            anti_dir = "Down" if t.direction == "Up" else "Up"
                            self._resolve_trade(t, anti_dir)
                            self.pause_until = now + 5
                        # else: still waiting for PM oracle, skip

                # ── Resolve pending fills ──
                for oid, info in list(self._pending_fills.items()):
                    ws = info.get("window_start")
                    tf_sec = info.get("window_tf", 5) * 60
                    if ws and now >= ws + tf_sec + 5:
                        self._pending_fills.pop(oid, None)
                        self._resolve_from_open_order(oid, info)

                # ── CLOB reconciliation (only when we have open orders) ──
                if self._open_orders and now - last_reconcile >= 10:
                    self._reconcile_open_orders()
                    last_reconcile = now
                # Only refresh balance every 60s to avoid blocking the main loop
                if now - getattr(self, "_last_balance_refresh", 0) >= 60:
                    self._refresh_balance()  # use fast version, not retry
                    self._last_balance_refresh = now
                    if abs(self._balance - (self.bankroll or 0)) > 1.0:
                        log(
                            f"[CLOB] Balance drift: local={self.bankroll:.2f} clob={self._balance:.2f}"
                        )

                # ── Signal check + trade (per timeframe) ──
                for tf in self._timeframes:
                    win = self.windows[tf]
                    if win.market_id is None or win.traded_this_window:
                        continue
                    if now - last_check.get(tf, 0) < 1:
                        continue
                    last_check[tf] = int(now)
                    checks += 1
                    sig = self._check_signal(win, tf)
                    if sig:
                        sig_counts[sig["direction"]] += 1
                        trade = self._place_trade(sig, win, tf)
                        if trade:
                            trade.window_tf = tf
                            win.traded_this_window = True

                # ── Status log (every 30s) ──
                if now - last_status >= 30:
                    tt = self.wins + self.losses
                    wr = self.wins / tt if tt > 0 else 0
                    elapsed_h = (now - self.t0) / 3600
                    pnl_pct = (
                        (self._balance - self.starting) / max(self.starting, 1) * 100
                    )
                    sugg = self.lessons.suggest_params()
                    sugg_str = f" LESSONS:{len(sugg)}" if sugg else ""
                    log(
                        f"[{datetime.fromtimestamp(now).strftime('%H:%M:%S')}] elapsed={elapsed_h:.1f}h "
                        f"trades={tt}(W:{self.wins} L:{self.losses} U:{self.unfilled}) WR={wr:.0%} "
                        f"Bk=${self._balance:.2f}({pnl_pct:+.1f}%) "
                        f"BTC=${self.btc_price or 0:.0f} signals={dict(sig_counts)}{sugg_str}"
                    )
                    last_status = now

                # ── GA evolution (every 5 min if enough trades) ──
                if (self.wins + self.losses) >= 5 and now - getattr(
                    self, "_last_ga_evolve", 0
                ) >= 300:
                    self._last_ga_evolve = now
                    self._ga_evolve()

                time.sleep(0.5)
            except Exception as e:
                import traceback

                log(f"[ERROR] Main loop: {e}")
                log(f"[ERROR] Traceback: {traceback.format_exc()[-500:]}")
                time.sleep(1)

        self._report(checks, sig_counts)

    def _ga_evolve(self):
        """Background GA evolution — scores params against journal, switches if better."""
        import threading

        def bg():
            try:
                if self.live and not LIVE_ALLOW_PARAM_MUTATION:
                    log("[GA] SHADOW ONLY: live param mutation disabled; paper training must promote params")
                    return
                recent = []
                try:
                    with open(JOURNAL_FILE) as f:
                        lines = f.readlines()
                    for line in lines[-500:]:
                        try:
                            t = json.loads(line)
                            if isinstance(t, dict) and "btc_delta" in t:
                                recent.append(t)
                        except:
                            pass
                except:
                    pass

                recent = self._scored_trades(recent)
                wins = [t for t in recent if t.get("won")]
                losses = [t for t in recent if not t.get("won")]
                total = wins + losses
                min_sample = PARAM_MIN_SAMPLE if self.live else 5
                if len(total) < min_sample:
                    log(f"[GA] Waiting for more scored filled BTC trades ({len(total)}/{min_sample})")
                    return

                ga = SniperGA()
                ga.load_or_create()

                current_score = self._score_trades(total)
                log(
                    f"[GA] Current score={current_score:.2f} from {len(wins)}W/{len(losses)}L"
                )

                ga.best_score = current_score
                ga.best_params = copy.deepcopy(self.params)
                pop = [self.params] + [self.params.mutate(0.4) for _ in range(7)]

                results = []
                for p in pop:
                    s = self._score_params(p, total)
                    results.append((s, p))

                results.sort(key=lambda x: x[0], reverse=True)
                best_score, best_p = results[0]

                if best_score > current_score + 5:
                    log(
                        f"[GA] 🏆 BETTER: {best_p.name} score={best_score:.2f} > {current_score:.2f}"
                    )
                    log(
                        f"[GA]   delta={best_p.delta_thresh:.1f} conf={best_p.conf_thresh:.2f} spend={best_p.spend_ratio:.2f}"
                    )
                    self.params = copy.deepcopy(best_p)
                    ga.best_score = best_score
                    ga.best_params = copy.deepcopy(best_p)
                    ga.save()
                else:
                    log(
                        f"[GA] No improvement. Best={best_score:.2f} vs current={current_score:.2f}"
                    )
            except Exception as e:
                log(f"[GA] Evolution error: {e}")

        t = threading.Thread(target=bg, daemon=True)
        t.start()

    def _hour_adjust_conf(self, base_conf: float, direction: str, hour: int) -> float:
        """Adjust conf based on UTC hour directional bias from live journal data.

        Updated from 336-trade analysis (Apr 27 2026):
        Hard blocks: UTC 21 (both bad), UTC 05 (both bad), UTC 17 DOWN (10% WR), UTC 20 UP (14% WR)
        Penalties: UTC 12 UP (44% WR), UTC 04 DOWN (25% WR), UTC 13 DOWN (44% WR)
        Boosts (+0.20): UP at 08,09,10,11,15,17,18,19,22 | DOWN at 00,09,12,14,20,23
        """
        CONFLICT_DOWN = {17}
        CONFLICT_UP = {20}
        PENALIZE_UP = {12}
        PENALIZE_DOWN = {4, 13}  # 13 Down added: 44% WR from journal
        SKIP_ALL = {21, 5}

        if hour in SKIP_ALL:
            return 0.0

        if direction == "Down" and hour in CONFLICT_DOWN:
            return 0.0
        if direction == "Up" and hour in CONFLICT_UP:
            return 0.0

        penalty = 0.0
        if direction == "Up" and hour in PENALIZE_UP:
            penalty = 0.30
        if direction == "Down" and hour in PENALIZE_DOWN:
            penalty = 0.30

        boost = 0.0
        # Up boosts: 08(89%), 09(100%), 10(86%), 11(73%), 15(80%), 17(88%), 18(69%), 19(100%), 22(100%)
        if direction == "Up" and hour in {8, 9, 10, 11, 15, 17, 18, 19, 22}:
            boost = 0.20
        # Down boosts: 00(80%), 09(64% borderline), 12(90%), 14(88%), 20(100%), 23(80%)
        if direction == "Down" and hour in {0, 9, 12, 14, 20, 23}:
            boost = 0.20

        return max(0.0, base_conf - penalty + boost)

    def _score_params(self, params, trades):
        """Score a param set by simulating which trades it would have fired.

        Applies the same filtering as live trading: delta, conf, AND hour filter.
        """
        CONFLICT_DOWN = {17}
        CONFLICT_UP = {20}
        PENALIZE_UP = {12}
        PENALIZE_DOWN = {4, 13}  # synced with _hour_adjust_conf
        SKIP_ALL = {21, 5}

        fired = 0
        fired_wins = 0
        fired_pnl = 0.0
        for t in self._scored_trades(trades):
            delta = abs(t.get("btc_delta", 0))
            base_conf = t.get("conf", 0)
            if delta < params.delta_thresh:
                continue

            try:
                hr = int(t.get("placed_at", "00")[11:13])
            except:
                continue
            direction = t.get("direction", "?")
            if hr in SKIP_ALL:
                continue
            if direction == "Down" and hr in CONFLICT_DOWN:
                continue
            if direction == "Up" and hr in CONFLICT_UP:
                continue

            penalty = 0.0
            if direction == "Up" and hr in PENALIZE_UP:
                penalty = 0.30
            if direction == "Down" and hr in PENALIZE_DOWN:
                penalty = 0.30

            boost = 0.0
            if direction == "Up" and hr in {8, 9, 10, 11, 15, 17, 18, 19, 22}:
                boost = 0.20
            if direction == "Down" and hr in {0, 9, 12, 14, 20, 23}:
                boost = 0.20

            conf = max(0.0, base_conf - penalty + boost)
            if conf < params.conf_thresh:
                continue

            fired += 1
            if t.get("won"):
                fired_wins += 1
            fired_pnl += t.get("pnl", 0)

        min_fired = PARAM_MIN_SAMPLE if self.live else 3
        if fired < min_fired:
            return 0.0
        wr = fired_wins / fired
        return fired_pnl * 10 + wr * 20 - max(0, 10 - fired) * 0.5

    def _report(self, checks: int, sig_counts: dict):
        tt = self.wins + self.losses
        wr = self.wins / tt if tt > 0 else 0
        elapsed = self._now() - self.t0
        pnl_pct = (self._balance - self.starting) / self.starting * 100

        log(f"\n{'=' * 60}")
        log(f"FINAL RESULTS ({elapsed / 3600:.2f}h)")
        log(f"{'=' * 60}")
        log(f"Trades:   {tt}  W:{self.wins} L:{self.losses} Blocks:{self.blocks}")
        log(f"Win rate: {wr:.1%}")
        log(f"Start Bk: ${self.starting:.2f}")
        log(f"Final Bk: ${self._balance:.2f} ({pnl_pct:+.2f}%)")
        log(f"Total PnL: ${self.total_pnl:+.2f}")
        log(f"{'=' * 60}\n")

        result = {
            "mode": "live" if self.live else "paper",
            "duration": elapsed,
            "bankroll_start": self.starting,
            "bankroll_end": self.bankroll,
            "total_pnl": self.total_pnl,
            "pnl_pct": pnl_pct,
            "trades": tt,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": wr,
            "blocked": self.blocks,
            "params": self.params.to_dict(),
        }
        out = HARVEY_HOME / "tmp" / "sniper_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"→ Results: {out}")


# ── GA Optimization ────────────────────────────────────────────────────────────
class SniperGA:
    def __init__(self):
        self.best_params: Optional[SniperParams] = None
        self.best_score = float("-inf")
        self.generation = 0

    def load_or_create(self) -> list[SniperParams]:
        if BEST_PARAMS_FILE.exists():
            try:
                with open(BEST_PARAMS_FILE) as f:
                    d = json.load(f)
                loaded = SniperParams.from_dict(d)
                log(f"Loaded params: {loaded.name} (score={d.get('best_score', '?')})")
                self.best_params = loaded
                self.best_score = d.get("best_score", float("-inf"))
                self.generation = d.get("generation", 0)
                pop = [loaded]
                for _ in range(loaded.pop_size - 1):
                    pop.append(loaded.mutate(rate=0.35))
                return pop
            except Exception as e:
                log(f"Failed to load params: {e}")
        log(f"Creating new random population")
        return SniperParams.random_population(12)

    def evaluate(self, params: SniperParams, session_min: int = 30) -> dict:
        """Paper-trade with params for session_min minutes. Return score."""
        sniper = LiveSniper(params, live=False)
        sniper.run(duration=session_min * 60)

        tt = sniper.wins + sniper.losses
        wr = sniper.wins / tt if tt > 0 else 0
        pnl = sniper.total_pnl

        # Score: PnL + win rate bonus + trade count
        score = pnl * 100 + wr * 20 + min(tt, 20) * 0.5

        log(
            f"  {params.name}: PnL=${pnl:+.4f} WR={wr:.0%} trades={tt} score={score:.4f}"
        )
        return {"score": score, "pnl": pnl, "wr": wr, "trades": tt, "params": params}

    def evolve(self, population: list[SniperParams]) -> list[SniperParams]:
        """Run GA selection + breeding."""
        # Evaluate all
        results = []
        for p in population:
            r = self.evaluate(p, session_min=20)
            results.append(r)
            if r["score"] > self.best_score:
                self.best_score = r["score"]
                self.best_params = copy.deepcopy(p)
                log(f"  🏆 NEW BEST: {self.best_score:.4f} ({p.name})")

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        elite = [r["params"] for r in results[:3]]

        # Breed next generation
        next_pop = list(elite)
        while len(next_pop) < len(population):
            a, b = random.sample(list(zip(results, population)), 2)
            winner = a[1] if a[0]["score"] > b[0]["score"] else b[1]
            loser = b[1] if winner is a[1] else a[1]
            child = winner.crossover(winner, loser)
            child = child.mutate(rate=0.20)
            next_pop.append(child)

        # AI suggestion every generation
        ai_suggestion = self._ask_ai()
        if ai_suggestion:
            next_pop[-1] = ai_suggestion

        return next_pop

    def _ask_ai(self) -> Optional[SniperParams]:
        """Ask AI to analyze recent trades and suggest param improvements."""
        recent = []
        try:
            with open(JOURNAL_FILE) as f:
                lines = f.readlines()
            for line in lines[-50:]:
                try:
                    recent.append(json.loads(line))
                except:
                    pass
        except:
            return None

        if len(recent) < 5:
            return None

        wins = [t for t in recent if t.get("won")]
        losses = [t for t in recent if not t.get("won")]
        avg_win = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
        avg_loss = sum(t.get("pnl", 0) for t in losses) / max(len(losses), 1)

        prompt = f"""Analyze this BTC Polymarket sniper trading journal and suggest param improvements.

Recent trades ({len(recent)} total, {len(wins)}W/{len(losses)}L):
- Avg win: ${avg_win:+.4f}
- Avg loss: ${avg_loss:+.4f}

Sample trades (last 10):
{chr(10).join([f"  {t.get('direction')} {'WIN' if t.get('won') else 'LOSS'} pnl={t.get('pnl', 0):+.4f} {t.get('exit_reason', '')}" for t in recent[-10:]])}

Current params:
{json.dumps(self.best_params.to_dict() if self.best_params else {}, indent=2)}

Suggest 3 param changes. Return JSON: {{"suggestions": [{{"param": "...", "current": X, "suggested": Y, "reason": "..."}}]}}"""

        response = ai_complete(prompt, max_tokens=600)
        if not response:
            return None

        try:
            import re

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            suggestions = data.get("suggestions", [])
            if not suggestions or not self.best_params:
                return None

            new_params = self.best_params.mutate(rate=0.0)
            for s in suggestions[:2]:
                if s["param"] in new_params.to_dict():
                    val = s["suggested"]
                    if isinstance(val, (int, float)):
                        setattr(new_params, s["param"], val)
            new_params.name = f"ai_{self.best_params.name}_{suggestions[0]['param']}"
            log(f"  AI suggestion: {new_params.name}")
            return new_params
        except Exception as e:
            log(f"  AI parse error: {e}")
        return None

    def save(self):
        if not self.best_params:
            return
        state = {
            "generation": self.generation,
            "best_score": self.best_score,
            **self.best_params.to_dict(),
        }
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump(state, f, indent=2)
        log(f"Saved best params: {self.best_params.name} score={self.best_score:.4f}")




def assert_live_go_gate() -> None:
    """Hard block real-money mode unless the mechanical CLOB-only gate passes."""
    if LIVE_KILL_SWITCH_FILE.exists() and os.environ.get("BTC_LIVE_KILL_SWITCH_OVERRIDE") != "I_UNDERSTAND_REAL_MONEY_LOSS_RISK":
        reason = ""
        try:
            reason = json.loads(LIVE_KILL_SWITCH_FILE.read_text()).get("reason", "")
        except Exception:
            reason = LIVE_KILL_SWITCH_FILE.read_text(errors="replace").strip()[:240]
        suffix = f": {reason}" if reason else ""
        raise SystemExit(
            f"LIVE BLOCKED: kill switch active at {LIVE_KILL_SWITCH_FILE}{suffix}. "
            "Remove the file or set BTC_LIVE_KILL_SWITCH_OVERRIDE=I_UNDERSTAND_REAL_MONEY_LOSS_RISK after audit."
        )

    # Operator-override path: deliberate, distinct from the disabled-on-purpose
    # BTC_LIVE_DISABLE_GO_GATE trap. Requires a long magic string that the
    # operator must paste consciously, plus the canary ACK below. Used for
    # author-authorized $5-cap canary runs while gates are still NO_GO.
    if os.environ.get("BTC_LIVE_OPERATOR_OVERRIDE") == "AUTHOR_AUTHORIZED_CANARY_5USDC_1USDC_TICKET":
        log("LIVE GATE OVERRIDDEN by operator. Caps via BTC_MAX_* env vars must be set.")
        ack = os.environ.get("BTC_LIVE_CANARY_ACK")
        if ack != "I_ACCEPT_CANARY_RISK_MAX_5_USDC":
            raise SystemExit("LIVE BLOCKED: operator override requires BTC_LIVE_CANARY_ACK=I_ACCEPT_CANARY_RISK_MAX_5_USDC")
        return
    if os.environ.get("BTC_LIVE_DISABLE_GO_GATE", "0") == "1":
        raise SystemExit("LIVE BLOCKED: BTC_LIVE_DISABLE_GO_GATE override is disabled on purpose. Do not bypass gates.")
    gate = Path(__file__).with_name("btc_live_go_nogo.py")
    cmd = [sys.executable, str(gate), "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        log("LIVE BLOCKED: btc_live_go_nogo.py returned NO_GO")
        if result.stdout:
            log(result.stdout.strip()[-1200:])
        raise SystemExit("LIVE BLOCKED: paper CLOB-only GO/NO-GO gates failed")
    try:
        report = json.loads(result.stdout)
    except Exception as exc:
        raise SystemExit(f"LIVE BLOCKED: invalid GO/NO-GO report: {exc}")
    if report.get("verdict") != "GO_CANARY":
        raise SystemExit(f"LIVE BLOCKED: verdict={report.get('verdict')}")
    ack = os.environ.get("BTC_LIVE_CANARY_ACK")
    if ack != "I_ACCEPT_CANARY_RISK_MAX_5_USDC":
        raise SystemExit("LIVE BLOCKED: set BTC_LIVE_CANARY_ACK=I_ACCEPT_CANARY_RISK_MAX_5_USDC after reviewing GO_CANARY")
    log("LIVE CANARY GATE PASSED: $5 wallet cap / $1 order cap only")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="BTC Sniper Pro")
    parser.add_argument("--live", action="store_true", help="Use real money")
    parser.add_argument("--paper", action="store_true", help="Paper trading (default)")
    parser.add_argument("--evolve", action="store_true", help="Run GA evolution first")
    parser.add_argument(
        "--duration", type=int, default=None, help="Run duration in seconds"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Comma-separated timeframes to trade, e.g. '5' or '15' or '5,15'",
    )
    args = parser.parse_args()

    if args.timeframes:
        os.environ["BTC_TIMEFRAMES"] = args.timeframes

    live = args.live and not args.paper
    if live:
        assert_live_go_gate()

    ga = SniperGA()

    if args.evolve:
        log("\n### GA EVOLUTION MODE ###")
        population = ga.load_or_create()
        for gen in range(3):
            ga.generation = gen + 1
            log(f"\n--- Generation {gen + 1} ---")
            population = ga.evolve(population)
        ga.save()
        log("\nEvolution done. Best params saved.")
        return

    # Load best params or create new
    population = ga.load_or_create()
    params = ga.best_params or population[0]

    log(f"\nStarting sniper with params: {params.name}")
    sniper = LiveSniper(params, live=live)

    def stop_handler(sig, frame):
        log("STOP received — shutting down...")
        sniper.running = False
        if ga.best_score > ga.best_score:
            ga.best_params = sniper.params
            ga.save()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    sniper.run(duration=args.duration)
    ga.best_params = sniper.params
    if ga.best_score > 0:
        ga.save()
        log(f"Session best: {ga.best_params.name} score={ga.best_score:.2f}")


if __name__ == "__main__":
    main()
