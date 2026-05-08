#!/usr/local/Cellar/python@3.11/3.11.10/Frameworks/Python.framework/Versions/3.11/bin/python3.11
"""
BTC Sniper — Fast Paper Mode
============================
Paper trading with relaxed thresholds to generate MAXIMUM trades rapidly.
Fires on delta>=8 (79% WR historically) and conf>=0.4 to collect data fast.
Logs every trade to intraday_journal.jsonl for GA analysis.
"""

import argparse, copy, json, math, os, random, signal, sys, time, requests, threading
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from btc_fee_model import resolved_buy_pnl, taker_fee_usdc
from btc_prob_gate import BetDecision, ProbabilityGate  # Phase 3: probability edge gate
from btc_param_contract import (
    DEFAULT_DYNAMIC_PARAMS,
    DYNAMIC_PARAM_FIELDS,
    FIXED_EXECUTION_PARAMS,
    STRATEGY_VERSION,
    as_runtime_dict,
    clamp_dynamic_params,
    params_file_lock,
    read_params_file,
    write_params_file,
)

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.path.expanduser(os.environ.get(name, str(default))))


def _safe_slug(raw: str) -> str:
    out = []
    for ch in str(raw or ""):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "btc-agent"


def parse_timeframes(raw) -> list[int]:
    """Parse isolated agent timeframe config.

    Accepts "5", "15", "5,15", "5m,15m", [5], [5, 15].
    Anything else is a config error: BTC markets here are only 5m/15m.
    """
    if raw is None or raw == "":
        parts = ["5", "15"]
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]

    out: list[int] = []
    for part in parts:
        token = str(part).strip().lower().removesuffix("m")
        try:
            tf = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid BTC timeframe {part!r}; expected 5 or 15") from exc
        if tf not in (5, 15):
            raise ValueError(f"invalid BTC timeframe {tf}; expected 5 or 15")
        if tf not in out:
            out.append(tf)
    if not out:
        raise ValueError("at least one BTC timeframe required")
    return out


LOG_DIR = _env_path("BTC_LOG_DIR", DATA_DIR / "logs")
JOURNAL_FILE = _env_path("BTC_JOURNAL_FILE", STATE_DIR / "intraday_journal.jsonl")
DEFAULT_BEST_PARAMS_FILE = STATE_DIR / "sniper_best_params.json"
BEST_PARAMS_FILE = _env_path("BTC_BEST_PARAMS_FILE", DEFAULT_BEST_PARAMS_FILE)
PAPER_LOG_FILE = _env_path("BTC_PAPER_LOG_FILE", LOG_DIR / "btc_sniper_paper_fast.log")
PROB_MODEL_PATH = _env_path("BTC_PROB_MODEL_PATH", DATA_DIR / "model" / "btc_prob_model_current.pkl")
STRATEGY_NAME = os.environ.get("BTC_STRATEGY_NAME", "main")
AGENT_ID = os.environ.get("BTC_AGENT_ID", STRATEGY_NAME or "btc-combined")
ACTIVE_TIMEFRAMES = parse_timeframes(os.environ.get("BTC_TIMEFRAMES", "5,15"))
DISABLE_PARAM_RELOAD = os.environ.get("BTC_DISABLE_PARAM_RELOAD", "0") == "1"
PARAM_OVERRIDES_JSON = os.environ.get("BTC_PARAM_OVERRIDES_JSON", "")
INCLUDE_LIVE_TRAINING = os.environ.get("BTC_INCLUDE_LIVE_TRAINING", "0") == "1"
LIVE_TRADE_WEIGHT = float(os.environ.get("BTC_LIVE_TRADE_WEIGHT", "1.0"))
PAPER_CAPITAL = float(os.environ.get("BTC_PAPER_CAPITAL", "100.0"))
MIN_SPEND = float(os.environ.get("BTC_MIN_SPEND", "2.50"))
MAX_TRADE_COST = float(os.environ.get("BTC_MAX_TRADE_COST", "3.00"))
MAX_OPEN_BTC_TRADES = int(os.environ.get("BTC_MAX_OPEN_BTC_TRADES", "1"))
MAX_OPEN_BTC_TRADES_PER_TF = int(os.environ.get("BTC_MAX_OPEN_BTC_TRADES_PER_TF", "0"))
ALLOW_CROSS_TF_CORRELATED_OPEN = os.environ.get("BTC_ALLOW_CROSS_TF_CORRELATED_OPEN", "0") == "1"
_TOTAL_EXPOSURE_RAW = os.environ.get("BTC_MAX_TOTAL_BTC_EXPOSURE_PCT", "").strip()
MAX_TOTAL_BTC_EXPOSURE_PCT = float(_TOTAL_EXPOSURE_RAW) if _TOTAL_EXPOSURE_RAW else None
RISK_HALT_DRAWDOWN_PCT = float(os.environ.get("BTC_RISK_HALT_DRAWDOWN_PCT", "0.20"))
REQUIRE_CLOB_QUOTE = os.environ.get("BTC_REQUIRE_CLOB_QUOTE", "1") != "0"
LOOP_SLEEP_SECONDS = float(os.environ.get("BTC_LOOP_SLEEP_SECONDS", "0.5"))
MARKET_CHECK_SECONDS = float(os.environ.get("BTC_MARKET_CHECK_SECONDS", "5.0"))
STATUS_SECONDS = float(os.environ.get("BTC_STATUS_SECONDS", "30.0"))
PAPER_EXPLORATION_ENABLED = os.environ.get("BTC_PAPER_EXPLORATION", "0") == "1"
PAPER_EXPLORATION_ALLOW = os.environ.get("BTC_ALLOW_PAPER_EXPLORATION", "0") == "1"
PAPER_EXPLORATION_FORCE = os.environ.get("BTC_PAPER_EXPLORATION_FORCE", "0") == "1"
PAPER_EXPLORATION_MIN_EDGE = float(os.environ.get("BTC_EXPLORATION_MIN_EDGE", "0.02"))
PAPER_EXPLORATION_MIN_CONF = float(os.environ.get("BTC_EXPLORATION_MIN_CONF", "0.10"))
PAPER_EXPLORATION_MIN_POLY = float(os.environ.get("BTC_EXPLORATION_MIN_POLY", "0.25"))
PAPER_EXPLORATION_MAX_POLY = float(os.environ.get("BTC_EXPLORATION_MAX_POLY", "0.74"))
PAPER_EXPLORATION_MAX_SPEND = float(os.environ.get("BTC_EXPLORATION_MAX_SPEND", "2.50"))
PAPER_EXPLORATION_DELTA_FLOOR = float(os.environ.get("BTC_EXPLORATION_DELTA_FLOOR", "2.00"))
PAPER_EXPLORATION_MAX_SLIPPAGE_BPS = float(os.environ.get("BTC_EXPLORATION_MAX_SLIPPAGE_BPS", "250"))
_FORCE_DELTA_THRESHOLD_RAW = os.environ.get("BTC_FORCE_DELTA_THRESHOLD", "").strip()
FORCE_DELTA_THRESHOLD = (
    float(_FORCE_DELTA_THRESHOLD_RAW) if _FORCE_DELTA_THRESHOLD_RAW else None
)
DELTA_HINT_SIGNAL_ENABLED = os.environ.get("BTC_DELTA_HINT_SIGNAL", "0") == "1"
DELTA_HINT_MIN_CONF = float(os.environ.get("BTC_DELTA_HINT_MIN_CONF", "0.45"))
PROB_EDGE_THRESHOLD = float(os.environ.get("BTC_PROB_EDGE_THRESHOLD", "0.05"))
PROB_POLY_PRICE_CEILING = float(os.environ.get("BTC_PROB_POLY_PRICE_CEILING", "0.70"))
PROB_POLY_CEILING_EDGE_BUFFER = float(os.environ.get("BTC_PROB_POLY_CEILING_EDGE_BUFFER", "0.12"))
PROB_HARD_POLY_CAP = float(os.environ.get("BTC_PROB_HARD_POLY_CAP", "0.80"))
_MAX_CLOB_GAMMA_GAP_RAW = os.environ.get("BTC_MAX_CLOB_GAMMA_GAP", "0.12").strip()
MAX_CLOB_GAMMA_GAP = (
    float(_MAX_CLOB_GAMMA_GAP_RAW) if _MAX_CLOB_GAMMA_GAP_RAW else None
)
REQUIRE_EXEC_PRICE_WITHIN_GAMMA = os.environ.get("BTC_REQUIRE_EXEC_PRICE_WITHIN_GAMMA", "1") == "1"
CLOB_GAMMA_GAP_OVERRIDE_PROB = float(os.environ.get("BTC_CLOB_GAMMA_GAP_OVERRIDE_PROB", "0.75"))
ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE = os.environ.get("BTC_ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE", "0") == "1"
PROB_MIN_DIRECTION_PROB = float(os.environ.get("BTC_PROB_MIN_DIRECTION_PROB", "0.55"))
PROB_MIN_POLY_PRICE = float(os.environ.get("BTC_PROB_MIN_POLY_PRICE", "0.15"))
REQUIRE_PRE_EXEC_EDGE = os.environ.get("BTC_REQUIRE_PRE_EXEC_EDGE", "1") == "1"
PRE_EXEC_EDGE_THRESHOLD = float(os.environ.get("BTC_PRE_EXEC_EDGE_THRESHOLD", "0.03"))
REQUIRE_EXTERNAL_FLOW_AGREEMENT = os.environ.get("BTC_REQUIRE_EXTERNAL_FLOW_AGREEMENT", "1") == "1"
EXTERNAL_FLOW_SOFT_THRESHOLD = float(os.environ.get("BTC_EXTERNAL_FLOW_SOFT_THRESHOLD", "0.05"))
_EXTERNAL_BULL_SCORE_MIN_RAW = os.environ.get("BTC_EXTERNAL_BULL_SCORE_MIN", "").strip()
_EXTERNAL_BULL_SCORE_MAX_RAW = os.environ.get("BTC_EXTERNAL_BULL_SCORE_MAX", "").strip()
EXTERNAL_BULL_SCORE_MIN = (
    float(_EXTERNAL_BULL_SCORE_MIN_RAW) if _EXTERNAL_BULL_SCORE_MIN_RAW else None
)
EXTERNAL_BULL_SCORE_MAX = (
    float(_EXTERNAL_BULL_SCORE_MAX_RAW) if _EXTERNAL_BULL_SCORE_MAX_RAW else None
)
# PnL/fee model lives in btc_fee_model.py (Polymarket crypto taker-fee formula).
BINANCE_REST = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_BOOK = "https://api.exchange.coinbase.com/products/BTC-USD/book"
BITGET_SPOT_TICKERS = "https://api.bitget.com/api/v2/spot/market/tickers"
BITGET_SPOT_CANDLES = "https://api.bitget.com/api/v2/spot/market/candles"
BITGET_SPOT_ORDERBOOK = "https://api.bitget.com/api/v2/spot/market/orderbook"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
PAPER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if AGENT_ID and STRATEGY_NAME and AGENT_ID != STRATEGY_NAME:
        prefix = f"[{AGENT_ID}/{STRATEGY_NAME}] "
    elif AGENT_ID:
        prefix = f"[{AGENT_ID}] "
    elif STRATEGY_NAME:
        prefix = f"[{STRATEGY_NAME}] "
    else:
        prefix = ""
    line = f"[{ts}] {prefix}{msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        log_path = PAPER_LOG_FILE
        if log_path.exists() and log_path.stat().st_size > 5_000_000:
            os.replace(log_path, Path(str(log_path) + ".1"))
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        # Disk pressure must not kill the paper trader. Losing logs is better
        # than stopping a live paper experiment mid-window.
        pass


@dataclass
class SniperParams:
    version: str = "pro1.0"
    name: str = "paper_fast"
    delta_thresh: float = DEFAULT_DYNAMIC_PARAMS["delta_thresh"]
    conf_thresh: float = DEFAULT_DYNAMIC_PARAMS["conf_thresh"]
    ens_thresh: float = DEFAULT_DYNAMIC_PARAMS["ens_thresh"]
    spend_ratio: float = FIXED_EXECUTION_PARAMS["spend_ratio"]
    max_bet_pct: float = FIXED_EXECUTION_PARAMS["max_bet_pct"]
    max_hold_seconds: int = FIXED_EXECUTION_PARAMS["max_hold_seconds"]
    min_market_volume: int = FIXED_EXECUTION_PARAMS["min_market_volume"]
    max_spread_bps: int = FIXED_EXECUTION_PARAMS["max_spread_bps"]
    pop_size: int = FIXED_EXECUTION_PARAMS["pop_size"]

    def to_dict(self):
        return asdict(self)

    def from_dict(d):
        runtime = as_runtime_dict(d or {})
        return SniperParams(
            **{k: v for k, v in runtime.items() if k in asdict(SniperParams())}
        )

    def mutate(self, rate=0.4):
        import copy

        p = copy.deepcopy(self)
        p.name = f"gen_{int(time.time()) % 1000000}_{random.randint(1000, 9999)}"
        for attr in DYNAMIC_PARAM_FIELDS:
            if random.random() < rate:
                delta = getattr(p, attr) * 0.3
                setattr(p, attr, getattr(p, attr) + random.uniform(-delta, delta))
        clean = clamp_dynamic_params(p.to_dict(), mode="paper")
        for attr, val in clean.items():
            setattr(p, attr, val)
        return p


def apply_param_overrides(params: SniperParams) -> SniperParams:
    """Apply one-process paper-lab overrides after contract sanitization.

    This is intentionally env-scoped: production/live code still gets guarded by
    the frozen contract, while isolated fake-money workers can sweep strategy
    variants without fighting over the global best-params file.
    """
    if not PARAM_OVERRIDES_JSON:
        return params
    try:
        raw = json.loads(PARAM_OVERRIDES_JSON)
        if not isinstance(raw, dict):
            raise ValueError("BTC_PARAM_OVERRIDES_JSON must be a JSON object")
        valid = asdict(SniperParams()).keys()
        for key, value in raw.items():
            if key in valid:
                setattr(params, key, value)
        clean = clamp_dynamic_params(params.to_dict(), mode="paper")
        for key, value in clean.items():
            setattr(params, key, value)
        log(
            "[LAB] Overrides applied: "
            + ", ".join(f"{k}={getattr(params, k)}" for k in sorted(raw) if hasattr(params, k))
        )
    except Exception as e:
        log(f"[LAB] Override error: {e}")
    return params


# ── Data fetchers ──────────────────────────────────────────────────────────
def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _interval_seconds(interval: str) -> int:
    raw = str(interval).strip().lower()
    if raw.endswith("m"):
        return int(float(raw[:-1]) * 60)
    if raw.endswith("h"):
        return int(float(raw[:-1]) * 3600)
    return int(float(raw))


def _bitget_granularity(interval: str) -> str:
    seconds = _interval_seconds(interval)
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{max(1, seconds // 60)}min"


def get_btc_price():
    providers = (
        ("binance", BINANCE_REST, {}, lambda d: d.get("price")),
        ("coinbase", COINBASE_TICKER, {}, lambda d: d.get("price")),
        ("bitget_spot", BITGET_SPOT_TICKERS, {"symbol": "BTCUSDT"}, lambda d: (d.get("data") or [{}])[0].get("lastPr")),
    )
    for _, url, params, pick in providers:
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code != 200:
                continue
            px = _safe_float(pick(r.json()))
            if px and px > 0:
                return px
        except Exception:
            continue
    return None


def get_klines(interval="5m", limit=30):
    try:
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
        r = requests.get(BINANCE_KLINES, params=params, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"binance_klines_status={r.status_code}")
        data = r.json()
        closes = [float(k[4]) for k in data]
        highs = [float(k[2]) for k in data]
        lows = [float(k[3]) for k in data]
        vols = [float(k[5]) for k in data]
        return closes, highs, lows, vols
    except Exception:
        pass
    try:
        granularity = _interval_seconds(interval)
        r = requests.get(COINBASE_CANDLES, params={"granularity": granularity}, timeout=8)
        if r.status_code == 200:
            rows = sorted(r.json(), key=lambda k: int(k[0]))[-limit:]
            closes = [float(k[4]) for k in rows]
            highs = [float(k[2]) for k in rows]
            lows = [float(k[1]) for k in rows]
            vols = [float(k[5]) for k in rows]
            if closes:
                return closes, highs, lows, vols
    except Exception:
        pass
    try:
        r = requests.get(
            BITGET_SPOT_CANDLES,
            params={"symbol": "BTCUSDT", "granularity": _bitget_granularity(interval), "limit": str(limit)},
            timeout=8,
        )
        if r.status_code == 200:
            d = r.json()
            rows = d.get("data") or []
            rows = sorted(rows, key=lambda k: int(k[0]))[-limit:]
            closes = [float(k[4]) for k in rows]
            highs = [float(k[2]) for k in rows]
            lows = [float(k[3]) for k in rows]
            vols = [float(k[5]) for k in rows]
            if closes:
                return closes, highs, lows, vols
    except Exception:
        pass
    return [None] * limit, [None] * limit, [None] * limit, [None] * limit


def get_orderbook_depth(limit=20):
    """Spot BTC order-book imbalance with Binance→Coinbase→Bitget fallback."""
    provider_specs = (
        (
            "binance",
            "https://api.binance.com/api/v3/depth",
            {"symbol": "BTCUSDT", "limit": limit},
            lambda d: (d.get("bids", []), d.get("asks", [])),
        ),
        (
            "coinbase",
            COINBASE_BOOK,
            {"level": 2},
            lambda d: (d.get("bids", [])[:limit], d.get("asks", [])[:limit]),
        ),
        (
            "bitget_spot",
            BITGET_SPOT_ORDERBOOK,
            {"symbol": "BTCUSDT", "type": "step0", "limit": str(limit)},
            lambda d: ((d.get("data") or {}).get("bids", []), (d.get("data") or {}).get("asks", [])),
        ),
    )
    for _, url, params, pick in provider_specs:
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code != 200:
                continue
            raw_bids, raw_asks = pick(r.json())
            bids = [(float(p), float(q)) for p, q, *_ in raw_bids[:limit]]
            asks = [(float(p), float(q)) for p, q, *_ in raw_asks[:limit]]
            sum_b = sum(q for _, q in bids)
            sum_a = sum(q for _, q in asks)
            if sum_b + sum_a <= 0:
                continue
            imbalance = (sum_b - sum_a) / (sum_b + sum_a + 1e-9)
            return {"imbalance": imbalance, "bids": bids, "asks": asks}
        except Exception:
            continue
    return {"imbalance": 0.0, "bids": [], "asks": []}


def get_btc_now():
    now_ts = int(time.time())
    window_sec = 300
    current_window = (now_ts // window_sec) * window_sec
    return current_window, get_btc_price()


def fetch_btc_markets(tf_minutes=5):
    now_ts = int(time.time())
    window_sec = tf_minutes * 60
    current_window = (now_ts // window_sec) * window_sec
    slug_prefix = f"btc-updown-{tf_minutes}m"
    for offset in [0, 1, 2]:
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
        except:
            continue
    return None


def _json_list(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else default
        except Exception:
            return default
    return default


def clob_token_id_for_direction(market, direction):
    """Return the CLOB token id for a BTC Up/Down market direction."""
    outcomes = [str(x).lower() for x in _json_list(market.get("outcomes"), ["Up", "Down"])]
    token_ids = [str(x) for x in _json_list(market.get("clobTokenIds"), [])]
    if not token_ids:
        return None
    wanted = str(direction).lower()
    for idx, outcome in enumerate(outcomes):
        if outcome == wanted and idx < len(token_ids):
            return token_ids[idx]
    # BTC up/down markets are conventionally [Up, Down].
    idx = 0 if wanted == "up" else 1
    return token_ids[idx] if idx < len(token_ids) else None


def fetch_clob_buy_quote(token_id, target_spend, min_size=5.0, max_spend=None):
    """Simulate a marketable BUY using public CLOB ask depth.

    Paper mode must pay executable ask/depth, not Gamma midpoint/outcomePrice.
    This is read-only: only GET /book is used. No wallet. No order POST.
    """
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
        # API returns asks high->low; executable best ask is the lowest ask.
        asks.sort(key=lambda x: x[0])
        best_ask = asks[0][0]
        target_size = max(float(min_size), float(target_spend) / best_ask)
        cap = float(max_spend if max_spend is not None else max(float(target_spend), target_size * best_ask))
        filled = 0.0
        cost = 0.0
        levels_used = 0
        for price, available in asks:
            if filled >= target_size:
                break
            take = min(available, target_size - filled)
            if cost + take * price > cap:
                take = max(0.0, (cap - cost) / price)
            if take <= 0:
                break
            filled += take
            cost += take * price
            levels_used += 1
        if filled + 1e-9 < float(min_size):
            return None
        if filled <= 0 or cost <= 0:
            return None
        avg_price = cost / filled
        return {
            "price": avg_price,
            "size": filled,
            "cost": cost,
            "best_ask": best_ask,
            "levels_used": levels_used,
            "slippage_bps": ((avg_price - best_ask) / best_ask * 10_000) if best_ask else 0.0,
        }
    except Exception:
        return None


def fetch_market_resolution(market_id):
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
            except:
                pass
        return None
    except:
        return None


# ── Signal Engine ──────────────────────────────────────────────────────────
class SignalEngine:
    def __init__(self):
        self.ph = []
        self.btc_price = 0
        self.prices_5m = [None] * 30
        self.highs_5m = [None] * 30
        self.lows_5m = [None] * 30
        self.vols_5m = [None] * 30
        self.prices_1m = [None] * 20

    def update(self, btc_price):
        self.btc_price = btc_price
        now_ts = int(time.time())
        self.ph.append((now_ts, btc_price))
        if len(self.ph) > 120:
            self.ph = self.ph[-120:]
        closes, highs, lows, vols = get_klines("5m", 30)
        if closes[0] is not None:
            self.prices_5m = closes
            self.highs_5m = highs
            self.lows_5m = lows
            self.vols_5m = vols
        closes_1m, _, _, _ = get_klines("1m", 20)
        if closes_1m[0] is not None:
            self.prices_1m = closes_1m

    def ensemble(self, ens_thresh, window_delta):
        if len(self.ph) < 20:
            return {
                "direction": "Neutral",
                "conf": 0.0,
                "reasons": [],
                "conditions": {},
            }
        conf_ups, conf_downs = [], []
        reasons = []

        # Window delta (primary)
        if window_delta >= 15:
            conf_ups.append(0.80)
        elif window_delta >= 10:
            conf_ups.append(0.65)
        elif window_delta >= 8:
            conf_ups.append(0.55)
        elif window_delta >= 5:
            conf_ups.append(0.30)
        elif window_delta >= 3:
            conf_ups.append(0.15)
        elif window_delta >= 1:
            conf_ups.append(0.05)
        if window_delta <= -15:
            conf_downs.append(0.80)
        elif window_delta <= -10:
            conf_downs.append(0.65)
        elif window_delta <= -8:
            conf_downs.append(0.55)
        elif window_delta <= -5:
            conf_downs.append(0.30)
        elif window_delta <= -3:
            conf_downs.append(0.15)
        elif window_delta <= -1:
            conf_downs.append(0.05)

        # RSI 14
        if len(self.prices_5m) >= 15:
            deltas = [
                self.prices_5m[i] - self.prices_5m[i - 1]
                for i in range(1, len(self.prices_5m))
            ]
            gains = [d for d in deltas[-14:] if d > 0]
            losses = [-d for d in deltas[-14:] if d < 0]
            ag = sum(gains) / 14 if gains else 0
            al = sum(losses) / 14 if losses else 1e-9
            rs = ag / al
            rsi = 100 - (100 / (1 + rs))
            if rsi < 35 and conf_ups:
                conf_ups.append(0.10)
            elif rsi > 65 and conf_downs:
                conf_downs.append(0.10)

        # MACD (12, 26, 9)
        if len(self.prices_5m) >= 26:
            ema12 = self._ema(self.prices_5m, 12)
            ema26 = self._ema(self.prices_5m, 26)
            macd = ema12 - ema26
            signal_line = self._ema([macd] * len(self.prices_5m[-9:]), 9)
            if macd > signal_line and conf_ups:
                conf_ups.append(0.08)
            elif macd < signal_line and conf_downs:
                conf_downs.append(0.08)

        total_conf = max(sum(conf_ups), sum(conf_downs))
        direction = (
            "Up"
            if sum(conf_ups) > sum(conf_downs)
            else "Down"
            if conf_downs
            else "Neutral"
        )
        return {
            "direction": direction,
            "conf": total_conf,
            "reasons": reasons,
            "conditions": {"window_delta": window_delta},
        }

    def _ema(self, data, n):
        if len(data) < n:
            return data[-1] if data else 0
        k = 2 / (n + 1)
        ema = sum(data[:n]) / n
        for v in data[n:]:
            ema = v * k + ema * (1 - k)
        return ema


def journal_trade_pnl(t):
    try:
        price = float(t.get("poly_price", 0.5) or 0.5)
        shares = float(t.get("size") or 0.0)
        if shares <= 0:
            spend = float(t.get("spend", 0.0) or 0.0)
            shares = spend / price if price > 0 else 0.0
        return resolved_buy_pnl(bool(t.get("won")), shares, price)
    except Exception:
        return float(t.get("pnl", 0.0) or 0.0)


# ── Replay parity for FastGA ───────────────────────────────────────────────
# Keep this aligned with btc_backtest_autoresearch.py and the runtime signal gate.
# Previous FastGA scoring only checked delta+conf, so ens_thresh mutations were
# effectively random and the in-process GA could overwrite better external
# backtest params. Paper optimization must score the exact gate we execute.
HOUR_CONFLICT_DOWN = {17}
HOUR_CONFLICT_UP = {20}
HOUR_PENALIZE_UP = {12}
HOUR_PENALIZE_DOWN = {4, 13}
HOUR_SKIP_ALL = {21, 5}
HOUR_BOOST_UP = {8, 9, 10, 11, 15, 17, 18, 19, 22}
HOUR_BOOST_DOWN = {0, 9, 12, 14, 20, 23}


def _hour_adjust_conf(base_conf: float, direction: str, hour: int) -> float:
    if hour in HOUR_SKIP_ALL:
        return 0.0
    if direction == "Down" and hour in HOUR_CONFLICT_DOWN:
        return 0.0
    if direction == "Up" and hour in HOUR_CONFLICT_UP:
        return 0.0

    penalty = 0.0
    if direction == "Up" and hour in HOUR_PENALIZE_UP:
        penalty = 0.30
    if direction == "Down" and hour in HOUR_PENALIZE_DOWN:
        penalty = 0.30

    boost = 0.0
    if direction == "Up" and hour in HOUR_BOOST_UP:
        boost = 0.20
    if direction == "Down" and hour in HOUR_BOOST_DOWN:
        boost = 0.20
    return max(0.0, base_conf - penalty + boost)


def fast_ga_would_fire(trade: dict, params: SniperParams) -> bool:
    try:
        btc_delta = abs(float(trade.get("btc_delta", 0) or 0))
        conf = float(trade.get("conf", 0) or 0)
        poly_price = float(trade.get("poly_price", 0.5) or 0.5)
    except Exception:
        return False

    direction = trade.get("direction", "Up")
    placed = str(trade.get("placed_at", ""))
    try:
        hour = int(placed[11:13]) if len(placed) >= 13 else 0
    except Exception:
        hour = 0
    conf = _hour_adjust_conf(conf, direction, hour)

    poly_conv = abs(poly_price - 0.5) * 2
    if poly_conv >= 0.90:
        return conf >= params.ens_thresh
    if btc_delta < params.delta_thresh:
        return False
    if conf < params.conf_thresh:
        return False
    if conf < params.ens_thresh:
        return False
    return True


# ── Fast GA ────────────────────────────────────────────────────────────────
class FastGA:
    def __init__(self, timeframes=None, include_live: bool | None = None, live_weight: float | None = None):
        self.best_params = SniperParams()
        self.best_score = float("-inf")
        self.population = []
        self.timeframes = set(parse_timeframes(timeframes or ACTIVE_TIMEFRAMES))
        self.include_live = INCLUDE_LIVE_TRAINING if include_live is None else bool(include_live)
        self.live_weight = LIVE_TRADE_WEIGHT if live_weight is None else float(live_weight)

    def run(self, journal_file, generations=20, session_min=5):
        log("### FAST GA STARTING ###")
        # Load resolved paper trades by default; optionally include resolved live fills.
        trades = self._load_trades(journal_file)
        mode_counts = Counter(t.get("mode") or "unknown" for t in trades)
        mode_summary = ", ".join(f"{mode}={count}" for mode, count in sorted(mode_counts.items())) or "none"
        log(f"Loaded {len(trades)} trades for GA ({mode_summary}; include_live={self.include_live})")

        # Start with proven good params
        base = SniperParams(
            delta_thresh=10.0,
            conf_thresh=0.45,
            ens_thresh=0.50,
            spend_ratio=0.30,
            name="ga_baseline",
        )
        self.population = [base] + [base.mutate(0.5) for _ in range(19)]

        for gen in range(generations):
            results = []
            for p in self.population:
                score = self._score_params(p, trades)
                results.append((score, p))
                log(
                    f"  Gen{gen + 1} {p.name}: score={score:.2f} delta={p.delta_thresh:.1f} conf={p.conf_thresh:.2f}"
                )

            results.sort(key=lambda x: x[0], reverse=True)
            best_score, best_p = results[0]
            log(
                f"Gen {gen + 1}/{generations}: BEST {best_p.name} score={best_score:.2f} delta={best_p.delta_thresh:.1f} conf={best_p.conf_thresh:.2f}"
            )

            if best_score > self.best_score:
                self.best_score = best_score
                self.best_params = copy.deepcopy(best_p)
                self.best_params.name = f"ga_best_gen{gen + 1}"
                self._save()
                log(f"  🏆 NEW BEST: score={best_score:.2f}")

            # Selection + mutation
            elite = [p for _, p in results[:5]]
            next_pop = list(elite)
            while len(next_pop) < 20:
                parent = random.choice(elite)
                child = parent.mutate(rate=0.5)
                next_pop.append(child)
            self.population = next_pop[:20]

            trades = self._load_trades(journal_file)  # Refresh with new trades

        log(
            f"### FAST GA DONE. Best: delta={self.best_params.delta_thresh:.1f} conf={self.best_params.conf_thresh:.2f} score={self.best_score:.2f}"
        )

    def _load_trades(self, journal_file):
        trades = []
        try:
            with open(journal_file) as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except:
                        pass
        except:
            pass
        out = []
        allowed_modes = {"paper"}
        if self.include_live:
            allowed_modes.add("live")
        for t in trades:
            if t.get("mode") not in allowed_modes:
                continue
            if t.get("won") is None or t.get("pnl") is None:
                continue
            try:
                tf = int(t.get("window_tf") or 0)
            except Exception:
                tf = 0
            if self.timeframes and tf not in self.timeframes:
                continue
            out.append(t)
        return out

    def _trade_weight(self, trade: dict) -> float:
        if trade.get("mode") == "live":
            return max(0.0, float(self.live_weight))
        return 1.0

    def _score_params(self, params, trades):
        if not trades:
            return 0.0
        eligible = [t for t in trades if fast_ga_would_fire(t, params)]
        if not eligible:
            return -50.0
        weights = [self._trade_weight(t) for t in eligible]
        sample_n = sum(weights)
        if sample_n <= 0:
            return -50.0
        wins = sum(w for t, w in zip(eligible, weights) if t.get("won"))
        wr = wins / sample_n
        pnl = sum(journal_trade_pnl(t) * w for t, w in zip(eligible, weights))
        pnls = [journal_trade_pnl(t) * w for t, w in zip(eligible, weights)]
        mean_pnl = sum(pnls) / len(pnls)
        if len(pnls) > 1:
            var = sum((x - mean_pnl) ** 2 for x in pnls) / (len(pnls) - 1)
            std_pnl = var ** 0.5 or 1e-9
        else:
            std_pnl = 1e-9
        sharpe_like = mean_pnl / std_pnl * (min(len(eligible), 50) ** 0.5)

        # Same shape as btc_backtest_autoresearch: money first, WR quality,
        # per-trade normalization, and a hard penalty for low sample/low WR.
        penalty = 0.0
        if sample_n < 15:
            penalty += (15 - sample_n) * 5.0
        if wr < 0.60:
            penalty += 200.0
        score = (
            pnl * 0.5
            + wr * 30.0
            + sharpe_like * 5.0
            + math.log(1 + sample_n) * 2.0
            - penalty
        )
        return score

    def _save(self):
        try:
            d = self.best_params.to_dict()
            d.update({
                "best_score": self.best_score,
                "generation": int(time.time()),
                "source": f"btc_paper_fast.FastGA:{AGENT_ID}:{','.join(str(x) for x in sorted(self.timeframes))}m",
                "strategy_version": STRATEGY_VERSION,
            })
            with params_file_lock(BEST_PARAMS_FILE):
                if BEST_PARAMS_FILE.exists():
                    existing = read_params_file(BEST_PARAMS_FILE, mode="paper")
                    existing_score = self._score_params(SniperParams.from_dict(existing), self._load_trades(JOURNAL_FILE))
                    if existing_score > self.best_score + 0.01:
                        log(f"GA skip save: existing score={existing_score:.2f} > candidate={self.best_score:.2f}")
                        return
                write_params_file(BEST_PARAMS_FILE, d, mode="paper")
        except Exception as e:
            log(f"GA save error: {e}")


# ── Paper Trader ────────────────────────────────────────────────────────────
class PaperTrader:
    def __init__(self, params=None, timeframes=None, agent_id: str | None = None):
        self.params = params or SniperParams()
        self.agent_id = agent_id or AGENT_ID
        self.timeframes = parse_timeframes(timeframes or ACTIVE_TIMEFRAMES)
        self.live = False
        self.bankroll = PAPER_CAPITAL
        self._balance = PAPER_CAPITAL
        self.starting = PAPER_CAPITAL
        self.peak_equity = PAPER_CAPITAL
        self.risk_halted = False
        self.risk_halt_reason = None
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.blocks = 0
        self.t0 = time.time()
        self.se_5m = SignalEngine()
        self.se_15m = SignalEngine()
        # Phase 3: ProbabilityGate — replaces broken ensemble confidence gate
        # Falls back gracefully if model is not yet trained.
        self.prob_gate = ProbabilityGate(
            model_path=PROB_MODEL_PATH,
            edge_threshold=PROB_EDGE_THRESHOLD,
            poly_price_ceiling=PROB_POLY_PRICE_CEILING,
            poly_ceiling_edge_buffer=PROB_POLY_CEILING_EDGE_BUFFER,
            hard_poly_cap=PROB_HARD_POLY_CAP,
        )
        self.windows = {
            tf: {"start": 0, "price": 0, "traded": False, "market_id": None, "market": None}
            for tf in self.timeframes
        }
        self.trades = []
        self.running = True
        self._params_generation = None
        self._last_skip_log = {}
        self.telemetry = defaultdict(int)

    def _reload_params_from_best(self):
        if DISABLE_PARAM_RELOAD:
            return
        try:
            if not BEST_PARAMS_FILE.exists():
                return
            d = read_params_file(BEST_PARAMS_FILE, mode="paper")
            gen = d.get("generation")
            if gen == self._params_generation:
                return
            candidate = SniperParams.from_dict(d)
            self.params = apply_param_overrides(candidate)
            self._params_generation = gen
            log(
                f"[GA] Loaded params: delta={self.params.delta_thresh:.1f} conf={self.params.conf_thresh:.2f} ens={self.params.ens_thresh:.2f}"
            )
        except Exception as e:
            log(f"[GA] Reload error: {e}")

    def run(self, duration=28800):
        log(
            f"Starting PAPER sniper agent={self.agent_id} timeframes={','.join(str(tf)+'m' for tf in self.timeframes)} "
            f"delta>={self.params.delta_thresh}, conf>={self.params.conf_thresh}"
        )
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "running", False))
        last_market_check = {tf: 0 for tf in self.timeframes}
        last_status = 0
        last_btc = 0
        last_param_reload = 0

        deadline = self.t0 + duration
        while self.running and time.time() < deadline:
            now = time.time()
            if now - last_param_reload >= 60:
                self._reload_params_from_best()
                last_param_reload = now
            btc = get_btc_price()
            if btc is None:
                time.sleep(1)
                continue

            if abs(btc - last_btc) > 0.5:
                self.se_5m.update(btc)
                self.se_15m.update(btc)
                last_btc = btc

            # Check signals every second for each configured timeframe.
            # Split-agent mode runs exactly one tf per process; combined mode
            # still supports [5, 15] for backward compatibility.
            for tf in self.timeframes:
                if now - last_market_check.get(tf, 0) < MARKET_CHECK_SECONDS:
                    continue
                win = self.windows[tf]

                # Refetch market
                mkt = fetch_btc_markets(tf)
                if mkt:
                    win["market"] = mkt
                    from datetime import datetime as dt_cls

                    ts_str = mkt.get("endDate", "") or mkt.get("endDate_iso", "")
                    try:
                        dt = dt_cls.fromisoformat(ts_str.replace("Z", "+00:00"))
                        win_end = int(dt.timestamp())
                    except:
                        win_end = None

                    # Calculate current tf-aligned window start (UTC)
                    cur_win_start = (now // (tf * 60)) * (tf * 60)

                    # Only init/reset price once per actual tf-minute window boundary
                    if not win["start"] or win["start"] != cur_win_start:
                        win["start"] = cur_win_start
                        win["price"] = btc
                        win["traded"] = False
                        win["market_id"] = mkt.get("id")
                        win["market"] = mkt

                    # Check trade
                    window_delta = btc - win["price"]
                    if (
                        abs(window_delta) >= self._active_delta_threshold()
                        and not win["traded"]
                        and win["market_id"]
                        and (deadline - now) > (tf * 60 + 120)
                    ):
                        engine = self.se_15m if tf == 15 else self.se_5m
                        sig = engine.ensemble(self.params.ens_thresh, window_delta)
                        if sig["direction"] == "Neutral":
                            delta_hint_sig = self._delta_hint_signal(window_delta)
                            if delta_hint_sig:
                                sig = delta_hint_sig
                            else:
                                explore_sig = self._neutral_exploration_signal(window_delta)
                                if explore_sig:
                                    sig = explore_sig
                        if sig["direction"] != "Neutral":
                            # Phase 3: ProbabilityGate — edge check replaces conf_thresh
                            prob_feats = self._build_prob_features(sig, win, tf, btc)
                            prob_decision = self.prob_gate.evaluate(
                                prob_feats, direction_hint=sig["direction"]
                            )
                            exploration_decision = None
                            if not prob_decision.should_bet:
                                exploration_decision = self._exploration_decision(
                                    sig, prob_feats, prob_decision
                                )
                            if not prob_decision.should_bet and not (
                                exploration_decision and exploration_decision.should_bet
                            ):
                                self._log_skip(
                                    f"prob_gate:{tf}:{sig['direction']}",
                                    f"  ⏸️ {prob_decision.summary}"
                                )
                            else:
                                if exploration_decision and exploration_decision.should_bet:
                                    prob_decision = exploration_decision
                                    sig["_exploration_mode"] = True
                                    sig["_exploration_reason"] = prob_decision.gate_reason
                                sig["_prob_features"] = prob_feats
                                sig["_prob_decision"] = {
                                    "model_prob": prob_decision.model_prob,
                                    "market_prob": prob_decision.market_prob,
                                    "edge": prob_decision.edge,
                                    "prob_up": prob_decision.prob_up,
                                    "prob_down": prob_decision.prob_down,
                                    "gate_reason": prob_decision.gate_reason,
                                }
                                placed = self._place_trade(sig, win, tf, btc)
                                if placed is not None:
                                    win["traded"] = True
                                    log_decision = placed.get("prob_decision") or sig.get("_prob_decision", {})
                                    log_model_prob = float(log_decision.get("model_prob", prob_decision.model_prob))
                                    log_market_prob = float(log_decision.get("market_prob", prob_decision.market_prob))
                                    log_edge = float(log_decision.get("edge", prob_decision.edge))
                                    log(
                                        f"  {'🟣 PAPER EXPLORE' if sig.get('_exploration_mode') else '🟢 BET'} {tf}m: {prob_decision.direction} "
                                        f"ΔBTC={window_delta:+.0f} model={log_model_prob:.1%} "
                                        f"poly={log_market_prob:.1%} edge={log_edge:+.1%} "
                                        f"cost=${placed['spend']:.2f} px={placed.get('poly_price', 0):.3f}"
                                    )
                        else:
                            self._log_skip(
                                f"neutral:{tf}",
                                f"  ⏸️ NEUTRAL signal, no bet"
                            )

                last_market_check[tf] = now

            # Resolve trades
            resolved = []
            for t in self.trades:
                if t.get("resolved"):
                    continue
                tf_sec = t["window_tf"] * 60
                if now >= t["window_start"] + tf_sec + 10:
                    actual = (
                        fetch_market_resolution(t["market_id"])
                        if t.get("market_id")
                        else None
                    )
                    if actual:
                        won = actual == t["direction"]
                        pnl = self._calc_pnl(
                            won, t["size"], t["poly_price"], t["direction"]
                        )
                        t["resolved"] = True
                        t["won"] = won
                        t["pnl"] = pnl
                        self.total_pnl += pnl
                        if won:
                            self.wins += 1
                            self._bump_telemetry("resolved_win")
                        else:
                            self.losses += 1
                            self._bump_telemetry("resolved_loss")
                        self._balance += t["spend"] + pnl
                        self._journal_trade(t)
                        log(
                            f"  {'🟢' if won else '🔴'} RESOLVED {t['direction']}: {'WIN' if won else 'LOSS'} pnl={pnl:+.2f}"
                        )
                    elif now >= t["window_start"] + tf_sec + 600:
                        t["resolved"] = True
                        t["won"] = False
                        t["pnl"] = -abs(t["size"] * t["poly_price"])
                        self.total_pnl += t["pnl"]
                        self.losses += 1
                        self._bump_telemetry("resolved_loss")
                        self._journal_trade(t)
                        log(f"  🔴 TIMEOUT LOSS pnl={t['pnl']:+.2f}")

            # Status
            if now - last_status >= STATUS_SECONDS:
                tt = self.wins + self.losses
                wr = self.wins / tt if tt > 0 else 0
                locked = sum(t["spend"] for t in self._open_trades())
                equity, dd = self._refresh_risk_state()
                pnl_pct = (equity - self.starting) / self.starting * 100
                risk_suffix = f" DD={dd:.0%}" + (f" HALT={self.risk_halt_reason}" if self.risk_halted else "")
                telemetry_suffix = self._telemetry_brief()
                open_count = len(self._open_trades())
                log(
                    f"[{datetime.fromtimestamp(now).strftime('%H:%M:%S')}] elapsed={(now - self.t0) / 3600:.1f}h "
                    f"trades={tt}(W:{self.wins} L:{self.losses}) WR={wr:.0%} Cash=${self._balance:.2f} "
                    f"locked=${locked:.2f} open={open_count} Eq=${equity:.2f}({pnl_pct:+.1f}%){risk_suffix} "
                    f"BTC=${btc:.0f}{telemetry_suffix}"
                )
                last_status = now

            time.sleep(LOOP_SLEEP_SECONDS)

        tt = self.wins + self.losses
        locked = sum(t["spend"] for t in self.trades if not t.get("resolved"))
        equity = self._balance + locked
        log(
            f"\nFINAL: {tt} trades, {self.wins}W/{self.losses}L, Cash=${self._balance:.2f} locked=${locked:.2f} Eq=${equity:.2f} PnL=${equity - self.starting:+.2f}"
        )

    def _open_trades(self):
        return [t for t in self.trades if not t.get("resolved")]

    def _current_equity(self):
        locked = sum(t["spend"] for t in self._open_trades())
        return self._balance + locked

    def _refresh_risk_state(self):
        equity = self._current_equity()
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= RISK_HALT_DRAWDOWN_PCT:
            self.risk_halted = True
            self.risk_halt_reason = f"drawdown {dd:.1%} >= {RISK_HALT_DRAWDOWN_PCT:.0%}"
        return equity, dd

    def _telemetry_category(self, key: str) -> str:
        prefix = str(key).split(":", 1)[0]
        return {
            "prob_gate": "model_edge_skip",
            "open_cap": "open_cap",
            "open_cap_tf": "open_cap_tf",
            "corr": "correlated_open",
            "risk_halt": "risk_halt",
            "no_clob": "no_clob",
            "no_market": "no_market",
            "market_mismatch": "market_mismatch",
            "exec_poly_min": "exec_min_poly",
            "direction_prob": "exec_direction_prob",
            "pre_exec_edge": "pre_exec_edge",
            "clob_gamma_gap": "exec_clob_gamma_gap",
            "external_flow": "flow_skip",
            "exec_poly_cap": "exec_poly_cap",
            "exec_edge": "exec_edge_skip",
            "neutral": "neutral",
            "explore_price_guard": "explore_price_guard",
            "explore_slippage_guard": "explore_slippage_guard",
        }.get(prefix, prefix or "other_skip")

    def _bump_telemetry(self, category: str, amount: int = 1) -> None:
        try:
            self.telemetry[str(category)] += int(amount)
        except Exception:
            pass

    def _telemetry_brief(self) -> str:
        keys = (
            "bet_placed",
            "model_edge_skip",
            "open_cap",
            "open_cap_tf",
            "correlated_open",
            "correlated_cross_tf_allowed",
            "no_clob",
            "flow_skip",
            "exec_edge_skip",
            "exec_clob_gamma_gap",
            "exec_min_poly",
            "exec_direction_prob",
            "pre_exec_edge",
            "neutral",
            "resolved_win",
            "resolved_loss",
        )
        parts = [f"{key}={int(self.telemetry.get(key, 0))}" for key in keys if self.telemetry.get(key, 0)]
        return (" | telemetry " + " ".join(parts)) if parts else ""

    def _log_skip(self, key, msg, every=60.0):
        self._bump_telemetry(self._telemetry_category(key))
        now = time.time()
        last = self._last_skip_log.get(key, 0.0)
        if now - last >= every:
            self._last_skip_log[key] = now
            log(msg)

    def _build_prob_features(self, sig, win, tf, btc) -> dict:
        """Build feature dict for ProbabilityGate from current market state."""
        from datetime import datetime as dt_cls
        window_start = win.get("start", 0)
        try:
            dt = dt_cls.fromtimestamp(window_start, tz=timezone.utc)
            hour_utc = dt.hour
        except Exception:
            hour_utc = 12
        # Get market prices from Gamma. `poly_price_enter` is normalized to the
        # Up/Yes probability because the probability model predicts absolute Up.
        # The chosen side price stays in `chosen_poly_price` for edge/hard-cap
        # checks on Down trades.
        poly_up_price = 0.50
        poly_down_price = 0.50
        chosen_poly_price = 0.50
        try:
            mkt = self._market_for_window(win, tf)
            if mkt:
                outcomes_list = _json_list(mkt.get("outcomes"), ["Up", "Down"])
                op = _json_list(mkt.get("outcomePrices"), [])
                if len(op) >= 1:
                    poly_up_price = float(op[0])
                if len(op) >= 2:
                    poly_down_price = float(op[1])
                else:
                    poly_down_price = 1.0 - poly_up_price
                chosen_poly_price = poly_up_price if sig["direction"] == "Up" else poly_down_price
        except Exception:
            pass
        delta = btc - win.get("price", btc)
        window_start_price = win.get("price", btc)
        # RSI from signal engine
        engine = self.se_15m if tf == 15 else self.se_5m
        closes_5m, _, _, _ = get_klines("5m", 30)
        closes_15m, _, _, _ = get_klines("15m", 20)
        closes_5m = self._numeric_series(closes_5m)
        closes_15m = self._numeric_series(closes_15m)
        rsi_5m  = self._compute_rsi(closes_5m,  14) if len(closes_5m)  >= 15 else 50.0
        rsi_15m = self._compute_rsi(closes_15m, 14) if len(closes_15m) >= 15 else 50.0
        macd_5m  = self._macd_hist(closes_5m,  12, 26) if len(closes_5m)  >= 26 else 0.0
        macd_15m = self._macd_hist(closes_15m, 12, 26) if len(closes_15m) >= 26 else 0.0
        bb_5m  = self._bb_pos(closes_5m,  20) if len(closes_5m)  >= 20 else 0.5
        bb_15m = self._bb_pos(closes_15m, 20) if len(closes_15m) >= 20 else 0.5
        vol_5m  = self._vol_ratio(closes_5m,  30) if len(closes_5m)  >= 30 else 1.0
        vol_15m = self._vol_ratio(closes_15m, 12) if len(closes_15m) >= 12 else 1.0
        mom_5m  = self._momentum_pct(closes_5m,  20) if len(closes_5m)  >= 21 else 0.0
        mom_15m = self._momentum_pct(closes_15m, 12) if len(closes_15m) >= 13 else 0.0
        ob = get_orderbook_depth()
        external_feats = {}
        try:
            from btc_external_metrics import context_feature_subset, fetch_external_market_context
            external_feats = context_feature_subset(fetch_external_market_context())
        except Exception:
            external_feats = {}
        # Direction sign
        dir_sign = 1.0 if sig["direction"] == "Up" else -1.0
        feats = {
            "poly_price_enter":  poly_up_price,
            "poly_price_down":   poly_down_price,
            "chosen_poly_price": chosen_poly_price,
            "poly_vs_50":        (poly_up_price - 0.5) * 2,
            "btc_delta":          delta,
            "btc_delta_pct":     delta / window_start_price if window_start_price else 0.0,
            "rsi_5m":            rsi_5m,
            "rsi_15m":           rsi_15m,
            "macd_hist_5m":      macd_5m,
            "macd_hist_15m":     macd_15m,
            "bb_pos_5m":         bb_5m,
            "bb_pos_15m":        bb_15m,
            "vol_ratio_5m":       vol_5m,
            "vol_ratio_15m":      vol_15m,
            "ob_imbalance":       ob.get("imbalance", 0.0),
            "momentum_5m_pct":    mom_5m,
            "momentum_15m_pct":   mom_15m,
            "realized_vol":       0.0,  # TODO: compute from 1m bars
            "hour_utc":           hour_utc,
            "tf_minutes":         float(tf),
            "direction_sign":     dir_sign,
            "poly_bucket_high":    1.0 if poly_up_price > 0.65 else 0.0,
            "poly_bucket_very_high": 1.0 if poly_up_price > 0.80 else 0.0,
            # Legacy: pass through ensemble conf for GA backward compat
            "ensemble_conf":      sig.get("conf", 0.0),
        }
        feats.update(external_feats)
        return feats

    # ── Helpers for _build_prob_features ────────────────────────────────────
    def _numeric_series(self, values: list) -> list[float]:
        clean = []
        for value in values or []:
            try:
                f = float(value)
                if math.isfinite(f):
                    clean.append(f)
            except Exception:
                continue
        return clean

    def _compute_rsi(self, closes: list, period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        ag = sum(gains) / period if gains else 0
        al = sum(losses) / period if losses else 1e-9
        return 100 - (100 / (1 + ag / al))

    def _macd_hist(self, closes: list, fast: int, slow: int) -> float:
        if len(closes) < slow:
            return 0.0
        def ema(data, n):
            k = 2 / (n + 1); e = sum(data[:n]) / n
            for v in data[n:]: e = v * k + e * (1 - k)
            return e
        return ema(closes, fast) - ema(closes, slow)

    def _bb_pos(self, closes: list, period: int, std_dev: float = 2.0) -> float:
        if len(closes) < period:
            return 0.5
        recent = closes[-period:]
        mid = sum(recent) / len(recent)
        std = math.sqrt(sum((p - mid) ** 2 for p in recent) / len(recent))
        if std < 1e-9:
            return 0.5
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return (closes[-1] - lower) / (upper - lower)

    def _vol_ratio(self, closes: list, n: int) -> float:
        if len(closes) < 3:
            return 1.0
        deltas = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        lookback = deltas[-n:] if n > 0 else deltas
        if len(lookback) < 2:
            return 1.0
        half = max(1, len(lookback) // 2)
        older_slice = lookback[:-half]
        recent_slice = lookback[-half:]
        if not older_slice or not recent_slice:
            return 1.0
        recent = sum(recent_slice) / len(recent_slice)
        older = sum(older_slice) / len(older_slice)
        return recent / (older + 1e-9)

    def _momentum_pct(self, closes: list, period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        return (closes[-1] - closes[-period]) / closes[-period]

    def _risk_allows_new_trade(self, direction, tf):
        equity, dd = self._refresh_risk_state()
        if self.risk_halted:
            self._log_skip("risk_halt", f"  ⛔ RISK HALT: {self.risk_halt_reason}; skip {tf}m {direction}")
            return False
        open_trades = self._open_trades()
        if len(open_trades) >= MAX_OPEN_BTC_TRADES:
            self._log_skip(
                f"open_cap:{direction}",
                f"  ⏸️ RISK SKIP {tf}m {direction}: {len(open_trades)} open BTC trade(s), cap={MAX_OPEN_BTC_TRADES}",
            )
            return False
        if MAX_OPEN_BTC_TRADES_PER_TF > 0:
            tf_open = [t for t in open_trades if int(t.get("window_tf") or 0) == int(tf)]
            if len(tf_open) >= MAX_OPEN_BTC_TRADES_PER_TF:
                self._log_skip(
                    f"open_cap_tf:{tf}:{direction}",
                    f"  ⏸️ RISK SKIP {tf}m {direction}: {len(tf_open)} open trade(s) in {tf}m, tf_cap={MAX_OPEN_BTC_TRADES_PER_TF}",
                )
                return False
        same_direction = [t for t in open_trades if t.get("direction") == direction]
        if same_direction:
            same_tf_same_direction = any(int(t.get("window_tf") or 0) == int(tf) for t in same_direction)
            if not (ALLOW_CROSS_TF_CORRELATED_OPEN and not same_tf_same_direction):
                self._log_skip(
                    f"corr:{direction}",
                    f"  ⏸️ RISK SKIP {tf}m {direction}: correlated {direction} exposure already open",
                )
                return False
            self._bump_telemetry("correlated_cross_tf_allowed")
        return True

    def _market_for_window(self, win: dict, tf: int) -> Optional[dict]:
        """Return the exact Polymarket market attached to the active window.

        Feature pricing, CLOB token ids, and resolution must use the same Gamma
        market id. Around 5m/15m boundaries, an internal refetch can otherwise
        drift to a different market and create fake edge from the wrong token.
        """
        expected_id = win.get("market_id")
        cached = win.get("market")
        if isinstance(cached, dict):
            cached_id = cached.get("id")
            if not expected_id or str(cached_id) == str(expected_id):
                return cached

        fetched = fetch_btc_markets(tf)
        if not fetched:
            return None
        fetched_id = fetched.get("id")
        if expected_id and str(fetched_id) != str(expected_id):
            self._log_skip(
                f"market_mismatch:{tf}",
                f"  ⏸️ MARKET SKIP {tf}m: cached market_id={expected_id} but refetch returned {fetched_id}",
                every=30.0,
            )
            return None
        win["market"] = fetched
        if not expected_id:
            win["market_id"] = fetched_id
        return fetched

    def _exploration_allowed(self) -> bool:
        """Paper-only firehose fallback for labs when no probability model exists.

        Requires two env flags plus either --parallel-lab or FORCE. This keeps the
        main/live path conservative while allowing fake-money workers to collect
        labels instead of sitting idle behind MODEL_DOWN.
        """
        if not (PAPER_EXPLORATION_ENABLED and PAPER_EXPLORATION_ALLOW):
            return False
        if PAPER_EXPLORATION_FORCE:
            return True
        return "--parallel-lab" in sys.argv

    def _active_delta_threshold(self) -> float:
        if FORCE_DELTA_THRESHOLD is not None:
            return max(0.0, float(FORCE_DELTA_THRESHOLD))
        if self._exploration_allowed():
            return min(float(self.params.delta_thresh), max(0.0, PAPER_EXPLORATION_DELTA_FLOOR))
        return float(self.params.delta_thresh)

    def _delta_hint_signal(self, window_delta: float) -> Optional[dict]:
        """Low-latency direction hint for model-gated paper canaries.

        This is not the heuristic exploration fallback: it only supplies Up/Down
        direction from raw BTC movement so the probability model can evaluate
        more windows during fake-money validation. The model gate still decides
        whether a trade has edge.
        """
        if not DELTA_HINT_SIGNAL_ENABLED:
            return None
        if abs(window_delta) < self._active_delta_threshold():
            return None
        direction = "Up" if window_delta > 0 else "Down"
        conf = min(
            0.90,
            max(
                DELTA_HINT_MIN_CONF,
                abs(window_delta) / max(1.0, self.params.delta_thresh * 2.0),
            ),
        )
        return {
            "direction": direction,
            "conf": conf,
            "reasons": ["model_gate_delta_hint"],
            "conditions": {"window_delta": window_delta, "model_gate_delta_hint": True},
        }

    def _exploration_decision(self, sig, features: dict, blocked: BetDecision) -> Optional[BetDecision]:
        if not self._exploration_allowed():
            return None
        if not (PAPER_EXPLORATION_FORCE or str(blocked.summary).startswith("MODEL_DOWN")):
            return None
        direction = sig.get("direction")
        if direction not in ("Up", "Down"):
            return None
        try:
            if direction == "Down":
                poly_price = float(features.get("poly_price_down", features.get("chosen_poly_price", 0.5)) or 0.5)
            else:
                poly_price = float(features.get("poly_price_enter", features.get("chosen_poly_price", 0.5)) or 0.5)
        except Exception:
            poly_price = 0.5
        if poly_price <= PAPER_EXPLORATION_MIN_POLY or poly_price >= PAPER_EXPLORATION_MAX_POLY:
            return None
        try:
            conf = float(sig.get("conf", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < PAPER_EXPLORATION_MIN_CONF:
            return None

        direction_sign = 1.0 if direction == "Up" else -1.0
        btc_delta_aligned = float(features.get("btc_delta", 0.0) or 0.0) * direction_sign
        external_bull = float(features.get("external_bull_score", 0.0) or 0.0)
        flow_keys = (
            "cg_taker_30m_imbalance",
            "cg_cvd_30m_imbalance",
            "bg_taker_5m_imbalance",
            "bg_depth_imbalance",
            "bg_recent_trade_imbalance",
        )
        flow_vals = []
        for key in flow_keys:
            try:
                flow_vals.append(max(-1.0, min(1.0, float(features.get(key, 0.0) or 0.0))))
            except Exception:
                pass
        flow = (sum(flow_vals) / len(flow_vals)) if flow_vals else 0.0

        # Heuristic only used for fake-money label collection. It blends aligned
        # BTC momentum, external derivatives flow, and ensemble confidence.
        momentum_term = math.tanh(btc_delta_aligned / max(1.0, self.params.delta_thresh * 2.0))
        conf_term = max(-1.0, min(1.0, (conf - 0.45) / 0.45))
        heuristic_prob = (
            0.50
            + 0.24 * max(-1.0, min(1.0, external_bull * direction_sign))
            + 0.16 * max(-1.0, min(1.0, flow * direction_sign))
            + 0.16 * momentum_term
            + 0.08 * conf_term
        )
        heuristic_prob = max(0.01, min(0.99, heuristic_prob))
        edge = heuristic_prob - poly_price
        if edge < PAPER_EXPLORATION_MIN_EDGE:
            return None
        prob_up = heuristic_prob if direction == "Up" else 1.0 - heuristic_prob
        prob_down = 1.0 - prob_up
        return BetDecision(
            should_bet=True,
            direction=direction,
            model_prob=heuristic_prob,
            market_prob=poly_price,
            edge=edge,
            prob_up=prob_up,
            prob_down=prob_down,
            gate_reason=(
                "PAPER_EXPLORATION_MODEL_DOWN "
                f"heuristic={heuristic_prob:.3f} edge={edge:+.3f} "
                f"ext={external_bull:+.3f} flow={flow:+.3f} conf={conf:.2f}"
            ),
            summary=(
                f"EXPLORATION_BET poly={poly_price:.2f} heuristic_prob={heuristic_prob:.1%} "
                f"edge={edge:+.1%} min_edge={PAPER_EXPLORATION_MIN_EDGE:.1%}"
            ),
        )

    def _neutral_exploration_signal(self, window_delta: float) -> Optional[dict]:
        """Derive a lab-only signal from raw BTC movement during warmup.

        SignalEngine needs price-history warmup and can return Neutral even when
        the current 5m/15m window already moved enough to collect a useful paper
        label. Exploration labs should not sit idle during that warmup.
        """
        if not self._exploration_allowed():
            return None
        if abs(window_delta) < self._active_delta_threshold():
            return None
        direction = "Up" if window_delta > 0 else "Down"
        conf = min(0.90, max(PAPER_EXPLORATION_MIN_CONF, abs(window_delta) / max(1.0, self.params.delta_thresh * 2.0)))
        return {
            "direction": direction,
            "conf": conf,
            "reasons": ["paper_exploration_delta_warmup"],
            "conditions": {"window_delta": window_delta, "exploration_warmup": True},
        }

    def _place_trade(self, sig, win, tf, btc):
        direction = sig["direction"]
        if not self._risk_allows_new_trade(direction, tf):
            return None
        poly_price = 0.50
        gamma_price = None
        token_id = None
        price_source = "gamma_outcomePrices_fallback"
        clob_quote = None
        locked = sum(t["spend"] for t in self._open_trades())
        total_exposure_pct = (
            float(MAX_TOTAL_BTC_EXPOSURE_PCT)
            if MAX_TOTAL_BTC_EXPOSURE_PCT is not None
            else float(self.params.max_bet_pct)
        )
        exposure_cap = self.starting * total_exposure_pct
        remaining_exposure = max(0.0, exposure_cap - locked)
        max_spend = min(
            self._balance * 0.95,
            self._balance * self.params.max_bet_pct,
            remaining_exposure,
        )
        if max_spend < MIN_SPEND:
            self._log_skip(
                f"exposure_cap:{tf}:{direction}",
                f"  ⏸️ RISK SKIP {tf}m {direction}: remaining_exposure=${remaining_exposure:.2f} "
                f"< min_spend=${MIN_SPEND:.2f} total_exposure_cap={total_exposure_pct:.0%}",
            )
            return None
        spend = min(self._balance * self.params.spend_ratio, MAX_TRADE_COST, max_spend)
        if sig.get("_exploration_mode"):
            spend = min(spend, max(MIN_SPEND, PAPER_EXPLORATION_MAX_SPEND))

        # Fetch market for price
        mkt = None
        try:
            mkt = self._market_for_window(win, tf)
            if mkt:
                token_id = clob_token_id_for_direction(mkt, direction)
                op = _json_list(mkt.get("outcomePrices"), [])
                if len(op) >= 2:
                    # outcomes are ["Up", "Down"]; buy the token matching our direction.
                    gamma_price = float(op[0]) if direction == "Up" else float(op[1])
                    poly_price = gamma_price
                clob_quote = fetch_clob_buy_quote(
                    token_id,
                    target_spend=spend,
                    min_size=5.0,
                    max_spend=max_spend,
                )
                if clob_quote:
                    poly_price = float(clob_quote["price"])
                    price_source = "clob_book_ask_depth"
                    feats = sig.get("_prob_features")
                    if isinstance(feats, dict):
                        feats["clob_exec_price"] = poly_price
                        feats["clob_best_ask"] = float(clob_quote.get("best_ask") or poly_price)
                        if gamma_price is not None:
                            feats["clob_gamma_gap"] = float(gamma_price) - poly_price
        except:
            pass

        if not mkt:
            self._log_skip(
                f"no_market:{tf}:{direction}",
                f"  ⏸️ MARKET SKIP {tf}m {direction}: no active window market for pricing",
            )
            return None

        if REQUIRE_CLOB_QUOTE and not clob_quote:
            self._log_skip(
                f"no_clob:{tf}:{direction}",
                f"  ⏸️ RISK SKIP {tf}m {direction}: no executable CLOB ask depth",
            )
            return None

        if sig.get("_exploration_mode"):
            if poly_price <= PAPER_EXPLORATION_MIN_POLY or poly_price >= PAPER_EXPLORATION_MAX_POLY:
                self._log_skip(
                    f"explore_price_guard:{tf}:{direction}",
                    f"  ⏸️ PAPER EXPLORE SKIP {tf}m {direction}: executable px={poly_price:.3f} "
                    f"outside [{PAPER_EXPLORATION_MIN_POLY:.2f}, {PAPER_EXPLORATION_MAX_POLY:.2f}]",
                )
                return None
            if clob_quote and float(clob_quote.get("slippage_bps") or 0.0) > PAPER_EXPLORATION_MAX_SLIPPAGE_BPS:
                self._log_skip(
                    f"explore_slippage_guard:{tf}:{direction}",
                    f"  ⏸️ PAPER EXPLORE SKIP {tf}m {direction}: slippage={float(clob_quote.get('slippage_bps') or 0.0):.1f}bps "
                    f"> max={PAPER_EXPLORATION_MAX_SLIPPAGE_BPS:.1f}bps",
                )
                return None

        if not self._runtime_guards_allow(sig, tf, direction, poly_price, gamma_price):
            return None

        if not self._execution_edge_allows(sig, tf, direction, poly_price):
            return None

        if clob_quote:
            size = float(clob_quote["size"])
            cost = float(clob_quote["cost"])
        else:
            size = spend / poly_price
            if size < 5:
                size = 5.0
            cost = size * poly_price
            if cost > max_spend:
                capped_size = max_spend / poly_price
                if capped_size < 5.0:
                    return None
                size = capped_size
            cost = size * poly_price
        if cost > max_spend + 1e-9:
            return None

        trade = {
            "direction": direction,
            "size": size,
            "poly_price": poly_price,
            "price_source": price_source,
            "gamma_price": gamma_price,
            "clob_token_id": token_id,
            "clob_best_ask": clob_quote.get("best_ask") if clob_quote else None,
            "clob_levels_used": clob_quote.get("levels_used") if clob_quote else None,
            "clob_slippage_bps": clob_quote.get("slippage_bps") if clob_quote else None,
            "spend": cost,
            "btc_price_enter": btc,
            "btc_delta": btc - win["price"],
            "conf": sig["conf"],
            "prob_features": sig.get("_prob_features", {}),
            "prob_decision": sig.get("_prob_decision", {}),
            "external_bull_score": (sig.get("_prob_features", {}) or {}).get("external_bull_score"),
            "exploration_mode": bool(sig.get("_exploration_mode")),
            "exploration_reason": sig.get("_exploration_reason"),
            "window_start": win["start"],
            "window_tf": tf,
            "market_id": win.get("market_id"),
            "market_slug": mkt.get("slug") if isinstance(mkt, dict) else None,
            "condition_id": (mkt.get("conditionId") or mkt.get("condition_id")) if isinstance(mkt, dict) else None,
            "resolved": False,
            "placed_at": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._balance -= cost
        self._bump_telemetry("bet_placed")
        return trade  # PnL calculated at resolution

    def _direction_model_prob(self, decision: dict, direction: str) -> Optional[float]:
        key = "prob_up" if direction == "Up" else "prob_down"
        for candidate in (decision.get(key), decision.get("model_prob")):
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    def _pre_exec_edge(self, decision: dict) -> Optional[float]:
        for candidate in (decision.get("pre_exec_edge"), decision.get("edge")):
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        try:
            model_prob = float(decision.get("model_prob"))
            market_prob = float(decision.get("market_prob"))
            value = model_prob - market_prob
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _external_flow_allows(self, sig, tf: int, direction: str) -> bool:
        if not REQUIRE_EXTERNAL_FLOW_AGREEMENT:
            return True
        feats = sig.get("_prob_features") or {}
        sign = 1.0 if direction == "Up" else -1.0
        threshold = max(0.0, float(EXTERNAL_FLOW_SOFT_THRESHOLD))
        flow_keys = (
            # Binance spot/futures
            "ob_imbalance",
            "bn_taker_15m_imbalance",
            "bn_top_ls_15m_imbalance",
            "bn_global_ls_15m_imbalance",
            # CoinGlass
            "cg_taker_30m_imbalance",
            "cg_cvd_30m_imbalance",
            "cg_orderbook_30m_imbalance",
            "cg_longs_30m_imbalance",
            # Bitget
            "bg_taker_5m_imbalance",
            "bg_taker_15m_imbalance",
            "bg_depth_imbalance",
            "bg_recent_trade_imbalance",
            "bg_ls_5m_imbalance",
            "bg_ls_15m_imbalance",
            "bg_trader_ls_5m_imbalance",
            "bg_position_ls_5m_imbalance",
        )
        aligned = []
        for key in flow_keys:
            if key not in feats:
                continue
            try:
                raw = float(feats.get(key))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(raw) or abs(raw) < threshold:
                continue
            aligned.append(max(-1.0, min(1.0, raw)) * sign)
        if "external_bull_score" in feats:
            try:
                raw = float(feats.get("external_bull_score"))
                if math.isfinite(raw) and abs(raw) >= threshold:
                    aligned.append(max(-1.0, min(1.0, raw)) * sign)
            except (TypeError, ValueError):
                pass
        if not aligned:
            return True
        avg = sum(aligned) / len(aligned)
        opposing = sum(1 for value in aligned if value <= -threshold)
        supportive = sum(1 for value in aligned if value >= threshold)
        if avg <= -threshold or (opposing >= 2 and supportive == 0):
            self._log_skip(
                f"external_flow:{tf}:{direction}",
                f"  ⏸️ FLOW SKIP {tf}m {direction}: external flow disagrees "
                f"avg_aligned={avg:+.3f} opposing={opposing} supportive={supportive} threshold={threshold:.3f}",
            )
            return False
        return True

    def _runtime_guards_allow(
        self,
        sig,
        tf: int,
        direction: str,
        executable_price: float,
        gamma_price: Optional[float],
    ) -> bool:
        """Conservative live-readiness guards for model-gated paper entries."""
        decision = sig.get("_prob_decision") or {}
        model_prob = self._direction_model_prob(decision, direction)
        high_prob_override = (
            model_prob is not None
            and model_prob >= CLOB_GAMMA_GAP_OVERRIDE_PROB
        )
        try:
            px = float(executable_price)
        except (TypeError, ValueError):
            px = float("nan")

        # Never let an overconfident model buy lottery-tail contracts by
        # default. Today's 14:06 CEST paper swarm bought ~9c Up contracts
        # because model_prob crossed the high-prob override, then every first
        # resolved trade lost. Gamma/CLOB gap override can still bypass gap
        # checks, but min executable price is a hard liquidity/late-window guard
        # unless explicitly enabled for an isolated experiment.
        if (
            math.isfinite(px)
            and px <= PROB_MIN_POLY_PRICE
            and not (ALLOW_MIN_POLY_HIGH_PROB_OVERRIDE and high_prob_override)
        ):
            self._log_skip(
                f"exec_poly_min:{tf}:{direction}",
                f"  ⏸️ EXEC SKIP {tf}m {direction}: executable px={px:.3f} <= min_poly={PROB_MIN_POLY_PRICE:.2f}",
            )
            return False

        if model_prob is not None and model_prob < PROB_MIN_DIRECTION_PROB:
            self._log_skip(
                f"direction_prob:{tf}:{direction}",
                f"  ⏸️ EXEC SKIP {tf}m {direction}: direction_prob={model_prob:.1%} "
                f"< min_direction_prob={PROB_MIN_DIRECTION_PROB:.1%}",
            )
            return False

        if REQUIRE_PRE_EXEC_EDGE:
            pre_edge = self._pre_exec_edge(decision)
            if pre_edge is not None and pre_edge <= PRE_EXEC_EDGE_THRESHOLD and not high_prob_override:
                self._log_skip(
                    f"pre_exec_edge:{tf}:{direction}",
                    f"  ⏸️ EXEC SKIP {tf}m {direction}: pre_exec_edge={pre_edge:+.1%} "
                    f"<= threshold={PRE_EXEC_EDGE_THRESHOLD:.1%}",
                )
                return False

        if (
            REQUIRE_EXEC_PRICE_WITHIN_GAMMA
            and MAX_CLOB_GAMMA_GAP is not None
            and math.isfinite(px)
            and gamma_price is not None
        ):
            try:
                gamma = float(gamma_price)
            except (TypeError, ValueError):
                gamma = float("nan")
            if math.isfinite(gamma):
                gap = abs(px - gamma)
                if gap > MAX_CLOB_GAMMA_GAP and not high_prob_override:
                    self._log_skip(
                        f"clob_gamma_gap:{tf}:{direction}",
                        f"  ⏸️ EXEC SKIP {tf}m {direction}: |clob_exec-gamma|={gap:.3f} "
                        f"> max_gap={MAX_CLOB_GAMMA_GAP:.3f} "
                        f"and direction_prob={(model_prob if model_prob is not None else 0.0):.1%} "
                        f"< override={CLOB_GAMMA_GAP_OVERRIDE_PROB:.1%}",
                    )
                    return False

        if EXTERNAL_BULL_SCORE_MIN is not None or EXTERNAL_BULL_SCORE_MAX is not None:
            feats = sig.get("_prob_features") or {}
            try:
                external_bull = float(feats.get("external_bull_score"))
            except (TypeError, ValueError):
                external_bull = float("nan")
            if not math.isfinite(external_bull):
                self._log_skip(
                    f"external_score_missing:{tf}:{direction}",
                    f"  ⏸️ EXEC SKIP {tf}m {direction}: external_bull_score missing for score-window guard",
                )
                return False
            if EXTERNAL_BULL_SCORE_MIN is not None and external_bull < EXTERNAL_BULL_SCORE_MIN:
                self._log_skip(
                    f"external_score_low:{tf}:{direction}",
                    f"  ⏸️ EXEC SKIP {tf}m {direction}: external_bull_score={external_bull:+.3f} "
                    f"< min={EXTERNAL_BULL_SCORE_MIN:+.3f}",
                )
                return False
            if EXTERNAL_BULL_SCORE_MAX is not None and external_bull > EXTERNAL_BULL_SCORE_MAX:
                self._log_skip(
                    f"external_score_high:{tf}:{direction}",
                    f"  ⏸️ EXEC SKIP {tf}m {direction}: external_bull_score={external_bull:+.3f} "
                    f"> max={EXTERNAL_BULL_SCORE_MAX:+.3f}",
                )
                return False

        return self._external_flow_allows(sig, tf, direction)

    def _execution_edge_allows(self, sig, tf: int, direction: str, executable_price: float) -> bool:
        """Recheck edge against the actual executable ask, not stale Gamma price."""
        decision = sig.get("_prob_decision") or {}
        try:
            model_prob = float(decision.get("model_prob"))
            px = float(executable_price)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(model_prob) or not math.isfinite(px):
            return True
        if px >= PROB_HARD_POLY_CAP:
            self._log_skip(
                f"exec_poly_cap:{tf}:{direction}",
                f"  ⏸️ EXEC SKIP {tf}m {direction}: executable px={px:.3f} >= hard_cap={PROB_HARD_POLY_CAP:.2f}",
            )
            return False
        threshold = PROB_EDGE_THRESHOLD
        if sig.get("_exploration_mode"):
            threshold = max(threshold, PAPER_EXPLORATION_MIN_EDGE)
        if px >= PROB_POLY_PRICE_CEILING:
            threshold += PROB_POLY_CEILING_EDGE_BUFFER
        exec_edge = model_prob - px
        if exec_edge <= threshold:
            self._log_skip(
                f"exec_edge:{tf}:{direction}",
                f"  ⏸️ EXEC SKIP {tf}m {direction}: model={model_prob:.1%} "
                f"exec_px={px:.1%} edge={exec_edge:+.1%} <= eff_thresh={threshold:.1%}",
            )
            return False
        decision.setdefault("pre_exec_market_prob", decision.get("market_prob"))
        decision.setdefault("pre_exec_edge", decision.get("edge"))
        decision["market_prob"] = px
        decision["edge"] = exec_edge
        decision["exec_edge_checked"] = True
        sig["_prob_decision"] = decision
        return True

    def _calc_pnl(self, won, size, poly_price, direction):
        return resolved_buy_pnl(won, size, poly_price)

    def _journal_trade(self, t):
        try:
            d = {
                "mode": "paper",
                "agent_id": self.agent_id,
                "strategy": STRATEGY_NAME,
                "params": self.params.to_dict(),
                "window_start": t["window_start"],
                "window_tf": t["window_tf"],
                "direction": t["direction"],
                "spend": t["spend"],
                "poly_price": t["poly_price"],
                "price_source": t.get("price_source"),
                "gamma_price": t.get("gamma_price"),
                "clob_token_id": t.get("clob_token_id"),
                "clob_best_ask": t.get("clob_best_ask"),
                "clob_levels_used": t.get("clob_levels_used"),
                "clob_slippage_bps": t.get("clob_slippage_bps"),
                "market_slug": t.get("market_slug"),
                "condition_id": t.get("condition_id"),
                "fee_model": "polymarket_crypto_taker_v1",
                "entry_fee": taker_fee_usdc(t["size"], t["poly_price"]),
                "btc_delta": t["btc_delta"],
                "btc_price_enter": t["btc_price_enter"],
                "conf": t["conf"],
                "prob_features": t.get("prob_features", {}),
                "prob_decision": t.get("prob_decision", {}),
                "external_bull_score": t.get("external_bull_score"),
                "exploration_mode": bool(t.get("exploration_mode")),
                "exploration_reason": t.get("exploration_reason"),
                "won": t["won"],
                "pnl": t["pnl"],
                "placed_at": t.get("placed_at"),
            }
            with open(JOURNAL_FILE, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception as e:
            log(f"Journal error: {e}")


def continuous_ga_loop():
    while True:
        try:
            FastGA(timeframes=ACTIVE_TIMEFRAMES).run(JOURNAL_FILE, generations=8, session_min=5)
        except Exception as e:
            log(f"[GA] Continuous loop error: {e}")
        time.sleep(300)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BTC 5m/15m paper sniper")
    parser.add_argument("duration_pos", nargs="?", type=int, help="run duration in seconds (legacy positional)")
    parser.add_argument("--duration", type=int, default=None, help="run duration in seconds")
    parser.add_argument("--timeframes", default=None, help="comma-separated subset: 5, 15, or 5,15")
    parser.add_argument("--agent-id", default=None, help="stable process id, e.g. btc-5m or btc-15m")
    parser.add_argument("--capital", type=float, default=None, help="paper bankroll for this isolated process")
    parser.add_argument("--best-params-file", default=None, help="agent-specific params JSON")
    parser.add_argument("--log-file", default=None, help="agent-specific trader log")
    parser.add_argument("--journal-file", default=None, help="paper journal file")
    parser.add_argument("--include-live-training", action="store_true", help="include resolved live trades in FastGA training")
    parser.add_argument("--live-trade-weight", type=float, default=None, help="weight for live rows in FastGA scoring")
    # Process-owner markers used by launchers/watchdogs; parsed so argparse does
    # not reject existing calls: `btc_paper_fast.py 21600 --watchdog-main`.
    parser.add_argument("--watchdog-main", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parallel-lab", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _default_params_path_for_agent(agent_id: str, timeframes: list[int]) -> Path:
    if len(timeframes) == 1:
        return STATE_DIR / f"sniper_best_params_{timeframes[0]}m.json"
    return STATE_DIR / f"sniper_best_params_{_safe_slug(agent_id)}.json"


def configure_runtime(args) -> int:
    """Apply CLI split-agent config to module globals before starting threads."""
    global AGENT_ID, ACTIVE_TIMEFRAMES, PAPER_CAPITAL, BEST_PARAMS_FILE, PAPER_LOG_FILE, JOURNAL_FILE
    global INCLUDE_LIVE_TRAINING, LIVE_TRADE_WEIGHT

    duration = args.duration if args.duration is not None else args.duration_pos
    duration = duration if duration is not None else 28800

    if args.timeframes is not None:
        ACTIVE_TIMEFRAMES = parse_timeframes(args.timeframes)
        os.environ["BTC_TIMEFRAMES"] = ",".join(str(tf) for tf in ACTIVE_TIMEFRAMES)

    if args.agent_id:
        AGENT_ID = args.agent_id
        os.environ["BTC_AGENT_ID"] = AGENT_ID

    if args.capital is not None:
        PAPER_CAPITAL = float(args.capital)
        os.environ["BTC_PAPER_CAPITAL"] = str(PAPER_CAPITAL)

    if args.journal_file:
        JOURNAL_FILE = Path(os.path.expanduser(args.journal_file))
        os.environ["BTC_JOURNAL_FILE"] = str(JOURNAL_FILE)

    if args.include_live_training:
        INCLUDE_LIVE_TRAINING = True
        os.environ["BTC_INCLUDE_LIVE_TRAINING"] = "1"
    if args.live_trade_weight is not None:
        LIVE_TRADE_WEIGHT = float(args.live_trade_weight)
        os.environ["BTC_LIVE_TRADE_WEIGHT"] = str(LIVE_TRADE_WEIGHT)

    env_params_explicit = bool(os.environ.get("BTC_BEST_PARAMS_FILE"))
    if args.best_params_file:
        BEST_PARAMS_FILE = Path(os.path.expanduser(args.best_params_file))
        os.environ["BTC_BEST_PARAMS_FILE"] = str(BEST_PARAMS_FILE)
    elif not env_params_explicit and (args.agent_id or args.timeframes):
        BEST_PARAMS_FILE = _default_params_path_for_agent(AGENT_ID, ACTIVE_TIMEFRAMES)
        os.environ["BTC_BEST_PARAMS_FILE"] = str(BEST_PARAMS_FILE)

    if args.log_file:
        PAPER_LOG_FILE = Path(os.path.expanduser(args.log_file))
        os.environ["BTC_PAPER_LOG_FILE"] = str(PAPER_LOG_FILE)
    elif not os.environ.get("BTC_PAPER_LOG_FILE") and (args.agent_id or args.timeframes):
        PAPER_LOG_FILE = LOG_DIR / f"btc_paper_fast_{_safe_slug(AGENT_ID)}.log"
        os.environ["BTC_PAPER_LOG_FILE"] = str(PAPER_LOG_FILE)

    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # First split run starts from the current combined params, then diverges.
    if BEST_PARAMS_FILE != DEFAULT_BEST_PARAMS_FILE and not BEST_PARAMS_FILE.exists():
        seed = read_params_file(DEFAULT_BEST_PARAMS_FILE, mode="paper") if DEFAULT_BEST_PARAMS_FILE.exists() else {}
        seed["name"] = str(seed.get("name") or f"{AGENT_ID}_seed")
        seed["source"] = f"split-agent-seed:{AGENT_ID}:{','.join(str(tf) for tf in ACTIVE_TIMEFRAMES)}m"
        write_params_file(BEST_PARAMS_FILE, seed, mode="paper")
        log(f"Seeded params file {BEST_PARAMS_FILE} from {DEFAULT_BEST_PARAMS_FILE}")

    return int(duration)


if __name__ == "__main__":
    args = parse_args()
    duration = configure_runtime(args)

    # Continuous in-process GA is useful as a fallback, but the preferred
    # trainer is the isolated parallel autoresearch launchd agent. Disable this
    # with BTC_FAST_GA_ENABLED=0 to keep one training authority.
    if os.environ.get("BTC_FAST_GA_ENABLED", "1") != "0":
        ga_thread = threading.Thread(target=continuous_ga_loop, daemon=True)
        ga_thread.start()
    else:
        log("[GA] FastGA disabled by BTC_FAST_GA_ENABLED=0")

    p = SniperParams()
    try:
        if BEST_PARAMS_FILE.exists():
            p = SniperParams.from_dict(read_params_file(BEST_PARAMS_FILE, mode="paper"))
    except Exception as e:
        log(f"Param load error: {e}")
    p = apply_param_overrides(p)
    trader = PaperTrader(p, timeframes=ACTIVE_TIMEFRAMES, agent_id=AGENT_ID)
    trader.run(duration=duration)
