#!/usr/bin/env python3
"""
Phase 0: BTC Polymarket Training Dataset Generator
===================================================
Produces (features, outcome) rows from:
  - Historical journal entries (intraday_journal.jsonl)
  - Binance klines for technical feature computation
  - Gamma/Polymarket resolution data for outcome labels

Output: data/arbitrage-agent/v2/model/train_dataset.jsonl
Each line: {"features": {...}, "outcome": 0|1, "meta": {...}}
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import glob
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
MODEL_DIR = DATA_DIR / "model"
JOURNAL_FILE = DATA_DIR / "state" / "intraday_journal.jsonl"
LAB_JOURNAL_GLOB = DATA_DIR / "parallel_lab" / "*" / "*" / "state" / "intraday_journal.jsonl"
OUT_FILE = MODEL_DIR / "train_dataset.jsonl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    from btc_external_metrics import context_feature_subset
except Exception:
    context_feature_subset = None

# ── Binance ──────────────────────────────────────────────────────────────────
BINANCE_REST = "https://api.binance.com/api/v3"
GAMMA_API = "https://gamma-api.polymarket.com"


def btc_klines(symbol="BTCUSDT", interval="1m", limit=1000,
               start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
    if end_time_ms:
        params["endTime"] = end_time_ms
    try:
        r = requests.get(f"{BINANCE_REST}/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = []
        for k in data:
            rows.append({
                "open_time":  float(k[0]),
                "open":        float(k[1]),
                "high":        float(k[2]),
                "low":         float(k[3]),
                "close":       float(k[4]),
                "volume":      float(k[5]),
                "close_time":  float(k[6]),
                "quote_volume": float(k[7]),
            })
        return rows
    except Exception as e:
        print(f"[klines] error: {e}", file=sys.stderr)
        return []


def btc_ticker_24hr():
    try:
        r = requests.get(f"{BINANCE_REST}/ticker/24hr", params={"symbol": "BTCUSDT"}, timeout=5)
        r.raise_for_status()
        d = r.json()
        return {
            "price":          float(d["lastPrice"]),
            "quote_volume":    float(d["quoteVolume"]),
            "price_chg_pct":   float(d["priceChangePercent"]),
        }
    except:
        return None


def btc_orderbook_depth(limit=20):
    try:
        r = requests.get(f"{BINANCE_REST}/depth", params={"symbol": "BTCUSDT", "limit": limit}, timeout=5)
        r.raise_for_status()
        d = r.json()
        bids = [(float(p), float(q)) for p, q in d.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in d.get("asks", [])]
        sum_b = sum(q for _, q in bids)
        sum_a = sum(q for _, q in asks)
        imbalance = (sum_b - sum_a) / (sum_b + sum_a + 1e-9)
        return {"imbalance": imbalance, "bids": bids, "asks": asks}
    except:
        return {"imbalance": 0.0, "bids": [], "asks": []}


# ── Gamma / Polymarket ────────────────────────────────────────────────────────
def gamma_market_resolution(market_id: str) -> Optional[str]:
    """Fetch resolution for a market. Returns 'Up' or 'Down' or None."""
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=8)
        if r.status_code != 200:
            return None
        m = r.json()
        # Check for resolution
        resolved = m.get("resolved")
        if not resolved:
            return None
        # outcomePrices: ["0.45", "0.55"] for ["Up", "Down"]
        prices_raw = m.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        labels_raw = m.get("outcomes", '["Up","Down"]')
        labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
        for i, p in enumerate(prices):
            try:
                pf = float(p)
                if pf >= 0.99:
                    lbl = labels[i] if i < len(labels) else ""
                    if str(lbl).lower() in ("up", "yes"):
                        return "Up"
                    elif str(lbl).lower() in ("down", "no"):
                        return "Down"
            except:
                pass
        return None
    except Exception as e:
        return None


def fetch_all_btc_markets(tf_minutes: int = 5, lookback_hours: int = 24):
    """Fetch all BTC markets for a timeframe within lookback period."""
    markets = []
    now_ts = int(time.time())
    window_sec = tf_minutes * 60
    # Scan ~lookback_hours worth of windows
    current_window = (now_ts // window_sec) * window_sec
    for offset in range(-int(lookback_hours * 3600 / window_sec), 2):
        window = current_window + offset * window_sec
        slug = f"btc-updown-{tf_minutes}m-{window}"
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            mkt_list = data if isinstance(data, list) else data.get("data", [])
            for m in mkt_list:
                if not m.get("closed"):
                    continue
                markets.append(m)
        except:
            continue
    return markets


# ── Technical Indicators ─────────────────────────────────────────────────────
def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    ag = sum(gains) / period if gains else 0
    al = sum(losses) / period if losses else 1e-9
    rs = ag / al
    return 100 - (100 / (1 + rs))


def ema(data: list[float], n: int) -> float:
    if len(data) < n:
        return data[-1] if data else 0.0
    k = 2 / (n + 1)
    result = sum(data[:n]) / n
    for v in data[n:]:
        result = v * k + result * (1 - k)
    return result


def macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if len(closes) < slow:
        return 0.0, 0.0, 0.0
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # Approximate signal line (simplified)
    hist = macd_line  # use macd_line as histogram proxy
    return ema_fast, ema_slow, hist


def bollinger_position(closes: list[float], period: int = 20, std_dev: float = 2.0) -> float:
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


def volume_ratio(closes: list[float], volumes: list[float], n: int = 30) -> float:
    if len(closes) < n or len(volumes) < n:
        return 1.0
    recent_vol = sum(volumes[-n//2:]) / (n // 2)
    older_vol  = sum(volumes[-n:-n//2]) / (n // 2) if n > 1 else recent_vol
    return recent_vol / (older_vol + 1e-9)


def momentum_pct(closes: list[float], period: int = 20) -> float:
    if len(closes) < period + 1:
        return 0.0
    return (closes[-1] - closes[-period]) / closes[-period]


def realized_volatility(closes: list[float], period: int = 20) -> float:
    if len(closes) < period + 1:
        return 0.0
    returns = [math.log(closes[i] / (closes[i-1] + 1e-9)) for i in range(1, len(closes))]
    rets = returns[-period:]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var * 252 * 1440)  # annualized from minute bars


# ── Feature extraction per window ─────────────────────────────────────────────
def extract_features_for_window(window_start: float, window_end: float,
                                 tf_minutes: int, direction: str,
                                 poly_price_enter: float,
                                 btc_delta: float) -> dict:
    """Extract all features for a window using Binance klines."""
    start_ms = int(window_start * 1000)
    end_ms   = int(window_end   * 1000 + 60_000)  # buffer

    # 1-minute bars covering the window + lookback
    lookback_ms = 60 * 60 * 1000  # 1hr lookback
    klines_1m = btc_klines(
        interval="1m", limit=120,
        start_time_ms=max(start_ms - lookback_ms, 0),
        end_time_ms=min(end_ms, int(time.time() * 1000))
    )

    closes_1m = [k["close"] for k in klines_1m]
    highs_1m = [k["high"] for k in klines_1m]
    lows_1m  = [k["low"]  for k in klines_1m]
    vols_1m  = [k["quote_volume"] for k in klines_1m]

    # 5-minute bars
    klines_5m = btc_klines(
        interval="5m", limit=60,
        start_time_ms=max(start_ms - lookback_ms, 0),
        end_time_ms=min(end_ms, int(time.time() * 1000))
    )
    closes_5m = [k["close"] for k in klines_5m]
    vols_5m    = [k["quote_volume"] for k in klines_5m]

    # 15-minute bars
    klines_15m = btc_klines(
        interval="15m", limit=40,
        start_time_ms=max(start_ms - lookback_ms, 0),
        end_time_ms=min(end_ms, int(time.time() * 1000))
    )
    closes_15m = [k["close"] for k in klines_15m]

    if not closes_1m:
        return None

    window_price_start = closes_1m[0] if closes_1m else None

    # Orderbook
    ob = btc_orderbook_depth()

    # Hour of day
    try:
        dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
        hour_utc = dt.hour
    except:
        hour_utc = 12

    poly_prices = normalize_poly_prices(direction, poly_price_enter)
    poly_up = poly_prices["poly_price_up"]
    poly_down = poly_prices["poly_price_down"]
    chosen_poly = poly_prices["chosen_poly_price"]

    feats = {
        # Market-implied
        # `poly_price_enter` is normalized to the Up/Yes probability. The
        # journal stores the chosen side price, so Down trades are inverted here.
        "poly_price_enter": poly_up,
        "poly_price_down": poly_down,
        "chosen_poly_price": chosen_poly,
        "poly_vs_50": (poly_up - 0.5) * 2,  # [-1, 1]

        # Delta features
        "btc_delta": btc_delta,
        "btc_delta_pct": btc_delta / window_price_start if window_price_start else 0.0,

        # RSI
        "rsi_5m": rsi(closes_5m, 14),
        "rsi_15m": rsi(closes_15m, 14),

        # MACD histogram
        "macd_hist_5m": macd(closes_5m)[2],
        "macd_hist_15m": macd(closes_15m)[2],

        # Bollinger position
        "bb_pos_5m": bollinger_position(closes_5m, 20, 2.0),
        "bb_pos_15m": bollinger_position(closes_15m, 20, 2.0),

        # Volume
        "vol_ratio_5m": volume_ratio(closes_5m, vols_5m, 30),
        "vol_ratio_15m": volume_ratio(closes_15m, vols_5m, 12),  # reuse 5m vols

        # Orderbook
        "ob_imbalance": ob.get("imbalance", 0.0),

        # Momentum
        "momentum_5m_pct": momentum_pct(closes_5m, 20),
        "momentum_15m_pct": momentum_pct(closes_15m, 12),

        # Volatility
        "realized_vol": realized_volatility(closes_1m, 20),

        # Time
        "hour_utc": hour_utc,
        "tf_minutes": float(tf_minutes),

        # Direction (signed)
        "direction_sign": 1.0 if direction == "Up" else -1.0,

        # UTC hour cyclical encoding
        f"hour_sin_{hour_utc}": 1.0,  # sparse; model uses hour_utc directly

        # Poly price bucket (for non-linearity)
        f"poly_bucket_high": 1.0 if poly_up > 0.65 else 0.0,
        f"poly_bucket_very_high": 1.0 if poly_up > 0.80 else 0.0,
    }
    return feats


# ── Journal loader ────────────────────────────────────────────────────────────
def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _expand_journal_pattern(raw: str) -> list[Path]:
    raw = os.path.expandvars(os.path.expanduser(str(raw)))
    matches = [Path(p) for p in glob.glob(raw)]
    if matches:
        return sorted({p.resolve() for p in matches if p.is_file()})
    p = Path(raw).resolve()
    return [p] if p.is_file() else []


def journal_sources() -> list[Path]:
    """Return journal files to use for training.

    Defaults to the main paper journal. Lab journals are opt-in via
    BTC_PROB_DATASET_INCLUDE_LABS=1 so ad-hoc experiments do not silently change
    old workflows. Extra journals/globs are colon-separated in
    BTC_PROB_DATASET_EXTRA_JOURNALS.
    """
    sources: list[Path] = []
    if _env_bool("BTC_PROB_DATASET_INCLUDE_MAIN", "1"):
        sources.append(JOURNAL_FILE)
    if _env_bool("BTC_PROB_DATASET_INCLUDE_LABS", "0"):
        pattern = os.environ.get("BTC_PROB_DATASET_LAB_GLOB", str(LAB_JOURNAL_GLOB))
        sources.extend(_expand_journal_pattern(pattern))
    extra_raw = os.environ.get("BTC_PROB_DATASET_EXTRA_JOURNALS", "")
    if extra_raw:
        for part in extra_raw.split(":"):
            part = part.strip()
            if part:
                sources.extend(_expand_journal_pattern(part))

    deduped: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        key = str(Path(source).resolve())
        if key not in seen and Path(source).exists():
            seen.add(key)
            deduped.append(Path(source).resolve())
    return deduped


def load_journal_trades_from_sources(sources: list[Path]):
    """Load resolved paper trades from explicit journal files."""
    trades = []
    loaded_sources = 0
    for source in sources:
        source = Path(source)
        if not source.exists():
            print(f"[dataset] Journal not found: {source}", file=sys.stderr)
            continue
        loaded_sources += 1
        with source.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    if t.get("mode") != "paper":
                        continue
                    if t.get("won") is None:
                        continue  # unresolved
                    t = dict(t)
                    t["_source_journal"] = str(source)
                    t["_source_line"] = line_no
                    if not t.get("strategy"):
                        try:
                            t["strategy"] = source.parents[1].name
                        except Exception:
                            t["strategy"] = "unknown"
                    trades.append(t)
                except Exception:
                    continue
    print(f"[dataset] Loaded {len(trades)} paper trades from {loaded_sources} journal(s)")
    return trades


def load_journal_trades():
    """Load all resolved paper trades from configured journal sources."""
    sources = journal_sources()
    if not sources:
        print("[dataset] No journal sources found", file=sys.stderr)
        return []
    trades = load_journal_trades_from_sources(sources)
    return trades


def _clamp_prob(value: float, default: float = 0.5) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if not math.isfinite(out):
        out = default
    return min(0.999999, max(0.000001, out))


def normalize_poly_prices(direction: str, chosen_poly_price: float) -> dict:
    """Normalize a journal's chosen-side price into Up/Down market prices.

    The probability model predicts absolute Up probability. Paper journals store
    the price of whichever side we bought. For a Down trade, a chosen-side price
    of 0.62 means Up-implied probability is roughly 0.38 in a binary market.
    """
    direction = str(direction or "Up")
    chosen = _clamp_prob(chosen_poly_price)
    if direction == "Down":
        poly_down = chosen
        poly_up = 1.0 - chosen
    else:
        poly_up = chosen
        poly_down = 1.0 - chosen
    return {
        "poly_price_up": _clamp_prob(poly_up),
        "poly_price_down": _clamp_prob(poly_down),
        "chosen_poly_price": chosen,
    }


def actual_up_outcome(direction: str, won: bool) -> int:
    """Return the actual market outcome label: 1=Up won, 0=Down won.

    Journals store whether our chosen direction won. The probability model and
    backtest expect an absolute Up/Down label, so Down+won means outcome=Down,
    not Up.
    """
    direction = str(direction or "Up")
    if direction == "Up":
        return 1 if bool(won) else 0
    return 0 if bool(won) else 1


def journal_external_features(trade: dict) -> dict:
    """Preserve external derivatives features captured at entry time."""
    if context_feature_subset is None:
        return {}
    raw = {}
    if isinstance(trade.get("prob_features"), dict):
        raw.update(trade.get("prob_features") or {})
    if trade.get("external_bull_score") is not None:
        raw["external_bull_score"] = trade.get("external_bull_score")
    try:
        return context_feature_subset(raw)
    except Exception:
        return {}


def journal_poly_price_for_features(trade: dict) -> float:
    """Return the pre-trade chosen-side probability seen by the gate.

    Prefer `prob_features.poly_price_enter` because it is captured before
    execution. Fallback to journal `poly_price` for old rows.
    """
    prob_features = trade.get("prob_features") if isinstance(trade.get("prob_features"), dict) else {}
    if prob_features and prob_features.get("poly_price_enter") is not None:
        return _clamp_prob(prob_features.get("poly_price_enter"))
    return _clamp_prob(trade.get("poly_price", 0.5))


def journal_feature_snapshot(trade: dict) -> Optional[dict]:
    """Use the feature snapshot captured at entry time when present.

    Parallel labs already journal `prob_features` built before the paper trade.
    Reusing that snapshot is faster and more honest than fetching today's
    orderbook while reconstructing an old window. We still normalize the market
    price fields to absolute Up/Down semantics.
    """
    raw = trade.get("prob_features")
    if not isinstance(raw, dict):
        return None
    # Require at least a technical feature to avoid treating a tiny partial map
    # as complete.
    if not any(k in raw for k in ("rsi_5m", "macd_hist_5m", "bb_pos_5m", "ob_imbalance")):
        return None
    feats = dict(raw)
    direction = trade.get("direction", "Up")
    prices = normalize_poly_prices(direction, journal_poly_price_for_features(trade))
    poly_up = prices["poly_price_up"]
    feats["poly_price_enter"] = poly_up
    feats["poly_price_down"] = prices["poly_price_down"]
    feats["chosen_poly_price"] = prices["chosen_poly_price"]
    feats["poly_vs_50"] = (poly_up - 0.5) * 2
    feats["poly_bucket_high"] = 1.0 if poly_up > 0.65 else 0.0
    feats["poly_bucket_very_high"] = 1.0 if poly_up > 0.80 else 0.0
    feats["direction_sign"] = 1.0 if direction == "Up" else -1.0
    feats["tf_minutes"] = float(trade.get("window_tf", feats.get("tf_minutes", 5.0)) or 5.0)
    if trade.get("btc_delta") is not None:
        feats["btc_delta"] = float(trade.get("btc_delta") or 0.0)
    return feats


def dataset_key_from_meta(meta: dict) -> tuple:
    """Stable dedupe key for train rows.

    Lab journals intentionally contain multiple strategy rows for the same
    Polymarket window. Dedupe by source line when available; legacy rows fall
    back to the old window/market key.
    """
    if meta.get("source_journal") and meta.get("source_line") is not None:
        return ("source", meta.get("source_journal"), int(meta.get("source_line")))
    if meta.get("strategy") and meta.get("placed_at"):
        return (
            "trade",
            meta.get("window_start"),
            meta.get("market_id"),
            meta.get("strategy"),
            meta.get("placed_at"),
        )
    return ("window", meta.get("window_start"), meta.get("market_id"))


def dataset_key_from_trade(trade: dict) -> tuple:
    if trade.get("_source_journal") and trade.get("_source_line") is not None:
        return ("source", trade.get("_source_journal"), int(trade.get("_source_line")))
    if trade.get("strategy") and trade.get("placed_at"):
        return (
            "trade",
            trade.get("window_start"),
            trade.get("market_id"),
            trade.get("strategy"),
            trade.get("placed_at"),
        )
    return ("window", trade.get("window_start"), trade.get("market_id"))


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(append: bool = False):
    """Build or append to the training dataset."""
    require_feature_snapshot = _env_bool("BTC_PROB_DATASET_REQUIRE_FEATURE_SNAPSHOT", "0")
    existing = []
    if append and OUT_FILE.exists():
        with OUT_FILE.open() as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except:
                    continue
        print(f"[dataset] Append mode: {len(existing)} existing rows")

    trades = load_journal_trades()
    existing_keys = {
        dataset_key_from_meta(r.get("meta") or {})
        for r in existing
        if isinstance(r.get("meta"), dict)
    }

    new_rows = []
    skipped = 0

    for i, t in enumerate(trades):
        key = dataset_key_from_trade(t)
        if key in existing_keys:
            skipped += 1
            continue

        ws = t.get("window_start")
        tf = t.get("window_tf", 5)
        if not ws:
            skipped += 1
            continue

        we = ws + tf * 60
        direction = t.get("direction", "Up")
        poly_price = journal_poly_price_for_features(t)
        delta = float(t.get("btc_delta", 0))

        feats = journal_feature_snapshot(t)
        if feats is None and require_feature_snapshot:
            skipped += 1
            continue
        if feats is None:
            feats = extract_features_for_window(ws, we, tf, direction, poly_price, delta)
        if feats is None:
            skipped += 1
            continue

        feats.update(journal_external_features(t))

        # Outcome: 1 = Up won, 0 = Down won. Journal `won` is relative to the
        # chosen direction, so convert it back to an absolute market label.
        outcome = actual_up_outcome(direction, bool(t.get("won")))

        row = {
            "features": feats,
            "outcome": outcome,
            "meta": {
                "window_start": ws,
                "window_end": we,
                "window_tf": tf,
                "market_id": t.get("market_id"),
                "journal_idx": i,
                "placed_at": t.get("placed_at"),
                "strategy": t.get("strategy"),
                "source_journal": t.get("_source_journal"),
                "source_line": t.get("_source_line"),
                "price_source": t.get("price_source"),
                "direction": direction,
            }
        }
        new_rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"[dataset] Processed {i+1}/{len(trades)} trades")

    all_rows = existing + new_rows
    print(f"[dataset] New rows: {len(new_rows)}, Skipped: {skipped}, Total: {len(all_rows)}")

    # Write
    with OUT_FILE.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"[dataset] Written to {OUT_FILE}")

    # Stats
    outcomes = [r["outcome"] for r in all_rows]
    n_up = sum(outcomes)
    n_down = len(outcomes) - n_up
    wr = n_up / len(outcomes) if outcomes else 0.5
    print(f"[dataset] Labels: {n_up} Up / {n_down} Down, WR={wr:.1%}")
    return all_rows


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--append", action="store_true", help="Append to existing dataset")
    p.add_argument("--stats", action="store_true", help="Print dataset stats and exit")
    args = p.parse_args()

    if args.stats:
        if not OUT_FILE.exists():
            print("No dataset found. Run without --stats first.")
            sys.exit(1)
        rows = []
        with OUT_FILE.open() as f:
            for line in f:
                try: rows.append(json.loads(line))
                except: continue
        feats = rows[0]["features"] if rows else {}
        outcomes = [r["outcome"] for r in rows]
        print(f"Total rows: {len(rows)}")
        print(f"Features ({len(feats)}): {sorted(feats.keys())}")
        print(f"Up: {sum(outcomes)}, Down: {len(outcomes)-sum(outcomes)}, WR: {sum(outcomes)/len(outcomes):.1%}")
        sys.exit(0)

    build_dataset(append=args.append)
