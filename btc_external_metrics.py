#!/usr/bin/env python3
"""
BTC external derivatives metrics.

Pulls usable non-ordering data from:
  - Coinalyze free API: aggregated OI, liquidations, funding, predicted funding,
    long/short ratio.
  - CoinGlass Startup: OI, liquidation, funding, long/short, taker volume, CVD,
    orderbook ask/bid history.
  - Binance Futures / Bybit public REST: free per-venue OI, L/S, taker flow,
    funding.
  - Bitget public futures API: 5m/15m account long/short, current OI, funding,
    ticker spread, depth imbalance.
  - Hyperliquid public info API: DEX L2 depth + funding/premium.

Hard rule: this module is read-only. No trading/account POSTs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import fcntl
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
STATE_DIR = DATA_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_FILE = Path(
    os.path.expanduser(
        os.environ.get("BTC_EXTERNAL_CONTEXT_FILE", str(STATE_DIR / "external_market_context.json"))
    )
)
CONTEXT_TTL_SECONDS = float(os.environ.get("BTC_EXTERNAL_CONTEXT_TTL", "120"))
CONTEXT_LOCK_FILE = Path(
    os.path.expanduser(
        os.environ.get("BTC_EXTERNAL_CONTEXT_LOCK_FILE", str(STATE_DIR / "external_market_context.lock"))
    )
)

COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINALYZE_BASE = "https://api.coinalyze.net"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
BITGET_BASE = "https://api.bitget.com"
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"

EXTERNAL_MODEL_FEATURE_KEYS = [
    "ca_oi_30m_chg_pct",
    "ca_liq_1h_imbalance",
    "ca_funding_1h",
    "ca_pred_funding_1h",
    "ca_ls_30m_imbalance",
    "ca_ls_30m_chg",
    "ca_oi_now",
    "ca_funding_now",
    "cg_oi_30m_chg_pct",
    "cg_oi_1h_chg_pct",
    "cg_liq_30m_imbalance",
    "cg_liq_1h_imbalance",
    "cg_funding_30m",
    "cg_funding_1h",
    "cg_longs_30m_imbalance",
    "cg_taker_30m_imbalance",
    "cg_cvd_30m_imbalance",
    "cg_orderbook_30m_imbalance",
    "cg_oi_binance_share",
    "bn_oi_30m_chg_pct",
    "bn_top_ls_15m_imbalance",
    "bn_taker_15m_imbalance",
    "bn_global_ls_15m_imbalance",
    "bn_funding_latest",
    "by_oi_30m_chg_pct",
    "by_funding_latest",
    "bg_ls_5m_imbalance",
    "bg_ls_5m_chg",
    "bg_ls_15m_imbalance",
    "bg_taker_5m_imbalance",
    "bg_taker_15m_imbalance",
    "bg_trader_ls_5m_imbalance",
    "bg_position_ls_5m_imbalance",
    "bg_recent_trade_imbalance",
    "bg_depth_imbalance",
    "bg_spread_bps",
    "bg_mark_basis_bps",
    "bg_index_basis_bps",
    "bg_funding_hours_to_next",
    "hl_depth_imbalance",
    "hl_spread_bps",
    "hl_funding_latest",
    "hl_premium_latest",
    "hl_pred_funding_hl",
    "hl_pred_funding_binance",
    "hl_pred_funding_bybit",
    "external_bull_score",
]

# Same feature set for the legacy probability-gate model. The gate adapts to the
# feature list stored in the pickle, so adding these is backward-compatible.
EXTERNAL_PROB_FEATURE_KEYS = list(EXTERNAL_MODEL_FEATURE_KEYS)

_COINGLASS_ERRORS: Counter[str] = Counter()


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _imbalance(pos: float, neg: float) -> float:
    denom = abs(pos) + abs(neg)
    return (pos - neg) / denom if denom > 0 else 0.0


def _latest(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        return {}
    return max(
        (r for r in rows if isinstance(r, dict)),
        key=lambda r: _float(r.get("time") or r.get("ts") or r.get("t") or r.get("timestamp") or r.get("fundingRateTimestamp")),
        default={},
    )


def _pct_change(rows: Any, field: str) -> float:
    if not isinstance(rows, list) or len(rows) < 2:
        return 0.0
    clean = [r for r in rows if isinstance(r, dict) and field in r]
    clean.sort(key=lambda r: _float(r.get("time") or r.get("ts") or r.get("t") or r.get("timestamp") or r.get("fundingRateTimestamp")))
    if len(clean) < 2:
        return 0.0
    first = _float(clean[0].get(field))
    last = _float(clean[-1].get(field))
    return (last - first) / abs(first) if first else 0.0


def _delta_change(rows: Any, field: str) -> float:
    if not isinstance(rows, list) or len(rows) < 2:
        return 0.0
    clean = [r for r in rows if isinstance(r, dict) and field in r]
    clean.sort(key=lambda r: _float(r.get("time") or r.get("ts") or r.get("t") or r.get("timestamp") or r.get("fundingRateTimestamp")))
    if len(clean) < 2:
        return 0.0
    return _float(clean[-1].get(field)) - _float(clean[0].get(field))


@lru_cache(maxsize=8)
def _secret(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    try:
        cp = subprocess.run(
            ["makakoo", "secret", "get", name],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return ""


def _get_json(url: str, *, params: dict[str, Any], headers: Optional[dict[str, str]] = None, timeout: int = 10) -> dict:
    try:
        r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"code": str(r.status_code), "msg": r.text[:200]}
    except Exception as e:
        return {"code": "EXC", "msg": f"{type(e).__name__}: {e}"}


def _post_json(url: str, *, payload: dict[str, Any], timeout: int = 10) -> Any:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"code": str(r.status_code), "msg": r.text[:200]}
    except Exception as e:
        return {"code": "EXC", "msg": f"{type(e).__name__}: {e}"}


def _history(data: Any) -> list[dict[str, Any]]:
    """Coinalyze returns [{symbol, history:[...]}]. Normalize to history rows."""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("history"), list):
            return [r for r in first["history"] if isinstance(r, dict)]
        return [r for r in data if isinstance(r, dict)]
    return []


def _coinalyze(path: str, params: dict[str, Any]) -> Any:
    key = _secret("COINALYZE_API_KEY")
    if not key:
        return None
    return _get_json(
        COINALYZE_BASE + path,
        params={**params, "api_key": key},
        timeout=12,
    )


def _coinglass(path: str, params: dict[str, Any]) -> Any:
    key = _secret("COINGLASS_API_KEY")
    if not key:
        _COINGLASS_ERRORS["missing_key"] += 1
        return None
    d = _get_json(
        COINGLASS_BASE + path,
        params=params,
        headers={"accept": "application/json", "CG-API-KEY": key},
        timeout=12,
    )
    if str(d.get("code")) == "0":
        return d.get("data")
    _COINGLASS_ERRORS[str(d.get("code") or "unknown")] += 1
    return None


def _bitget(path: str, params: dict[str, Any]) -> Any:
    d = _get_json(BITGET_BASE + path, params=params, timeout=10)
    if str(d.get("code")) == "00000":
        return d.get("data")
    return None


def _binance_futures(path: str, params: dict[str, Any]) -> Any:
    return _get_json(BINANCE_FUTURES_BASE + path, params=params, timeout=10)


def _bybit(path: str, params: dict[str, Any]) -> Any:
    d = _get_json(BYBIT_BASE + path, params=params, timeout=10)
    if str(d.get("retCode")) == "0":
        return d.get("result")
    return None


def fetch_coinalyze_context() -> dict[str, float | int]:
    """Fetch free Coinalyze aggregated futures metrics. Primary $0 regime feed."""
    ctx: dict[str, float | int] = {"ca_ok": 0}
    if not _secret("COINALYZE_API_KEY"):
        ctx["ca_missing_key"] = 1
        return ctx

    ok = 0
    now = int(time.time())
    frm = now - int(os.environ.get("BTC_COINALYZE_LOOKBACK_SECONDS", str(3 * 24 * 3600)))
    symbol = os.environ.get("BTC_COINALYZE_SYMBOL", "BTCUSDT_PERP.A")
    common = {"symbols": symbol, "from": frm, "to": now}

    rows = _history(_coinalyze("/v1/open-interest-history", {**common, "interval": "30min"}))
    row = _latest(rows)
    if row:
        ok += 1
        ctx["ca_oi_30m"] = _float(row.get("c"))
        ctx["ca_oi_30m_chg_pct"] = _pct_change(rows[-8:], "c")

    rows = _history(_coinalyze("/v1/liquidation-history", {**common, "interval": "1hour"}))
    row = _latest(rows)
    if row:
        ok += 1
        long_liq = _float(row.get("l"))
        short_liq = _float(row.get("s"))
        ctx["ca_liq_1h_long"] = long_liq
        ctx["ca_liq_1h_short"] = short_liq
        ctx["ca_liq_1h_imbalance"] = _imbalance(short_liq, long_liq)

    rows = _history(_coinalyze("/v1/funding-rate-history", {**common, "interval": "1hour"}))
    row = _latest(rows)
    if row:
        ok += 1
        ctx["ca_funding_1h"] = _float(row.get("c"))
        ctx["ca_funding_1h_chg"] = _delta_change(rows[-8:], "c")

    rows = _history(_coinalyze("/v1/predicted-funding-rate-history", {**common, "interval": "1hour"}))
    row = _latest(rows)
    if row:
        ok += 1
        ctx["ca_pred_funding_1h"] = _float(row.get("c"))
        ctx["ca_pred_funding_1h_chg"] = _delta_change(rows[-8:], "c")

    rows = _history(_coinalyze("/v1/long-short-ratio-history", {**common, "interval": "30min"}))
    row = _latest(rows)
    if row:
        ok += 1
        long_pct = _float(row.get("l")) / 100.0
        short_pct = _float(row.get("s")) / 100.0
        ctx["ca_long_30m_pct"] = long_pct
        ctx["ca_short_30m_pct"] = short_pct
        ctx["ca_ls_30m_ratio"] = _float(row.get("r"))
        ctx["ca_ls_30m_imbalance"] = long_pct - short_pct
        ctx["ca_ls_30m_chg"] = _delta_change(rows[-8:], "l") / 100.0

    rows = _coinalyze("/v1/open-interest", {"symbols": symbol})
    if isinstance(rows, list) and rows:
        ok += 1
        ctx["ca_oi_now"] = _float(rows[0].get("value"))

    rows = _coinalyze("/v1/funding-rate", {"symbols": symbol})
    if isinstance(rows, list) and rows:
        ok += 1
        ctx["ca_funding_now"] = _float(rows[0].get("value"))

    ctx["ca_ok"] = 1 if ok else 0
    ctx["ca_endpoint_count"] = ok
    return ctx


def fetch_coinglass_context() -> dict[str, float | int]:
    """Fetch working CoinGlass Startup metrics. Returns numeric feature dict."""
    _COINGLASS_ERRORS.clear()
    ctx: dict[str, float | int] = {"cg_ok": 0, "cg_heatmap_available": 0}
    if not _secret("COINGLASS_API_KEY"):
        ctx["cg_missing_key"] = 1
        return ctx

    exchange_list = "Binance,OKX,Bybit"
    ok = 0

    for interval in ("30m", "1h", "4h"):
        rows = _coinglass(
            "/api/futures/open-interest/aggregated-history",
            {"symbol": "BTC", "interval": interval, "limit": 8, "unit": "usd"},
        )
        row = _latest(rows)
        if row:
            ok += 1
            ctx[f"cg_oi_{interval}_usd"] = _float(row.get("close"))
            ctx[f"cg_oi_{interval}_chg_pct"] = _pct_change(rows, "close")

        rows = _coinglass(
            "/api/futures/liquidation/aggregated-history",
            {"exchange_list": exchange_list, "symbol": "BTC", "interval": interval, "limit": 8},
        )
        row = _latest(rows)
        if row:
            ok += 1
            long_usd = _float(row.get("aggregated_long_liquidation_usd"))
            short_usd = _float(row.get("aggregated_short_liquidation_usd"))
            ctx[f"cg_liq_{interval}_long_usd"] = long_usd
            ctx[f"cg_liq_{interval}_short_usd"] = short_usd
            # Positive = more shorts liquidated than longs (bullish pressure feature).
            ctx[f"cg_liq_{interval}_imbalance"] = _imbalance(short_usd, long_usd)

    for interval in ("30m", "1h", "4h"):
        rows = _coinglass(
            "/api/futures/funding-rate/history",
            {"exchange": "Binance", "symbol": "BTCUSDT", "interval": interval, "limit": 8},
        )
        row = _latest(rows)
        if row:
            ok += 1
            ctx[f"cg_funding_{interval}"] = _float(row.get("close"))
            ctx[f"cg_funding_{interval}_chg"] = _delta_change(rows, "close")

        rows = _coinglass(
            "/api/futures/global-long-short-account-ratio/history",
            {"exchange": "Binance", "symbol": "BTCUSDT", "interval": interval, "limit": 8},
        )
        row = _latest(rows)
        if row:
            ok += 1
            long_pct = _float(row.get("global_account_long_percent")) / 100.0
            short_pct = _float(row.get("global_account_short_percent")) / 100.0
            ctx[f"cg_longs_{interval}_pct"] = long_pct
            ctx[f"cg_longs_{interval}_imbalance"] = long_pct - short_pct

    for interval in ("30m", "1h"):
        rows = _coinglass(
            "/api/futures/aggregated-taker-buy-sell-volume/history",
            {"exchange_list": exchange_list, "symbol": "BTC", "interval": interval, "limit": 8, "unit": "usd"},
        )
        row = _latest(rows)
        if row:
            ok += 1
            buy = _float(row.get("aggregated_buy_volume_usd"))
            sell = _float(row.get("aggregated_sell_volume_usd"))
            ctx[f"cg_taker_{interval}_buy_usd"] = buy
            ctx[f"cg_taker_{interval}_sell_usd"] = sell
            ctx[f"cg_taker_{interval}_imbalance"] = _imbalance(buy, sell)

        rows = _coinglass(
            "/api/futures/cvd/history",
            {"exchange": "Binance", "symbol": "BTCUSDT", "interval": interval, "limit": 8},
        )
        row = _latest(rows)
        if row:
            ok += 1
            buy = _float(row.get("taker_buy_vol"))
            sell = _float(row.get("taker_sell_vol"))
            ctx[f"cg_cvd_{interval}_delta"] = _float(row.get("cum_vol_delta"))
            ctx[f"cg_cvd_{interval}_imbalance"] = _imbalance(buy, sell)

    # Startup plan supports aggregated bid/ask history. Prefer multi-exchange
    # depth pressure over a single Binance pair snapshot.
    rows = _coinglass(
        "/api/futures/orderbook/aggregated-ask-bids-history",
        {"exchange_list": exchange_list, "symbol": "BTC", "interval": "30m", "limit": 4},
    )
    row = _latest(rows)
    if row:
        ok += 1
        bids = _float(row.get("aggregated_bids_usd") or row.get("bids_usd"))
        asks = _float(row.get("aggregated_asks_usd") or row.get("asks_usd"))
        ctx["cg_orderbook_30m_bids_usd"] = bids
        ctx["cg_orderbook_30m_asks_usd"] = asks
        ctx["cg_orderbook_30m_imbalance"] = _imbalance(bids, asks)
        ctx["cg_orderbook_30m_aggregated"] = 1

    chart = _coinglass(
        "/api/futures/open-interest/exchange-history-chart",
        {"symbol": "BTC", "range": "4h", "unit": "usd"},
    )
    if isinstance(chart, dict) and isinstance(chart.get("data_map"), dict):
        last_by_exchange = {
            exch: _float(vals[-1]) for exch, vals in chart["data_map"].items()
            if isinstance(vals, list) and vals
        }
        total = sum(last_by_exchange.values())
        if total > 0:
            ok += 1
            ctx["cg_oi_binance_share"] = last_by_exchange.get("BINANCE", 0.0) / total
            ctx["cg_oi_bybit_share"] = last_by_exchange.get("BYBIT", 0.0) / total
            ctx["cg_oi_cme_share"] = last_by_exchange.get("CME", 0.0) / total

    ctx["cg_ok"] = 1 if ok else 0
    ctx["cg_endpoint_count"] = ok
    if _COINGLASS_ERRORS:
        ctx["cg_error_count"] = sum(_COINGLASS_ERRORS.values())
        if _COINGLASS_ERRORS.get("429"):
            ctx["cg_rate_limited"] = 1
        ctx["cg_last_error_code"] = sorted(_COINGLASS_ERRORS, key=lambda k: _COINGLASS_ERRORS[k], reverse=True)[0]
    return ctx


def fetch_bitget_context() -> dict[str, float | int]:
    """Fetch Bitget public futures metrics. No auth required."""
    ctx: dict[str, float | int] = {"bg_ok": 0}
    ok = 0
    common = {"symbol": "BTCUSDT", "productType": "USDT-FUTURES"}

    ticker = _bitget("/api/v2/mix/market/ticker", common)
    if isinstance(ticker, list) and ticker:
        t = ticker[0]
        ok += 1
        bid = _float(t.get("bidPr"))
        ask = _float(t.get("askPr"))
        mid = (bid + ask) / 2.0 if bid and ask else _float(t.get("lastPr"))
        ctx["bg_price"] = _float(t.get("lastPr"))
        ctx["bg_mark_price"] = _float(t.get("markPrice"))
        ctx["bg_funding_rate"] = _float(t.get("fundingRate"))
        ctx["bg_oi_btc_ticker"] = _float(t.get("holdingAmount"))
        ctx["bg_spread_bps"] = ((ask - bid) / mid * 10_000.0) if mid else 0.0

    oi = _bitget("/api/v2/mix/market/open-interest", common)
    if isinstance(oi, dict):
        rows = oi.get("openInterestList") or []
        if rows:
            ok += 1
            ctx["bg_oi_btc"] = _float(rows[0].get("size"))

    funding = _bitget(
        "/api/v2/mix/market/history-fund-rate",
        {**common, "pageSize": "8"},
    )
    if isinstance(funding, list) and funding:
        ok += 1
        latest = _latest([{"ts": r.get("fundingTime"), **r} for r in funding])
        ctx["bg_funding_latest"] = _float(latest.get("fundingRate"))
        ctx["bg_funding_chg"] = _delta_change(
            [{"ts": r.get("fundingTime"), "fundingRate": r.get("fundingRate")} for r in funding],
            "fundingRate",
        )

    for period in ("5m", "15m"):
        rows = _bitget(
            "/api/v2/mix/market/account-long-short",
            {**common, "period": period},
        )
        if isinstance(rows, list) and rows:
            ok += 1
            latest = _latest(rows)
            long_ratio = _float(latest.get("longAccountRatio"))
            short_ratio = _float(latest.get("shortAccountRatio"))
            ctx[f"bg_long_{period}_pct"] = long_ratio
            ctx[f"bg_short_{period}_pct"] = short_ratio
            ctx[f"bg_ls_{period}_imbalance"] = long_ratio - short_ratio
            ctx[f"bg_ls_{period}_chg"] = _delta_change(rows, "longAccountRatio")

    for period in ("5m", "15m"):
        rows = _bitget(
            "/api/v2/mix/market/taker-buy-sell",
            {"symbol": "BTCUSDT", "period": period},
        )
        if isinstance(rows, list) and rows:
            ok += 1
            latest = _latest(rows)
            buy = _float(latest.get("buyVolume"))
            sell = _float(latest.get("sellVolume"))
            ctx[f"bg_taker_{period}_buy_btc"] = buy
            ctx[f"bg_taker_{period}_sell_btc"] = sell
            ctx[f"bg_taker_{period}_imbalance"] = _imbalance(buy, sell)

    rows = _bitget("/api/v2/mix/market/long-short", {"symbol": "BTCUSDT", "period": "5m"})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        long_ratio = _float(latest.get("longRatio"))
        short_ratio = _float(latest.get("shortRatio"))
        ctx["bg_trader_long_5m_pct"] = long_ratio
        ctx["bg_trader_short_5m_pct"] = short_ratio
        ctx["bg_trader_ls_5m_imbalance"] = long_ratio - short_ratio
        ctx["bg_trader_ls_5m_chg"] = _delta_change(rows, "longRatio")

    rows = _bitget("/api/v2/mix/market/position-long-short", {"symbol": "BTCUSDT", "period": "5m"})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        long_ratio = _float(latest.get("longPositionRatio"))
        short_ratio = _float(latest.get("shortPositionRatio"))
        ctx["bg_position_long_5m_pct"] = long_ratio
        ctx["bg_position_short_5m_pct"] = short_ratio
        ctx["bg_position_ls_5m_imbalance"] = long_ratio - short_ratio
        ctx["bg_position_ls_5m_chg"] = _delta_change(rows, "longPositionRatio")

    fills = _bitget("/api/v2/mix/market/fills", {**common, "limit": "100"})
    if isinstance(fills, list) and fills:
        ok += 1
        buy_qty = sum(_float(r.get("size")) for r in fills if str(r.get("side", "")).lower() == "buy")
        sell_qty = sum(_float(r.get("size")) for r in fills if str(r.get("side", "")).lower() == "sell")
        ctx["bg_recent_trade_buy_btc"] = buy_qty
        ctx["bg_recent_trade_sell_btc"] = sell_qty
        ctx["bg_recent_trade_count"] = len(fills)
        ctx["bg_recent_trade_imbalance"] = _imbalance(buy_qty, sell_qty)

    price_rows = _bitget("/api/v2/mix/market/symbol-price", common)
    if isinstance(price_rows, list) and price_rows:
        ok += 1
        row = price_rows[0]
        price = _float(row.get("price"))
        mark = _float(row.get("markPrice"))
        index = _float(row.get("indexPrice"))
        ctx["bg_symbol_price"] = price
        ctx["bg_mark_basis_bps"] = ((mark - price) / price * 10_000.0) if price else 0.0
        ctx["bg_index_basis_bps"] = ((index - price) / price * 10_000.0) if price else 0.0

    cur_funding = _bitget("/api/v2/mix/market/current-fund-rate", common)
    if isinstance(cur_funding, list) and cur_funding:
        ok += 1
        row = cur_funding[0]
        ctx["bg_current_funding_rate"] = _float(row.get("fundingRate"))
        next_update = _float(row.get("nextUpdate")) / 1000.0
        if next_update:
            ctx["bg_funding_hours_to_next"] = max(0.0, (next_update - time.time()) / 3600.0)

    contract = _bitget("/api/v2/mix/market/contracts", common)
    if isinstance(contract, list) and contract:
        ok += 1
        row = contract[0]
        ctx["bg_taker_fee_rate"] = _float(row.get("takerFeeRate"))
        ctx["bg_min_trade_usdt"] = _float(row.get("minTradeUSDT"))
        ctx["bg_symbol_status_ok"] = 1.0 if row.get("symbolStatus") == "normal" else 0.0

    depth = _bitget(
        "/api/v2/mix/market/merge-depth",
        {**common, "precision": "scale0", "limit": "20"},
    )
    if isinstance(depth, dict):
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        if bids and asks:
            ok += 1
            bid_qty = sum(_float(b[1]) for b in bids[:20] if isinstance(b, list) and len(b) >= 2)
            ask_qty = sum(_float(a[1]) for a in asks[:20] if isinstance(a, list) and len(a) >= 2)
            best_bid = _float(bids[0][0])
            best_ask = _float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
            ctx["bg_depth_bid_qty_20"] = bid_qty
            ctx["bg_depth_ask_qty_20"] = ask_qty
            ctx["bg_depth_imbalance"] = _imbalance(bid_qty, ask_qty)
            ctx["bg_depth_spread_bps"] = ((best_ask - best_bid) / mid * 10_000.0) if mid else 0.0

    ctx["bg_ok"] = 1 if ok else 0
    ctx["bg_endpoint_count"] = ok
    return ctx


def fetch_binance_futures_context() -> dict[str, float | int]:
    """Fetch free Binance Futures public regime metrics."""
    ctx: dict[str, float | int] = {"bn_ok": 0}
    ok = 0

    rows = _binance_futures("/futures/data/openInterestHist", {"symbol": "BTCUSDT", "period": "30m", "limit": 8})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        ctx["bn_oi_30m"] = _float(latest.get("sumOpenInterest"))
        ctx["bn_oi_value_30m"] = _float(latest.get("sumOpenInterestValue"))
        ctx["bn_oi_30m_chg_pct"] = _pct_change(rows, "sumOpenInterest")

    rows = _binance_futures("/futures/data/topLongShortPositionRatio", {"symbol": "BTCUSDT", "period": "15m", "limit": 8})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        long_pct = _float(latest.get("longAccount"))
        short_pct = _float(latest.get("shortAccount"))
        ctx["bn_top_ls_15m_imbalance"] = long_pct - short_pct

    rows = _binance_futures("/futures/data/takerlongshortRatio", {"symbol": "BTCUSDT", "period": "15m", "limit": 8})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        buy = _float(latest.get("buyVol"))
        sell = _float(latest.get("sellVol"))
        ctx["bn_taker_15m_imbalance"] = _imbalance(buy, sell)

    rows = _binance_futures("/futures/data/globalLongShortAccountRatio", {"symbol": "BTCUSDT", "period": "15m", "limit": 8})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        long_pct = _float(latest.get("longAccount"))
        short_pct = _float(latest.get("shortAccount"))
        ctx["bn_global_ls_15m_imbalance"] = long_pct - short_pct

    rows = _binance_futures("/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 8})
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        ctx["bn_funding_latest"] = _float(latest.get("fundingRate"))
        ctx["bn_funding_chg"] = _delta_change(rows, "fundingRate")

    ctx["bn_ok"] = 1 if ok else 0
    ctx["bn_endpoint_count"] = ok
    return ctx


def fetch_bybit_context() -> dict[str, float | int]:
    """Fetch free Bybit public OI/funding metrics."""
    ctx: dict[str, float | int] = {"by_ok": 0}
    ok = 0

    result = _bybit("/v5/market/open-interest", {"category": "linear", "symbol": "BTCUSDT", "intervalTime": "30min", "limit": 8})
    rows = result.get("list") if isinstance(result, dict) else []
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        ctx["by_oi_30m"] = _float(latest.get("openInterest"))
        ctx["by_oi_30m_chg_pct"] = _pct_change(rows, "openInterest")

    result = _bybit("/v5/market/funding/history", {"category": "linear", "symbol": "BTCUSDT", "limit": 8})
    rows = result.get("list") if isinstance(result, dict) else []
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        ctx["by_funding_latest"] = _float(latest.get("fundingRate"))
        ctx["by_funding_chg"] = _delta_change(rows, "fundingRate")

    ctx["by_ok"] = 1 if ok else 0
    ctx["by_endpoint_count"] = ok
    return ctx


def fetch_hyperliquid_context() -> dict[str, float | int]:
    """Fetch free Hyperliquid DEX depth + funding/premium."""
    ctx: dict[str, float | int] = {"hl_ok": 0}
    ok = 0

    book = _post_json(HYPERLIQUID_INFO, payload={"type": "l2Book", "coin": "BTC"}, timeout=12)
    if isinstance(book, dict):
        levels = book.get("levels") or []
        if isinstance(levels, list) and len(levels) >= 2:
            bids = levels[0] or []
            asks = levels[1] or []
            if bids and asks:
                ok += 1
                bid_qty = sum(_float(r.get("sz")) for r in bids[:20] if isinstance(r, dict))
                ask_qty = sum(_float(r.get("sz")) for r in asks[:20] if isinstance(r, dict))
                best_bid = _float(bids[0].get("px")) if isinstance(bids[0], dict) else 0.0
                best_ask = _float(asks[0].get("px")) if isinstance(asks[0], dict) else 0.0
                mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
                ctx["hl_depth_bid_qty_20"] = bid_qty
                ctx["hl_depth_ask_qty_20"] = ask_qty
                ctx["hl_depth_imbalance"] = _imbalance(bid_qty, ask_qty)
                ctx["hl_spread_bps"] = ((best_ask - best_bid) / mid * 10_000.0) if mid else 0.0

    start_ms = int((time.time() - 3 * 24 * 3600) * 1000)
    rows = _post_json(HYPERLIQUID_INFO, payload={"type": "fundingHistory", "coin": "BTC", "startTime": start_ms}, timeout=12)
    if isinstance(rows, list) and rows:
        ok += 1
        latest = _latest(rows)
        ctx["hl_funding_latest"] = _float(latest.get("fundingRate"))
        ctx["hl_premium_latest"] = _float(latest.get("premium"))
        ctx["hl_funding_chg"] = _delta_change(rows[-8:], "fundingRate")

    rows = _post_json(HYPERLIQUID_INFO, payload={"type": "predictedFundings"}, timeout=12)
    if isinstance(rows, list):
        btc = next((r for r in rows if isinstance(r, list) and r and r[0] == "BTC"), None)
        if btc and len(btc) >= 2 and isinstance(btc[1], list):
            ok += 1
            for venue, dst in (
                ("HlPerp", "hl_pred_funding_hl"),
                ("BinPerp", "hl_pred_funding_binance"),
                ("BybitPerp", "hl_pred_funding_bybit"),
            ):
                entry = next((x for x in btc[1] if isinstance(x, list) and len(x) >= 2 and x[0] == venue), None)
                if entry and isinstance(entry[1], dict):
                    ctx[dst] = _float(entry[1].get("fundingRate"))

    ctx["hl_ok"] = 1 if ok else 0
    ctx["hl_endpoint_count"] = ok
    return ctx


def add_composite_features(ctx: dict[str, Any]) -> dict[str, Any]:
    """Add normalized composite score. Feature only; not a trading decision."""
    score = 0.0
    # Coinalyze is the free primary aggregate regime feed. CoinGlass remains a
    # paid fallback until cancellation and adds CVD/orderbook history while active.
    score += 0.16 * _clamp(_float(ctx.get("ca_liq_1h_imbalance") or ctx.get("cg_liq_30m_imbalance")))
    score += 0.12 * _clamp(_float(ctx.get("ca_oi_30m_chg_pct") or ctx.get("cg_oi_30m_chg_pct")) * 20.0)
    score += 0.10 * _clamp(_float(ctx.get("ca_ls_30m_imbalance") or ctx.get("cg_longs_30m_imbalance")))
    score += 0.06 * _clamp(-_float(ctx.get("ca_funding_now") or ctx.get("ca_funding_1h") or ctx.get("cg_funding_30m")) * 100.0)
    score += 0.07 * _clamp(_float(ctx.get("cg_taker_30m_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("cg_cvd_30m_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("cg_orderbook_30m_imbalance")))
    score += 0.08 * _clamp(_float(ctx.get("bn_taker_15m_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("bn_top_ls_15m_imbalance")))
    score += 0.05 * _clamp(_float(ctx.get("bn_global_ls_15m_imbalance")))
    score += 0.05 * _clamp(_float(ctx.get("by_oi_30m_chg_pct")) * 20.0)
    score += 0.07 * _clamp(_float(ctx.get("bg_ls_5m_imbalance")))
    score += 0.08 * _clamp(_float(ctx.get("bg_taker_5m_imbalance")))
    score += 0.05 * _clamp(_float(ctx.get("bg_trader_ls_5m_imbalance")))
    score += 0.05 * _clamp(_float(ctx.get("bg_position_ls_5m_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("bg_recent_trade_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("bg_depth_imbalance")))
    score += 0.06 * _clamp(_float(ctx.get("hl_depth_imbalance")))
    score += 0.04 * _clamp(-_float(ctx.get("hl_premium_latest")) * 100.0)
    score += 0.03 * _clamp(_float(ctx.get("bg_ls_5m_chg")) * 20.0)
    ctx["external_bull_score"] = _clamp(score)
    return ctx


def _read_cache(ignore_ttl: bool = False) -> Optional[dict[str, Any]]:
    try:
        if not CONTEXT_FILE.exists():
            return None
        d = json.loads(CONTEXT_FILE.read_text())
        age = time.time() - float(d.get("fetched_at", 0))
        if ignore_ttl or age <= CONTEXT_TTL_SECONDS:
            d["cache_age_sec"] = age
            return d
    except Exception:
        return None
    return None


def _write_cache(ctx: dict[str, Any]) -> None:
    try:
        tmp = CONTEXT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(ctx, sort_keys=True) + "\n")
        tmp.replace(CONTEXT_FILE)
    except Exception:
        pass


class _RefreshLock:
    def __init__(self, path: Path):
        self.path = path
        self.f = None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a+")
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def __exit__(self, *exc) -> None:
        if not self.f:
            return
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()


def fetch_external_market_context(use_cache: bool = True) -> dict[str, Any]:
    """Return cached/fresh external context. Never raises."""
    if use_cache:
        cached = _read_cache(ignore_ttl=False)
        if cached:
            return cached

    stale = _read_cache(ignore_ttl=True)
    if use_cache:
        with _RefreshLock(CONTEXT_LOCK_FILE) as got_lock:
            if not got_lock:
                # Parallel paper workers can stampede on a shared stale cache.
                # One worker refreshes; the rest reuse stale data instead of
                # burning CoinGlass quota and triggering 429s.
                cached = _read_cache(ignore_ttl=False)
                if cached:
                    return cached
                if stale:
                    stale["stale_due_to_refresh_lock"] = 1
                    return stale
                return {"ok": 0, "source": "coinalyze+coinglass+binance+bybit+bitget+hyperliquid", "refresh_locked": 1}
            cached = _read_cache(ignore_ttl=False)
            if cached:
                return cached
            return _fetch_external_market_context_locked(stale)

    return _fetch_external_market_context_locked(stale)


def _fetch_external_market_context_locked(stale: Optional[dict[str, Any]]) -> dict[str, Any]:
    try:
        ctx: dict[str, Any] = {
            "fetched_at": time.time(),
            "source": "coinalyze+coinglass+binance+bybit+bitget+hyperliquid",
        }
        ctx.update(fetch_coinalyze_context())
        ctx.update(fetch_coinglass_context())
        ctx.update(fetch_binance_futures_context())
        ctx.update(fetch_bybit_context())
        ctx.update(fetch_bitget_context())
        ctx.update(fetch_hyperliquid_context())
        add_composite_features(ctx)
        ctx["ok"] = 1 if any(ctx.get(k) for k in ("ca_ok", "cg_ok", "bn_ok", "by_ok", "bg_ok", "hl_ok")) else 0
        _write_cache(ctx)
        return ctx
    except Exception as e:
        if stale:
            stale["stale_due_to_error"] = f"{type(e).__name__}: {e}"
            return stale
        return {"ok": 0, "source": "coinalyze+coinglass+binance+bybit+bitget+hyperliquid", "error": f"{type(e).__name__}: {e}"}


def context_feature_subset(ctx: dict[str, Any], keys: list[str] | None = None) -> dict[str, float]:
    keys = keys or EXTERNAL_MODEL_FEATURE_KEYS
    return {k: _float(ctx.get(k)) for k in keys}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore cache and fetch live")
    ap.add_argument("--features-only", action="store_true", help="print only model feature subset")
    args = ap.parse_args()

    context = fetch_external_market_context(use_cache=not args.fresh)
    if args.features_only:
        context = context_feature_subset(context)
    print(json.dumps(context, indent=2, sort_keys=True))
