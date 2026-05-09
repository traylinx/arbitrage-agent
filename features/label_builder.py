"""label_builder.py — V2 §3a/3b core: joins Polymarket resolutions to
feature snapshots at decision_ts, producing a training-ready parquet.

This is the bridge between raw data (`polymarket_resolutions/` +
`binance_klines/` + ...) and the trainer (`features/train.py`, future).

Decision semantics (V2 §3b):
    Mid-window mode confirmed by `_place_trade` call site (line 2634
    of btc_sniper_live.py): the bot fires INSIDE an active window.

    For backfill, we standardize:
      decision_ts = window_start + DECISION_OFFSET_SEC (default 60s)

    This gives the model a fixed offset from window-open, which:
      - matches a realistic moment the bot could fire
      - leaves enough trailing data for ret_60m / realized_vol_30m
      - is reproducible (same offset for every market)

Output columns: all registered feature names + binary_label_up_won +
window_start + decision_ts + tf_minutes + market metadata (slug,
condition_id, fee_rate, volume).

Schema-compatible with V2 §3c trainer: train.py reads this parquet
directly, fits walk-forward folds on `decision_ts`.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .feature_base import (
    Snapshot,
    all_features,
    compute_all_features,
    feature_schema_hash,
)
from .external_metrics_loader import (
    DEFAULT_BASE as DEFAULT_EXT_ROOT,
    DEFAULT_CROSS_EX_BASE,
    load_coinalyze_wide,
    load_coinglass_wide,
    load_cross_exchange_all,
)

DEFAULT_PM_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/polymarket_resolutions"
DEFAULT_BN_ROOT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/binance_klines"
DEFAULT_OUT = Path.home() / "MAKAKOO/data/arbitrage-agent/v2/parquet-cache/labeled_dataset"

DECISION_OFFSET_SEC_DEFAULT = 60  # 60s after window-open


# ─── Loaders ─────────────────────────────────────────────────────────────────


def load_pm_resolutions(root: Path, tf: Optional[int] = None) -> pd.DataFrame:
    """Read all polymarket_resolutions parquet shards into one DataFrame."""
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            dfs.append(d)
        except Exception as e:
            print(f"[label_builder] warning: failed to read {f}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if tf is not None:
        df = df[df["tf_minutes"] == tf].reset_index(drop=True)
    return df


def load_binance_klines(
    root: Path, symbol: str = "BTCUSDT", interval: str = "1m"
) -> pd.DataFrame:
    """Read all binance_klines parquet shards for one (symbol, interval)."""
    base = root / f"symbol={symbol}" / f"interval={interval}"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"[label_builder] warning: failed to read {f}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")
    df = df.reset_index(drop=True)
    return df


# ─── Snapshot builder for backfill ───────────────────────────────────────────


def klines_to_synthetic_trades(klines: pd.DataFrame) -> pd.DataFrame:
    """Adapt 1m OHLCV to the trades-frame shape that price.py features expect.

    price.py reads `available_at` and `price`. We approximate with the
    candle close: each candle becomes one row at `close_time` with
    price = close. Sub-minute resolution is lost but trailing 1m/5m/15m
    returns and minute-bar realized vol are recoverable.

    Production live data will produce real trade rows; this adapter is
    backfill-only.
    """
    if klines.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "available_at": klines["close_time"],
        "price": klines["close"],
        "side": "close",  # not real trade side; not used by price features
        "size": klines["volume"],
    })
    return df


def klines_to_synthetic_bookticker(klines: pd.DataFrame) -> pd.DataFrame:
    """Best-effort bookticker frame from OHLCV.

    True bid/ask are not in OHLCV. We approximate `bid = low`, `ask = high`,
    `bid_size = ask_size = volume / 2` so spread_bps and microprice features
    can compute SOMETHING (typically wider than reality). Production live
    data overrides this.
    """
    if klines.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "available_at": klines["close_time"],
        "bid": klines["low"],
        "ask": klines["high"],
        "bid_size": klines["volume"] / 2.0,
        "ask_size": klines["volume"] / 2.0,
    })
    return df


def build_global_snapshot(
    binance_klines: pd.DataFrame,
    derivatives: Optional[dict[str, pd.DataFrame]] = None,
    cross_exchange_trades: Optional[dict[str, pd.DataFrame]] = None,
) -> Snapshot:
    """Builds a Snapshot containing ALL historical data. The as-of
    contract is enforced inside the feature `compute()` calls — they
    filter by `available_at <= decision_ts`.

    For training we don't filter upfront because the same Snapshot is
    reused for every decision_ts. The features themselves do the
    truncation (and the as-of harness verifies they do it correctly).

    `cross_exchange_trades` is keyed by exchange name (coinbase/bybit/
    hyperliquid) and merged into `btc_trades` alongside Binance's
    synthetic-trades frame so basis features can compare prices.
    """
    btc_trades = {}
    btc_bookticker = {}
    if not binance_klines.empty:
        btc_trades["binance"] = klines_to_synthetic_trades(binance_klines)
        btc_bookticker["binance"] = klines_to_synthetic_bookticker(binance_klines)
    if cross_exchange_trades:
        for ex, frame in cross_exchange_trades.items():
            if frame is not None and not frame.empty:
                btc_trades[ex] = frame
    return Snapshot(
        btc_trades=btc_trades,
        btc_bookticker=btc_bookticker,
        derivatives=derivatives or {},
        pm_book=pd.DataFrame(),
        pm_trades=pd.DataFrame(),
        pm_market={},
    )


# ─── Joining: PM rows × decision_ts × features ──────────────────────────────


def build_labeled_dataset(
    pm_resolutions: pd.DataFrame,
    snapshot: Snapshot,
    decision_offset_sec: int = DECISION_OFFSET_SEC_DEFAULT,
    progress_every: int = 2000,
) -> pd.DataFrame:
    """For each PM market resolution row, compute features at
    `decision_ts = window_start + decision_offset_sec` and join the
    binary outcome.

    Returns a DataFrame with one row per market: features + label +
    metadata. NaN columns indicate missing source data at that
    decision_ts (e.g. for old markets pre-Binance-backfill window).
    """
    if pm_resolutions.empty:
        return pd.DataFrame()

    feature_names = [s.name for s in all_features()]
    schema_hash = feature_schema_hash()
    rows: list[dict] = []

    for i, mkt in pm_resolutions.iterrows():
        ws = int(mkt["window_start"])
        decision_ts = datetime.fromtimestamp(ws + decision_offset_sec, tz=timezone.utc)
        feats = compute_all_features(snapshot, decision_ts)
        rec = dict(feats)
        rec.update({
            "tf_minutes": int(mkt["tf_minutes"]),
            "window_start": ws,
            "decision_ts": decision_ts,
            "decision_offset_sec": decision_offset_sec,
            "binary_label_up_won": int(mkt["binary_label_up_won"]),
            "winner": str(mkt.get("winner", "")),
            "slug": str(mkt.get("slug", "")),
            "condition_id": str(mkt.get("condition_id", "")),
            "volume": float(mkt.get("volume", 0.0) or 0.0),
            "fee_rate": float(mkt.get("fee_rate", 0.0) or 0.0),
            "fee_taker_only": bool(mkt.get("fee_taker_only", False)),
            "schema_hash": schema_hash,
        })
        rows.append(rec)

        if progress_every and (i + 1) % progress_every == 0:
            non_nan = sum(1 for v in feats.values() if not (isinstance(v, float) and math.isnan(v)))
            print(f"  [label_builder] {i + 1}/{len(pm_resolutions)} "
                  f"non_nan_features={non_nan}/{len(feature_names)}")

    df = pd.DataFrame(rows)
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labeled training dataset from PM + Binance parquet.")
    p.add_argument("--pm-root", type=Path, default=DEFAULT_PM_ROOT)
    p.add_argument("--bn-root", type=Path, default=DEFAULT_BN_ROOT)
    p.add_argument("--ext-root", type=Path, default=DEFAULT_EXT_ROOT,
                   help="root for external_metrics/ parquet shards (coinalyze + coinglass)")
    p.add_argument("--cross-ex-root", type=Path, default=DEFAULT_CROSS_EX_BASE,
                   help="root for cross_exchange_klines/ parquet shards (coinbase + bybit + hyperliquid)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tf", type=int, choices=[5, 15], required=True,
                   help="Timeframe to build (one of 5 or 15)")
    p.add_argument("--decision-offset-sec", type=int, default=DECISION_OFFSET_SEC_DEFAULT)
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, only build the first N markets (smoke)")
    p.add_argument("--no-derivatives", action="store_true",
                   help="Skip loading external_metrics/ (cross-ex features will be NaN)")
    return p.parse_args()


def main() -> int:
    # Lazy-import so this script is runnable without first registering features
    # (you should still register them — this guard just avoids an import-loop).
    import features.price  # noqa: F401  (registers price features)
    import features.crossex  # noqa: F401

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[label_builder] tf={args.tf}m offset={args.decision_offset_sec}s")
    print(f"[label_builder] loading PM resolutions from {args.pm_root}")
    pm = load_pm_resolutions(args.pm_root, tf=args.tf)
    print(f"[label_builder]   {len(pm)} markets loaded")
    if pm.empty:
        print("[error] no PM rows", file=sys.stderr)
        return 2

    print(f"[label_builder] loading Binance klines from {args.bn_root}")
    bn = load_binance_klines(args.bn_root)
    print(f"[label_builder]   {len(bn)} candles loaded "
          f"({bn['close_time'].min()} → {bn['close_time'].max()})" if not bn.empty else "")
    if bn.empty:
        print("[warn] no Binance klines — features will be NaN", file=sys.stderr)

    derivatives: dict[str, pd.DataFrame] = {}
    if not args.no_derivatives:
        print(f"[label_builder] loading external_metrics from {args.ext_root}")
        ca = load_coinalyze_wide(args.ext_root)
        cg = load_coinglass_wide(args.ext_root)
        if not ca.empty:
            derivatives["coinalyze"] = ca
            print(f"[label_builder]   coinalyze: {len(ca)} rows "
                  f"({ca['available_at'].min()} → {ca['available_at'].max()})")
        if not cg.empty:
            derivatives["coinglass"] = cg
            print(f"[label_builder]   coinglass: {len(cg)} rows "
                  f"({cg['available_at'].min()} → {cg['available_at'].max()})")
        if not derivatives:
            print("[warn] no external_metrics found — cross-ex features will be NaN",
                  file=sys.stderr)

    print(f"[label_builder] loading cross_exchange_klines from {args.cross_ex_root}")
    cross_ex_trades = load_cross_exchange_all(base=args.cross_ex_root)
    for ex, df in cross_ex_trades.items():
        print(f"[label_builder]   {ex}: {len(df)} rows "
              f"({df['available_at'].min()} → {df['available_at'].max()})")
    if not cross_ex_trades:
        print("[warn] no cross_exchange_klines found — basis features will be NaN",
              file=sys.stderr)

    snap = build_global_snapshot(bn, derivatives=derivatives,
                                 cross_exchange_trades=cross_ex_trades)

    if args.limit > 0:
        pm = pm.head(args.limit)
        print(f"[label_builder] LIMIT {args.limit}, building partial dataset")

    print(f"[label_builder] building dataset for {len(pm)} markets, "
          f"{len(all_features())} features...")
    dataset = build_labeled_dataset(
        pm_resolutions=pm,
        snapshot=snap,
        decision_offset_sec=args.decision_offset_sec,
    )

    print(f"[label_builder] dataset shape: {dataset.shape}")

    out_path = args.out / f"tf={args.tf}m_offset={args.decision_offset_sec}s.parquet"
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(dataset, preserve_index=False), out_path,
                   compression="snappy")
    print(f"[label_builder] wrote {out_path}")
    print(f"[label_builder] file size: {out_path.stat().st_size / 1024:.1f} KB")

    # Quick sanity report
    print()
    print("=== sanity ===")
    print(f"  total rows: {len(dataset)}")
    print(f"  Up won: {dataset['binary_label_up_won'].sum()} "
          f"({dataset['binary_label_up_won'].mean() * 100:.1f}%)")
    feat_cols = [s.name for s in all_features()]
    for c in feat_cols:
        non_nan = dataset[c].notna().sum()
        if non_nan > 0:
            mean = dataset[c].mean()
            print(f"  {c:40s} non_nan={non_nan} mean={mean:+.6f}")
        else:
            print(f"  {c:40s} all NaN (waiting for live collector)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
