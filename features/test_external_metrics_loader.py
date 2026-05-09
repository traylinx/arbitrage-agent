"""Regression tests for `external_metrics_loader`.

Captures bugs that bit us during v0.6/v0.7 ship:
- v0.7 bug: `load_cross_exchange_trades` stripped UTC tz from `available_at`
  when copying via `.values`, causing `Snapshot._filtered.searchsorted` to
  silently return 0 rows for every basis-feature compute. The fix preserves
  tz-aware Series; this test verifies the dtype + that filtering works.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .external_metrics_loader import load_cross_exchange_trades
from .feature_base import Snapshot


def _write_fake_kline_shard(out_root: Path, exchange: str, day_iso: str,
                            n_minutes: int = 60) -> None:
    """Write a small fake parquet shard matching the real backfill schema."""
    base = datetime.fromisoformat(day_iso).replace(tzinfo=timezone.utc)
    rows = []
    for i in range(n_minutes):
        t = base + timedelta(minutes=i)
        t_close = t + timedelta(minutes=1)
        rows.append({
            "open_time_ms": int(t.timestamp() * 1000),
            "open_time": t,
            "open": 100_000.0 + i,
            "high": 100_100.0 + i,
            "low":  99_900.0 + i,
            "close": 100_000.0 + i,
            "volume": 1.0,
            "close_time": t_close,
            "close_time_ms": int(t_close.timestamp() * 1000),
            "available_at": t_close,
            "exchange": exchange,
        })
    df = pd.DataFrame(rows)
    day_dir = out_root / f"exchange={exchange}"
    day_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   day_dir / f"date={day_iso}.parquet", compression="snappy")


class TestCrossExchangeLoaderTzAware(unittest.TestCase):
    """v0.7 regression: cross-exchange klines must produce tz-aware
    `available_at` so `Snapshot._filtered` can searchsorted against
    a tz-aware decision_ts.
    """

    def test_available_at_is_tz_aware_utc(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_kline_shard(root, "coinbase", "2026-04-15")
            df = load_cross_exchange_trades("coinbase", root)
            self.assertFalse(df.empty)
            dtype = str(df["available_at"].dtype)
            self.assertIn("UTC", dtype,
                          f"available_at must be tz-aware UTC, got {dtype}")

    def test_filter_with_tz_aware_decision_ts_returns_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_kline_shard(root, "bybit", "2026-04-15", n_minutes=30)
            df = load_cross_exchange_trades("bybit", root)
            snap = Snapshot(btc_trades={"bybit": df})
            # decision_ts ~halfway through the shard
            ts = datetime(2026, 4, 15, 0, 20, 0, tzinfo=timezone.utc)
            sub = snap.trades_as_of("bybit", ts)
            self.assertGreater(len(sub), 0,
                               "filter on tz-aware decision_ts must return rows; "
                               "v0.7 bug returned 0")
            # And it must respect the cut: latest available_at <= ts
            latest = sub["available_at"].max()
            self.assertLessEqual(latest, ts)


class TestCrossExchangeLoaderEmpty(unittest.TestCase):
    """Returning an empty frame for a missing exchange is graceful behavior."""

    def test_missing_exchange_returns_empty_df(self):
        with TemporaryDirectory() as tmp:
            df = load_cross_exchange_trades("hyperliquid", Path(tmp))
            self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
