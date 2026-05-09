"""as_of_test.py — V2 §2g leakage gate.

Codex finding (HIGH): "no lookahead leakage verified by walk-forward"
is insufficient. Walk-forward only protects time splits. It does NOT
catch features computed with centered windows, current incomplete
candles, scaler fit on full data, global z-scores, late-arriving
Coinglass values, or post-hoc market resolution joins.

The fix per V2 §2: every feature must implement an explicit as-of
contract, and this test enforces it.

The contract: for any decision_ts T, the feature value is identical
whether the snapshot contains rows after T or only rows up to T.
Equivalently: features depend ONLY on the past at every T.

This module exposes:
  - `assert_as_of_invariant(spec, snapshot, decision_ts)`: asserts the
    feature is invariant to future data being added/removed.
  - `run_full_suite(snapshot, decision_ts_list)`: runs every registered
    feature against every decision_ts and reports any leakage.
  - A unittest-compatible TestCase that pytest can pick up in CI.

Usage in CI:
    python3.11 -m unittest features.as_of_test
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .feature_base import (
    FeatureSpec,
    Snapshot,
    all_features,
    compute_all_features,
    get_feature,
)


# ─── Core invariant assertion ────────────────────────────────────────────────


@dataclass
class LeakageReport:
    feature: str
    decision_ts: datetime
    full_value: float
    truncated_value: float

    @property
    def differs(self) -> bool:
        # NaN == NaN, otherwise tight tolerance.
        if math.isnan(self.full_value) and math.isnan(self.truncated_value):
            return False
        if math.isnan(self.full_value) or math.isnan(self.truncated_value):
            return True
        return abs(self.full_value - self.truncated_value) > 1e-9


def truncate_snapshot(snapshot: Snapshot, decision_ts: datetime) -> Snapshot:
    """Build a new Snapshot containing ONLY rows where available_at <= decision_ts.

    A correctly-implemented feature returns the same value on `snapshot`
    and on `truncate_snapshot(snapshot, T)` because its contract only
    permits reading rows where available_at <= T anyway.
    """
    def trunc_dict(d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {
            k: (df[df["available_at"] <= decision_ts] if not df.empty else df)
            for k, df in d.items()
        }

    def trunc_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df[df["available_at"] <= decision_ts]

    return Snapshot(
        btc_trades=trunc_dict(snapshot.btc_trades),
        btc_bookticker=trunc_dict(snapshot.btc_bookticker),
        derivatives=trunc_dict(snapshot.derivatives),
        pm_book=trunc_df(snapshot.pm_book),
        pm_trades=trunc_df(snapshot.pm_trades),
        pm_market=dict(snapshot.pm_market),
    )


def check_feature_at(
    spec: FeatureSpec,
    snapshot_full: Snapshot,
    decision_ts: datetime,
) -> LeakageReport:
    """Compute the feature on the full and truncated snapshots and
    report whether they match.
    """
    full = spec.compute(snapshot_full, decision_ts)
    trunc_snap = truncate_snapshot(snapshot_full, decision_ts)
    truncated = spec.compute(trunc_snap, decision_ts)
    return LeakageReport(
        feature=spec.name,
        decision_ts=decision_ts,
        full_value=full,
        truncated_value=truncated,
    )


def run_full_suite(
    snapshot: Snapshot, decision_ts_list: list[datetime]
) -> list[LeakageReport]:
    """Run every registered feature against every decision_ts and
    return the list of LeakageReports that differ (empty list = clean).
    """
    bad: list[LeakageReport] = []
    for spec in all_features():
        for ts in decision_ts_list:
            r = check_feature_at(spec, snapshot, ts)
            if r.differs:
                bad.append(r)
    return bad


# ─── Synthetic snapshot generator for self-tests ─────────────────────────────


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0,
         second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def make_synthetic_snapshot(num_rows: int = 600) -> Snapshot:
    """Produce a Snapshot of `num_rows` minute bars across 5 exchanges +
    2 derivatives sources + a Polymarket book. Values are deterministic
    so as-of tests are reproducible.
    """
    base = _utc(2026, 5, 1, 12, 0, 0)
    times = [base + timedelta(minutes=i) for i in range(num_rows)]

    btc_trades = {}
    btc_bookticker = {}
    for i, ex in enumerate(["binance", "bybit", "coinbase", "bitget", "hyperliquid"]):
        # Each exchange has a tiny offset so basis-style features have non-zero
        # output. The sin pattern matches across exchanges so as-of invariance
        # is testable (truncating the snapshot must not change the last value).
        prices = [100000 + 50 * math.sin(j * 0.05) + i * 5 for j in range(num_rows)]
        btc_trades[ex] = pd.DataFrame({
            "available_at": times,
            "price": prices,
            "side": ["buy" if j % 2 == 0 else "sell" for j in range(num_rows)],
            "size": [0.1 + 0.01 * j for j in range(num_rows)],
        })
        btc_bookticker[ex] = pd.DataFrame({
            "available_at": times,
            "bid": [p - 1 for p in prices],
            "ask": [p + 1 for p in prices],
            "bid_size": [10 + 0.1 * j for j in range(num_rows)],
            "ask_size": [12 + 0.1 * j for j in range(num_rows)],
        })

    deriv_times = times[::5]  # 5-minute cadence
    n_deriv = num_rows // 5
    derivatives = {
        "coinalyze": pd.DataFrame({
            "available_at": deriv_times,
            "oi_open": [95_000 + 50 * (i % 100) for i in range(n_deriv)],
            "oi_high": [95_100 + 50 * (i % 100) for i in range(n_deriv)],
            "oi_low":  [94_900 + 50 * (i % 100) for i in range(n_deriv)],
            "oi_close": [95_000 + 50 * (i % 100) for i in range(n_deriv)],
            "funding_close": [0.0001 * math.sin(i * 0.1) for i in range(n_deriv)],
            "predicted_funding_close": [0.0001 * math.sin(i * 0.1) + 0.00002 for i in range(n_deriv)],
            "liquidations_long": [1000 + (i * 13 % 500) for i in range(n_deriv)],
            "liquidations_short": [800 + (i * 17 % 400) for i in range(n_deriv)],
            "long_short_ratio": [1.0 + 0.01 * (i % 30) for i in range(n_deriv)],
            "long_pct": [55 + 0.5 * (i % 10) for i in range(n_deriv)],
            "short_pct": [45 - 0.5 * (i % 10) for i in range(n_deriv)],
        }),
        "coinglass": pd.DataFrame({
            "available_at": deriv_times,
            "oi_agg_open":  [70_000 + 100 * (i % 50) for i in range(n_deriv)],
            "oi_agg_high":  [70_200 + 100 * (i % 50) for i in range(n_deriv)],
            "oi_agg_low":   [69_800 + 100 * (i % 50) for i in range(n_deriv)],
            "oi_agg_close": [70_000 + 100 * (i % 50) for i in range(n_deriv)],
            "funding_binance_close": [0.00012 * math.sin(i * 0.1) for i in range(n_deriv)],
            "liq_long_usd":  [1_000_000 + (i * 13 % 500_000) for i in range(n_deriv)],
            "liq_short_usd": [800_000 + (i * 17 % 400_000) for i in range(n_deriv)],
            "binance_taker_buy_vol":  [10_000 + (i * 7 % 2_000) for i in range(n_deriv)],
            "binance_taker_sell_vol": [10_500 + (i * 11 % 2_000) for i in range(n_deriv)],
            "binance_cvd": [-500 + (i * 3 % 800) - 400 for i in range(n_deriv)],
            "taker_buy_agg_usd":  [50_000_000 + (i * 17 % 5_000_000) for i in range(n_deriv)],
            "taker_sell_agg_usd": [51_000_000 + (i * 19 % 5_000_000) for i in range(n_deriv)],
            "lsr_long_pct": [62 + 0.3 * (i % 10) for i in range(n_deriv)],
            "lsr_short_pct": [38 - 0.3 * (i % 10) for i in range(n_deriv)],
            "lsr_ratio": [1.6 + 0.05 * (i % 10) for i in range(n_deriv)],
            "depth_bids_usd": [200_000_000 + 1_000_000 * (i % 20) for i in range(n_deriv)],
            "depth_bids_qty": [2_000 + 50 * (i % 20) for i in range(n_deriv)],
            "depth_asks_usd": [195_000_000 + 1_000_000 * (i % 20) for i in range(n_deriv)],
            "depth_asks_qty": [1_950 + 50 * (i % 20) for i in range(n_deriv)],
        }),
    }

    pm_book = pd.DataFrame({
        "available_at": times,
        "yes_bid": [0.50 + 0.01 * math.sin(i * 0.1) for i in range(num_rows)],
        "yes_ask": [0.51 + 0.01 * math.sin(i * 0.1) for i in range(num_rows)],
        "no_bid": [0.49 - 0.01 * math.sin(i * 0.1) for i in range(num_rows)],
        "no_ask": [0.50 - 0.01 * math.sin(i * 0.1) for i in range(num_rows)],
    })
    pm_trades = pd.DataFrame({
        "available_at": times[::3],
        "side": ["YES" if i % 2 == 0 else "NO" for i in range(num_rows // 3)],
        "size": [10 + 0.1 * i for i in range(num_rows // 3)],
        "price": [0.50 for _ in range(num_rows // 3)],
    })
    pm_market = {
        "slug": "btc-updown-5m-fixture",
        "endDate": (base + timedelta(minutes=num_rows)).isoformat(),
        "tickSize": 0.01,
    }

    return Snapshot(
        btc_trades=btc_trades,
        btc_bookticker=btc_bookticker,
        derivatives=derivatives,
        pm_book=pm_book,
        pm_trades=pm_trades,
        pm_market=pm_market,
    )


def make_decision_ts_list(num: int = 5) -> list[datetime]:
    """A handful of decision timestamps inside the synthetic snapshot."""
    base = _utc(2026, 5, 1, 12, 0, 0)
    return [base + timedelta(minutes=100 * (i + 1)) for i in range(num)]


# ─── Unittest TestCase — runs against the live registry ──────────────────────


class TestAsOfInvariant(unittest.TestCase):
    """Pytest-discoverable harness: every registered feature must pass
    the as-of invariant. Empty registry passes trivially (the harness
    is itself tested by `TestHarness` below using a poison feature).
    """

    def test_all_registered_features_clean(self):
        snap = make_synthetic_snapshot()
        timestamps = make_decision_ts_list()
        bad = run_full_suite(snap, timestamps)
        if bad:
            msg = "as-of leakage detected:\n"
            for r in bad[:10]:
                msg += (f"  feature={r.feature} ts={r.decision_ts.isoformat()} "
                        f"full={r.full_value} truncated={r.truncated_value}\n")
            if len(bad) > 10:
                msg += f"  ... and {len(bad) - 10} more\n"
            self.fail(msg)


# ─── Self-test: prove the harness catches leakage with a poison feature ─────


class TestHarnessSelfCheck(unittest.TestCase):
    """The harness itself must catch a known-bad feature. Without this
    self-test, an empty harness would silently pass everything.
    """

    def test_poison_feature_is_caught(self):
        from .feature_base import register, reset_registry, all_features
        # Snapshot of registry to restore after
        from .feature_base import _REGISTRY  # noqa: WPS437
        saved = dict(_REGISTRY)
        try:
            reset_registry()

            @register("future_peeker", category="poison",
                      description="reads beyond decision_ts on purpose")
            def future_peeker(snap: Snapshot, ts: datetime) -> float:
                # Deliberately uses ALL trades, not as-of-truncated.
                # When the harness gives it a truncated snapshot, the value
                # changes — that's the leakage signal.
                df = snap.btc_trades.get("binance", pd.DataFrame())
                if df.empty:
                    return 0.0
                return float(df["price"].iloc[-1])  # last row regardless of ts

            snap = make_synthetic_snapshot()
            ts = make_decision_ts_list()[0]  # first decision_ts (~minute 100)
            spec = get_feature("future_peeker")
            r = check_feature_at(spec, snap, ts)
            self.assertTrue(r.differs,
                            f"harness MUST detect leakage. full={r.full_value} "
                            f"trunc={r.truncated_value}")
        finally:
            reset_registry()
            _REGISTRY.update(saved)

    def test_correct_feature_passes(self):
        """A trailing-only feature (price at last bar <= ts) should pass."""
        from .feature_base import register, reset_registry
        from .feature_base import _REGISTRY  # noqa: WPS437
        saved = dict(_REGISTRY)
        try:
            reset_registry()

            @register("trailing_close", category="test")
            def trailing_close(snap: Snapshot, ts: datetime) -> float:
                df = snap.trades_as_of("binance", ts)
                if df.empty:
                    return float("nan")
                return float(df["price"].iloc[-1])

            snap = make_synthetic_snapshot()
            ts = make_decision_ts_list()[0]
            spec = get_feature("trailing_close")
            r = check_feature_at(spec, snap, ts)
            self.assertFalse(r.differs,
                             f"correct trailing feature must NOT trip harness. "
                             f"full={r.full_value} trunc={r.truncated_value}")
        finally:
            reset_registry()
            _REGISTRY.update(saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
