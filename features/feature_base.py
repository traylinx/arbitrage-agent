"""feature_base.py — V2 §2 feature contract.

Every feature is a callable that takes a `Snapshot` and a
`decision_ts`, and returns a single float (or NaN). The contract is:

    For any decision_ts T, the value returned MUST be computable using
    only rows where available_at <= T. Any reliance on data after T is
    leakage and the as-of test will fail it.

This file defines:
  - `Snapshot`: a typed view over the historical data needed to compute
    one feature row at a specific moment in time.
  - `Feature`: the callable Protocol every feature implements.
  - `register` decorator: makes the feature discoverable to the as-of
    test harness and the schema-hash machinery.
  - `feature_schema_hash`: deterministic sha256 of registered feature
    names + version tags, used by the predictor to refuse mismatched
    artifacts (V2 §4b).

Feature implementations live in adjacent modules (`price.py`,
`flow.py`, `crossex.py`, `polymarket.py`, ...). This file should never
import them — circularity-resistant.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd


# ─── Snapshot ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Snapshot:
    """A typed bag of historical dataframes available to a feature
    computation, one per source. Feature code MUST filter each frame
    by `df[df["available_at"] <= decision_ts]` before computing.

    Frames are pandas DataFrames with at least an `available_at`
    column (timezone-aware UTC datetime). Other columns vary per source.
    """
    # Per-exchange BTC trades / book frames. Keys: "binance", "bybit",
    # "coinbase", "bitget", "hyperliquid". Each frame must contain at
    # minimum: available_at, price, side. May contain bid, ask, sizes.
    btc_trades: dict[str, pd.DataFrame] = field(default_factory=dict)
    btc_bookticker: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Aggregate derivatives data.
    derivatives: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Polymarket book + trade frames for this market's two tokens.
    pm_book: pd.DataFrame = field(default_factory=pd.DataFrame)
    pm_trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Market metadata for the ACTIVE market. Always read-only at
    # decision time.
    pm_market: dict = field(default_factory=dict)

    def _filtered(self, df: pd.DataFrame, decision_ts: datetime) -> pd.DataFrame:
        """Internal: returns df[df.available_at <= decision_ts].

        Uses pandas `searchsorted` for O(log N) cut, requiring the input
        frame to be pre-sorted by `available_at`. For backfill builds
        this is critical: 20k markets × 16 features × 304k-row binance
        klines = >10⁸ row comparisons under naive O(N) filtering.
        """
        if df is None or df.empty:
            return df
        # If sorted (typical), use binary search; else fall back to mask.
        col = df["available_at"]
        if col.is_monotonic_increasing:
            idx = col.searchsorted(decision_ts, side="right")
            return df.iloc[:idx]
        return df[col <= decision_ts]

    def trades_as_of(self, exchange: str, decision_ts: datetime) -> pd.DataFrame:
        return self._filtered(self.btc_trades.get(exchange, pd.DataFrame()), decision_ts)

    def bookticker_as_of(self, exchange: str, decision_ts: datetime) -> pd.DataFrame:
        return self._filtered(self.btc_bookticker.get(exchange, pd.DataFrame()), decision_ts)

    def derivatives_as_of(self, source: str, decision_ts: datetime) -> pd.DataFrame:
        return self._filtered(self.derivatives.get(source, pd.DataFrame()), decision_ts)

    def pm_book_as_of(self, decision_ts: datetime) -> pd.DataFrame:
        return self._filtered(self.pm_book, decision_ts)

    def pm_trades_as_of(self, decision_ts: datetime) -> pd.DataFrame:
        return self._filtered(self.pm_trades, decision_ts)


# ─── Feature contract + registry ─────────────────────────────────────────────


@dataclass
class FeatureSpec:
    """Metadata for one registered feature."""
    name: str
    fn: Callable[[Snapshot, datetime], float]
    version: str
    category: str  # e.g. "price", "flow", "crossex", "pm", "regime"
    description: str = ""

    def compute(self, snapshot: Snapshot, decision_ts: datetime) -> float:
        try:
            v = self.fn(snapshot, decision_ts)
        except Exception:
            return float("nan")
        if v is None:
            return float("nan")
        if not isinstance(v, (int, float)):
            return float("nan")
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return float(v)


_REGISTRY: dict[str, FeatureSpec] = {}


def register(
    name: str,
    version: str = "v1",
    category: str = "uncategorized",
    description: str = "",
):
    """Decorator: registers a feature function in the global registry.

    Example:
        @register("ret_5m", category="price", description="5-minute log return, trailing-only")
        def ret_5m(snap: Snapshot, ts: datetime) -> float: ...
    """
    def deco(fn: Callable[[Snapshot, datetime], float]) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} already registered")
        _REGISTRY[name] = FeatureSpec(
            name=name, fn=fn, version=version,
            category=category, description=description,
        )
        return fn
    return deco


def get_feature(name: str) -> Optional[FeatureSpec]:
    return _REGISTRY.get(name)


def all_features() -> list[FeatureSpec]:
    """Snapshot of registered features in deterministic order."""
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def reset_registry() -> None:
    """Test helper: clears the registry. Production code never uses this."""
    _REGISTRY.clear()


def feature_schema_hash() -> str:
    """Deterministic sha256 of (name, version) pairs across all
    registered features. Used by V2 §4b ModelDecisionEngine to refuse
    artifacts whose schema doesn't match the live feature set.
    """
    parts = [f"{s.name}@{s.version}" for s in all_features()]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ─── Convenience: build FeatureVector for one decision timestamp ─────────────


def compute_all_features(
    snapshot: Snapshot, decision_ts: datetime
) -> dict[str, float]:
    """Returns {feature_name: value} for every registered feature.

    Production callers wrap this in a `FeatureVector` (decision_engine.py)
    which carries the schema hash for fail-closed checks.
    """
    return {s.name: s.compute(snapshot, decision_ts) for s in all_features()}


def build_feature_vector(snapshot: Snapshot, decision_ts: datetime):
    """Produce a `FeatureVector` (from decision_engine) ready for the
    ModelDecisionEngine.

    Imports `decision_engine.FeatureVector` lazily to avoid creating
    a hard dependency from `features` → `decision_engine` (the
    decision engine also imports from features at runtime in the
    production wiring).
    """
    from decision_engine import FeatureVector  # local import; runtime-only

    values = compute_all_features(snapshot, decision_ts)
    return FeatureVector(
        decision_ts=decision_ts,
        schema_hash=feature_schema_hash(),
        values=values,
    )
