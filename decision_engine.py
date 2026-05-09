"""decision_engine.py — V2 Phase 4a/4c/4d core decision logic.

Replaces the heuristic decision pipeline in btc_sniper_live.py
(`_external_vote` + `_hour_adjust_conf` + flat `spend_ratio`) with:

  1. A clean DecisionEngine Protocol so a model and the legacy heuristic
     can be swapped via a config flag (V2 §4a).
  2. Two-sided executable EV using actual yes_ask / no_ask (V2 §4c) —
     replacing V1's single-sided abstract Kelly that ignored which side
     was actually buyable at what price.
  3. Maker price helper that posts INSIDE the spread (V2 §4d) — fixing
     V1's `bid - 1 tick` bug which posted BEHIND the best bid for a buy.
  4. Fail-closed: missing features / inverted book / stale wallet / bad
     prediction → SKIP, never falls back to the heuristic.

Codex review reference: SPEC.md:246-249, :248-249, :250-252.

Tests live in test_decision_engine.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol


# ─── Domain types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureVector:
    """One row fed to the predictor. V2 §2 enforces all values were
    `available_at <= decision_ts`. The decision_ts is carried on the
    vector so leakage tests can replay.
    """
    decision_ts: datetime
    schema_hash: str
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = float("nan")) -> float:
        return self.values.get(name, default)

    def has_required(self, required: list[str]) -> bool:
        for k in required:
            v = self.values.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
        return True


@dataclass(frozen=True)
class PMOrderbook:
    """Polymarket book snapshot at decision_ts.

    Both YES and NO sides are needed because the decision picks the
    cheaper directional bet, not just YES (V2 §4c).
    """
    yes_token_id: str
    no_token_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    no_bid_size: float = 0.0
    no_ask_size: float = 0.0
    tick_size: float = 0.01

    @property
    def implied_p_up(self) -> float:
        """Mid-price of the YES (Up) token."""
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def is_inverted(self) -> bool:
        """In a well-formed binary book, yes_ask + no_ask should be ~1.
        Anything outside [0.99, 1.01] is a sign of crossed/stale data.
        """
        s = self.yes_ask + self.no_ask
        return s < 0.99 or s > 1.01


@dataclass(frozen=True)
class Wallet:
    usdc_balance: float
    open_orders: int = 0
    last_refreshed_age_sec: float = 0.0


@dataclass(frozen=True)
class Decision:
    action: str  # "bet" | "skip"
    side: Optional[str] = None  # "YES" | "NO"
    fraction: Optional[float] = None  # Kelly-derived fraction of bankroll
    edge: Optional[float] = None  # post-fee edge magnitude
    p_model: Optional[float] = None  # for downstream calibration logging
    p_market: Optional[float] = None  # what we paid against
    reason: str = ""

    @classmethod
    def skip(cls, reason: str) -> "Decision":
        return cls(action="skip", reason=reason)

    @classmethod
    def bet(
        cls,
        side: str,
        fraction: float,
        edge: float,
        p_model: float,
        p_market: float,
        reason: str = "ev_positive",
    ) -> "Decision":
        return cls(
            action="bet",
            side=side,
            fraction=fraction,
            edge=edge,
            p_model=p_model,
            p_market=p_market,
            reason=reason,
        )


# ─── Pure pricing / sizing helpers ───────────────────────────────────────────


def maker_price(
    side: str, best_bid: float, best_ask: float, tick: float = 0.01
) -> float:
    """Post inside the spread, ahead in queue. V2 §4d.

    V1 said `bid - 1 tick` for buys — that's BEHIND the best bid, worse
    queue, lower fill odds. Codex flagged this as `[HIGH] SPEC.md:250-252`.
    The fix is to post `best_bid + tick` (one tick inside the spread)
    capped strictly below `best_ask`.

    Args:
        side: "BUY" or "SELL"
        best_bid: top-of-book bid (must be < ask)
        best_ask: top-of-book ask
        tick: minimum tick size, default $0.01 per Polymarket
              `orderPriceMinTickSize`.

    Returns the limit price to post.
    """
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        raise ValueError(
            f"invalid book: bid={best_bid} ask={best_ask}"
        )
    if tick <= 0:
        raise ValueError(f"tick must be positive: {tick}")

    s = side.upper()
    if s == "BUY":
        # Inside-spread = better than best_bid; cap strictly below ask.
        return min(best_bid + tick, best_ask - tick)
    if s == "SELL":
        return max(best_ask - tick, best_bid + tick)
    raise ValueError(f"unknown side: {side!r}")


def compute_executable_ev(
    p_up: float,
    yes_ask: float,
    no_ask: float,
    fees_pct: float = 0.0,
) -> tuple[float, float]:
    """Two-sided executable EV per V2 §4c.

    For YES side: edge = p_up - yes_ask - fees
    For NO side:  edge = (1 - p_up) - no_ask - fees

    Note we use the *ask* (executable buy price), not the mid. Codex review:
      [HIGH] SPEC.md:248-249 — `max(0, (p - p_market)/(1 - p_market))`
      ignores NO pricing, spread, fees, min size, partial fills.

    Args:
        p_up: model probability that Up wins, in (0, 1).
        yes_ask: executable ask for the YES (Up) token.
        no_ask: executable ask for the NO (Down) token.
        fees_pct: execution fee as a fraction (0.0 for makers, ~0.02 for takers).

    Returns:
        (yes_edge, no_edge) post-fee.
    """
    yes_edge = (p_up - yes_ask) - fees_pct
    no_edge = ((1.0 - p_up) - no_ask) - fees_pct
    return yes_edge, no_edge


def kelly_fraction(
    p_win: float, ask: float, multiplier: float = 0.25
) -> float:
    """Fractional Kelly on the chosen side using the executable ask.

    Standard Kelly for a binary: f = (p*b - q) / b, where b = (1-ask)/ask
    is the net odds. Simplifies to (p - ask) / (1 - ask) for binary
    contracts that pay $1 if right and lose `ask` if wrong.

    Conservative multiplier (default 0.25) per V2 §4c. Gemini agreed
    0.25 fractional Kelly is appropriate for a first-gen ML model.
    """
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    if p_win <= ask:
        return 0.0
    f = (p_win - ask) / (1.0 - ask)
    return max(0.0, min(1.0, f)) * multiplier


# ─── Predictor protocol and fail-closed model engine ─────────────────────────


class Predictor(Protocol):
    """Loaded model with calibrator + scaler. V2 §4b verifies schema_hash on load."""

    schema_hash: str

    def predict(self, features: FeatureVector) -> float:
        """Returns calibrated P(Up wins) ∈ (0, 1)."""
        ...


class DecisionEngine(Protocol):
    """V2 §4a — concrete adapter contract for both ML and heuristic paths."""

    def decide(
        self, features: FeatureVector, book: PMOrderbook, wallet: Wallet
    ) -> Decision:
        ...


@dataclass
class ModelDecisionEngine:
    """V2 default engine. Owns direction AND probability — no upstream
    heuristic filtering (codex SPEC.md:247).
    """
    predictor: Predictor
    edge_threshold: float = 0.04  # post-fee edge magnitude required
    kelly_multiplier: float = 0.25
    fees_maker_pct: float = 0.0  # makers pay zero on Polymarket
    fees_taker_pct: float = 0.02  # taker drag empirically ~2% per V1 diagnosis
    assume_maker_execution: bool = True
    required_features: list[str] = field(default_factory=list)
    min_balance_usdc: float = 1.0
    max_open_orders: int = 0  # V2: single-open-order discipline kept
    max_wallet_age_sec: float = 60.0

    def decide(
        self, features: FeatureVector, book: PMOrderbook, wallet: Wallet
    ) -> Decision:
        # ── Fail-closed gates (V2 §4e). Each one returns a SKIP with a
        # specific reason so monitoring can categorize without parsing.
        if features is None:
            return Decision.skip("features_missing")
        if features.schema_hash != self.predictor.schema_hash:
            return Decision.skip(
                f"schema_mismatch: features={features.schema_hash[:8]} "
                f"predictor={self.predictor.schema_hash[:8]}"
            )
        if self.required_features and not features.has_required(self.required_features):
            missing = [
                k for k in self.required_features
                if k not in features.values
                or (isinstance(features.values.get(k), float) and math.isnan(features.values[k]))
            ]
            return Decision.skip(f"required_feature_nan: {missing[:5]}")
        if book is None:
            return Decision.skip("book_missing")
        if book.yes_ask <= 0 or book.no_ask <= 0:
            return Decision.skip(f"bad_book: yes_ask={book.yes_ask} no_ask={book.no_ask}")
        if book.is_inverted:
            return Decision.skip(
                f"book_inverted: yes_ask+no_ask={book.yes_ask + book.no_ask:.4f}"
            )
        if wallet is None:
            return Decision.skip("wallet_missing")
        if wallet.last_refreshed_age_sec > self.max_wallet_age_sec:
            return Decision.skip(
                f"wallet_stale: age={wallet.last_refreshed_age_sec:.0f}s > {self.max_wallet_age_sec}s"
            )
        if wallet.usdc_balance < self.min_balance_usdc:
            return Decision.skip(f"insufficient_balance: ${wallet.usdc_balance:.2f}")
        if wallet.open_orders > self.max_open_orders:
            return Decision.skip(f"open_orders_present: {wallet.open_orders}")

        # ── Prediction. Bad probability = SKIP, not heuristic fallback.
        try:
            p_up = self.predictor.predict(features)
        except Exception as e:
            return Decision.skip(f"predictor_error: {type(e).__name__}: {str(e)[:80]}")
        if p_up is None or not isinstance(p_up, (int, float)) or math.isnan(p_up):
            return Decision.skip(f"bad_prediction: {p_up!r}")
        if not (0.0 < p_up < 1.0):
            return Decision.skip(f"prediction_out_of_range: {p_up}")

        # ── Two-sided executable EV. Maker fees apply if we post inside spread.
        fees_pct = self.fees_maker_pct if self.assume_maker_execution else self.fees_taker_pct
        yes_edge, no_edge = compute_executable_ev(p_up, book.yes_ask, book.no_ask, fees_pct)

        if yes_edge > self.edge_threshold and yes_edge >= no_edge:
            f = kelly_fraction(p_up, book.yes_ask, self.kelly_multiplier)
            if f <= 0:
                return Decision.skip(f"kelly_zero: yes_edge={yes_edge:.4f}")
            return Decision.bet(
                side="YES",
                fraction=f,
                edge=yes_edge,
                p_model=p_up,
                p_market=book.yes_ask,
            )
        if no_edge > self.edge_threshold and no_edge > yes_edge:
            f = kelly_fraction(1.0 - p_up, book.no_ask, self.kelly_multiplier)
            if f <= 0:
                return Decision.skip(f"kelly_zero: no_edge={no_edge:.4f}")
            return Decision.bet(
                side="NO",
                fraction=f,
                edge=no_edge,
                p_model=1.0 - p_up,
                p_market=book.no_ask,
            )
        return Decision.skip(
            f"no_edge: yes_edge={yes_edge:+.4f} no_edge={no_edge:+.4f} "
            f"thresh={self.edge_threshold}"
        )


@dataclass
class HeuristicDecisionEngine:
    """Legacy V1 heuristic, kept ONLY behind explicit operator flag for
    A/B and emergency rollback. Default config flag is DISABLED.

    This engine deliberately has no leakage tests — it is preserved for
    historical comparison, not for promotion.
    """
    enabled: bool = False
    base_conf_threshold: float = 0.88

    def decide(
        self, features: FeatureVector, book: PMOrderbook, wallet: Wallet
    ) -> Decision:
        if not self.enabled:
            return Decision.skip("heuristic_disabled_by_default")
        return Decision.skip("heuristic_engine_not_implemented_in_v2_path")


# ─── Helper: compute decision_ts from now ────────────────────────────────────


def now_decision_ts() -> datetime:
    """The wall-clock decision timestamp. V2 §3b mid-window mode: this
    is the actual moment the bot fires an order during an active
    market window — must match between training and live.
    """
    return datetime.now(timezone.utc)
