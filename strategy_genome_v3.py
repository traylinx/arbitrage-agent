#!/usr/local/opt/python@3.11/bin/python3.11
"""
StrategyGenome v3 — Directional Intraday Trading.

Evolvable params for BTC-momentum × PM-price directional strategy.
"""

import random
import json
import copy
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class StrategyGenome:
    version: str = "v3.0"
    name: str = ""

    # BTC momentum thresholds — reasonable for 15m timeframe
    bull_threshold: float = 0.10  # % BTC mom to be BULL (buy YES)
    bear_threshold: float = -0.10  # % BTC mom to be BEAR (buy NO)

    # PM momentum threshold (secondary confirmation, CLOB has 30s cache)
    pm_bull_threshold: float = 0.001  # YES price rising → BUY YES
    pm_bear_threshold: float = -0.001  # YES price falling → BUY NO

    # Confidence scaling
    base_confidence: float = 0.55
    max_confidence: float = 0.85
    min_confidence: float = 0.40  # lower min for BTC-leading entries
    conf_per_bps_mom: float = 0.010

    # Timeframe
    primary_timeframe: str = "15m"  # 15m for clearer signals
    use_composite_momentum: bool = False
    btc_weight_1m: float = 0.0
    btc_weight_5m: float = 0.30
    btc_weight_15m: float = 0.40
    btc_weight_1h: float = 0.30

    # BTC as filter
    use_btc_filter: bool = False
    btc_filter_threshold: float = 1.0
    btc_bear_multiplier: float = 0.50

    # Entry filters
    min_volume_24h: float = 30000  # higher min volume
    max_entry_price: float = 0.75
    min_entry_price: float = 0.05

    # RSI filter
    use_rsi_filter: bool = False
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    # RSI mean reversion (DISABLED)
    use_rsi_reversion: bool = False
    rsi_reversion_buy_threshold: float = 30
    rsi_reversion_sell_threshold: float = 70

    # Position sizing — MORE CONSERVATIVE
    max_position_pct: float = 0.08  # 8% per trade (was 10%)
    min_position_pct: float = 0.03
    max_positions: int = 2  # max 2 concurrent (was 3)
    kelly_fraction: float = 0.20

    # Exit rules — tighter for illiquid PM
    profit_target_pct: float = 0.04  # 4% — reasonable for PM moves
    stop_loss_pct: float = 0.03  # 3% — cut losses fast
    max_hold_hours: float = 2.0  # force exit after 2h

    # Poll / session
    poll_seconds: int = 10  # faster polling
    session_minutes: int = 60

    def __post_init__(self):
        if not self.name:
            self.name = f"gen_{datetime.now().strftime('%H%M%S')}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("version", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "StrategyGenome":
        d_copy = copy.deepcopy(d)
        version = d_copy.pop("version", "v3.0")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        for k in list(d_copy.keys()):
            if k not in known:
                d_copy.pop(k)
        return cls(version=version, **d_copy)

    def mutate(self, rate: float = 0.25) -> "StrategyGenome":
        """Create a mutated copy of this genome."""
        new = copy.deepcopy(self)
        new.name = (
            f"gen_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
        )

        mutations = [
            ("bull_threshold", 0.05, 0.25, 0.02),
            ("bear_threshold", -0.25, -0.05, 0.02),
            ("pm_bull_threshold", 0.0005, 0.004, 0.0005),
            ("pm_bear_threshold", -0.004, -0.0005, 0.0005),
            ("base_confidence", 0.50, 0.70, 0.05),
            ("max_confidence", 0.70, 0.95, 0.05),
            ("min_confidence", 0.35, 0.50, 0.05),
            ("conf_per_bps_mom", 0.005, 0.040, 0.005),
            ("btc_weight_1m", 0.0, 0.5, 0.05),
            ("btc_weight_5m", 0.0, 0.5, 0.05),
            ("btc_weight_15m", 0.0, 0.5, 0.05),
            ("btc_weight_1h", 0.0, 0.5, 0.05),
            ("btc_filter_threshold", 0.5, 2.0, 0.10),
            ("btc_bear_multiplier", 0.1, 1.0, 0.05),
            ("min_volume_24h", 20000, 100000, 10000),
            ("max_entry_price", 0.50, 0.90, 0.02),
            ("min_entry_price", 0.02, 0.20, 0.02),
            ("rsi_overbought", 60, 85, 2),
            ("rsi_oversold", 15, 40, 2),
            ("rsi_reversion_buy_threshold", 20, 40, 1),
            ("rsi_reversion_sell_threshold", 60, 80, 1),
            ("max_position_pct", 0.05, 0.15, 0.02),
            ("min_position_pct", 0.02, 0.08, 0.01),
            ("max_positions", 1, 3, 1),
            ("kelly_fraction", 0.10, 0.40, 0.05),
            ("profit_target_pct", 0.02, 0.10, 0.01),
            ("stop_loss_pct", 0.02, 0.08, 0.01),
            ("max_hold_hours", 0.5, 4.0, 0.5),
            ("poll_seconds", 5, 30, 5),
        ]

        for attr, lo, hi, step in mutations:
            if random.random() < rate:
                val = round(random.uniform(lo, hi) / step) * step
                setattr(new, attr, val)

        # Fixed: always use primary TF only (no composite) on 15m — avoid confusing signal
        new.use_composite_momentum = False
        new.primary_timeframe = "15m"
        if random.random() < rate:
            new.use_btc_filter = not self.use_btc_filter

        return new

    @classmethod
    def crossover(cls, a: "StrategyGenome", b: "StrategyGenome") -> "StrategyGenome":
        d_a = a.to_dict()
        d_b = b.to_dict()
        keys = list(d_a.keys())
        pt = random.randint(1, len(keys) - 1)
        child_dict = {}
        for i, k in enumerate(keys):
            child_dict[k] = d_a[k] if i < pt else d_b[k]
        child_dict["version"] = "v3.0"
        child_dict["name"] = f"x_{datetime.now().strftime('%H%M%S')}"
        return cls.from_dict(child_dict)

    @classmethod
    def random_population(cls, size: int) -> List["StrategyGenome"]:
        pop = []
        for _ in range(size):
            g = cls()
            g.name = (
                f"init_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
            )
            g = g.mutate(rate=1.0)
            pop.append(g)
        return pop
