#!/usr/bin/env python3
"""Frozen strategy / mutable parameter contract for BTC 5m/15m sniper.

Rule: strategy code is frozen by `STRATEGY_VERSION`; optimizers may only change
DYNAMIC_PARAM_FIELDS. Everything else is either metadata or fixed code/config.
This prevents autoresearch from accidentally turning code constants into a new
strategy while we think we are only tuning thresholds.
"""

from __future__ import annotations

import copy
import contextlib
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

STRATEGY_VERSION = "btc-window-sniper-frozen-v1"
PARAM_CONTRACT_VERSION = 1

# Only these are trainable. Additions require an explicit code review.
DYNAMIC_PARAM_FIELDS = (
    "delta_thresh",   # BTC move threshold inside the 5m/15m window
    "conf_thresh",    # signal confidence threshold
    "ens_thresh",     # ensemble confidence threshold
)

# Metadata may be written/read but must not affect strategy execution directly.
METADATA_FIELDS = (
    "version", "name", "strategy_version", "param_contract_version", "mode",
    "best_score", "generation", "metrics", "source", "rejected_fields",
)

# Frozen execution defaults. Keep these in code, not optimizer mutation space.
FIXED_EXECUTION_PARAMS = {
    "version": "pro1.0",
    # Risk-first paper/live candidate defaults. Polymarket BTC contracts have
    # a practical 5-share minimum, so $20 paper bankrolls cannot support tiny
    # position sizing. Keep total open exposure tight and let runtime reject
    # trades that cannot fit inside the cap instead of silently overbetting.
    "spend_ratio": 0.15,
    "max_bet_pct": 0.20,
    "max_hold_seconds": 300,
    "min_market_volume": 0,
    "max_spread_bps": 2000,
    "pop_size": 12,
}

DEFAULT_DYNAMIC_PARAMS = {
    "delta_thresh": 11.8,
    "conf_thresh": 0.45,
    "ens_thresh": 0.50,
}

# Paper bounds are intentionally wider for exploration. Live bounds must be
# stricter and are for future canary/live wiring only.
PAPER_BOUNDS = {
    "delta_thresh": (8.0, 24.0),
    "conf_thresh": (0.45, 0.90),
    "ens_thresh": (0.30, 0.85),
}

LIVE_BOUNDS = {
    "delta_thresh": (10.0, 30.0),
    "conf_thresh": (0.65, 0.95),
    "ens_thresh": (0.55, 0.90),
}


def bounds_for(mode: str = "paper") -> dict[str, tuple[float, float]]:
    return LIVE_BOUNDS if mode == "live" else PAPER_BOUNDS


def _as_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


def clamp_dynamic_params(raw: Mapping[str, Any] | None, mode: str = "paper") -> dict[str, float]:
    """Return only dynamic params, filled with defaults and clamped to bounds."""
    raw = raw or {}
    out = copy.deepcopy(DEFAULT_DYNAMIC_PARAMS)
    b = bounds_for(mode)
    for key in DYNAMIC_PARAM_FIELDS:
        if key in raw:
            val = _as_float(key, raw[key])
            lo, hi = b[key]
            out[key] = min(hi, max(lo, val))
    return out


def validate_dynamic_params(params: Mapping[str, Any], mode: str = "paper") -> None:
    """Raise ValueError if params violate the frozen contract."""
    b = bounds_for(mode)
    for key in DYNAMIC_PARAM_FIELDS:
        if key not in params:
            raise ValueError(f"missing dynamic param: {key}")
        val = _as_float(key, params[key])
        lo, hi = b[key]
        if val < lo or val > hi:
            raise ValueError(f"{key}={val:.6g} outside [{lo}, {hi}] for mode={mode}")


def sanitize_payload(raw: Mapping[str, Any] | None, mode: str = "paper") -> dict[str, Any]:
    """Normalize a params file payload.

    Unknown fields are stripped and recorded in metadata. This is deliberate:
    optimizers can no longer smuggle executable knobs into the runtime.
    """
    raw = dict(raw or {})
    rejected = sorted(
        k for k in raw
        if k not in DYNAMIC_PARAM_FIELDS and k not in METADATA_FIELDS and k not in FIXED_EXECUTION_PARAMS
    )
    dyn = clamp_dynamic_params(raw, mode=mode)
    validate_dynamic_params(dyn, mode=mode)

    out: dict[str, Any] = {}
    out.update(FIXED_EXECUTION_PARAMS)
    out.update(dyn)
    out["strategy_version"] = str(raw.get("strategy_version") or STRATEGY_VERSION)
    if out["strategy_version"] != STRATEGY_VERSION:
        raise ValueError(
            f"strategy_version mismatch: {out['strategy_version']} != {STRATEGY_VERSION}"
        )
    out["param_contract_version"] = PARAM_CONTRACT_VERSION
    out["mode"] = mode
    out["name"] = str(raw.get("name") or "contract_params")
    if "best_score" in raw:
        out["best_score"] = float(raw["best_score"])
    if "generation" in raw:
        out["generation"] = int(float(raw["generation"]))
    else:
        out["generation"] = int(time.time())
    if "metrics" in raw and isinstance(raw["metrics"], dict):
        out["metrics"] = dict(raw["metrics"])
    if "source" in raw:
        out["source"] = str(raw["source"])
    if rejected:
        out["rejected_fields"] = rejected
    return out


def read_params_file(path: Path, mode: str = "paper") -> dict[str, Any]:
    if not path.exists():
        return sanitize_payload({}, mode=mode)
    return sanitize_payload(json.loads(path.read_text()), mode=mode)


def write_params_file(path: Path, payload: Mapping[str, Any], mode: str = "paper") -> dict[str, Any]:
    clean = sanitize_payload(payload, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return clean


@contextlib.contextmanager
def params_file_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock for read/compare/write param updates.

    `write_params_file` is already atomic for JSON integrity. This lock prevents
    stale optimizer workers from doing last-writer-wins deployment over better
    params found by another paper-only optimizer.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def locked_write_params_file(path: Path, payload: Mapping[str, Any], mode: str = "paper") -> dict[str, Any]:
    with params_file_lock(path):
        return write_params_file(path, payload, mode=mode)


def locked_update_params_file(
    path: Path,
    candidate: Mapping[str, Any],
    *,
    mode: str = "paper",
    accept_fn: Callable[[dict[str, Any] | None, dict[str, Any]], bool] | None = None,
) -> dict[str, Any] | None:
    """Locked read/compare/write helper.

    `accept_fn(current, clean_candidate)` returns True to deploy. `current` is
    None if the params file does not exist. Returns written clean params, or
    None when rejected.
    """
    clean_candidate = sanitize_payload(candidate, mode=mode)
    with params_file_lock(path):
        current = read_params_file(path, mode=mode) if path.exists() else None
        if accept_fn and not accept_fn(current, clean_candidate):
            return None
        return write_params_file(path, clean_candidate, mode=mode)


def dynamic_fingerprint(payload: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(payload[k]) for k in DYNAMIC_PARAM_FIELDS)


def as_runtime_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return fixed + dynamic fields suitable for SniperParams construction."""
    clean = sanitize_payload(payload, mode=str(payload.get("mode") or "paper"))
    return {**FIXED_EXECUTION_PARAMS, **{k: clean[k] for k in DYNAMIC_PARAM_FIELDS}}
