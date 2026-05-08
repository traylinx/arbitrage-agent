#!/usr/bin/env python3
"""Read-only BTC decision/trade audit.

Parses live + paper journals and surfaces the exact failure shape:
WR/PnL by timeframe, direction, parameter set, external-data age, provider
availability, strategy family, and duplicated market windows.

No network. No orders. No state writes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


HARVEY_HOME = Path.home() / "MAKAKOO"
DEFAULT_BASE = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"

AGE_RE = re.compile(r"age=(\d+)s")
EXT_RE = re.compile(r"ext_score=([+-]?\d+(?:\.\d+)?)")
FLOW_RE = re.compile(r"flow5m=([+-]?\d+(?:\.\d+)?)")
OK_RE = re.compile(r"ok ca/cg/bn/by/bg/hl=([01])/([01])/([01])/([01])/([01])/([01])")


def parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value))
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def iter_jsonl(path: Path, base: Path) -> Iterable[dict]:
    try:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                row["_file"] = str(path.relative_to(base))
                row["_line"] = line_no
                yield row
    except FileNotFoundError:
        return


def pnl(row: dict) -> float:
    try:
        return float(row.get("pnl") or 0.0)
    except Exception:
        return 0.0


def is_trade_row(row: dict) -> bool:
    if row.get("mode") == "live":
        return bool(row.get("filled")) and row.get("exit_reason") != "unfilled"
    return "won" in row and row.get("pnl") is not None and float(row.get("spend") or 0) > 0


def won(row: dict) -> bool:
    return bool(row.get("won"))


def agg(label: str, rows: Iterable[dict]) -> str:
    rows = list(rows)
    n = len(rows)
    wins = sum(1 for row in rows if won(row))
    total_pnl = sum(pnl(row) for row in rows)
    return (
        f"{label:<48} n={n:>5} W={wins:>5} L={n-wins:>5} "
        f"WR={(100*wins/n if n else 0):>5.1f}% PnL=${total_pnl:>9.2f} "
        f"avg=${(total_pnl/n if n else 0):>6.2f}"
    )


def params_name(row: dict) -> str:
    return row.get("strategy") or (row.get("params") or {}).get("name") or "unknown"


def strategy_family(row: dict) -> str:
    name = params_name(row)
    name = re.sub(r"^sweep_g\d+_\d+_", "", name)
    name = re.sub(r"_d[\d.]+_c[\d.]+_e[\d.]+.*$", "", name)
    return name


def external_fields(row: dict) -> dict:
    text = " | ".join(map(str, row.get("reasons") or []))
    fields: dict = {}
    match = AGE_RE.search(text)
    fields["age"] = int(match.group(1)) if match else None
    match = EXT_RE.search(text)
    fields["ext"] = float(match.group(1)) if match else None
    match = FLOW_RE.search(text)
    fields["flow"] = float(match.group(1)) if match else None
    match = OK_RE.search(text)
    fields["ok"] = tuple(map(int, match.groups())) if match else None
    return fields


def age_bucket(age: int | None) -> str:
    if age is None:
        return "no_external_logged"
    if age <= 60:
        return "<=60s"
    if age <= 120:
        return "61-120s"
    if age <= 180:
        return "121-180s"
    return ">180s"


def load_rows(base: Path, since: datetime) -> tuple[list[dict], list[dict]]:
    live_paths = [
        base / "state" / "intraday_journal_live_5m.jsonl",
        base / "state" / "intraday_journal_live_15m.jsonl",
    ]
    live: list[dict] = []
    for path in live_paths:
        for row in iter_jsonl(path, base):
            dt = parse_dt(row.get("placed_at") or row.get("resolved_at"))
            if dt and dt < since:
                continue
            if is_trade_row(row):
                live.append(row)

    paper: list[dict] = []
    for path in base.rglob("intraday_journal*.jsonl"):
        if path in live_paths:
            continue
        for row in iter_jsonl(path, base):
            dt = parse_dt(row.get("placed_at") or row.get("resolved_at"))
            if dt and dt < since:
                continue
            # Drop synthetic unit-test rows.
            if float(row.get("btc_price_enter") or 0) <= 1000:
                continue
            if is_trade_row(row):
                paper.append(row)
    return live, paper


def print_group(title: str, rows: list[dict], key_fn, min_n: int = 1, limit: int | None = None) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    items = [(k, v) for k, v in groups.items() if len(v) >= min_n]
    items.sort(key=lambda kv: sum(pnl(row) for row in kv[1]))
    if limit:
        items = items[:limit] if limit > 0 else items[limit:]
    print(f"\n--- {title} ---")
    for key, value in items:
        print(agg(key[:48], value))


def provider_nonzero_rates(rows: list[dict]) -> dict[str, dict[str, float]]:
    prefixes = {"ca": "ca_", "cg": "cg_", "bn": "bn_", "by": "by_", "bg": "bg_", "hl": "hl_"}
    counts = {key: [0, 0] for key in prefixes}
    for row in rows:
        features = row.get("prob_features") or {}
        for provider, prefix in prefixes.items():
            values = [v for key, v in features.items() if key.startswith(prefix)]
            if not values:
                continue
            counts[provider][0] += 1
            if any(abs(float(value or 0.0)) > 1e-12 for value in values):
                counts[provider][1] += 1
    return {
        provider: {
            "rows": rows_seen,
            "nonzero_rows": nonzero,
            "nonzero_pct": round(100.0 * nonzero / rows_seen, 1) if rows_seen else 0.0,
        }
        for provider, (rows_seen, nonzero) in counts.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--hours", type=float, default=36.0)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    since = datetime.now() - timedelta(hours=args.hours)
    live, paper = load_rows(args.base, since)
    print(f"BTC decision audit base={args.base} since={since.isoformat(timespec='seconds')}")
    print("\n=== live strict filled ===")
    print(agg("LIVE TOTAL", live))
    print_group("live by param", live, params_name)
    print_group("live by direction", live, lambda row: row.get("direction"))
    print_group("live by timeframe", live, lambda row: f"{row.get('window_tf')}m")
    print_group("live by external data age", live, lambda row: age_bucket(external_fields(row)["age"]))

    provider_counts = Counter()
    rows_with_provider_summary = 0
    for row in live:
        ok = external_fields(row)["ok"]
        if not ok:
            continue
        rows_with_provider_summary += 1
        for name, value in zip(("coinalyze", "coinglass", "binance", "bybit", "bitget", "hyperliquid"), ok):
            provider_counts[name] += value
    print("\n--- live provider-ok summary ---")
    print(f"rows_with_provider_summary={rows_with_provider_summary} ok_counts={dict(provider_counts)}")

    print("\n=== paper/lab trade rows ===")
    print(agg("PAPER/LAB TOTAL", paper))
    print_group("paper/lab by strategy family", paper, strategy_family)
    print_group("worst individual strategies/runs", paper, params_name, min_n=3, limit=args.limit)
    print_group("best individual strategies/runs", paper, params_name, min_n=3, limit=-args.limit)
    print_group("paper/lab by timeframe", paper, lambda row: f"{row.get('window_tf')}m")

    gate_counts = Counter()
    for row in paper:
        reason = (row.get("prob_decision") or {}).get("gate_reason") or row.get("exploration_reason") or ""
        if "MODEL_DOWN" in reason:
            gate_counts["MODEL_DOWN exploration heuristic"] += 1
        elif reason:
            gate_counts["other_gate"] += 1
        else:
            gate_counts["no_gate_logged"] += 1
    print("\n--- paper/lab gate reasons ---")
    print(dict(gate_counts))
    print("\n--- paper/lab provider nonzero rates ---")
    print(json.dumps(provider_nonzero_rates(paper), sort_keys=True))

    duplicated: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in paper:
        slug = row.get("market_slug") or row.get("window_start")
        duplicated[(str(slug), str(row.get("direction")))].append(row)
    print("\n--- top duplicated market windows ---")
    for (slug, direction), rows in sorted(duplicated.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.limit]:
        print(f"{slug} {direction} duplicates={len(rows)} W={sum(1 for row in rows if won(row))} PnL={sum(pnl(row) for row in rows):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
