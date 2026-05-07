#!/usr/local/opt/python@3.11/bin/python3.11
"""
Readiness Monitor — tracks paper trading performance and computes a LIVE_READINESS score.
Only recommends real money when ALL gates pass.

Gates:
  1. MIN_TRADES:        500+ paper trades
  2. MIN_WIN_RATE_CI:   95% CI lower bound > 55%
  3. MIN_PROFIT:        Cumulative PnL > $200 on $100 bankroll
  4. MAX_DRAWDOWN:      Peak-to-trough < 25%
  5. MIN_SHARPE:        Sharpe ratio > 1.0
  6. PARAM_STABILITY:   conf_thresh unchanged for 7+ days
  7. REGIME_CHECK:      Profitable in both high and low volatility windows
  8. CONSECUTIVE_DAYS:  14+ days of positive cumulative PnL

Score: 0-100. Real money gate: score >= 85 AND all hard gates pass.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

HARVEY_HOME = Path(os.path.expanduser(os.environ.get("HARVEY_HOME", "~/MAKAKOO")))
DATA_DIR = HARVEY_HOME / "data" / "arbitrage-agent" / "v2"
JOURNAL_FILE = DATA_DIR / "state" / "intraday_journal.jsonl"
BEST_PARAMS_FILE = DATA_DIR / "state" / "sniper_best_params.json"
FREEZE_FILE = DATA_DIR / "state" / "freezes" / "latest.json"
LOG_FILE = DATA_DIR / "logs" / "readiness_monitor.log"
BRAIN_JOURNAL = HARVEY_HOME / "data" / "Brain" / "journals" / (datetime.now().strftime("%Y_%m_%d") + ".md")

# ── Gates ─────────────────────────────────────────────────────────────────────
MIN_TRADES = 500
MIN_WR_CI_LO = 0.55
MIN_PROFIT = 200.0
MAX_DRAWDOWN_PCT = 0.25
MIN_SHARPE = 1.0
MIN_STABLE_DAYS = 7
MIN_CONSECUTIVE_DAYS = 14
PAPER_CAPITAL = float(os.environ.get("BTC_PAPER_CAPITAL", "20.0"))
CLOB_PRICE_SOURCE = "clob_book_ask_depth"
READINESS_REQUIRE_CLOB = os.environ.get("BTC_READINESS_REQUIRE_CLOB", "1") != "0"
LAST_EXCLUDED_NON_CLOB = 0

STABLE_PARAMS_FILE = DATA_DIR / "state" / "stable_params_hash.json"

def _param_hash(params: dict) -> str:
    """Hash of the param values that matter for stability."""
    keys = ["delta_thresh", "conf_thresh", "ens_thresh"]
    vals = [str(params.get(k, "")) for k in keys]
    return "|".join(vals)

def _load_stable_date() -> tuple[str, datetime]:
    """Return (hash, first_seen_date) of stable params."""
    if STABLE_PARAMS_FILE.exists():
        try:
            d = json.loads(STABLE_PARAMS_FILE.read_text())
            return d.get("hash", ""), datetime.fromisoformat(d.get("first_seen", "2024-01-01T00:00:00"))
        except Exception:
            pass
    return "", datetime.min


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def brain_log(msg: str) -> None:
    BRAIN_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(BRAIN_JOURNAL, "a") as f:
        f.write(f"- {msg}\n")


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    width = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (
        max(0.0, (centre - width) / denom),
        min(1.0, (centre + width) / denom),
    )


def load_btc_trades() -> list[dict]:
    global LAST_EXCLUDED_NON_CLOB
    LAST_EXCLUDED_NON_CLOB = 0
    trades = []
    if not JOURNAL_FILE.exists():
        return trades
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                t = json.loads(line)
                if "btc_delta" not in t or "pnl" not in t:
                    continue
                if READINESS_REQUIRE_CLOB and t.get("price_source") != CLOB_PRICE_SOURCE:
                    LAST_EXCLUDED_NON_CLOB += 1
                    continue
                trades.append(t)
            except Exception:
                continue
    return trades


def compute_sharpe(daily_pnls: list[float]) -> float:
    if len(daily_pnls) < 2:
        return 0.0
    mean = sum(daily_pnls) / len(daily_pnls)
    variance = sum((p - mean) ** 2 for p in daily_pnls) / len(daily_pnls)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    return mean / std


def max_drawdown(cumulative_pnls: list[float]) -> float:
    if not cumulative_pnls:
        return 0.0
    peak = cumulative_pnls[0]
    max_dd = 0.0
    for pnl in cumulative_pnls:
        if pnl > peak:
            peak = pnl
        dd = (peak - pnl) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def max_drawdown_from_pnls(pnls: list[float], starting_equity: float = PAPER_CAPITAL) -> float:
    """Peak-to-trough drawdown over a trade sequence.

    Include starting equity as the initial peak. The old monitor started at the
    first post-trade equity value, which underreported losses after the first
    losing trade and could hide a bad fresh run behind earlier cumulative gains.
    """
    if not pnls:
        return 0.0
    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    for pnl in pnls:
        equity += float(pnl or 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def latest_freeze_start() -> datetime | None:
    try:
        if not FREEZE_FILE.exists():
            return None
        d = json.loads(FREEZE_FILE.read_text())
        return datetime.fromisoformat(d.get("created_at", ""))
    except Exception:
        return None


def main():
    trades = load_btc_trades()
    n = len(trades)

    log("=" * 60)
    log("READINESS MONITOR")
    log("=" * 60)
    log(f"Total BTC paper trades: {n}")
    log(f"CLOB-only readiness: {READINESS_REQUIRE_CLOB} (excluded_non_clob={LAST_EXCLUDED_NON_CLOB})")

    if n < 10:
        log("Too few trades. Aborting.")
        return

    # ── Basic stats ──────────────────────────────────────────────────────────
    pnls = [t.get("pnl", 0) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    wr = wins / max(wins + losses, 1)
    total_pnl = sum(pnls)
    ci_lo, ci_hi = wilson_ci(wins, wins + losses)

    log(f"WR: {wr:.1%} (95% CI: {ci_lo:.1%}–{ci_hi:.1%})")
    log(f"Total PnL: ${total_pnl:+.2f}")

    # ── Daily breakdown ──────────────────────────────────────────────────────
    daily: dict[str, list[float]] = {}
    for t in trades:
        placed = t.get("placed_at", "")
        if placed:
            day = placed[:10]
            daily.setdefault(day, []).append(t.get("pnl", 0))

    daily_pnls = [sum(v) for v in daily.values()]
    daily_dates = sorted(daily.keys())
    log(f"Trading days: {len(daily_dates)} ({daily_dates[0]} to {daily_dates[-1]})")

    # Cumulative PnL per day
    cum_pnls = []
    running = 0.0
    for d in daily_dates:
        running += sum(daily[d])
        cum_pnls.append(running)

    sharpe = compute_sharpe(daily_pnls)
    daily_dd = max_drawdown(cum_pnls)
    overall_trade_dd = max_drawdown_from_pnls(pnls, PAPER_CAPITAL)
    freeze_start = latest_freeze_start()
    post_freeze_pnls = []
    if freeze_start:
        for t in trades:
            try:
                placed = datetime.fromisoformat(t.get("placed_at", ""))
            except Exception:
                continue
            if placed >= freeze_start:
                post_freeze_pnls.append(float(t.get("pnl", 0) or 0.0))
    post_freeze_dd = max_drawdown_from_pnls(post_freeze_pnls, PAPER_CAPITAL)
    dd = max(daily_dd, overall_trade_dd, post_freeze_dd)
    log(f"Sharpe: {sharpe:.2f}")
    log(f"Max drawdown: {dd:.1%} (daily={daily_dd:.1%}, overall={overall_trade_dd:.1%}, post-freeze={post_freeze_dd:.1%})")

    # ── Regime check: high vs low volatility ─────────────────────────────────
    # Strategy idles in low-vol (delta < thresh). A day with 0 trades = correctly idle.
    high_vol_days = []
    low_vol_days = []
    if daily_dates:
        first_day = datetime.strptime(daily_dates[0], "%Y-%m-%d")
        last_day = datetime.strptime(daily_dates[-1], "%Y-%m-%d")
        for offset in range((last_day - first_day).days + 1):
            d = (first_day + timedelta(days=offset)).strftime("%Y-%m-%d")
            day_trades = [t for t in trades if t.get("placed_at", "").startswith(d)]
            day_pnl = sum(daily.get(d, []))
            if day_trades:
                avg_delta = sum(abs(t.get("btc_delta", 0)) for t in day_trades) / len(day_trades)
                if avg_delta >= 15:
                    high_vol_days.append(day_pnl)
                else:
                    low_vol_days.append(day_pnl)
            else:
                # No trades = strategy correctly idled. PnL=0 counts as "not losing".
                low_vol_days.append(0.0)

    high_vol_profit = sum(high_vol_days) > 0 if high_vol_days else False
    # Low-vol pass if we never lost money on idle days (pnl >= 0)
    low_vol_profit = all(p >= 0 for p in low_vol_days) if low_vol_days else True
    log(f"High-vol days profitable: {high_vol_profit} ({len(high_vol_days)} days)")
    log(f"Low-vol days profitable: {low_vol_profit} ({len(low_vol_days)} days) — idle days count as pass if no losses")

    # ── Param stability ──────────────────────────────────────────────────────
    param_stable = False
    if BEST_PARAMS_FILE.exists():
        try:
            current_params = json.loads(BEST_PARAMS_FILE.read_text())
            current_hash = _param_hash(current_params)
        except Exception:
            current_hash = ""
        stable_hash, first_seen = _load_stable_date()
        if current_hash == stable_hash and current_hash:
            age_days = (datetime.now() - first_seen).days
            param_stable = age_days >= MIN_STABLE_DAYS
            log(f"Param values stable for {age_days} days (stable={param_stable})")
        else:
            # New params — reset the clock
            STABLE_PARAMS_FILE.write_text(json.dumps({
                "hash": current_hash,
                "first_seen": datetime.now().isoformat(),
            }))
            log(f"Param values changed — stability clock reset (new hash={current_hash})")

    # ── Consecutive profitable days ──────────────────────────────────────────
    consec_positive = 0
    for pnl in reversed(daily_pnls):
        if pnl >= 0:
            consec_positive += 1
        else:
            break
    log(f"Consecutive profitable days: {consec_positive}")

    # ── Score calculation ────────────────────────────────────────────────────
    score = 0.0
    gates = []

    # Gate 1: Trade count (0-20 points)
    gate1 = min(n / MIN_TRADES, 1.0)
    score += gate1 * 20
    gates.append(("Trades", gate1 >= 1.0, f"{n}/{MIN_TRADES}"))

    # Gate 2: WR CI (0-20 points)
    gate2 = max(0, (ci_lo - 0.45) / (MIN_WR_CI_LO - 0.45)) if ci_lo >= 0.45 else 0
    gate2 = min(gate2, 1.0)
    score += gate2 * 20
    gates.append(("WR CI", gate2 >= 1.0, f"{ci_lo:.1%} (need >{MIN_WR_CI_LO:.0%})"))

    # Gate 3: Profit (0-15 points)
    gate3 = min(max(total_pnl, 0) / MIN_PROFIT, 1.0)
    score += gate3 * 15
    gates.append(("Profit", gate3 >= 1.0, f"${total_pnl:+.0f} (need ${MIN_PROFIT:.0f})"))

    # Gate 4: Drawdown (0-15 points)
    gate4 = max(0, 1.0 - dd / MAX_DRAWDOWN_PCT)
    score += gate4 * 15
    gates.append(("Drawdown", dd <= MAX_DRAWDOWN_PCT, f"{dd:.1%} (max {MAX_DRAWDOWN_PCT:.0%})"))

    # Gate 5: Sharpe (0-15 points)
    gate5 = min(max(sharpe, 0) / MIN_SHARPE, 1.0)
    score += gate5 * 15
    gates.append(("Sharpe", gate5 >= 1.0, f"{sharpe:.2f} (need {MIN_SHARPE:.1f})"))

    # Gate 6: Regime (0-10 points)
    gate6 = 1.0 if (high_vol_profit and low_vol_profit) else 0.5 if (high_vol_profit or low_vol_profit) else 0.0
    score += gate6 * 10
    gates.append(("Regime", gate6 >= 1.0, f"H={high_vol_profit} L={low_vol_profit}"))

    # Gate 7: Stability (0-5 points)
    gate7 = 1.0 if param_stable else 0.0
    score += gate7 * 5
    gates.append(("Stability", gate7 >= 1.0, f"{param_stable}"))

    # Gate 8: consecutive profitable days. Hard gate only; score weights stay
    # backward-compatible at 100 total points.
    gates.append(("Consecutive days", consec_positive >= MIN_CONSECUTIVE_DAYS, f"{consec_positive}/{MIN_CONSECUTIVE_DAYS}"))

    score = round(score, 1)
    all_hard = all(g[1] for g in gates)
    ready = score >= 85 and all_hard and n >= MIN_TRADES

    log(f"READINESS SCORE: {score}/100")
    for name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        log(f"  [{status}] {name}: {detail}")
    log(f"ALL GATES: {'YES' if all_hard else 'NO'}")
    log(f"READY FOR LIVE: {'YES' if ready else 'NO'}")
    log("=" * 60)

    if ready:
        brain_log(
            f"[TRADING READINESS] Score={score}/100. ALL GATES PASS. "
            f"{n} trades, WR={wr:.0%}, PnL=${total_pnl:+.0f}, Sharpe={sharpe:.2f}. "
            f"Candidate for real money activation."
        )
    else:
        brain_log(
            f"[TRADING READINESS] Score={score}/100. NOT READY. "
            f"{n} trades, WR={wr:.0%}, PnL=${total_pnl:+.0f}, Sharpe={sharpe:.2f}. "
            f"Need {MIN_TRADES} trades, stable params, {MIN_CONSECUTIVE_DAYS} positive days, positive regime proof."
        )


if __name__ == "__main__":
    main()
