#!/usr/bin/env python3
"""Polymarket crypto fee/PnL model for BTC 5m/15m paper/live parity.

Docs formula for taker fees: fee = C * feeRate * p * (1-p)
where C is shares traded, p is trade price, and crypto feeRate is 0.072.
"""

from __future__ import annotations

CRYPTO_TAKER_FEE_RATE = 0.072


def taker_fee_usdc(shares: float, price: float, fee_rate: float = CRYPTO_TAKER_FEE_RATE) -> float:
    shares = max(0.0, float(shares))
    price = min(0.999999, max(0.000001, float(price)))
    return shares * fee_rate * price * (1.0 - price)


def buy_cost_usdc(shares: float, price: float) -> float:
    return max(0.0, float(shares)) * min(0.999999, max(0.000001, float(price)))


def resolved_buy_pnl(won: bool, shares: float, entry_price: float) -> float:
    """Net PnL for a BUY held to binary resolution, after entry taker fee.

    Cost is paid at entry. If won, shares redeem at $1. If lost, redeem at $0.
    """
    cost = buy_cost_usdc(shares, entry_price)
    fee = taker_fee_usdc(shares, entry_price)
    payout = max(0.0, float(shares)) if won else 0.0
    return payout - cost - fee
