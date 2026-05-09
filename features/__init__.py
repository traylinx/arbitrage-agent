"""V2 §2 feature engine.

Every feature in this module MUST satisfy the as-of contract:
    compute(snapshot, decision_ts) only reads rows where
    available_at <= decision_ts.

The as-of test harness in `as_of_test.py` enforces this for every
registered feature on every commit.
"""
