#!/bin/bash
# Cron wrapper for Auto-Improver nightly skill review
# Runs skill improvement cycle with real LLM calls (no money involved)

MAKAKOO_HOME="${HARVEY_HOME:-$HOME/HARVEY}"
PYTHON="/usr/local/opt/python@3.11/bin/python3.11"
SCRIPT="$MAKAKOO_HOME/skills-shared/harvey/meta/autoimprover/run_improvements.py"
LOG="$MAKAKOO_HOME/tmp/autoimprover.log"

exec "$PYTHON" "$SCRIPT" >> "$LOG" 2>&1
