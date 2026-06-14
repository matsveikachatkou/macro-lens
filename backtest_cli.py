"""
backtest_cli.py — Command-line backtest runner for macro-lens v4.

Three arms of the experiment:
  Arm 1 — Structural fixes baseline (120m inflation window + CFNAI-MA3)
  Arm 2 — Validator advisory log (detects anomalies, weights unchanged)
  Arm 3 — Validator active (Gate 2 PCE-anchor override applied to weights)

Run one arm at a time by uncommenting the relevant block.
Results are printed to console including benchmark comparison,
validator decision summary, and average weights by regime.

Validator cache (validator_cache.json) is populated on Arm 2 first run.
Arm 3 uses cached decisions — no extra API calls if Arm 2 ran first.
"""

from macro_lens import run_backtest

START = "2015-01-01"
END   = "2024-12-31"

# --- Arm 1: Structural fixes only (fast, no LLM calls) ---
# run_backtest(start=START, end=END, use_cfnai_ma3=True, validate_regime="off")

# --- Arm 2: Validator advisory — run this before Arm 3 to populate cache ---
# run_backtest(start=START, end=END, use_cfnai_ma3=True, validate_regime="log")

# --- Arm 3: Validator active (Gate 2 only, cache hits after Arm 2) ---
run_backtest(start=START, end=END, use_cfnai_ma3=True, validate_regime="active")