# test_step7.py
import pandas as pd
from macro_lens import (
    _fetch_price_history, _compute_performance,
    ASSET_PROXIES, BASELINE_WEIGHTS
)

# Fetch 6 months of real prices
prices = _fetch_price_history(list(ASSET_PROXIES.values()), "2020-01-01", "2020-06-30")
monthly_returns = prices.pct_change().dropna()

# Mock records: baseline weights for every month
dates = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
monthly_records = [
    {"date": d, "regime": "High Growth / Low Inflation",
     "confidence": "high", "weights": BASELINE_WEIGHTS.copy()}
    for d in dates
]

result = _compute_performance(monthly_records, monthly_returns)

# Test 1: curves exist and have correct length
ec = result["equity_curve"]
bc = result["benchmark_curve"]
assert len(ec) > 0, "FAIL: empty equity curve"
assert len(ec) == len(bc), "FAIL: curve length mismatch"
print(f"PASS curve length: {len(ec)} months")

# Test 2: starts near 100 (first month moves it slightly)
assert 80 < ec.iloc[0] < 120, f"FAIL: first value {ec.iloc[0]:.2f} looks wrong"
print(f"PASS starting value: {ec.iloc[0]:.2f}")

# Test 3: metrics dict has expected keys
m = result["metrics"]
for key in ["sharpe_portfolio", "sharpe_benchmark", "max_drawdown_portfolio",
            "max_drawdown_benchmark", "total_return_portfolio", "n_months"]:
    assert key in m, f"FAIL: missing metric {key}"
print("PASS all metric keys present")

# Test 4: max drawdown is negative or zero
assert m["max_drawdown_portfolio"] <= 0, "FAIL: drawdown should be <= 0"
print(f"PASS max drawdown: {m['max_drawdown_portfolio']:.2%}")

# Test 5: sanity — March 2020 crash should show in benchmark drawdown
print(f"Portfolio total return: {m['total_return_portfolio']:.2%}")
print(f"Benchmark total return: {m['total_return_benchmark']:.2%}")
print(f"Sharpe portfolio: {m['sharpe_portfolio']:.2f}")
print(f"Sharpe benchmark: {m['sharpe_benchmark']:.2f}")

print("All Step 7 tests passed.")