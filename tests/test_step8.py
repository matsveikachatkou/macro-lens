# test_step8.py
from macro_lens import run_backtest, ASSET_PROXIES

print("Running 3-month backtest (2020-01-01 to 2020-03-31)...")
print("Expected: ~3 LLM invocations, ~35-45 seconds with sleep(1)\n")

result = run_backtest(
    start="2020-01-01",
    end="2020-03-31",
    progress_callback=lambda f, msg: print(f"  [{f:.0%}] {msg}"),
)

# Test 1: correct number of monthly records
assert len(result["monthly_records"]) == 3, f"FAIL: expected 3 records, got {len(result['monthly_records'])}"
print(f"\nPASS monthly records: {len(result['monthly_records'])}")

# Test 2: regimes are valid strings
valid_regimes = {
    "High Growth / High Inflation",
    "High Growth / Low Inflation",
    "Low Growth / High Inflation",
    "Low Growth / Low Inflation",
    "Unknown",
}
for rec in result["monthly_records"]:
    assert rec["regime"] in valid_regimes, f"FAIL: invalid regime {rec['regime']}"
    print(f"  {rec['date']} — {rec['regime']} ({rec['confidence']})")

# Test 3: weights sum to ~1.0 for each record
for rec in result["monthly_records"]:
    total = sum(rec["weights"].values())
    assert abs(total - 1.0) < 0.001, f"FAIL: weights sum to {total}"
print("PASS weights sum to 1.0")

# Test 4: equity curve exists
ec = result["equity_curve"]
assert len(ec) > 0, "FAIL: empty equity curve"
print(f"PASS equity curve length: {len(ec)}")
print(f"  Final value: {ec.iloc[-1]:.2f}")

# Test 5: metrics present
m = result["metrics"]
print(f"\nMetrics:")
print(f"  Sharpe:       {m['sharpe_portfolio']:.2f}")
print(f"  Max drawdown: {m['max_drawdown_portfolio']:.2%}")
print(f"  Total return: {m['total_return_portfolio']:.2%}")

print("\nAll Step 8 tests passed.")