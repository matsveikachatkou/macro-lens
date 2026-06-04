# test_step6.py
import pandas as pd
from macro_lens import _fetch_price_history, ASSET_PROXIES

tickers = list(ASSET_PROXIES.values())
print("Fetching:", tickers)

prices = _fetch_price_history(tickers, "2020-01-01", "2020-06-30")

# Test 1: all tickers present
for ticker in tickers:
    assert ticker in prices.columns, f"FAIL: {ticker} missing"
print("PASS all tickers present:", list(prices.columns))

# Test 2: correct shape — should have ~6 monthly rows
assert len(prices) >= 5, f"FAIL: only {len(prices)} rows"
print("PASS row count:", len(prices))

# Test 3: no NaNs after ffill
assert not prices.isnull().all().any(), "FAIL: column with all NaN"
print("PASS no all-NaN columns")

# Test 4: TIP present (not LQD)
assert "TIP" in prices.columns, "FAIL: TIP missing — check ASSET_PROXIES"
assert "LQD" not in prices.columns, "FAIL: LQD still present"
print("PASS TIP confirmed, LQD absent")

print(prices.tail())
print("All Step 6 tests passed.")