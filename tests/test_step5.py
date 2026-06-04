# test_step5.py
from macro_lens import _monthly_dates

# Test 1: basic count
dates = _monthly_dates("2020-01-01", "2020-06-30")
assert len(dates) == 6, f"FAIL: expected 6, got {len(dates)}"
print("PASS count:", dates)

# Test 2: correct month-ends including leap year Feb
assert dates[0] == "2020-01-31"
assert dates[1] == "2020-02-29"  # 2020 is a leap year
assert dates[5] == "2020-06-30"
print("PASS month-ends correct")

# Test 3: December wraps correctly
dates_dec = _monthly_dates("2020-12-01", "2021-01-31")
assert dates_dec[0] == "2020-12-31"
assert dates_dec[1] == "2021-01-31"
print("PASS December wrap:", dates_dec)

# Test 4: non-leap year Feb
dates_2019 = _monthly_dates("2019-02-01", "2019-02-28")
assert dates_2019[0] == "2019-02-28"
print("PASS non-leap Feb:", dates_2019)

print("All Step 5 tests passed.")