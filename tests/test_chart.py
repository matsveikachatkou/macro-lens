# test_chart.py
import pandas as pd
import numpy as np
from app import _build_equity_chart, REGIME_COLORS

# Mock data
dates = pd.date_range("2020-02-01", periods=5, freq="ME")
equity_curve = pd.Series([100, 97, 102, 105, 103], index=dates)
benchmark_curve = pd.Series([100, 96, 101, 104, 102], index=dates)
monthly_records = [
    {"date": "2020-01-31", "regime": "Low Growth / High Inflation", "weights": {}},
    {"date": "2020-02-29", "regime": "Low Growth / High Inflation", "weights": {}},
    {"date": "2020-03-31", "regime": "Low Growth / Low Inflation",  "weights": {}},
    {"date": "2020-04-30", "regime": "High Growth / Low Inflation", "weights": {}},
    {"date": "2020-05-31", "regime": "High Growth / Low Inflation", "weights": {}},
]

fig = _build_equity_chart(equity_curve, benchmark_curve, monthly_records)

assert len(fig.data) >= 3, f"FAIL: expected >=3 traces, got {len(fig.data)}"
print(f"PASS trace count: {len(fig.data)}")

trace_names = [t.name for t in fig.data]
assert "Macro-Lens" in trace_names, "FAIL: missing Macro-Lens trace"
assert "60/40 Benchmark" in trace_names, "FAIL: missing benchmark trace"
assert "Active Return" in trace_names, "FAIL: missing active return trace"
print("PASS all traces present:", trace_names)

print("All chart tests passed.")