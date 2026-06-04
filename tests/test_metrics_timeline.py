# test_metrics_timeline.py
import pandas as pd
from app import _metrics_html, _build_regime_timeline

# Test metrics HTML
metrics = {
    "total_return_portfolio":  0.123,
    "total_return_benchmark":  0.087,
    "ann_return_portfolio":    0.045,
    "ann_return_benchmark":    0.031,
    "sharpe_portfolio":        0.82,
    "sharpe_benchmark":        0.61,
    "max_drawdown_portfolio":  -0.058,
    "max_drawdown_benchmark":  -0.091,
    "n_months":                36,
}

html = _metrics_html(metrics)
assert "Macro-Lens" in html
assert "+12.3%" in html
assert "0.82" in html
print("PASS metrics HTML generated")

# Test regime timeline
monthly_records = [
    {"date": "2020-01-31", "regime": "High Growth / Low Inflation"},
    {"date": "2020-02-29", "regime": "Low Growth / High Inflation"},
    {"date": "2020-03-31", "regime": "Low Growth / Low Inflation"},
]

fig = _build_regime_timeline(monthly_records)
assert len(fig.data) >= 1
print(f"PASS timeline trace count: {len(fig.data)}")

# Empty input should not crash
fig_empty = _build_regime_timeline([])
print("PASS empty input handled")

print("All metrics/timeline tests passed.")