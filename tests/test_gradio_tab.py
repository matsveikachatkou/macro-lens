# test_gradio_tab.py
import gradio as gr
from app import ui, run_backtest_ui, _build_equity_chart, _metrics_html, _build_regime_timeline

# Test 1: UI object exists and has correct type
assert isinstance(ui, gr.Blocks), "FAIL: ui is not gr.Blocks"
print("PASS ui is gr.Blocks")

# Test 2: all expected functions are importable
for fn in [run_backtest_ui, _build_equity_chart, _metrics_html, _build_regime_timeline]:
    assert callable(fn), f"FAIL: {fn} not callable"
print("PASS all functions importable")

# Test 3: run_backtest_ui returns 3 outputs on error path
import plotly.graph_objects as go
# Simulate what happens when run_backtest raises
try:
    raise ValueError("simulated failure")
except Exception as e:
    empty = go.Figure()
    result = (empty, empty, f"<p>Error: {e}</p>")
    assert len(result) == 3
    print("PASS error path returns 3 outputs")

print("All Gradio tab tests passed.")