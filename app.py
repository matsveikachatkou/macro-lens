import gradio as gr
import pandas as pd
from datetime import datetime
import uuid
from macro_lens import build_graph, MacroState

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from macro_lens import run_backtest, ASSET_PROXIES


TILT_ICONS = {
    "Strong Overweight":  "▲▲",
    "Overweight":         "▲",
    "Neutral":            "—",
    "Underweight":        "▼",
    "Strong Underweight": "▼▼",
}

REGIME_COLORS = {
    "High Growth / High Inflation": "#f59e0b",
    "High Growth / Low Inflation":  "#22c55e",
    "Low Growth / High Inflation":  "#ef4444",
    "Low Growth / Low Inflation":   "#3b82f6",
}


def _build_equity_chart(equity_curve, benchmark_curve, monthly_records) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Cumulative Performance (Base 100)", "Active Return vs 60/40"),
    )

    # Equity curves
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name="Macro-Lens",
            line=dict(color="#6366f1", width=2.5),
            hovertemplate="%{x|%b %Y}<br>Portfolio: %{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve.values,
            name="60/40 Benchmark",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>60/40: %{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Regime shading
    if monthly_records:
        prev_regime = None
        band_start = None

        def add_band(start, end, regime):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=REGIME_COLORS.get(regime, "#e2e8f0"),
                opacity=0.12,
                line_width=0,
                row=1, col=1,
            )

        for rec in monthly_records:
            r = rec["regime"]
            d = pd.Timestamp(rec["date"])
            if r != prev_regime:
                if prev_regime is not None and band_start is not None:
                    add_band(band_start, d, prev_regime)
                band_start = d
                prev_regime = r

        if prev_regime and band_start is not None and len(equity_curve) > 0:
            add_band(band_start, equity_curve.index[-1], prev_regime)

    # Active return bars
    if len(equity_curve) > 0 and len(benchmark_curve) > 0:
        common_idx = equity_curve.index.intersection(benchmark_curve.index)
        active = (
            equity_curve.loc[common_idx].pct_change().dropna()
            - benchmark_curve.loc[common_idx].pct_change().dropna()
        ) * 100

        fig.add_trace(
            go.Bar(
                x=active.index,
                y=active.values,
                name="Active Return",
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in active.values],
                opacity=0.7,
                hovertemplate="%{x|%b %Y}<br>Active: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color="#64748b", row=2, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _metrics_html(metrics: dict) -> str:
    def fmt_pct(v):
        return f"{v*100:+.1f}%" if v is not None else "N/A"

    def fmt_f(v):
        return f"{v:.2f}" if v is not None and v == v else "N/A"  # nan check

    rows = [
        ("Total Return",  fmt_pct(metrics.get("total_return_portfolio")),  fmt_pct(metrics.get("total_return_benchmark"))),
        ("Ann. Return",   fmt_pct(metrics.get("ann_return_portfolio")),    fmt_pct(metrics.get("ann_return_benchmark"))),
        ("Sharpe Ratio",  fmt_f(metrics.get("sharpe_portfolio")),          fmt_f(metrics.get("sharpe_benchmark"))),
        ("Max Drawdown",  fmt_pct(metrics.get("max_drawdown_portfolio")),  fmt_pct(metrics.get("max_drawdown_benchmark"))),
        ("Months",        str(metrics.get("n_months", "N/A")),             ""),
    ]

    html = """
    <style>
      .mt { width:100%; border-collapse:collapse; font-size:14px; }
      .mt th { text-align:left; padding:8px 12px; color:#94a3b8;
               font-weight:500; border-bottom:1px solid #334155; }
      .mt td { padding:8px 12px; border-bottom:1px solid #1e293b; }
      .mt tr:last-child td { border-bottom:none; }
      .port { color:#6366f1; font-weight:600; }
      .bench { color:#94a3b8; }
    </style>
    <table class="mt">
      <thead><tr>
        <th>Metric</th><th>Macro-Lens</th><th>60/40</th>
      </tr></thead><tbody>
    """
    for label, port_val, bench_val in rows:
        html += f"<tr><td style='color:#cbd5e1'>{label}</td><td class='port'>{port_val}</td><td class='bench'>{bench_val}</td></tr>"

    html += "</tbody></table>"
    return html


def _build_regime_timeline(monthly_records: list) -> go.Figure:
    if not monthly_records:
        return go.Figure()

    dates  = [pd.Timestamp(r["date"]) for r in monthly_records]
    colors = [REGIME_COLORS.get(r["regime"], "#94a3b8") for r in monthly_records]
    labels = [r["regime"] for r in monthly_records]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=[1] * len(dates),
        marker_color=colors,
        marker_line_width=0,
        customdata=labels,
        hovertemplate="%{x|%b %Y}<br>%{customdata}<extra></extra>",
        showlegend=False,
    ))

    # Legend entries
    for regime, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=regime,
        ))

    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=4, b=0),
        bargap=0.0,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="top", y=-0.4,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
    )
    return fig


def run_analysis():
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: MacroState = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "macro_data": None,
        "fetch_attempts": 0,
        "data_requests": None,
        "growth_direction": None,
        "inflation_direction": None,
        "regime": None,
        "regime_confidence": None,
        "regime_rationale": None,
        "previous_regime": None,
        "tilts": None,
        "weights": None,
        "allocation_rationale": None,
        "report": None,
        "messages": [],
    }

    result = graph.invoke(initial_state, config=config)

    # Regime summary
    regime = result.get("regime", "Unknown")
    growth = result.get("growth_direction", "unknown").title()
    inflation = result.get("inflation_direction", "unknown").title()
    confidence = result.get("regime_confidence", "unknown").title()
    date = result.get("current_date", "")

    regime_summary = f"**{regime}**\n\nGrowth: {growth}  |  Inflation: {inflation}  |  Confidence: {confidence}  |  As of: {date}"

    # Indicators table
    macro_data = result.get("macro_data", {})
    indicator_labels = {
        "t10y2y":       "Yield Curve (T10Y2Y)",
        "bamlh0a0hym2": "HY Credit Spread",
        "t10yie":       "Breakeven Inflation",
        "pcepilfe":     "Core PCE",
        "indpro":       "Industrial Production",
        "sahmrealtime": "Sahm Rule",
        "stlfsi4":      "Financial Stress Index",
        "vix":          "VIX",
    }

    indicator_rows = []
    for key, label in indicator_labels.items():
        d = macro_data.get(key, {})
        if "error" in d:
            indicator_rows.append([label, "N/A", "N/A", "N/A"])
        else:
            latest = d.get("latest")
            change = d.get("change")
            date_str = d.get("date", "")
            change_str = f"{change:+.4f}" if change is not None else "N/A"
            indicator_rows.append([
                label,
                f"{latest:.4f}" if latest is not None else "N/A",
                change_str,
                date_str,
            ])

    indicators_df = pd.DataFrame(
        indicator_rows,
        columns=["Indicator", "Latest", "Change", "As Of"]
    )

    # Allocation table
    weights = result.get("weights", {})
    tilts = result.get("tilts", {})

    allocation_rows = []
    for asset, weight in weights.items():
        tilt = tilts.get(asset, "Neutral")
        icon = TILT_ICONS.get(tilt, "—")
        allocation_rows.append([
            asset.replace("_", " ").title(),
            f"{weight:.1%}",
            f"{icon}  {tilt}",
        ])

    allocation_df = pd.DataFrame(
        allocation_rows,
        columns=["Asset Class", "Weight", "Tilt"]
    )

    regime_rationale = result.get("regime_rationale", "")
    allocation_rationale = result.get("allocation_rationale", "")

    return (
        regime_summary,
        indicators_df,
        allocation_df,
        regime_rationale,
        allocation_rationale,
    )


def run_backtest_ui(start_year: int, end_year: int, progress=gr.Progress()):
    start_str = f"{int(start_year)}-01-01"
    end_str   = f"{int(end_year)}-12-31"

    def cb(frac, msg):
        progress(frac, desc=msg)

    try:
        result = run_backtest(start=start_str, end=end_str, progress_callback=cb)
    except Exception as e:
        empty = go.Figure()
        empty.add_annotation(
            text=f"Backtest failed: {e}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#ef4444", size=14),
        )
        return empty, empty, f"<p style='color:#ef4444'>Error: {e}</p>"

    equity_fig   = _build_equity_chart(result["equity_curve"], result["benchmark_curve"], result["monthly_records"])
    timeline_fig = _build_regime_timeline(result["monthly_records"])
    metrics_str  = _metrics_html(result["metrics"])

    return equity_fig, timeline_fig, metrics_str


with gr.Blocks(title="macro-lens") as ui:

    gr.Markdown("# macro-lens")
    gr.Markdown(
        "**Macro Regime Detection · Tactical Asset Allocation · Backtest Engine**  \n"
        "Powered by FRED (point-in-time vintages) · yFinance · GPT-4o-mini · LangGraph"
    )

    with gr.Tabs():

        with gr.Tab("Live Analysis"):
            run_btn = gr.Button("▶  Run Analysis", variant="primary", scale=0)

            gr.Markdown("### Regime")
            regime_box = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Key Indicators")
                    indicators_table = gr.Dataframe(
                        headers=["Indicator", "Latest", "Change", "As Of"],
                        interactive=False, wrap=False,
                    )
                with gr.Column():
                    gr.Markdown("### Tactical Allocation")
                    allocation_table = gr.Dataframe(
                        headers=["Asset Class", "Weight", "Tilt"],
                        interactive=False, wrap=False,
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Regime Rationale")
                    regime_rationale_box = gr.Textbox(interactive=False, show_label=False, lines=4)
                with gr.Column():
                    gr.Markdown("### Allocation Rationale")
                    allocation_rationale_box = gr.Textbox(interactive=False, show_label=False, lines=4)

            run_btn.click(
                fn=run_analysis,
                inputs=[],
                outputs=[regime_box, indicators_table, allocation_table,
                         regime_rationale_box, allocation_rationale_box],
            )

        with gr.Tab("Backtest"):
            gr.Markdown(
                "Runs the regime pipeline monthly using **point-in-time FRED vintages**. "
                "Weights at month-end *t* are applied to ETF returns in month *t+1*."
            )

            with gr.Row():
                start_slider = gr.Slider(minimum=2010, maximum=2024, value=2015, step=1, label="Start Year")
                end_slider   = gr.Slider(minimum=2011, maximum=2025, value=2024, step=1, label="End Year")
                bt_run_btn   = gr.Button("▶  Run Backtest", variant="primary", scale=0)

            gr.Markdown("### Performance")
            equity_chart = gr.Plot()

            gr.Markdown("### Regime Timeline")
            regime_timeline = gr.Plot()

            gr.Markdown("### Summary Metrics")
            metrics_box = gr.HTML()

            bt_run_btn.click(
                fn=run_backtest_ui,
                inputs=[start_slider, end_slider],
                outputs=[equity_chart, regime_timeline, metrics_box],
            )


if __name__ == "__main__":
        ui.launch(
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
            * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
            .gradio-container { max-width: 1400px !important; }
        """
)