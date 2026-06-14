"""
app.py — Gradio dashboard for macro-lens v4.

Two tabs:
  Live Analysis: runs the full pipeline (FRED fetch → HMM → [validator] → BL → LLM narrative)
                 and displays the regime call, validator decision, indicator readings,
                 joint HMM probabilities, BL allocation, and LLM rationale.
  Backtest:      runs the monthly point-in-time backtest over a user-selected
                 date range and renders the equity curve, regime timeline, and
                 summary metrics vs 60/40 and the static policy mix.

v4 changes vs v3:
  - run_analysis() called with validate_regime="active" so validator overrides
    propagate to BL weights in live mode
  - Core PCE displayed as YoY % (not raw index level)
  - Validator decision row added to live analysis UI
  - Version string updated to v4 throughout
  - Backtest metrics table adds Policy Mix column alongside 60/40
"""

import gradio as gr
import pandas as pd
from datetime import datetime
import uuid

from macro_lens import run_analysis, run_backtest, ASSET_PROXIES
from black_litterman import MARKET_WEIGHTS, ASSETS as BL_ASSETS
from features import get_pcepilfe_level

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Bridgewater 2x2 regime colour scheme — consistent across all charts
REGIME_COLORS = {
    "High Growth / High Inflation": "#f59e0b",   # amber
    "High Growth / Low Inflation":  "#22c55e",   # green
    "Low Growth / High Inflation":  "#ef4444",   # red
    "Low Growth / Low Inflation":   "#3b82f6",   # blue
}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_equity_chart(equity_curve, benchmark_curve, policy_curve, monthly_records) -> go.Figure:
    """
    Two-panel chart:
      Top: cumulative performance (base 100) — Macro-Lens, Policy Mix, 60/40
           with regime background shading
      Bottom: monthly active return vs policy mix (primary benchmark)
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Cumulative Performance (Base 100)", "Active Return vs Policy Mix"),
    )

    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, y=equity_curve.values,
            name="Macro-Lens v4",
            line=dict(color="#6366f1", width=2.5),
            hovertemplate="%{x|%b %Y}<br>Portfolio: %{y:.1f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=policy_curve.index, y=policy_curve.values,
            name="Policy Mix (40/30/10/8/7/5)",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>Policy: %{y:.1f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_curve.index, y=benchmark_curve.values,
            name="60/40 Benchmark",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>60/40: %{y:.1f}<extra></extra>",
        ), row=1, col=1,
    )

    # Regime background bands
    if monthly_records:
        prev_regime = None
        band_start  = None

        def add_band(start, end, regime):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=REGIME_COLORS.get(regime, "#e2e8f0"),
                opacity=0.12, line_width=0, row=1, col=1, layer="below"
            )

        for rec in monthly_records:
            r = rec["regime"]
            d = pd.Timestamp(rec["date"])
            if r != prev_regime:
                if prev_regime is not None and band_start is not None:
                    add_band(band_start, d, prev_regime)
                band_start  = d
                prev_regime = r

        if prev_regime and band_start is not None and len(equity_curve) > 0:
            add_band(band_start, equity_curve.index[-1], prev_regime)

    # Active return vs policy mix
    if len(equity_curve) > 0 and len(policy_curve) > 0:
        common_idx = equity_curve.index.intersection(policy_curve.index)
        active = (
            equity_curve.loc[common_idx].pct_change().dropna()
            - policy_curve.loc[common_idx].pct_change().dropna()
        ) * 100

        fig.add_trace(
            go.Bar(
                x=active.index, y=active.values,
                name="Active vs Policy",
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in active.values],
                opacity=0.7,
                hovertemplate="%{x|%b %Y}<br>Active: %{y:.2f}%<extra></extra>",
            ), row=2, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color="#64748b", row=2, col=1)

    fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"), 
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(gridcolor="rgba(100, 116, 139, 0.2)", zeroline=False), # Transparent mid-tone
            xaxis2=dict(showgrid=False, zeroline=False),
            yaxis2=dict(gridcolor="rgba(100, 116, 139, 0.2)", zerolinecolor="#64748b"),
        )

    return fig


def _build_regime_timeline(monthly_records: list) -> go.Figure:
    """Single-row bar chart showing the dominant regime for each month."""
    if not monthly_records:
        return go.Figure()

    dates  = [pd.Timestamp(r["date"]) for r in monthly_records]
    colors = [REGIME_COLORS.get(r["regime"], "#94a3b8") for r in monthly_records]
    labels = [r["regime"] for r in monthly_records]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=[1] * len(dates),
        marker_color=colors, marker_line_width=0,
        customdata=labels,
        hovertemplate="%{x|%b %Y}<br>%{customdata}<extra></extra>",
        showlegend=False,
    ))

    for regime, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
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
            font=dict(family="Inter, sans-serif", color="#64748b", size=11), # Mid-tone grayish-blue
            legend=dict(
                orientation="h", yanchor="top", y=-0.4,
                xanchor="center", x=0.5, font=dict(size=10),
            ),
        )

    return fig


def _metrics_html(metrics: dict) -> str:
    """Render the summary metrics table as an HTML string — three columns."""
    def fmt_pct(v):
        return f"{v*100:+.1f}%" if v is not None else "N/A"

    def fmt_f(v):
        return f"{v:.2f}" if v is not None and v == v else "N/A"

    rows = [
        ("Total Return",
            fmt_pct(metrics.get("total_return_portfolio")),
            fmt_pct(metrics.get("total_return_policy")),
            fmt_pct(metrics.get("total_return_benchmark"))),
        ("Ann. Return",
            fmt_pct(metrics.get("ann_return_portfolio")),
            fmt_pct(metrics.get("ann_return_policy")),
            fmt_pct(metrics.get("ann_return_benchmark"))),
        ("Sharpe Ratio",
            fmt_f(metrics.get("sharpe_portfolio")),
            fmt_f(metrics.get("sharpe_policy")),
            fmt_f(metrics.get("sharpe_benchmark"))),
        ("Max Drawdown",
            fmt_pct(metrics.get("max_drawdown_portfolio")),
            fmt_pct(metrics.get("max_drawdown_policy")),
            fmt_pct(metrics.get("max_drawdown_benchmark"))),
        ("Months", str(metrics.get("n_months", "N/A")), "", ""),
    ]

    html = """
    <style>
      .mt { width:100%; border-collapse:collapse; font-size:14px; }
      .mt th { text-align:left; padding:8px 12px; color:#94a3b8;
               font-weight:500; border-bottom:1px solid #334155; }
      .mt td { padding:8px 12px; border-bottom:1px solid #1e293b; }
      .mt tr:last-child td { border-bottom:none; }
      .port  { color:#6366f1; font-weight:600; }
      .policy { color:#f59e0b; font-weight:400; }
      .bench { color:#94a3b8; }
    </style>
    <table class="mt">
      <thead><tr>
        <th>Metric</th>
        <th>Macro-Lens v4</th>
        <th>Policy Mix</th>
        <th>60/40</th>
      </tr></thead><tbody>
    """
    for label, port_val, policy_val, bench_val in rows:
        html += (
            f"<tr><td style='color:inherit; font-weight:500;'>{label}</td>"
            f"<td class='port'>{port_val}</td>"
            f"<td class='policy'>{policy_val}</td>"
            f"<td class='bench'>{bench_val}</td></tr>"
        )
    html += "</tbody></table>"
    return html


# ---------------------------------------------------------------------------
# UI callback functions
# ---------------------------------------------------------------------------

def run_analysis_ui():
    """
    Live analysis callback: invoke the full pipeline and format outputs.

    v4: validate_regime="active" so validator OVERRIDEs propagate to BL weights.
    """
    result = run_analysis(validate_regime="active")

    # Regime summary — show override if active
    regime            = result.get("regime", "Unknown")
    original_regime   = result.get("original_hmm_regime")
    growth            = result.get("growth_direction", "unknown").title()
    inflation         = result.get("inflation_direction", "unknown").title()
    confidence        = result.get("regime_confidence", 0.0)
    date              = result.get("current_date", "")
    validator_decision = result.get("regime_validator_decision", "SKIPPED")
    validator_rationale = result.get("regime_validator_rationale", "")

    # Show override context if validator changed the regime
    if (validator_decision == "OVERRIDE"
            and original_regime
            and original_regime != regime):
        regime_line = (
            f"**{regime}** *(overridden from {original_regime})*\n\n"
            f"Growth: {growth}  |  Inflation: {inflation}  |  "
            f"Confidence: {confidence:.1%}  |  As of: {date}\n\n"
            f"🔍 **Validator [{validator_decision}]:** {validator_rationale}"
        )
    elif validator_decision == "FLAG":
        regime_line = (
            f"**{regime}** *(validator flagged — policy weights applied)*\n\n"
            f"Growth: {growth}  |  Inflation: {inflation}  |  "
            f"Confidence: {confidence:.1%}  |  As of: {date}\n\n"
            f"🔍 **Validator [{validator_decision}]:** {validator_rationale}"
        )
    elif validator_decision not in ("SKIPPED", None):
        regime_line = (
            f"**{regime}**\n\n"
            f"Growth: {growth}  |  Inflation: {inflation}  |  "
            f"Confidence: {confidence:.1%}  |  As of: {date}\n\n"
            f"✅ **Validator [CONFIRM]:** {validator_rationale}"
        )
    else:
        regime_line = (
            f"**{regime}**\n\n"
            f"Growth: {growth}  |  Inflation: {inflation}  |  "
            f"Confidence: {confidence:.1%}  |  As of: {date}"
        )

    # Indicators table
    # PCE: fetch YoY % separately since FRED series is an index level
    macro_data = result.get("macro_data", {})
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    pce_yoy = get_pcepilfe_level(fred, datetime.now().strftime("%Y-%m-%d"))
    pce_yoy_str = f"{pce_yoy:.2%}" if pce_yoy == pce_yoy else "N/A"

    indicator_labels = {
        "t10y2y":       "Yield Curve (T10Y2Y)",
        "bamlh0a0hym2": "HY Credit Spread",
        "t10yie":       "Breakeven Inflation",
        "pcepilfe":     "Core PCE (YoY %)",
        "cfnai":        "CFNAI (Activity Index)",
        "sahmrealtime": "Sahm Rule",
        "stlfsi4":      "Financial Stress Index",
        "vix":          "VIX",
    }

    indicator_rows = []
    for key, label in indicator_labels.items():
        d = macro_data.get(key, {})
        if "error" in d:
            indicator_rows.append([label, "N/A", "N/A", "N/A"])
            continue

        latest   = d.get("latest")
        change   = d.get("change")
        date_str = d.get("date", "")

        # Override PCE display: show YoY % instead of index level
        if key == "pcepilfe":
            latest_str = pce_yoy_str
            change_str = "N/A"  # MoM change in YoY % not meaningful to display
        else:
            latest_str = f"{latest:.4f}" if latest is not None else "N/A"
            change_str = f"{change:+.4f}" if change is not None else "N/A"

        indicator_rows.append([label, latest_str, change_str, date_str])

    indicators_df = pd.DataFrame(
        indicator_rows,
        columns=["Indicator", "Latest", "Change", "As Of"]
    )

    # HMM joint probabilities table
    hmm_probs        = result.get("hmm_probabilities", {})
    mkt_weights_dict = dict(zip(BL_ASSETS, MARKET_WEIGHTS))

    prob_rows = []
    for regime_name, prob in hmm_probs.items():
        bar = "█" * int(prob * 20)
        prob_rows.append([regime_name, f"{prob:.1%}", bar])

    hmm_df = pd.DataFrame(
        prob_rows, columns=["Regime", "Probability", ""]
    ) if prob_rows else pd.DataFrame(columns=["Regime", "Probability", ""])

    # BL allocation table
    weights = result.get("weights", {})
    allocation_rows = []
    for asset in BL_ASSETS:
        weight = weights.get(asset, 0.0)
        mkt_w  = mkt_weights_dict.get(asset, 0.0)
        delta  = weight - mkt_w
        allocation_rows.append([
            asset.replace("_", " ").title(),
            f"{weight:.1%}",
            f"{mkt_w:.1%}",
            f"{delta:+.1%}",
        ])

    allocation_df = pd.DataFrame(
        allocation_rows,
        columns=["Asset Class", "BL Weight", "Market Weight", "Active Tilt"]
    )

    regime_rationale     = result.get("regime_rationale", "")
    allocation_rationale = result.get("allocation_rationale", "")

    return (
        regime_line,
        indicators_df,
        hmm_df,
        allocation_df,
        regime_rationale,
        allocation_rationale,
    )


def run_backtest_ui(start_year: int, end_year: int, validator_mode: str, progress=gr.Progress()):
    """Backtest callback: run the monthly backtest and render charts + metrics."""
    start_str = f"{int(start_year)}-01-01"
    end_str   = f"{int(end_year)}-12-31"

    # Map UI label to internal parameter
    validate_regime = "active" if validator_mode == "With LLM Validator" else "off"

    def cb(frac, msg):
        progress(frac, desc=msg)

    try:
        result = run_backtest(
            start=start_str,
            end=end_str,
            progress_callback=cb,
            use_cfnai_ma3=True,
            validate_regime=validate_regime,
        )
    except Exception as e:
        empty = go.Figure()
        empty.add_annotation(
            text=f"Backtest failed: {e}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#ef4444", size=14),
        )
        return empty, empty, f"<p style='color:#ef4444'>Error: {e}</p>"

    equity_fig   = _build_equity_chart(
        result["equity_curve"],
        result["benchmark_curve"],
        result["policy_curve"],
        result["monthly_records"],
    )
    timeline_fig = _build_regime_timeline(result["monthly_records"])
    metrics_str  = _metrics_html(result["metrics"])

    return equity_fig, timeline_fig, metrics_str


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="macro-lens v4") as ui:

    gr.Markdown("# macro-lens v4")
    gr.Markdown(
        "**Macro Regime Detection · Tactical Asset Allocation · Backtest Engine**  \n"
        "HMM Regime Classification · Black-Litterman Portfolio Construction · "
        "FRED (ALFRED point-in-time vintages) · yFinance · GPT-4o-mini · LangGraph  \n"
        "*v4: 120m inflation window · CFNAI-MA3 · Gated LLM Regime Validator*"
    )

    with gr.Tabs():

        # --------------------------------------------------------------
        with gr.Tab("Live Analysis"):
            run_btn = gr.Button("▶  Run Analysis", variant="primary", scale=0)

            gr.Markdown("### Regime  `[HMM + Validator]`")
            regime_box = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Key Indicators")
                    indicators_table = gr.Dataframe(
                        headers=["Indicator", "Latest", "Change", "As Of"],
                        interactive=False, wrap=False,
                    )
                with gr.Column():
                    gr.Markdown("### Joint Regime Probabilities  `[HMM]`")
                    hmm_table = gr.Dataframe(
                        headers=["Regime", "Probability", ""],
                        interactive=False, wrap=False,
                    )

            gr.Markdown("### Tactical Allocation  `[Black-Litterman]`")
            allocation_table = gr.Dataframe(
                headers=["Asset Class", "BL Weight", "Market Weight", "Active Tilt"],
                interactive=False, wrap=False,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Regime Rationale  `[LLM Analyst]`")
                    regime_rationale_box = gr.Textbox(
                        interactive=False, show_label=False, lines=4
                    )
                with gr.Column():
                    gr.Markdown("### Allocation Rationale  `[LLM Analyst]`")
                    allocation_rationale_box = gr.Textbox(
                        interactive=False, show_label=False, lines=4
                    )

            run_btn.click(
                fn=run_analysis_ui,
                inputs=[],
                outputs=[
                    regime_box, indicators_table, hmm_table,
                    allocation_table, regime_rationale_box, allocation_rationale_box,
                ],
            )

        # --------------------------------------------------------------
        with gr.Tab("Backtest"):
            gr.Markdown(
                "Runs the HMM + Black-Litterman pipeline monthly using "
                "**point-in-time FRED vintages**. "
                "Weights at month-end *t* are applied to ETF returns in month *t+1*.  \n"
                "Benchmark: Policy Mix (40/30/10/8/7/5) · 60/40 (SPY/TLT)"
            )

            with gr.Row():
                start_slider = gr.Slider(
                    minimum=2010, maximum=2024, value=2015,
                    step=1, label="Start Year"
                )
                end_slider = gr.Slider(
                    minimum=2011, maximum=2025, value=2024,
                    step=1, label="End Year"
                )

            validator_toggle = gr.Radio(
                choices=["Structural Fixes Only", "With LLM Validator"],
                value="Structural Fixes Only",
                label="Regime Validator",
                info=(
                    "'Structural Fixes Only' — 120m inflation window + CFNAI-MA3 baseline.  "
                    "'With LLM Validator' — adds gated PCE-anchor override "
                    "(results cached, no extra API calls)."
                ),
            )

            bt_run_btn = gr.Button(
                "▶  Run Backtest", variant="primary", scale=0, interactive=True
            )

            gr.Markdown("### Performance")
            equity_chart = gr.Plot()

            gr.Markdown("### Regime Timeline")
            regime_timeline = gr.Plot()

            gr.Markdown("### Summary Metrics")
            metrics_box = gr.HTML()

            bt_run_btn.click(
                fn=lambda: gr.update(value="⏳  Running...", interactive=False),
                outputs=[bt_run_btn],
                queue=False,
            ).then(
                fn=run_backtest_ui,
                inputs=[start_slider, end_slider, validator_toggle],
                outputs=[equity_chart, regime_timeline, metrics_box],
            ).then(
                fn=lambda: gr.update(value="▶  Run Backtest", interactive=True),
                outputs=[bt_run_btn],
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