import gradio as gr
import pandas as pd
from datetime import datetime
import uuid
from macro_lens import build_graph, MacroState

TILT_ICONS = {
    "Strong Overweight":  "▲▲",
    "Overweight":         "▲",
    "Neutral":            "—",
    "Underweight":        "▼",
    "Strong Underweight": "▼▼",
}


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


with gr.Blocks(title="macro-lens") as ui:

    gr.Markdown("# macro-lens")
    gr.Markdown("**Macro Regime Detection · Tactical Asset Allocation**  \nPowered by FRED · yFinance · GPT-4o-mini · LangGraph")

    run_btn = gr.Button("▶  Run Analysis", variant="primary", scale=0)

    gr.Markdown("### Regime")
    regime_box = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Key Indicators")
            indicators_table = gr.Dataframe(
                headers=["Indicator", "Latest", "Change", "As Of"],
                interactive=False,
                wrap=False,
            )
        with gr.Column():
            gr.Markdown("### Tactical Allocation")
            allocation_table = gr.Dataframe(
                headers=["Asset Class", "Weight", "Tilt"],
                interactive=False,
                wrap=False,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Regime Rationale")
            regime_rationale_box = gr.Textbox(
                interactive=False,
                show_label=False,
                lines=4,
            )
        with gr.Column():
            gr.Markdown("### Allocation Rationale")
            allocation_rationale_box = gr.Textbox(
                interactive=False,
                show_label=False,
                lines=4,
            )

    run_btn.click(
        fn=run_analysis,
        inputs=[],
        outputs=[
            regime_box,
            indicators_table,
            allocation_table,
            regime_rationale_box,
            allocation_rationale_box,
        ],
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