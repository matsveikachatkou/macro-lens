# macro-lens

**Macro Regime Detection · Tactical Asset Allocation**

macro-lens is a stateful AI agent built with LangGraph that monitors macroeconomic conditions, classifies the current market regime, and generates a tactical asset allocation — mimicking the top-down analytical process used by institutional multi-asset portfolio managers.

---

## What it does

1. **Fetches** 8 live macroeconomic indicators from FRED (Federal Reserve Economic Data) plus VIX from yFinance — all free, no paid data dependencies
2. **Classifies** the current macro regime using the Bridgewater 2×2 framework (Growth rising/falling × Inflation rising/falling)
3. **Routes** based on confidence — if the regime signal is ambiguous, the graph loops back to fetch additional indicators before committing
4. **Generates** tactical tilts per asset class (5-point scale: Strong Underweight → Strong Overweight) using GPT-4o-mini
5. **Calculates** final portfolio weights deterministically in Python — normalised, long-only, sums to 100%
6. **Reports** via a clean Gradio dashboard

---

## Architecture

The core of macro-lens is a stateful LangGraph graph with a confidence-gated feedback loop — the key feature that differentiates it from linear pipeline frameworks.

```
START
  └─► data_fetcher        (Python — FRED + yFinance)
        └─► regime_classifier   (LLM — Bridgewater 2×2, Pydantic output)
              └─► confidence_router   (Python — conditional edge)
                    ├─► [low confidence, retries < 2] ──► data_fetcher
                    └─► [high confidence or retry cap] ──► allocation_generator
                                                               └─► weight_calculator  (Python)
                                                                     └─► reporter
                                                                           └─► END
```

**Key design decisions:**

- The confidence router is pure Python — no LLM involved in routing, keeping token costs down and preventing hallucinated transitions
- LLM outputs *tilts* (qualitative), Python enforces the *math* (weights) — strict separation of conviction from constraint
- `RegimeType` and tilt levels are Pydantic enums — the LLM cannot hallucinate an off-schema value
- `temperature=0` on all LLM nodes for maximum determinism
- `previous_regime` in state enables hysteresis — conservative about declaring regime changes

---

## Macro indicators

| Series | Description | Axis |
|--------|-------------|------|
| T10Y2Y | 10Y–2Y Treasury yield spread | Growth proxy |
| INDPRO | Industrial Production Index | Growth |
| SAHMREALTIME | Sahm Rule Recession Indicator | Growth |
| BAMLH0A0HYM2 | ICE BofA HY Credit Spread | Risk / Growth |
| STLFSI4 | St. Louis Fed Financial Stress Index | Risk |
| PCEPILFE | Core PCE Price Index | Inflation |
| T10YIE | 10Y Breakeven Inflation Rate | Inflation |
| UNRATE | Unemployment Rate | Growth |
| ^VIX | CBOE Volatility Index | Risk sentiment |

On low-confidence retry, three additional series are fetched: CFNAI, PERMIT, UMCSENT.

---

## Regime framework

Based on the Bridgewater All Weather 2×2 matrix:

| Regime | Growth | Inflation | Favoured assets |
|--------|--------|-----------|-----------------|
| High Growth / High Inflation | Rising | Rising | Equities, Commodities |
| High Growth / Low Inflation | Rising | Falling | Equities, Bonds |
| Low Growth / High Inflation | Falling | Rising | Commodities, Gold, Cash |
| Low Growth / Low Inflation | Falling | Falling | Bonds |

---

## Stack

- **LangGraph** — stateful graph with conditional routing and MemorySaver checkpointing
- **LangChain + OpenAI** — GPT-4o-mini for regime classification and allocation generation
- **fredapi** — FRED macroeconomic data (free API key required)
- **yfinance** — VIX and asset class price data
- **Gradio** — dashboard UI
- **Pydantic** — structured LLM output validation

---

## Setup

```bash
git clone https://github.com/matsveikachatkou/macro-lens.git
cd macro-lens

uv venv
source .venv/bin/activate
uv sync
```

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
FRED_API_KEY=your_fred_key
```

FRED API keys are free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

Run the dashboard:

```bash
uv run python app.py
```

---

## Phase 2 (planned)

- **Historical backtest mode** — pass a target date, graph runs on that snapshot of FRED data
- **Black-Litterman weight optimisation** — replace the tilt map with a full BL model using yFinance covariance matrices
- **Regime persistence** — write final state to disk, enabling genuine hysteresis across daily runs
- **Conversational layer** — ask follow-up questions about the current allocation via a chat interface

---

## Related projects

- [**equity-lens**](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI pipeline for global equity screening via yFinance
- [**edgar-research-rag**](https://github.com/matsveikachatkou/edgar-research-rag) — RAG pipeline over SEC EDGAR filings with buy/hold/sell recommendations
- [**fund-lens**](https://github.com/matsveikachatkou/fund-lens) — automated fund due diligence brief generator: scrapes factsheet PDFs, enriches with live benchmark data, generates structured briefs via GPT-4o