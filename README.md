# macro-lens

**Macro Regime Detection · Tactical Asset Allocation · Backtest Engine**

macro-lens is a stateful AI agent built with LangGraph that monitors macroeconomic conditions, classifies the current market regime, and generates a tactical asset allocation — mimicking the top-down analytical process used by institutional multi-asset portfolio managers.

---

## What it does

1. **Fetches** 8 live macroeconomic indicators from FRED (Federal Reserve Economic Data) plus VIX from yFinance — all free, no paid data dependencies
2. **Classifies** the current macro regime using the Bridgewater 2×2 framework (Growth rising/falling × Inflation rising/falling)
3. **Routes** based on confidence — if the regime signal is ambiguous, the graph loops back to fetch additional indicators before committing
4. **Generates** tactical tilts per asset class (5-point scale: Strong Underweight → Strong Overweight) using GPT-4o-mini
5. **Calculates** final portfolio weights deterministically in Python — normalised, long-only, sums to 100%
6. **Reports** via a Gradio dashboard with live analysis and backtest tabs

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

## v2: Backtest engine

### Methodology

The backtest runs the full LangGraph pipeline monthly from a user-defined start date to end date, applying two layers of no-look-ahead discipline:

**1. Point-in-time FRED data (ALFRED vintages)**

Every FRED call in backtest mode passes `realtime_start=realtime_end=observation_date`, querying the ALFRED database — FRED's archive of historical data vintages. This returns the data that was *actually published* on that date, not the current revised figure.

Example: Core PCE for February 2020 was initially released as 1.8%. It was later revised to 1.9%. A backtest observation on `2020-03-31` receives 1.8% — what a portfolio manager would have known. Without vintage locking, FRED silently returns the revised figure, introducing look-ahead bias from data revisions.

**2. Execution lag (weights[t] → returns[t+1])**

Weights determined at month-end *t* are applied to ETF returns in month *t+1*. The PM decides on January 31st but can only trade on February 1st.

**Asset class proxies**

| Asset class | ETF | Rationale |
|-------------|-----|-----------|
| Equities | SPY | S&P 500, liquid benchmark |
| Bonds | TLT | 20Y+ Treasuries |
| Inflation Linked | TIP | TIPS, correct IL proxy |
| Commodities | GSG | Broad commodity index |
| Gold | GLD | Direct gold exposure |
| Cash | BIL | 1-3M T-Bills |

**Regime hysteresis across months**

`previous_regime` is carried forward via the Python loop variable — each month's pipeline receives last month's classification and requires strong contradictory evidence before switching quadrants. LangGraph's checkpointer is intentionally bypassed (fresh thread ID per month) to avoid context bloat over long backtests.

**Rate limit handling**

Each month sleeps 1 second between iterations. A `tenacity` retry wrapper with exponential backoff handles transient OpenAI rate limit errors without losing prior results.

### Backtest results

| Period | Macro-Lens | 60/40 | Max DD (ML) | Max DD (60/40) |
|--------|-----------|-------|-------------|----------------|
| 2015 (mid-cycle) | -9.7% | -2.0% | -10.3% | -6.1% |
| 2017–2018 (Goldilocks) | +7.2% | +11.1% | -6.3% | -7.6% |
| 2020–2022 (COVID + stagflation) | +11.3% | +1.7% | -14.7% | -26.2% |

**Pattern:** The model earns its keep in genuine macro stress and transition regimes (2020–2022), and underperforms in smooth bull markets and commodity-bearish environments (2015, 2017). This is expected behaviour for a regime-aware strategy — it is not designed to beat 60/40 in Goldilocks, it is designed to avoid the tail drawdowns.

### Known limitations

- **Commodity sensitivity** — the model tends to overweight commodities during Low Growth / High Inflation regimes. In periods where inflation is driven by demand rather than supply (e.g. 2015 oil crash), this tilt hurts.
- **LLM non-determinism** — `temperature=0` gives high but not perfect reproducibility across runs. Minor classification differences are possible.
- **ETF history** — some proxies (BIL, TIP) have limited history pre-2007. Backtests starting before 2007 may have sparser data.
- **No transaction costs** — weights change monthly with no slippage or rebalancing cost modelled.

---

## Stack

- **LangGraph** — stateful graph with conditional routing and MemorySaver checkpointing
- **LangChain + OpenAI** — GPT-4o-mini for regime classification and allocation generation
- **fredapi** — FRED macroeconomic data with ALFRED vintage support (free API key required)
- **yfinance** — VIX and asset class price data
- **Gradio** — dashboard UI with live analysis and backtest tabs
- **Plotly** — interactive equity curve, regime timeline, active return chart
- **Pydantic** — structured LLM output validation
- **tenacity** — retry logic for OpenAI rate limit handling
- **LangSmith** — full trace visibility per monthly pipeline run (optional)

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

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=macro-lens
```

FRED API keys are free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

Run the dashboard:

```bash
uv run python app.py
```

---

## Planned (v3)

- **Result caching** — persist backtest output to JSON/SQLite; reload instantly without re-running 180 LLM calls
- **Async FRED fetching** — fetch all series concurrently within a month, cutting per-month time from ~15s to ~3s
- **Black-Litterman weight optimisation** — replace the tilt map with a full BL model using yFinance covariance matrices
- **Regime persistence** — write final state to disk, enabling genuine hysteresis across daily runs
- **Conversational layer** — ask follow-up questions about the current allocation via a chat interface

---

## Related projects

- [**equity-lens**](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI pipeline for global equity screening via yFinance
- [**edgar-research-rag**](https://github.com/matsveikachatkou/edgar-research-rag) — RAG pipeline over SEC EDGAR filings with buy/hold/sell recommendations
- [**fund-lens**](https://github.com/matsveikachatkou/fund-lens) — automated fund due diligence brief generator: scrapes factsheet PDFs, enriches with live benchmark data, generates structured briefs via GPT-4o