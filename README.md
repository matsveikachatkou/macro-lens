# macro-lens

**Macro Regime Detection · Tactical Asset Allocation · Backtest Engine**

macro-lens is a hybrid quant/AI system built with LangGraph that monitors macroeconomic conditions, classifies the current market regime using Hidden Markov Models, and generates a tactical asset allocation via Black-Litterman portfolio construction — mimicking the top-down analytical process used by institutional multi-asset portfolio managers.

---

## What it does

1. **Fetches** live macroeconomic indicators from FRED (Federal Reserve Economic Data) plus VIX from yFinance — all free, no paid data dependencies
2. **Classifies** the current macro regime using two independent 2-state Gaussian HMMs (one for growth, one for inflation), combined into the Bridgewater 2×2 framework (High/Low Growth × High/Low Inflation)
3. **Constructs** a tactical portfolio via Black-Litterman: policy-anchored implied returns + Bridgewater heuristic views + constrained mean-variance optimisation with ±15% TAA bands
4. **Explains** the quant model output via GPT-4o-mini — the LLM writes the rationale, the HMM and BL layer make all allocation decisions
5. **Reports** via a Gradio dashboard with live analysis and backtest tabs

---

## Architecture

### v3 — Hybrid Quant/AI Pipeline

```
START
  └─► data_fetcher           (Python — FRED + yFinance, live mode only)
        └─► quant_regime_node    (Python — RegimeHMMEngine)
              └─► quant_allocation_node  (Python — Black-Litterman)
                    └─► weight_calculator   (Python — normalisation safety net)
                          └─► narrative_generator  (LLM — explanation only, live mode only)
                                └─► reporter
                                      └─► END
```

**Key design decisions:**

- **HMM over LLM for regime classification** — two independent 2-state GaussianHMMs (hmmlearn) fit on point-in-time FRED data via ALFRED vintage API. The LLM is strictly explanatory and cannot influence the regime call or allocation weights.
- **Black-Litterman over tilt maps** — BL generates posterior expected returns from a policy-anchored prior and Bridgewater heuristic views. Constrained SLSQP optimisation (scipy) with policy-relative bounds [±15%] replaces the unconstrained MV step that would otherwise corner into single assets.
- **Two views per regime** — each Bridgewater quadrant expresses two relative views, ensuring both the growth and inflation dimensions differentiate the final allocation (HG/HI vs HG/LI produce meaningfully different commodity exposures).
- **Confidence-scaled effective_Q** — view magnitude scales by HMM confidence, preventing optimizer cornering when prior implied returns are near zero (documented deviation from canonical BL).
- **Linear blend** — `w_final = conf × w_bl + (1-conf) × policy_weights` provides smooth confidence-to-conviction scaling.
- **Independence assumption** — growth and inflation HMMs are fit separately; joint regime probability = product of marginals. Deliberate bias-variance tradeoff following Kritzman, Page & Turkington (2012, FAJ).

### v2 — LLM-only pipeline (historical)

v2 used GPT-4o-mini for both regime classification and allocation generation, with a confidence-gated feedback loop. Replaced in v3 by the HMM + BL layer. The LLM now explains quant model outputs rather than producing them.

---

## HMM Feature Set

| Series | Transform | Window | Role |
|--------|-----------|--------|------|
| CFNAI | level | 36m | Primary growth feature — Chicago Fed composite of 85 monthly indicators across production, employment, consumption, and orders. Replaces INDPRO (too narrow: goods sector only, ~11% of GDP). |
| UNRATE | diff | 36m | Secondary growth feature — labour market confirmation signal |
| PCEPILFE | yoy | 60m | Primary inflation feature — Fed's preferred realized gauge |
| CPIAUCSL | yoy | 60m | Secondary inflation feature — realized CPI. Replaces T10YIE (breakeven inflation embeds risk premium and TIPS liquidity premium — not a realized series) |

All features are z-scored on a rolling window using only data published as of the observation date (ALFRED vintage API, no look-ahead bias). Z-scores are clipped at ±3.5 to prevent outlier months (e.g. COVID March 2020) from dominating HMM EM fitting.

---

## Black-Litterman Setup

**Policy prior (neutral allocation):**

| Asset | Policy Weight | ETF Proxy |
|-------|--------------|-----------|
| Equities | 40% | SPY |
| Bonds | 30% | TLT |
| Inflation Linked | 10% | TIP |
| Commodities | 8% | GSG |
| Gold | 7% | GLD |
| Cash | 5% | BIL |

Note: this is a policy/risk-balanced prior, not a true market-cap equilibrium. BL implied returns are policy-anchored rather than market-clearing.

**View table (Bridgewater heuristic, Path 1 — external prior):**

| Regime | View 1 | View 2 |
|--------|--------|--------|
| High Growth / High Inflation | Equities over Bonds +2% | Commodities over Bonds +2% |
| High Growth / Low Inflation | Equities over Bonds +3% | Equities over Commodities +2% |
| Low Growth / High Inflation | Commodities over Equities +2.5% | Gold over Bonds +2% |
| Low Growth / Low Inflation | Bonds over Equities +2% | Gold over Commodities +2% |

**Optimizer bounds (SLSQP):**

| Asset | Bounds |
|-------|--------|
| Equities | [20%, 60%] — wider band for equity-benchmarked TAA |
| Bonds | [15%, 45%] |
| Inflation Linked | [0%, 25%] |
| Commodities | [0%, 23%] |
| Gold | [0%, 22%] |
| Cash | [0%, 8%] — residual liquidity buffer only |

---

## Regime Framework

Based on the Bridgewater All Weather 2×2 matrix:

| Regime | Growth | Inflation | Primary tilt |
|--------|--------|-----------|--------------|
| High Growth / High Inflation | Rising | Rising | Equities + Commodities |
| High Growth / Low Inflation | Rising | Falling | Equities (max conviction) |
| Low Growth / High Inflation | Falling | Rising | Commodities + Gold |
| Low Growth / Low Inflation | Falling | Falling | Bonds + Gold |

---

## Backtest Methodology

### Point-in-time discipline

Every FRED call passes `realtime_start=realtime_end=observation_date`, querying the ALFRED database — FRED's archive of historical data vintages. This returns the data *actually published* on that date, not the current revised figure, eliminating look-ahead bias from data revisions.

A disk-persisted z-score cache keyed by `(series_id, observation_date)` reduces ALFRED API calls from O(N²) to O(N) across backtest runs.

### Execution convention

Weights at month-end *t* are applied to ETF returns in month *t+1* — the PM decides January 31st but trades February 1st.

### Benchmarks

Three benchmarks are computed:

| Benchmark | Definition | Purpose |
|-----------|-----------|---------|
| Policy mix | Static 40/30/10/8/7/5, monthly rebalanced | **Primary** — isolates regime-overlay alpha from strategic asset mix |
| 60/40 | 60% SPY + 40% TLT | Reference — what a typical investor holds |
| Information Ratio | Sharpe of (portfolio - policy) monthly excess returns | Key metric for overlay value-add |

The policy mix is the correct primary benchmark for a TAA overlay strategy. 60/40 is retained as a reference but is not the primary success criterion — a 6-asset strategy with 40% policy equity weight is structurally different from a 2-asset 60/40 portfolio.

### Backtest results (2015–2024, 119 months)

| Metric | Macro-Lens v3 | Policy Mix | 60/40 |
|--------|--------------|------------|-------|
| Total Return | +114.5% | +78.4% | +103.3% |
| Ann. Return | +8.0% | +6.0% | +7.4% |
| Sharpe Ratio | 0.61 | 0.49 | 0.53 |
| Max Drawdown | -17.5% | -19.6% | -26.2% |
| Info Ratio vs Policy | +0.02 | — | — |

**Sub-period performance (regime-conditional):**

| Regime | n months | Mean active return vs 60/40 | Total contribution |
|--------|----------|----------------------------|-------------------|
| High Growth / High Inflation | 68 | +0.06% | +3.8% |
| High Growth / Low Inflation | 27 | +0.28% | +7.6% |
| Low Growth / High Inflation | 10 | -1.28% | -11.5% |
| Low Growth / Low Inflation | 15 | +0.27% | +3.8% |

**Pattern:** The strategy adds value in High Growth periods (dominant regime, 95 of 119 months) and Low Growth/Low Inflation defensive periods. The -11.5% drag from Low Growth/High Inflation is concentrated in 9 months around COVID (Feb-Apr 2020), where the model correctly identified slowing growth but held stagflation hedges (commodities + gold) into a deflationary liquidity crash — the worst environment for real assets. Without the COVID misclassification, total active contribution would be approximately +4pp positive.

**Average weights by regime:**

| Regime | Equities | Bonds | IL | Commodities | Gold | Cash |
|--------|----------|-------|----|-------------|------|------|
| HG/HI | 58% | 17% | 1% | 22% | 1% | 2% |
| HG/LI | 58% | 17% | 12% | 8% | 1% | 5% |
| LG/HI | 27% | 17% | 10% | 21% | 20% | 5% |
| LG/LI | 22% | 44% | 9% | 1% | 21% | 4% |

---

## Live Analysis (June 2026)

Current regime call: **Low Growth / Low Inflation** (94.1% confidence)

Key drivers: CFNAI at 0.14 (mildly positive but below trend), flat yield curve (T10Y2Y = 0.39), Sahm Rule at 0.10 (not triggered but rising), Financial Stress at -0.87 (low). Current BL allocation: 44% bonds, 21% gold, 21% equities, minimal commodities.

---

## Known Limitations and Model Misspecification

### 1. Inflation signal lag — the most important current limitation

The HMM inflation features (PCEPILFE, CPIAUCSL) are z-scored on a 60-month rolling window. When the trailing window includes the 2021-2023 high-inflation period (PCE at 5-6%), a current reading of 2.7% PCE looks "low" relative to that window mean, producing a negative inflation z-score even though 2.7% is above the Fed's 2% target. As of June 2026, the model calls Low Inflation while consensus forecasts (US Bank, JPMorgan, Morningstar) put 2026 PCE at 2.7-3.6% with Fed on hold and rate hike risk rising. The 60m window is too long for detecting the current inflation regime boundary — the model is a coincident indicator of realized data relative to recent history, not an absolute inflation level detector.

### 2. COVID regime misclassification (known, structural)

February-April 2020 is classified as Low Growth/High Inflation because PCEPILFE was running above its trailing mean going into the crash. COVID was a deflationary demand shock — the correct regime was Low Growth/Low Inflation. The stagflation allocation (commodities + gold) lost significantly relative to 60/40 during the March crash (equities) and April recovery (missed the V-shape). This accounts for -11.5pp of total active return drag from just 9 months. FRED reporting lag (INDPRO/UNRATE for March 2020 published in April) means the growth signal was also a month behind. This is expected behaviour for a monthly coincident indicator — no fix without introducing look-ahead.

### 3. Independence assumption between growth and inflation HMMs

Joint regime probability = P(growth) × P(inflation) assumes the two dimensions are independent. They are not — the business cycle creates empirical correlation between growth and inflation states. The cost lands precisely on the off-diagonal cells (stagflation and Goldilocks) where the All-Weather logic earns its keep. A joint 4-state HMM over 4 features would be more correct but has ~47 parameters estimated from ~120 autocorrelated months (roughly 5-10 distinct regime episodes) — severely overparameterized. The independence factorization follows Kritzman, Page & Turkington (2012, FAJ) who used the same approach.

### 4. CFNAI as growth proxy — services blind spot

CFNAI aggregates 85 indicators but is still weighted toward production, employment, and orders. During the 2015-2016 industrial recession (oil price collapse), CFNAI correctly called weak industrial activity — but equity markets returned ~12% because the service sector was fine. A growth indicator better correlated with equity earnings (e.g. Conference Board coincident index, GDP nowcasts) would be more appropriate for an equity-benchmarked TAA strategy.

### 5. Unconstrained MV replaced but BL prior is not true market-cap

MARKET_WEIGHTS (40/30/10/8/7/5) is a policy/risk-balanced prior, not a true market-cap equilibrium. BL reverse-optimization produces implied returns that rationalize the policy mix, not market-clearing returns. The equilibrium prior is policy-anchored — defensible as a TAA starting point but loses the CAPM grounding that motivates canonical BL.

### 6. Single equity sleeve — no sector differentiation

The equity allocation is entirely in SPY (cap-weighted S&P 500). The model has no mechanism to differentiate sector exposure within equities based on the macro regime — e.g. overweighting energy in High Inflation, overweighting technology in Goldilocks. This is the primary v4 opportunity (see Roadmap).

### 7. TLT as bond proxy — duration mismatch

TLT tracks 20Y+ Treasuries, far more volatile than aggregate bond exposure. During 2022, TLT fell ~30% while the Bloomberg Aggregate (AGG) fell ~13%. The bond sleeve performance is amplified by duration risk beyond what "30% bonds" implies for most investors.

### 8. No transaction costs

Monthly rebalancing with no slippage, bid-ask spread, or market impact modelled. In practice, moving 10-15% of a portfolio monthly would incur meaningful execution costs, particularly for less liquid ETFs (GSG, TIP).

### 9. In-sample backtest — data mining risk

The five architectural changes in v3 (CFNAI over INDPRO, CPIAUCSL over T10YIE, constrained MVO, ±15% TAA bands, two views per regime) were motivated by a combination of literature (KPT 2012, NBER, institutional practice) and observed backtest performance over the 2015-2024 window. The feature selection (CFNAI) is grounded in independent literature; the optimizer bounds and view structure are partially fitted to the known sample. An out-of-sample test (2010-2014 or forward) would be needed to validate whether the IR vs policy (+0.02) is genuine signal or in-sample optimism.

---

## Stack

- **LangGraph** — stateful graph with MemorySaver checkpointing; fresh thread ID per backtest month to prevent context bloat
- **hmmlearn** — GaussianHMM for regime classification (2-state, diagonal covariance)
- **scipy** — SLSQP constrained optimization for BL portfolio construction
- **LangChain + OpenAI** — GPT-4o-mini for narrative rationale only (live mode)
- **fredapi** — FRED macroeconomic data with ALFRED vintage support (free API key required)
- **yfinance** — ETF price history and VIX
- **Gradio** — dashboard UI with live analysis and backtest tabs
- **Plotly** — interactive equity curve, regime timeline, active return chart
- **tenacity** — retry logic for FRED rate limit handling

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

## v4 Roadmap

- **Sector rotation layer** — allocate the equity sleeve across sector ETFs (XLK, XLF, XLV, XLY, XLP, XLI, XLE, XLB, XLC, XLU, XLRE) using macro regime + LLM sentiment analysis, following the ICLR 2025 paper "Leveraging LLMs for Top-Down Sector Allocation in Automated Trading" (Quek et al., 2025)
- **LLM macro memory module** — store FOMC minutes summaries and regime history across runs; give the narrative agent longitudinal context rather than single-month snapshots
- **News sentiment overlay** — dual-stream processing of macro data + news sentiment for sector selection within the equity sleeve
- **Confirmation filter** — only rebalance when dominant regime has been stable for 2 consecutive months, reducing whipsaw transaction costs
- **Absolute inflation anchor** — supplement the rolling z-score inflation signal with an absolute level check (e.g. PCE > 2.5% flags High Inflation regardless of z-score) to fix the window-anchoring limitation in post-high-inflation periods
- **Quarterly rebalancing** — align rebalancing cadence with HMM refit cadence to reduce noise from monthly regime flips

---

## Related projects

- [**equity-lens**](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI pipeline for global equity screening via yFinance
- [**edgar-research-rag**](https://github.com/matsveikachatkou/edgar-research-rag) — RAG pipeline over SEC EDGAR filings with buy/hold/sell recommendations
- [**fund-lens**](https://github.com/matsveikachatkou/fund-lens) — automated fund due diligence brief generator: scrapes factsheet PDFs, enriches with live benchmark data, generates structured briefs via GPT-4o