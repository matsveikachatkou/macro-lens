# macro-lens

**Macro Regime Detection · Tactical Asset Allocation · Backtest Engine**

macro-lens is a hybrid quant/AI system built with LangGraph that monitors macroeconomic conditions, classifies the current market regime using Hidden Markov Models, and generates a tactical asset allocation via Black-Litterman portfolio construction — mimicking the top-down analytical process used by institutional multi-asset portfolio managers.

---

## What it does

1. **Fetches** live macroeconomic indicators from FRED (Federal Reserve Economic Data) plus VIX from yFinance — all free, no paid data dependencies
2. **Classifies** the current macro regime using two independent 2-state Gaussian HMMs (one for growth, one for inflation), combined into the Bridgewater 2×2 framework (High/Low Growth × High/Low Inflation)
3. **Validates** the HMM regime call via a gated LLM sanity check — an absolute PCE anchor and VIX stress gate catch structural HMM failure modes that rolling z-scores cannot detect
4. **Constructs** a tactical portfolio via Black-Litterman: policy-anchored implied returns + Bridgewater heuristic views + constrained mean-variance optimisation with ±15% TAA bands
5. **Explains** the quant model output via GPT-4o-mini — the LLM writes the rationale, the HMM and BL layer make all allocation decisions
6. **Reports** via a Gradio dashboard with live analysis and backtest tabs

---

## Architecture

### v4 — Hybrid Quant/AI Pipeline with Gated Validator

```
START
  └─► data_fetcher              (Python — FRED + yFinance, live mode only)
        └─► quant_regime_node       (Python — RegimeHMMEngine)
              └─► llm_regime_validator   (LLM — gated PCE-anchor + stress check)
                    └─► quant_allocation_node  (Python — Black-Litterman)
                          └─► weight_calculator   (Python — normalisation safety net)
                                └─► narrative_generator  (LLM — explanation only, live mode only)
                                      └─► reporter
                                            └─► END
```

**Key design decisions:**

- **HMM over LLM for regime classification** — two independent 2-state GaussianHMMs (hmmlearn) fit on point-in-time FRED data via ALFRED vintage API. The LLM validator is a second opinion with veto power in specific failure modes, not a replacement for the HMM.
- **Gated validator, not free-judging** — the LLM is only invoked when a deterministic gate fires. Gate 2 (PCE anchor) fires when PCE YoY > 2.5% and the HMM calls Low Inflation. Gate 3 (stress) fires when VIX > 35 and the HMM calls High Growth. No gate = no LLM call = no cost and no noise.
- **Blinded inputs** — the validator receives no calendar date. It reasons from indicator values only (PCE level, CFNAI z-score, HY spread, yield curve). This forces contemporaneous reasoning rather than episode recall.
- **Frozen prompt with integrity hash** — `validator_prompt_FROZEN.py` and its SHA-256 are committed before any backtest results are examined, enforcing discipline against prompt iteration on results.
- **Black-Litterman over tilt maps** — BL generates posterior expected returns from a policy-anchored prior and Bridgewater heuristic views. Constrained SLSQP optimisation (scipy) with policy-relative bounds [±15%] replaces the unconstrained MV step that would otherwise corner into single assets.
- **Two views per regime** — each Bridgewater quadrant expresses two relative views, ensuring both the growth and inflation dimensions differentiate the final allocation (HG/HI vs HG/LI produce meaningfully different commodity exposures).
- **Confidence-scaled effective_Q** — view magnitude scales by HMM confidence, preventing optimizer cornering when prior implied returns are near zero (documented deviation from canonical BL).
- **Linear blend** — `w_final = conf × w_bl + (1-conf) × policy_weights` provides smooth confidence-to-conviction scaling.
- **Independence assumption** — growth and inflation HMMs are fit separately; joint regime probability = product of marginals. Deliberate bias-variance tradeoff following Kritzman, Page & Turkington (2012, FAJ).

---

## v4 Changes vs v3

### Problem 1 — Inflation window anchoring (structural fix)

The 60m rolling z-score for PCEPILFE and CPIAUCSL was anchored to the trailing window mean. When the window included 2021-2023 high-inflation years (PCE at 5-6%), a current reading of 2.7% PCE looked "low" by comparison, producing a Low Inflation call even though 2.7% is above the Fed's 2% target.

**Fix:** Extended both inflation windows from 60m to 120m. At 120m the window spans a full decade and always includes at least one complete low-inflation era, making the z-score more stable across regime transitions. This is a parameter-free fix — 120m (10 years) is a standard business cycle span, not tuned on backtest results.

### Problem 2 — COVID growth signal noise (structural fix)

The March 2020 CFNAI reading was a one-month cliff (-4.97 z-score) driven by the manufacturing collapse. The 3-month MA3 version was already negative in January-February, giving the HMM a more honest read of the underlying trend without introducing a free parameter — CFNAI-MA3 is a published Chicago Fed series.

**Fix:** Added `use_cfnai_ma3=True` flag to `RegimeHMMEngine`. When active, `CFNAI_MA3` replaces `CFNAI` as the primary growth feature. The 3m rolling mean is computed in `features.py` before z-scoring; everything downstream is identical.

### Problem 3 — Post-high-inflation era misclassification (LLM validator)

After the 2021-2023 surge, PCE declined from 6% to ~2.7-3.3% by 2023-2024. The HMM, trained on a window that included the surge, placed its High Inflation state mean around 4-5%. PCE at 3% sat inside the Low Inflation state's support. The 120m window fix improved the z-score direction but didn't fully resolve the HMM's state geometry — 3% PCE was still closer to the Low Inflation Gaussian mean than the High Inflation mean.

**Fix:** Gated LLM validator with absolute PCE anchor (Gate 2). When PCE YoY > 2.5% and the HMM calls Low Inflation, the validator is invoked to adjudicate. With blinded inputs and a frozen prompt, it correctly calls High Inflation based on the raw PCE level vs the Fed's 2% target — reasoning the HMM cannot do because it only sees z-scores.

---

## HMM Feature Set

| Series | Transform | Window | Role |
|--------|-----------|--------|------|
| CFNAI-MA3 | level | 36m | Primary growth feature — 3-month MA of Chicago Fed composite of 85 monthly indicators. MA3 smooths one-off monthly spikes (hurricanes, survey noise) without introducing a free parameter. Replaces INDPRO (goods sector only, ~11% of GDP). |
| UNRATE | diff | 36m | Secondary growth feature — labour market confirmation signal |
| PCEPILFE | yoy | 120m | Primary inflation feature — Fed's preferred realized gauge. Window extended from 60m to 120m in v4 to prevent a single high-inflation era from dominating the rolling mean. |
| CPIAUCSL | yoy | 120m | Secondary inflation feature — realized CPI. Replaces T10YIE (breakeven inflation embeds risk premium and TIPS liquidity premium — not a realized series). Window extended from 60m to 120m in v4. |

All features are z-scored on a rolling window using only data published as of the observation date (ALFRED vintage API, no look-ahead bias). Z-scores are clipped at ±3.5 to prevent outlier months from dominating HMM EM fitting.

---

## LLM Regime Validator

### Design

The validator is a gated sanity check inserted between `quant_regime_node` and `quant_allocation_node`. It does not replace the HMM — it acts as a second opinion with veto power only in specific, economically-motivated failure modes.

### Gate logic (deterministic — no LLM involved)

| Gate | Condition | Failure mode caught |
|------|-----------|-------------------|
| Gate 2 — PCE anchor | PCE YoY > 2.5% AND HMM calls Low Inflation | Rolling z-score anchored to high-inflation era misclassifies above-target PCE as Low Inflation |
| Gate 3 — Stress signal | VIX > 35 AND HMM calls High Growth | Growth indicators lag by 1-2 months during liquidity crises; VIX reacts immediately |

Gate 1 (confidence threshold < 0.65) was tested and removed. It produced economically unjustified overrides in the 2015 backtest window — the LLM free-associated regimes without a strong anchor when only confidence was low. Gates 2 and 3 have explicit economic anchors that constrain the LLM to specific failure modes.

### Threshold justification (all first-principles, none tuned on backtest)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| PCE threshold | 2.5% | Fed target (2.0%) + 25bp buffer — standard "demonstrably above target" definition in Fed communication |
| VIX threshold | 35 | Practitioner threshold for crisis-level volatility; referenced in Whaley (2009) and CBOE literature |

### Blinding and freeze discipline

- Inputs are stripped of calendar date before LLM call — forces reasoning from indicator values, not episode recall
- `validator_prompt_FROZEN.py` SHA-256 committed before first backtest run
- Contamination audit built in: `audit_rationale_for_contamination()` checks rationales for episode-specific language (year references, named events, forward-looking statements)
- Validator cache keyed by SHA-256 of blinded inputs — same macro readings always produce same cached decision

### Three-arm experiment

| Arm | Config | Result |
|-----|--------|--------|
| Arm 1 | Structural fixes only (`validate_regime="off"`) | Baseline for comparison |
| Arm 2 | Advisory log (`validate_regime="log"`) | Weights unchanged; decision audit primary artifact |
| Arm 3 | Active override (`validate_regime="active"`) | Gate 2 OVERRIDEs applied to BL weights |

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
| Equities | [20%, 60%] |
| Bonds | [15%, 45%] |
| Inflation Linked | [0%, 25%] |
| Commodities | [0%, 23%] |
| Gold | [0%, 22%] |
| Cash | [0%, 8%] |

---

## Backtest Methodology

### Point-in-time discipline

Every FRED call passes `realtime_start=realtime_end=observation_date`, querying the ALFRED database — FRED's archive of historical data vintages. This returns the data *actually published* on that date, not the current revised figure, eliminating look-ahead bias from data revisions.

A disk-persisted z-score cache keyed by `(series_id, observation_date)` reduces ALFRED API calls from O(N²) to O(N) across backtest runs. A separate validator cache keyed by SHA-256 of blinded inputs ensures LLM decisions are reproducible and cost nothing on re-runs.

### Execution convention

Weights at month-end *t* are applied to ETF returns in month *t+1*.

### Benchmarks

| Benchmark | Definition | Purpose |
|-----------|-----------|---------|
| Policy mix | Static 40/30/10/8/7/5, monthly rebalanced | **Primary** — isolates regime-overlay alpha from strategic asset mix |
| 60/40 | 60% SPY + 40% TLT | Reference |
| Information Ratio | Sharpe of (portfolio − policy) monthly excess returns | Key metric for overlay value-add |

### Backtest results (2015–2024, 119 months)

| Metric | v3 | v4 Arm 1 (structural) | v4 Arm 3 (+ validator) | Policy Mix | 60/40 |
|--------|----|-----------------------|------------------------|------------|-------|
| Total Return | +114.5% | +123.7% | +130.1% | +78.4% | +103.3% |
| Ann. Return | +7.9% | +8.5% | +8.8% | +6.0% | +7.4% |
| Sharpe Ratio | 0.61 | 0.67 | 0.74 | 0.49 | 0.53 |
| Max Drawdown | -17.5% | -17.5% | -17.5% | -19.6% | -26.2% |
| IR vs Policy | +0.02 | +0.12 | +0.16 | — | — |

Each step adds value monotonically. MDD is unchanged across all three — the improvements come from better regime classification, not from taking more risk.

**What drove the improvement:**

- Arm 1 vs v3 (+0.06 Sharpe): longer inflation window correctly z-scores the post-2021 disinflation period; CFNAI-MA3 reduces growth signal noise
- Arm 3 vs Arm 1 (+0.07 Sharpe): Gate 2 correctly reclassified 22 months in 2023-2024 from Low Inflation to High Inflation, shifting allocation from bonds (44%) to commodities (22%) + gold (21%) in a period where bonds were underperforming and real assets were holding up

**Average weights by regime (Arm 3):**

| Regime | n months | Equities | Bonds | IL | Commodities | Gold | Cash |
|--------|----------|----------|-------|----|-------------|------|------|
| HG/HI | 39 | 58% | 16% | 1% | 22% | 1% | 2% |
| HG/LI | 21 | 58% | 16% | 14% | 3% | 1% | 8% |
| LG/HI | 38 | 25% | 16% | 9% | 22% | 21% | 7% |
| LG/LI | 22 | 22% | 44% | 13% | 1% | 21% | 0% |

### Validator experiment findings

- **22 Gate 2 OVERRIDEs** across 2023-2024 — all months where PCE was 2.57-4.70% and HMM called Low Inflation
- **Zero contamination markers** in any rationale — validator cited only contemporaneous indicator values, no episode names or year references
- **Do-no-harm confirmed** — log mode (Arm 2) produced identical performance to Arm 1, confirming no state mutation
- **Honest caveat** — the 22 override months are a single contiguous period within LLM training data. Statistical significance cannot be claimed. The validator's primary value is forward-looking: live mode, genuinely out-of-sample, where it catches the structural z-score failure in real time

---

## Live Analysis (June 2026)

Current regime call: **Low Growth / High Inflation** (validator override from HMM's Low Growth / Low Inflation)

- HMM called Low Inflation because PCE z-score is slightly above rolling window mean but within the Low Inflation Gaussian's support
- Gate 2 fired: PCE YoY at 3.29% > 2.5% threshold
- Validator overrode to Low Growth / High Inflation with HIGH confidence
- BL allocation: 22% commodities, 21% gold, 21% equities, 16% bonds — stagflation tilt

---

## Known Limitations

### 1. Post-high-inflation era misclassification (partially addressed in v4)

The 120m window fix and Gate 2 validator address the rolling z-score anchoring problem. The HMM's state geometry still places its Low Inflation mean below 2%, meaning PCE readings of 2.5-3% sit in an ambiguous zone between the two Gaussians. Gate 2 catches this when PCE is demonstrably above target; readings just above 2% (2.0-2.5%) remain in a grey zone.

### 2. COVID regime misclassification (structural, not fixable without lookahead)

February-April 2020 classified as Low Growth / High Inflation. COVID was a deflationary demand shock — Low Growth / Low Inflation was correct. CFNAI-MA3 partially mitigates (MA3 was already negative in Jan-Feb 2020) but the inflation HMM still saw above-mean PCE going into the crash. Gate 3 (VIX > 35) would fire in this episode in live mode, giving the validator a chance to override — but VIX is unavailable in backtest mode (skip_fred=True), so the COVID months remain misclassified in the historical backtest.

### 3. Independence assumption between growth and inflation HMMs

Joint P(regime) = P(growth) × P(inflation) assumes independence. They are empirically correlated across the business cycle. A joint 4-state HMM is more correct but has ~47 parameters from ~120 autocorrelated months — severely overparameterized. Independence factorization follows Kritzman, Page & Turkington (2012, FAJ).

### 4. Validator backtest has small-N limitation

22 override events, all in one contiguous 2023-2024 window. The Sharpe improvement (+0.07) is plausible but not statistically significant. The experiment is honest about this — the primary artifact is the decision audit log, not the performance delta.

### 5. No transaction costs

Monthly rebalancing with no slippage or execution costs modelled.

### 6. TLT duration mismatch

TLT tracks 20Y+ Treasuries. During 2022, TLT fell ~30% vs AGG ~13%. The bond sleeve performance is amplified by duration beyond what "30% bonds" implies.

### 7. Single equity sleeve

No sector differentiation within equities. The equity allocation is entirely SPY regardless of regime — the primary v5 opportunity.

---

## Stack

- **LangGraph** — stateful graph with MemorySaver checkpointing
- **hmmlearn** — GaussianHMM (2-state, diagonal covariance)
- **scipy** — SLSQP constrained optimization for BL portfolio construction
- **LangChain + OpenAI** — GPT-4o-mini for validator and narrative (live mode)
- **fredapi** — FRED macroeconomic data with ALFRED vintage support
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

Run the backtest experiment:

```bash
uv run python backtest_cli.py
```

---

## Roadmap

- **Sector rotation layer** — allocate the equity sleeve across sector ETFs (XLK, XLF, XLV, XLY, XLP, XLI, XLE, XLB, XLC, XLU, XLRE) using macro regime + LLM sentiment analysis, following the ICLR 2025 paper "Leveraging LLMs for Top-Down Sector Allocation in Automated Trading" (Quek et al., 2025)
- **Gate 3 in backtest** — add a point-in-time VIX fetch to `run_backtest()` so the stress gate fires in historical runs; currently VIX is only available in live mode via yfinance
- **3-state inflation HMM ablation** — test Low / Moderate / High inflation states to address the 2.0-2.5% grey zone; requires careful treatment of parameter count vs training window length
- **Probability blending** — use the full `prob_i_high` posterior as a blend weight between Low and High Inflation BL views rather than a binary classification, capturing the ambiguous zone more faithfully
- **LLM macro memory module** — store FOMC minutes summaries and regime history across runs; give the narrative agent longitudinal context
- **Quarterly rebalancing** — align rebalancing cadence with HMM refit cadence to reduce noise from monthly regime flips

---

## Related projects

- [**equity-lens**](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI pipeline for global equity screening via yFinance
- [**edgar-research-rag**](https://github.com/matsveikachatkou/edgar-research-rag) — RAG pipeline over SEC EDGAR filings with buy/hold/sell recommendations
- [**fund-lens**](https://github.com/matsveikachatkou/fund-lens) — automated fund due diligence brief generator: scrapes factsheet PDFs, enriches with live benchmark data, generates structured briefs via GPT-4o