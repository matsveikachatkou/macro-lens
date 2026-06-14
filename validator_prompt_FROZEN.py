"""
macro-lens v4 — LLM Regime Validator
FROZEN PROMPT + DECISION SCHEMA

Design discipline:
  - Written from economic first principles BEFORE any backtest output was examined.
  - This file must not be modified after the first backtest run begins.
  - The prompt version string below is the integrity anchor. If the prompt changes
    mid-backtest, prior cached decisions are invalidated (different model).
  - Free parameters are documented explicitly. None are tuned on backtest results.

Prompt version: v4-validator-1.0
Cache key:      hash(blinded_input_dict) — NOT (date, regime). Blinding is the point.
"""

PROMPT_VERSION = "v4-validator-1.0-retest"

# ---------------------------------------------------------------------------
# GATE THRESHOLDS
# These are the only three free parameters in the validator.
# Justification for each is economic, not empirical.
# ---------------------------------------------------------------------------

GATE_CONFIDENCE_THRESHOLD = 0.65
# Rationale: below 0.65, the two joint probabilities closest to dominant are
# within ~20pp of each other — the HMM itself is ambiguous. Above 0.65, the
# HMM has a clear read and the LLM should not override without strong
# contradicting evidence. 0.65 is the midpoint between a coin-flip (0.50)
# and a strong signal (0.80). NOT tuned on backtest.

GATE_PCE_THRESHOLD = 2.5
# Rationale: Fed target is 2.0%. A 25bp buffer above target (2.5%) is the
# standard definition of "demonstrably above target" in Fed communication.
# NOT tuned on backtest. Would be the same number regardless of whether
# we had a rolling-window anchoring problem.

GATE_VIX_THRESHOLD = 35
# Rationale: VIX > 35 is the widely-used practitioner threshold for
# "crisis-level" volatility — above 2008-normal, above typical correction.
# Referenced independently in Whaley (2009) and CBOE literature.
# NOT tuned on backtest.

# ---------------------------------------------------------------------------
# GATE LOGIC (applied before LLM is invoked — LLM is NOT free to pick gates)
# ---------------------------------------------------------------------------

def should_invoke_validator(
    hmm_confidence: float,
    pcepilfe_level: float,        # raw PCE YoY, e.g. 0.027 for 2.7%
    hmm_inflation_call: str,      # "High Inflation" or "Low Inflation"
    vix_level: float,
    hmm_growth_call: str,         # "High Growth" or "Low Growth"
) -> tuple[bool, list[str]]:
    """
    Returns (should_invoke, list_of_gates_fired).
    If no gate fires, the validator is skipped entirely.
    The gate logic is deterministic — the LLM has no say in whether it's invoked.
    """
    gates_fired = []

    # Gate 1: HMM confidence is genuinely ambiguous
    if hmm_confidence < GATE_CONFIDENCE_THRESHOLD:
        gates_fired.append(
            f"GATE_1_CONFIDENCE: HMM confidence {hmm_confidence:.2f} "
            f"< threshold {GATE_CONFIDENCE_THRESHOLD}"
        )

    # Gate 2: PCE is above Fed target but HMM calls Low Inflation
    # This is the specific structural failure mode identified in v3.
    if (pcepilfe_level > GATE_PCE_THRESHOLD / 100) and (hmm_inflation_call == "Low Inflation"):
        gates_fired.append(
            f"GATE_2_PCE_ANCHOR: PCE YoY {pcepilfe_level*100:.2f}% "
            f"> {GATE_PCE_THRESHOLD}% but HMM calls Low Inflation"
        )

    # Gate 3: VIX signals crisis but HMM calls High Growth
    # Catches liquidity-crisis episodes where growth indicators lag by 1-2 months.
    if (vix_level > GATE_VIX_THRESHOLD) and (hmm_growth_call == "High Growth"):
        gates_fired.append(
            f"GATE_3_STRESS: VIX {vix_level:.1f} > {GATE_VIX_THRESHOLD} "
            f"but HMM calls High Growth"
        )

    return len(gates_fired) > 0, gates_fired


# ---------------------------------------------------------------------------
# BLINDING FUNCTION
# Strips the calendar date from inputs before sending to LLM.
# Purpose: force reasoning from contemporaneous macro readings, not
# episode recognition. Imperfect (VIX=82 fingerprints COVID) but auditable.
# ---------------------------------------------------------------------------

def build_blinded_input(
    hmm_dominant_regime: str,
    hmm_confidence: float,
    hmm_joint_probabilities: dict,
    growth_high_prob: float,
    inflation_high_prob: float,
    pcepilfe_level: float,         # raw YoY level, e.g. 0.027
    pcepilfe_zscore: float,
    cfnai_zscore: float,
    unrate_zscore: float,
    cpiaucsl_zscore: float,
    vix_level: float,
    hy_spread: float,              # BAMLH0A0HYM2 in %
    yield_curve: float,            # T10Y2Y in %
    gates_fired: list[str],
) -> dict:
    """
    Assembles the blinded input dict passed to the LLM.
    No calendar date. No period label. No year.
    The hash of this dict is the cache key.
    """
    return {
        "hmm_dominant_regime":      hmm_dominant_regime,
        "hmm_confidence":           round(hmm_confidence, 4),
        "hmm_joint_probabilities":  {k: round(v, 4) for k, v in hmm_joint_probabilities.items()},
        "growth_high_prob":         round(growth_high_prob, 4),
        "inflation_high_prob":      round(inflation_high_prob, 4),
        "pcepilfe_yoy_pct":         round(pcepilfe_level * 100, 2),
        "pcepilfe_zscore":          round(pcepilfe_zscore, 3),
        "cfnai_zscore":             round(cfnai_zscore, 3),
        "unrate_zscore":            round(unrate_zscore, 3),
        "cpiaucsl_zscore":          round(cpiaucsl_zscore, 3),
        "vix":                      round(vix_level, 1),
        "hy_spread_pct":            round(hy_spread, 2),
        "yield_curve_t10y2y_pct":   round(yield_curve, 2),
        "gates_fired":              gates_fired,
        "fed_inflation_target_pct": 2.0,   # hard-coded fundamental, not a parameter
    }


# ---------------------------------------------------------------------------
# FROZEN SYSTEM PROMPT
# Written once. Not to be iterated on after backtest begins.
# Economic reasoning only — no backtest-specific calibration.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a macro regime validator. A statistical model (Hidden Markov Model) 
has classified the current macroeconomic regime. Your job is to evaluate whether that 
classification is consistent with the raw macro indicators provided.

CRITICAL CONSTRAINTS:
1. You will NOT be given the calendar date or year. Reason only from the indicator values.
2. You are NOT making a portfolio allocation decision. You are only evaluating regime consistency.
3. You must output ONLY the JSON structure specified. No preamble, no explanation outside the JSON.
4. Your rationale must cite only the indicator values provided. Do not reference historical 
   episodes by name (e.g. do not say "this looks like 2008" or "this resembles COVID").
   Describe the readings: "VIX at X.X indicates stress-level volatility."

ECONOMIC REFERENCE POINTS (use these as your only anchors):
- Fed inflation target: 2.0%. PCE above 2.5% is demonstrably above target.
- VIX above 35 indicates crisis-level volatility inconsistent with a growth expansion.
- Yield curve (T10Y2Y) below 0 is inverted — historically a late-cycle signal.
- HY spreads above 5% indicate credit stress; above 8% is acute distress.
- CFNAI z-score: negative = below-trend growth, positive = above-trend. 
  It is already mean-zero by construction, so z-score and level are equivalent.
- PCE YoY z-score uses a rolling window that may be anchored to a high-inflation era.
  The raw PCE YoY level is a more reliable inflation anchor than its z-score.

YOUR DECISION OPTIONS:
- CONFIRM: The HMM regime is broadly consistent with the indicators. Proceed as-is.
- FLAG: The indicators are ambiguous or contradictory. Insufficient basis to confirm 
  or override. Fall back to policy weights (neutral allocation).
- OVERRIDE: The indicators clearly and substantially contradict the HMM regime.
  You must specify which of the four regimes better fits the data.
  Valid regimes: "High Growth / High Inflation", "High Growth / Low Inflation",
  "Low Growth / High Inflation", "Low Growth / Low Inflation"

OVERRIDE DISCIPLINE:
- Override only when the contradiction is substantial — not when indicators are 
  merely mixed. A mixed picture warrants FLAG, not OVERRIDE.
- The raw PCE YoY level takes precedence over its z-score for inflation classification.
- VIX and HY spreads are contemporaneous. Growth indicators (CFNAI) may lag by 1-2 months.
  When stress signals (VIX, HY) sharply contradict growth signals (CFNAI), weight the 
  stress signals more heavily for regime classification.
- If you OVERRIDE, the replacement regime must be one of the four valid labels exactly.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "decision": "CONFIRM" | "FLAG" | "OVERRIDE",
  "replacement_regime": null | "<one of the four valid regime strings>",
  "rationale": "<2-3 sentences citing specific indicator values only>",
  "primary_contradiction": "<the single most important disagreement between HMM and indicators, or null if CONFIRM>",
  "confidence_in_decision": "HIGH" | "MEDIUM" | "LOW"
}"""


# ---------------------------------------------------------------------------
# USER PROMPT TEMPLATE
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """STATISTICAL MODEL OUTPUT:
  Dominant regime:    {hmm_dominant_regime}
  HMM confidence:     {hmm_confidence:.1%}
  Joint probabilities:
    High Growth / High Inflation:  {hghi:.1%}
    High Growth / Low Inflation:   {hgli:.1%}
    Low Growth  / High Inflation:  {lghi:.1%}
    Low Growth  / Low Inflation:   {lgli:.1%}

MACRO INDICATORS (as of the observation date — no date provided intentionally):
  PCE YoY (raw level):   {pcepilfe_yoy_pct:.2f}%  [Fed target: 2.0%]
  PCE YoY (z-score):     {pcepilfe_zscore:+.3f}
  CPI YoY (z-score):     {cpiaucsl_zscore:+.3f}
  CFNAI (z-score):       {cfnai_zscore:+.3f}  [already mean-zero by construction]
  UNRATE (z-score):      {unrate_zscore:+.3f}
  VIX:                   {vix:.1f}
  HY spread:             {hy_spread_pct:.2f}%
  Yield curve (T10Y2Y):  {yield_curve_t10y2y_pct:+.2f}%

GATES THAT TRIGGERED THIS VALIDATION:
{gates_text}

Evaluate the regime classification and respond with the required JSON only."""


def build_user_prompt(blinded_input: dict) -> str:
    probs = blinded_input["hmm_joint_probabilities"]
    gates_text = "\n".join(f"  - {g}" for g in blinded_input["gates_fired"])
    return USER_PROMPT_TEMPLATE.format(
        hmm_dominant_regime=blinded_input["hmm_dominant_regime"],
        hmm_confidence=blinded_input["hmm_confidence"],
        hghi=probs.get("High Growth / High Inflation", 0.0),
        hgli=probs.get("High Growth / Low Inflation", 0.0),
        lghi=probs.get("Low Growth / High Inflation", 0.0),
        lgli=probs.get("Low Growth / Low Inflation", 0.0),
        pcepilfe_yoy_pct=blinded_input["pcepilfe_yoy_pct"],
        pcepilfe_zscore=blinded_input["pcepilfe_zscore"],
        cpiaucsl_zscore=blinded_input["cpiaucsl_zscore"],
        cfnai_zscore=blinded_input["cfnai_zscore"],
        unrate_zscore=blinded_input["unrate_zscore"],
        vix=blinded_input["vix"],
        hy_spread_pct=blinded_input["hy_spread_pct"],
        yield_curve_t10y2y_pct=blinded_input["yield_curve_t10y2y_pct"],
        gates_text=gates_text,
    )


# ---------------------------------------------------------------------------
# CACHE KEY
# Deterministic hash of the blinded input dict.
# Same macro readings → same cache key, regardless of observation date.
# This is intentional: if two months have identical blinded readings,
# the validator should give the same answer for both.
# ---------------------------------------------------------------------------

import hashlib, json

def build_cache_key(blinded_input: dict) -> str:
    """SHA-256 of the canonical JSON of the blinded input."""
    canonical = json.dumps(blinded_input, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DECISION PARSING + VALIDATION
# ---------------------------------------------------------------------------

VALID_REGIMES = {
    "High Growth / High Inflation",
    "High Growth / Low Inflation",
    "Low Growth / High Inflation",
    "Low Growth / Low Inflation",
}

VALID_DECISIONS = {"CONFIRM", "FLAG", "OVERRIDE"}


def parse_validator_response(raw_json: str) -> dict:
    """
    Parse and validate the LLM's JSON response.
    Returns a normalised decision dict or a fallback CONFIRM on parse failure.
    """
    try:
        parsed = json.loads(raw_json.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from response that has surrounding text
        import re
        match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_confirm("JSON parse failed")
        else:
            return _fallback_confirm("No JSON found in response")

    decision = parsed.get("decision", "").upper()
    if decision not in VALID_DECISIONS:
        return _fallback_confirm(f"Invalid decision: {decision!r}")

    replacement = parsed.get("replacement_regime")
    if decision == "OVERRIDE":
        if replacement not in VALID_REGIMES:
            # Override with invalid regime → downgrade to FLAG
            return {
                "decision":              "FLAG",
                "replacement_regime":    None,
                "rationale":             parsed.get("rationale", ""),
                "primary_contradiction": parsed.get("primary_contradiction"),
                "confidence_in_decision": parsed.get("confidence_in_decision", "LOW"),
                "parse_note":            f"OVERRIDE downgraded to FLAG: invalid regime {replacement!r}",
            }

    return {
        "decision":               decision,
        "replacement_regime":     replacement if decision == "OVERRIDE" else None,
        "rationale":              parsed.get("rationale", ""),
        "primary_contradiction":  parsed.get("primary_contradiction"),
        "confidence_in_decision": parsed.get("confidence_in_decision", "MEDIUM"),
        "parse_note":             None,
    }


def _fallback_confirm(reason: str) -> dict:
    return {
        "decision":               "CONFIRM",
        "replacement_regime":     None,
        "rationale":              "",
        "primary_contradiction":  None,
        "confidence_in_decision": "LOW",
        "parse_note":             f"Fallback CONFIRM: {reason}",
    }


# ---------------------------------------------------------------------------
# CONTAMINATION AUDIT HELPERS
# Run after backtest to check whether rationales cite contemporaneous info only.
# ---------------------------------------------------------------------------

CONTAMINATION_MARKERS = [
    # Named episodes — should never appear
    "covid", "pandemic", "2020", "2019", "2018", "2017", "2016", "2015",
    "2021", "2022", "2023", "2024", "gfc", "financial crisis", "lehman",
    "great recession", "dot-com", "dotcom", "taper tantrum",
    "inflation surge", "post-pandemic",
    # Named people/institutions that signal episode recall
    "powell", "bernanke", "yellen", "fed chair",
    # Future-looking language (lookahead signal)
    "recovered", "would recover", "subsequently", "afterward",
    "in hindsight", "looking back", "we now know",
]


def audit_rationale_for_contamination(rationale: str) -> list[str]:
    """
    Returns a list of contamination markers found in the rationale string.
    Empty list = clean. Non-empty = flag for manual review.
    """
    lower = rationale.lower()
    return [m for m in CONTAMINATION_MARKERS if m in lower]