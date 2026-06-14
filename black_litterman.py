"""
Black-Litterman portfolio construction for macro-lens v3.

Translates HMM regime probabilities into posterior portfolio weights via
the Black-Litterman model:

  1. Equilibrium prior: implied returns from policy weights + Σ
  2. Views (P, Q): two relative views per regime (Bridgewater heuristic)
  3. View uncertainty (Ω): scaled by HMM confidence — higher confidence
     → lower Ω → views pull harder on the posterior
  4. Posterior: BL formula blends prior and views into updated expected
     returns, fed into a CONSTRAINED mean-variance optimizer with
     policy-relative box constraints
  5. Final blend: w_final = conf × w_bl + (1-conf) × policy weights

Design parameters:
  DELTA    = 2.5  — risk aversion (standard default)
  TAU      = 0.05 — prior uncertainty scalar (industry standard)
  TAA_BAND = 0.15 — default maximum deviation from policy weight per asset
                    (±15% is the standard institutional TAA mandate band);
                    equities use ±20% given the equity-benchmarked mandate

Prior note:
  MARKET_WEIGHTS is a policy/risk-balanced prior (40/30/10/8/7/5), NOT a
  true market-cap equilibrium. BL implied returns are therefore policy-
  anchored rather than market-clearing. This is intentional — the policy
  mix is the TAA strategy's neutral allocation and the correct anchor for
  tactical tilts. True market-cap weights would be ~55% equities with
  negligible explicit commodity/gold allocations.

Key design choice — constrained MVO replaces unconstrained:
  The canonical BL step w = (δΣ)⁻¹μ_BL is the "error maximizer" —
  it amplifies small return differences into extreme corner allocations
  (80-92% single-asset). Post-hoc clipping is self-defeating: when all
  other assets have near-zero weights, clip(0.92 → 0.55) then renormalize
  gives 0.84, not 0.55. The fix is scipy SLSQP with policy-relative bounds
  enforced DURING optimization, producing moderate multi-asset tilts. The
  linear confidence blend then provides smooth conviction scaling on top.

View table (Bridgewater heuristic, Path 1 — external prior):
  High Growth / High Inflation:
    Equities over Bonds       +2%  (growth premium, duration pain)
    Commodities over Bonds    +2%  (inflation hedge + growth demand)
  High Growth / Low Inflation:
    Equities over Bonds       +3%  (Goldilocks, max equity conviction)
    Equities over Commodities +2%  (no inflation hedge needed)
  Low Growth / High Inflation:
    Commodities over Equities +2.5% (stagflation, margin compression)
    Gold over Bonds           +2%   (real rates falling)
  Low Growth / Low Inflation:
    Bonds over Equities       +2%  (recession, duration is king)
    Gold over Commodities     +2%  (deflation collapses commodity demand)

Two views per regime (vs one in v3 original) ensure the inflation dimension
differentiates the allocation — HG/HI and HG/LI produce meaningfully
different weights (different commodity exposure) rather than identical
portfolios where only the equity/bond split changes.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Asset ordering — must be consistent across all arrays in this module
ASSETS = ["equities", "bonds", "inflation_linked", "commodities", "gold", "cash"]
N = len(ASSETS)

# BL parameters
DELTA = 2.5    # risk aversion coefficient (standard multi-asset default)
TAU   = 0.05   # prior uncertainty scalar (industry standard fixed value)

# Default TAA band; equities use a wider band given the equity-benchmarked mandate
TAA_BAND = 0.15

# Policy weights — the TAA strategy's neutral allocation.
# Serves three roles:
#   (1) BL prior anchor: reverse-optimized to produce equilibrium implied returns
#   (2) Optimizer bound center: WEIGHT_BOUNDS are ±TAA_BAND around each weight
#   (3) Blend target: w_final approaches policy weights as confidence → 0
MARKET_WEIGHTS = np.array([
    0.40,   # equities
    0.30,   # bonds
    0.10,   # inflation_linked
    0.08,   # commodities
    0.07,   # gold
    0.05,   # cash
])

# Per-asset weight bounds enforced during SLSQP optimization.
# Equities use ±20% (wider band for equity-benchmarked TAA strategy).
# All others use ±15% standard institutional TAA band.
# Cash is capped at 8% — treated as a residual liquidity buffer, not
# a tactical position, consistent with institutional multi-asset mandates.
WEIGHT_BOUNDS = [
    (0.20, 0.60),   # equities:         40% ± 20%
    (0.15, 0.45),   # bonds:            30% ± 15%
    (0.00, 0.25),   # inflation_linked: 10% ± 15%, floored at 0
    (0.00, 0.23),   # commodities:       8% ± 15%, floored at 0
    (0.00, 0.22),   # gold:              7% ± 15%, floored at 0
    (0.00, 0.08),   # cash:              5% + 3% max — residual only
]


def _build_views(regime: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (P, Q) for the current dominant regime.

    Each regime expresses two relative views to ensure both the growth and
    inflation dimensions influence the final allocation. With a single view
    (e.g. "equities over bonds"), the four unconstrained assets drift to
    whatever the covariance structure implies — causing HG/HI and HG/LI to
    produce identical weights despite different inflation signals. The second
    view anchors a second asset pair and differentiates the regimes.

    P: (2, N) pick matrix  — +1 long side, -1 short side per view
    Q: (2,)  annualised excess return views

    Raises ValueError for unrecognised regime strings.
    """
    idx = {asset: i for i, asset in enumerate(ASSETS)}

    def relative_view(long_asset: str, short_asset: str) -> np.ndarray:
        row = np.zeros(N)
        row[idx[long_asset]]  = +1.0
        row[idx[short_asset]] = -1.0
        return row

    if regime == "High Growth / High Inflation":
        # Growth supports equities vs bonds (duration pain from rising inflation).
        # Commodities over bonds as second view: both growth demand and inflation hedge.
        P = np.array([
            relative_view("equities",    "bonds"),
            relative_view("commodities", "bonds"),
        ])
        Q = np.array([0.02, 0.02])

    elif regime == "High Growth / Low Inflation":
        # Goldilocks: maximum equity conviction.
        # Equities over commodities: no inflation hedge needed in low-inflation growth.
        # This second view differentiates HG/LI from HG/HI by explicitly suppressing
        # the commodity overweight that the covariance structure would otherwise produce.
        P = np.array([
            relative_view("equities", "bonds"),
            relative_view("equities", "commodities"),
        ])
        Q = np.array([0.03, 0.02])

    elif regime == "Low Growth / High Inflation":
        # Stagflation: commodities primary inflation hedge; equities suffer from
        # margin compression and rising real rates.
        # Gold over bonds: real rates falling as Fed lags behind inflation,
        # gold benefits from negative real yield environment.
        P = np.array([
            relative_view("commodities", "equities"),
            relative_view("gold",        "bonds"),
        ])
        Q = np.array([0.025, 0.02])

    elif regime == "Low Growth / Low Inflation":
        # Recession/deflation: duration is king, equities de-rate on falling earnings.
        # Gold over commodities: deflation collapses commodity demand while gold
        # retains safe-haven bid.
        P = np.array([
            relative_view("bonds", "equities"),
            relative_view("gold",  "commodities"),
        ])
        Q = np.array([0.02, 0.02])

    else:
        raise ValueError(f"Unrecognised regime: {regime!r}")

    return P, Q


def _implied_returns(sigma: np.ndarray) -> np.ndarray:
    """
    Policy-anchored implied returns: Π = δ · Σ · w_policy

    Reverse-optimizes the policy weights into the expected returns that
    would make them the mean-variance optimal portfolio at risk aversion δ.
    These serve as the BL prior — the starting point before views are applied.
    """
    return DELTA * sigma @ MARKET_WEIGHTS


def _bl_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.

    μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

    Bayesian update of the prior implied returns Π with the investor views
    (P, Q) weighted by view uncertainty Ω. Higher confidence → smaller Ω
    → views dominate the posterior.
    """
    tau_sigma_inv = np.linalg.inv(TAU * sigma)
    omega_inv     = np.linalg.inv(omega)
    M             = tau_sigma_inv + P.T @ omega_inv @ P
    mu_bl         = np.linalg.inv(M) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    return mu_bl


def _posterior_weights(
    mu_bl: np.ndarray,
    sigma: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """
    Translate BL posterior returns into portfolio weights via constrained
    mean-variance optimization, then blend with policy weights by confidence.

    Two-stage construction:

    Stage 1 — Constrained MVO:
      argmax_w { w·μ_BL - 0.5δ·w·Σ·w }
      subject to: sum(w) = 1, WEIGHT_BOUNDS per asset

      SLSQP enforces bounds during optimization (not post-hoc). This avoids
      the self-defeating clip+renormalize pattern: if the unconstrained
      solution puts 92% in equities, clipping to 55% and renormalizing gives
      84% — worse than before. Enforcing the bound during SLSQP finds the
      true constrained optimum directly.

    Stage 2 — Confidence blend:
      w_final = conf × w_bl + (1-conf) × policy_weights

      Smooth monotonic confidence-to-conviction relationship:
        conf = 0.30 → 30% tactical + 70% policy → mild tilt
        conf = 0.65 → 65% tactical + 35% policy → moderate tilt
        conf = 0.95 → 95% tactical + 5% policy  → strong conviction

      Falls back to policy weights at confidence = 0 (HMM has no view).
    """
    def neg_utility(w: np.ndarray) -> float:
        return -(w @ mu_bl - 0.5 * DELTA * w @ sigma @ w)

    result = minimize(
        neg_utility,
        MARKET_WEIGHTS.copy(),          # warm-start from policy weights
        method="SLSQP",
        bounds=WEIGHT_BOUNDS,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if result.success:
        w_bl = result.x
    else:
        # Fallback: policy weights if optimizer fails (should be rare)
        w_bl = MARKET_WEIGHTS.copy()

    # Safety clip + renorm (optimizer should already satisfy bounds)
    w_bl = np.clip(w_bl, 0.0, None)
    w_bl = w_bl / w_bl.sum()

    # Blend with policy weights via confidence
    w_final = confidence * w_bl + (1.0 - confidence) * MARKET_WEIGHTS
    return w_final / w_final.sum()


def generate_target_weights(
    returns_df: pd.DataFrame,
    dominant_regime: str,
    confidence: float,
) -> Dict[str, float]:
    """
    Main entry point for the BL portfolio construction layer.

    Args:
        returns_df: Monthly returns DataFrame with columns matching ASSETS,
                    at least 60 rows. Missing columns are filled with 0
                    (e.g. cash proxy with near-zero returns).
        dominant_regime: One of the four Bridgewater quadrant labels from
                         RegimeHMMEngine.infer().
        confidence: Joint regime probability from HMM (0.0 to 1.0).
                    Clipped to [0.05, 0.95] internally to prevent Ω
                    singularity at the extremes.

    Returns:
        Dict mapping asset name → portfolio weight (sums to 1.0, long-only).
    """
    # --- 1. Covariance matrix from trailing returns ---
    aligned = returns_df.reindex(columns=ASSETS).fillna(0.0)
    sigma   = aligned.cov().values.copy()
    sigma  += np.eye(N) * 1e-6      # regularise for positive definiteness

    # --- 2. Policy-anchored implied returns ---
    pi = _implied_returns(sigma)

    # --- 3. Views ---
    P, Q = _build_views(dominant_regime)

    # --- 4. View uncertainty Ω + confidence-scaled Q ---
    bounded_conf = float(np.clip(confidence, 0.05, 0.95))

    # Ω scales view uncertainty inversely with confidence:
    #   high confidence → small omega_scalar → views dominate posterior
    #   low confidence  → large omega_scalar → prior dominates posterior
    omega_scalar = (1.0 - bounded_conf) / bounded_conf
    omega        = omega_scalar * TAU * (P @ sigma @ P.T)
    omega       += np.eye(len(Q)) * 1e-8   # numerical stability

    # Confidence-scaled view magnitude — deliberate deviation from canonical BL.
    # Canonical BL uses fixed Q with Ω handling all uncertainty. However, when
    # the prior implied return for an asset is near-zero (e.g. commodities:
    # Π ≈ 0.0003), even a modest Q overwhelms the prior at any confidence,
    # making Ω irrelevant and cornering the optimizer. Scaling Q by confidence
    # ensures the view magnitude itself shrinks toward zero at low confidence,
    # preventing cornering when the prior signal is weak. This is an Idzorek-
    # family idea (documented deviation, not a bug).
    effective_Q = Q * bounded_conf

    # --- 5. BL posterior expected returns ---
    mu_bl = _bl_posterior(pi, sigma, P, effective_Q, omega)

    # --- 6. Constrained weights + confidence blend ---
    w = _posterior_weights(mu_bl, sigma, bounded_conf)

    return {asset: round(float(w[i]), 6) for i, asset in enumerate(ASSETS)}


def weights_to_display(weights: Dict[str, float]) -> pd.DataFrame:
    """Format weights dict as a display DataFrame (for Gradio / reporter)."""
    return pd.DataFrame([
        {"Asset Class": k.replace("_", " ").title(), "Weight": f"{v:.1%}"}
        for k, v in weights.items()
    ])


if __name__ == "__main__":
    import yfinance as yf
    from macro_lens import ASSET_PROXIES

    tickers = list(ASSET_PROXIES.values())
    prices  = yf.download(tickers, period="5y", interval="1mo",
                          auto_adjust=True, progress=False)["Close"]
    returns = prices.pct_change().dropna()
    returns = returns.rename(columns={v: k for k, v in ASSET_PROXIES.items()})

    print("Testing all four regimes at confidence=0.80:\n")
    for regime in [
        "High Growth / High Inflation",
        "High Growth / Low Inflation",
        "Low Growth / High Inflation",
        "Low Growth / Low Inflation",
    ]:
        weights = generate_target_weights(returns, regime, confidence=0.80)
        print(f"{regime}")
        for asset, w in weights.items():
            bar = "█" * int(w * 40)
            print(f"  {asset:<22} {w:>6.1%}  {bar}")
        print()

    print("Confidence sensitivity — Low Growth / High Inflation:\n")
    for conf in [0.30, 0.50, 0.65, 0.80, 0.95]:
        weights = generate_target_weights(
            returns, "Low Growth / High Inflation", confidence=conf
        )
        print(f"  conf={conf:.2f}  eq={weights['equities']:.1%}  "
              f"bonds={weights['bonds']:.1%}  "
              f"comm={weights['commodities']:.1%}  "
              f"gold={weights['gold']:.1%}")