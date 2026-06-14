"""
Hidden Markov Model engine for macro-lens v3.

Two independent 2-state GaussianHMMs:
  - Growth HMM:    CFNAI (level z-score) + UNRATE (diff z-score)
  - Inflation HMM: PCEPILFE (yoy z-score) + CPIAUCSL (yoy z-score)

Feature choices:
  CFNAI replaces INDPRO as the primary growth signal. Industrial production
  (INDPRO) covers only the goods sector (~11% of GDP) and produced spurious
  Low Growth calls during the 2015-2016 oil-driven industrial recession while
  equity markets were fine. CFNAI aggregates 85 monthly indicators across
  production, employment, consumption, and orders — a much broader proxy for
  the macro environment that drives equity returns. CFNAI is already
  mean-zero by construction (no transform needed beyond "level").

  CPIAUCSL replaces T10YIE as the secondary inflation signal. T10YIE
  (10-year breakeven) embeds an inflation risk premium and a TIPS liquidity
  premium, making it a forward-looking market expectation rather than a
  realized inflation measure. Mixing it with PCEPILFE (realized core PCE)
  in the same Gaussian HMM blurs the state definition. Both PCEPILFE and
  CPIAUCSL are realized series, keeping the inflation HMM anchored to what
  inflation actually was.

Design choices:
  - Rolling training window (default 120 months / 10 years), refit at
    calendar quarter-ends. A rolling window keeps the model trained on a
    "recent enough" macro era rather than ever-expanding history that
    permanently drags in eras like the GFC.
  - covariance_type="diag": with only 2 features and training windows as
    short as ~36 months early in a backtest, a full 2x2 covariance matrix
    per state (3 free params) is overparameterized relative to a diagonal
    matrix (2 free params) and risks unstable/near-singular fits. Diag is
    the safer default; "full" can be tested later as an explicit ablation.
  - Monthly inference between refits uses predict_proba on the FULL
    historical sequence (not a single observation), taking the posterior
    for the latest timestep. This preserves the HMM's transition-matrix
    "memory" (hysteresis) at inference time, not just at fit time —
    predict_proba on a length-1 sequence would discard that and reduce to
    a static per-observation classification.
  - Label anchoring on the primary feature only: CFNAI mean for growth,
    PCEPILFE mean for inflation. The secondary feature (UNRATE / CPIAUCSL)
    participates in EM fitting (shaping cluster geometry) but does not
    affect label assignment — preventing the secondary signal from
    overriding the primary realized measure in ambiguous periods.
  - Independence assumption: growth and inflation HMMs are fit separately;
    the joint regime probability is the product of the two marginals. This
    is a deliberate bias-variance tradeoff — a joint 4-state HMM over 4
    features would have ~47 parameters estimated from ~120 autocorrelated
    months with only a handful of regime transitions, making it severely
    overparameterized. The independence assumption is the same factorization
    used in Kritzman, Page & Turkington (2012, FAJ).
  - Point-in-time z-score cache keyed by (series_id, observation_date)
    turns the O(N^2) ALFRED calls of a rolling-window backtest into O(N).
"""

from typing import Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
from hmmlearn import hmm

from features import get_pit_zscore, ZSCORE_TRANSFORMS


GROWTH_FEATURES    = ["CFNAI", "UNRATE"]
INFLATION_FEATURES = ["PCEPILFE", "CPIAUCSL"]

# Minimum rows required to attempt a fit. Below this, _fit_and_anchor raises.
MIN_TRAIN_ROWS = 36


class RegimeHMMEngine:
    """
    Quarterly-refit, rolling-window HMM engine for joint growth/inflation
    regime probabilities.

    Usage:
        engine = RegimeHMMEngine(fred)
        result = engine.infer(observation_date)

    `infer` raises ValueError if there isn't enough history to fit (caller
    should catch this and fall back to baseline weights, mirroring the
    existing run_backtest fallback pattern).
    """

    def __init__(
        self,
        fred_client: Fred,
        history_window_months: int = 120,
        request_delay: float = 0.6,
        cache_path: Optional[str] = "zscore_cache.json",
        min_covar: float = 0.05,
    ):
        """
        Args:
            fred_client: Authenticated fredapi client.
            history_window_months: Trailing window length for HMM training
                (rolling, not expanding). 120 months (10 years) is long
                enough to span a full business cycle while staying
                regime-relevant. Kept fixed rather than expanded — expanding
                the window to dilute outliers would trade current-era
                relevance for curve-fitting; outliers are handled via
                z-score clipping (features.get_pit_zscore) and min_covar.
            request_delay: Seconds to sleep after each uncached ALFRED
                call, to stay under FRED's free-tier rate limit
                (~120 req/min). 0.6s -> ~100 req/min ceiling.
            cache_path: Path to a JSON file persisting the point-in-time
                z-score cache across runs. Loaded on init if it exists;
                pass None to disable persistence (in-memory only).
            min_covar: Variance floor passed to GaussianHMM. The hmmlearn
                default (1e-3) lets EM collapse one state's variance to
                near-zero around a tight cluster while the other state's
                variance absorbs all remaining observations — separating
                states by volatility rather than by the economic anchor
                means. A higher floor (0.05) forces both states to maintain
                realistic minimum variance, pushing EM separation toward
                the mean differences that the label anchoring relies on.
        """
        self.fred = fred_client
        self.history_window_months = history_window_months
        self.request_delay = request_delay
        self.min_covar = min_covar
        self.cache_path = Path(cache_path) if cache_path else None

        # Point-in-time z-score cache: (series_id, obs_date) -> float
        self.cache: Dict[Tuple[str, str], float] = self._load_cache()

        self.growth_model: Optional[hmm.GaussianHMM] = None
        self.inflation_model: Optional[hmm.GaussianHMM] = None
        self.g_high_idx: Optional[int] = None
        self.g_low_idx: Optional[int] = None
        self.i_high_idx: Optional[int] = None
        self.i_low_idx: Optional[int] = None
        self.last_refit_date: Optional[str] = None

        # Cache the full training sequences used at last refit, so monthly
        # inference between refits can append the latest observation and
        # run predict_proba over the full sequence (preserving transition
        # matrix context) without rebuilding the whole matrix each time.
        self._growth_train_df: Optional[pd.DataFrame] = None
        self._inflation_train_df: Optional[pd.DataFrame] = None

    # Cache persistence

    def _load_cache(self) -> Dict[Tuple[str, str], float]:
        """Load the z-score cache from disk, if cache_path is set and exists."""
        if self.cache_path is None or not self.cache_path.exists():
            return {}

        with open(self.cache_path, "r") as f:
            raw = json.load(f)

        # JSON keys are strings "series_id|date" — convert back to tuples
        return {tuple(k.split("|")): v for k, v in raw.items()}

    def _save_cache(self) -> None:
        """Persist the z-score cache to disk, if cache_path is set."""
        if self.cache_path is None:
            return

        raw = {f"{sid}|{date}": v for (sid, date), v in self.cache.items()}
        with open(self.cache_path, "w") as f:
            json.dump(raw, f, indent=2)

    # Feature construction

    def _generate_trailing_dates(self, end_date_str: str) -> List[str]:
        """Month-end dates for the rolling training window ending at end_date_str."""
        end_dt = pd.to_datetime(end_date_str)
        start_dt = end_dt - pd.DateOffset(months=self.history_window_months)
        dates = pd.date_range(start=start_dt, end=end_dt, freq="ME")
        return [d.strftime("%Y-%m-%d") for d in dates]

    def _get_feature_row(self, series_ids: List[str], obs_date: str) -> Dict[str, float]:
        """Point-in-time z-scores for a single date, via cache (persisted to disk)."""
        row = {}
        for sid in series_ids:
            key = (sid, obs_date)
            if key not in self.cache:
                config = ZSCORE_TRANSFORMS[sid]
                self.cache[key] = get_pit_zscore(
                    self.fred,
                    sid,
                    obs_date,
                    transform=config["transform"],
                    window_months=config["window_months"],
                )
                self._save_cache()
                time.sleep(self.request_delay)
            row[sid] = self.cache[key]
        return row

    def _build_training_matrix(self, series_ids: List[str], dates: List[str]) -> pd.DataFrame:
        """Feature matrix for HMM fitting, indexed by date, NaN rows dropped."""
        rows = []
        for d in dates:
            row_data = self._get_feature_row(series_ids, d)
            row_data["date"] = d
            rows.append(row_data)

        df = pd.DataFrame(rows).set_index("date")
        return df.dropna()

    # Fitting + anchoring

    def _fit_and_anchor(self, df: pd.DataFrame, model_type: str):
        """
        Fit a 2-state diag-covariance GaussianHMM and anchor state labels.

        Label anchoring uses only the primary ground-truth feature (index 0):
          Growth HMM:    CFNAI  — the broad realized activity composite
          Inflation HMM: PCEPILFE — the Fed's preferred realized gauge

        The secondary feature (UNRATE / CPIAUCSL) still participates in EM
        fitting, shaping cluster geometry, but does not affect label
        assignment. This prevents the secondary signal from overriding the
        primary realized measure in ambiguous periods — e.g. CPIAUCSL
        diverging slightly from PCEPILFE during supply-chain episodes would
        not flip the inflation label if PCEPILFE anchoring is used.

        The state with the higher primary-feature mean is labelled "High";
        the other is labelled "Low". hmmlearn's arbitrary state indices
        (0/1) are remapped to economically meaningful indices via this
        comparison.
        """
        X = df.values
        if len(X) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Insufficient data ({len(X)} rows) to fit {model_type} HMM "
                f"(need >= {MIN_TRAIN_ROWS})."
            )

        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="diag",
            min_covar=self.min_covar,
            n_iter=100,
            random_state=42,
        )
        model.fit(X)

        # Anchor on primary feature (index 0) mean only
        score_0 = model.means_[0][0]
        score_1 = model.means_[1][0]

        high_idx = 0 if score_0 > score_1 else 1
        low_idx  = 1 - high_idx

        return model, high_idx, low_idx

    # Inference

    def infer(self, observation_date: str) -> Dict:
        """
        Returns point-in-time joint regime probabilities as of
        observation_date.

        Refits both HMMs (rolling window) on calendar quarter-ends (or on
        the first call). On non-refit months, appends the latest observation
        to the cached training sequence and runs predict_proba over the full
        sequence, taking the last timestep's posterior — preserving
        transition-matrix context rather than scoring the observation in
        isolation.

        Joint probability is computed as the product of the two independent
        marginals: P(growth, inflation) = P(growth) × P(inflation).
        See module docstring for justification of the independence assumption.

        Raises:
            ValueError if there isn't enough history to fit (only possible
            on the first refit, for very early observation dates).
        """
        obs_dt = pd.to_datetime(observation_date)
        is_quarter_end = obs_dt.month in (3, 6, 9, 12)
        needs_refit = self.last_refit_date is None or is_quarter_end

        if needs_refit:
            training_dates = self._generate_trailing_dates(observation_date)

            growth_df = self._build_training_matrix(GROWTH_FEATURES, training_dates)
            self.growth_model, self.g_high_idx, self.g_low_idx = self._fit_and_anchor(
                growth_df, "growth"
            )
            self._growth_train_df = growth_df

            inflation_df = self._build_training_matrix(INFLATION_FEATURES, training_dates)
            self.inflation_model, self.i_high_idx, self.i_low_idx = self._fit_and_anchor(
                inflation_df, "inflation"
            )
            self._inflation_train_df = inflation_df

            self.last_refit_date = observation_date

        # Latest observation's point-in-time features
        latest_g_row = self._get_feature_row(GROWTH_FEATURES, observation_date)
        latest_i_row = self._get_feature_row(INFLATION_FEATURES, observation_date)

        if any(pd.isna(v) for v in latest_g_row.values()) or any(
            pd.isna(v) for v in latest_i_row.values()
        ):
            return {"error": "Missing point-in-time data for inference."}

        prob_g_high = self._posterior_for_latest(
            self._growth_train_df, GROWTH_FEATURES, latest_g_row,
            self.growth_model, self.g_high_idx, observation_date,
        )
        prob_i_high = self._posterior_for_latest(
            self._inflation_train_df, INFLATION_FEATURES, latest_i_row,
            self.inflation_model, self.i_high_idx, observation_date,
        )

        prob_high_high = prob_g_high * prob_i_high
        prob_high_low  = prob_g_high * (1 - prob_i_high)
        prob_low_high  = (1 - prob_g_high) * prob_i_high
        prob_low_low   = (1 - prob_g_high) * (1 - prob_i_high)

        regimes = {
            "High Growth / High Inflation": prob_high_high,
            "High Growth / Low Inflation":  prob_high_low,
            "Low Growth / High Inflation":  prob_low_high,
            "Low Growth / Low Inflation":   prob_low_low,
        }
        dominant_regime = max(regimes, key=regimes.get)

        return {
            "growth_high_prob":   round(float(prob_g_high), 4),
            "inflation_high_prob": round(float(prob_i_high), 4),
            "joint_probabilities": {k: round(float(v), 4) for k, v in regimes.items()},
            "dominant_regime":    dominant_regime,
            "confidence":         round(float(regimes[dominant_regime]), 4),
        }

    def _posterior_for_latest(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        latest_row: Dict[str, float],
        model: hmm.GaussianHMM,
        high_idx: int,
        observation_date: str,
        posterior_floor: float = 0.03,
    ) -> float:
        """
        Append the latest observation to the cached training sequence and
        return P(state=high) at the final timestep via predict_proba over
        the FULL sequence (forward-backward), preserving the transition
        matrix's hysteresis effect.

        If observation_date is already the last row of train_df (the normal
        case on a quarter-end refit), train_df is used as-is. Otherwise the
        latest row is appended before running predict_proba.

        Posteriors are floored at `posterior_floor` and renormalized to
        prevent numerical saturation at 0.0/1.0. A -3.7 sigma distance from
        a state produces likelihood ratios that round to exactly 0.0 in
        float64, giving degenerate confidence inputs to Black-Litterman.
        Flooring at 0.03 preserves the strong directional signal while
        acknowledging irreducible model uncertainty.
        """
        if observation_date in train_df.index:
            X = train_df[feature_cols].values
        else:
            latest_series = pd.Series(latest_row, name=observation_date)
            extended = pd.concat([train_df[feature_cols], latest_series.to_frame().T])
            X = extended.values.astype(float)

        posteriors = model.predict_proba(X)

        # Floor and renormalise to prevent saturation at 0/1
        p = posteriors[-1].copy()
        p = np.clip(p, posterior_floor, 1.0 - posterior_floor)
        p = p / p.sum()

        return float(p[high_idx])


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    engine = RegimeHMMEngine(fred)

    # Sanity check across dates spanning the COVID shock.
    # Expected: growth should be High pre-COVID, drop during crash,
    # inflation should be mildly High (PCE was running above trend in early 2020).
    for date in ["2019-12-31", "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]:
        try:
            result = engine.infer(date)
        except ValueError as e:
            print(f"{date}: {e}")
            continue

        if "error" in result:
            print(f"{date}: {result['error']}")
            continue

        print(
            f"{date}  "
            f"P(G_high)={result['growth_high_prob']:.3f}  "
            f"P(I_high)={result['inflation_high_prob']:.3f}  "
            f"dominant={result['dominant_regime']}  "
            f"conf={result['confidence']:.3f}"
        )