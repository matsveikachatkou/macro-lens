"""
Hidden Markov Model engine for macro-lens v4.

Two independent 2-state GaussianHMMs:
  - Growth HMM:    CFNAI or CFNAI_MA3 (level z-score) + UNRATE (diff z-score)
  - Inflation HMM: PCEPILFE (yoy z-score) + CPIAUCSL (yoy z-score)

Feature choices: unchanged from v3. See v3 docstring for full justification.

v4 changes vs v3:
  1. use_cfnai_ma3 flag on RegimeHMMEngine.__init__():
       When True, GROWTH_FEATURES uses "CFNAI_MA3" instead of "CFNAI".
       The 3-month rolling mean is computed in features.get_pit_zscore()
       before z-scoring — no changes needed here beyond swapping the key.
       Default False so v3 behaviour is preserved unless explicitly opted in.

  2. infer() returns raw indicator z-scores in an "indicators" sub-dict:
       The values are already in self.cache (computed during _get_feature_row)
       so no additional ALFRED calls are made. The validator node in
       macro_lens.py reads these to build the blinded input without
       needing its own fetch path.

  3. get_pcepilfe_level() is imported from features and called inside infer()
       to populate "indicators.pcepilfe_level" — the raw PCE YoY decimal
       used by Gate 2 in the validator. It uses the same ALFRED vintage
       fetch as the z-score path, so it is point-in-time safe.
       Result is cached in self._level_cache (separate from zscore cache
       to keep cache key semantics clean).
"""

from typing import Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
from hmmlearn import hmm

from features import get_pit_zscore, get_pcepilfe_level, ZSCORE_TRANSFORMS


# Default growth features — overridden by use_cfnai_ma3=True at init time
_DEFAULT_GROWTH_FEATURES    = ["CFNAI", "UNRATE"]
INFLATION_FEATURES          = ["PCEPILFE", "CPIAUCSL"]

# Minimum rows required to attempt a fit. Below this, _fit_and_anchor raises.
MIN_TRAIN_ROWS = 36


class RegimeHMMEngine:
    """
    Quarterly-refit, rolling-window HMM engine for joint growth/inflation
    regime probabilities.

    Usage:
        engine = RegimeHMMEngine(fred)                        # v3-compatible
        engine = RegimeHMMEngine(fred, use_cfnai_ma3=True)   # v4 smoothed growth

        result = engine.infer(observation_date)

    `infer` raises ValueError if there isn't enough history to fit (caller
    should catch this and fall back to baseline weights).

    The result dict now includes an "indicators" sub-dict with raw z-scores
    and the PCE level, consumed by the LLM validator node.
    """

    def __init__(
        self,
        fred_client: Fred,
        history_window_months: int = 120,
        request_delay: float = 0.6,
        cache_path: Optional[str] = "zscore_cache.json",
        min_covar: float = 0.05,
        use_cfnai_ma3: bool = False,
    ):
        """
        Args:
            fred_client: Authenticated fredapi client.
            history_window_months: Trailing window length for HMM training
                (rolling, not expanding). 120 months (10 years) is long
                enough to span a full business cycle while staying
                regime-relevant.
            request_delay: Seconds to sleep after each uncached ALFRED
                call, to stay under FRED's free-tier rate limit.
            cache_path: Path to a JSON file persisting the point-in-time
                z-score cache across runs. Loaded on init if it exists;
                pass None to disable persistence (in-memory only).
            min_covar: Variance floor passed to GaussianHMM. Prevents EM
                from collapsing one state's variance to near-zero, which
                separates states by volatility rather than by the economic
                anchor means.
            use_cfnai_ma3: If True, use the 3-month moving average of CFNAI
                as the primary growth feature instead of raw CFNAI. The MA3
                smooths hurricane/one-off monthly spikes without introducing
                a free parameter — the 3m window is the Chicago Fed's own
                published smoothing of the index. Default False preserves
                v3 behaviour exactly.
        """
        self.fred = fred_client
        self.history_window_months = history_window_months
        self.request_delay = request_delay
        self.min_covar = min_covar
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_cfnai_ma3 = use_cfnai_ma3

        # Select growth features based on flag
        self.growth_features: List[str] = (
            ["CFNAI_MA3", "UNRATE"] if use_cfnai_ma3 else list(_DEFAULT_GROWTH_FEATURES)
        )

        # Point-in-time z-score cache: (series_id, obs_date) -> float
        self.cache: Dict[Tuple[str, str], float] = self._load_cache()

        # Separate cache for raw PCE level (not z-scored)
        # Keyed by obs_date string only — series is always PCEPILFE
        self._level_cache: Dict[str, float] = {}

        self.growth_model: Optional[hmm.GaussianHMM] = None
        self.inflation_model: Optional[hmm.GaussianHMM] = None
        self.g_high_idx: Optional[int] = None
        self.g_low_idx: Optional[int] = None
        self.i_high_idx: Optional[int] = None
        self.i_low_idx: Optional[int] = None
        self.last_refit_date: Optional[str] = None

        self._growth_train_df: Optional[pd.DataFrame] = None
        self._inflation_train_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict[Tuple[str, str], float]:
        """Load the z-score cache from disk, if cache_path is set and exists."""
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        with open(self.cache_path, "r") as f:
            raw = json.load(f)
        return {tuple(k.split("|")): v for k, v in raw.items()}

    def _save_cache(self) -> None:
        """Persist the z-score cache to disk, if cache_path is set."""
        if self.cache_path is None:
            return
        raw = {f"{sid}|{date}": v for (sid, date), v in self.cache.items()}
        with open(self.cache_path, "w") as f:
            json.dump(raw, f, indent=2)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

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

    def _get_pcepilfe_level(self, obs_date: str) -> float:
        """
        Raw point-in-time PCE YoY level (decimal), cached in-memory only.

        Separate from the z-score cache because the level is not z-scored
        and does not need disk persistence — it is only used by the
        validator node, which is optional and skipped in most backtest runs.
        On cache miss, calls features.get_pcepilfe_level() which uses the
        same ALFRED vintage fetch as the z-score path.
        """
        if obs_date not in self._level_cache:
            self._level_cache[obs_date] = get_pcepilfe_level(self.fred, obs_date)
            time.sleep(self.request_delay)
        return self._level_cache[obs_date]

    def _build_training_matrix(self, series_ids: List[str], dates: List[str]) -> pd.DataFrame:
        """Feature matrix for HMM fitting, indexed by date, NaN rows dropped."""
        rows = []
        for d in dates:
            row_data = self._get_feature_row(series_ids, d)
            row_data["date"] = d
            rows.append(row_data)
        df = pd.DataFrame(rows).set_index("date")
        return df.dropna()

    # ------------------------------------------------------------------
    # Fitting + anchoring
    # ------------------------------------------------------------------

    def _fit_and_anchor(self, df: pd.DataFrame, model_type: str):
        """
        Fit a 2-state diag-covariance GaussianHMM and anchor state labels.

        Label anchoring uses only the primary feature (index 0):
          Growth HMM:    CFNAI or CFNAI_MA3 — the broad activity composite
          Inflation HMM: PCEPILFE — the Fed's preferred realized gauge

        The secondary feature participates in EM fitting but does not affect
        label assignment. Unchanged from v3.
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

        score_0 = model.means_[0][0]
        score_1 = model.means_[1][0]
        high_idx = 0 if score_0 > score_1 else 1
        low_idx  = 1 - high_idx

        return model, high_idx, low_idx

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, observation_date: str) -> Dict:
        """
        Returns point-in-time joint regime probabilities as of
        observation_date.

        Refits both HMMs on calendar quarter-ends (or on first call).
        On non-refit months, appends the latest observation to the cached
        training sequence and runs predict_proba over the full sequence.

        v4 addition: result dict includes an "indicators" sub-dict with:
          - z-scores for all four HMM input series (read from cache,
            no extra ALFRED calls)
          - pcepilfe_level: raw PCE YoY decimal (point-in-time, cached
            separately in self._level_cache)
          - growth_feature_used: "CFNAI" or "CFNAI_MA3" for audit trail

        The "indicators" dict is consumed by the LLM validator node in
        macro_lens.py to build the blinded input. It is always populated
        regardless of whether the validator is enabled — zero marginal cost
        since the z-scores are already in cache from the HMM feature build.

        Raises:
            ValueError if there isn't enough history to fit.
        """
        obs_dt = pd.to_datetime(observation_date)
        is_quarter_end = obs_dt.month in (3, 6, 9, 12)
        needs_refit = self.last_refit_date is None or is_quarter_end

        if needs_refit:
            training_dates = self._generate_trailing_dates(observation_date)

            growth_df = self._build_training_matrix(self.growth_features, training_dates)
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

        # Latest observation features
        latest_g_row = self._get_feature_row(self.growth_features, observation_date)
        latest_i_row = self._get_feature_row(INFLATION_FEATURES, observation_date)

        if any(pd.isna(v) for v in latest_g_row.values()) or any(
            pd.isna(v) for v in latest_i_row.values()
        ):
            return {"error": "Missing point-in-time data for inference."}

        prob_g_high = self._posterior_for_latest(
            self._growth_train_df, self.growth_features, latest_g_row,
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

        # ------------------------------------------------------------------
        # v4: Build indicators dict from already-cached z-scores + PCE level.
        # The primary growth key is dynamic (CFNAI or CFNAI_MA3).
        # CFNAI z-score is always included for the audit trail even when
        # CFNAI_MA3 is the active feature — pull from cache if present,
        # otherwise it won't be there (no extra fetch is made).
        # ------------------------------------------------------------------
        primary_growth_key = self.growth_features[0]  # "CFNAI" or "CFNAI_MA3"

        indicators = {
            "growth_feature_used":  primary_growth_key,
            "cfnai_zscore":         self.cache.get((primary_growth_key, observation_date), float("nan")),
            "unrate_zscore":        self.cache.get(("UNRATE",    observation_date), float("nan")),
            "pcepilfe_zscore":      self.cache.get(("PCEPILFE",  observation_date), float("nan")),
            "cpiaucsl_zscore":      self.cache.get(("CPIAUCSL",  observation_date), float("nan")),
            "pcepilfe_level":       self._get_pcepilfe_level(observation_date),
        }

        return {
            "growth_high_prob":    round(float(prob_g_high), 4),
            "inflation_high_prob": round(float(prob_i_high), 4),
            "joint_probabilities": {k: round(float(v), 4) for k, v in regimes.items()},
            "dominant_regime":     dominant_regime,
            "confidence":          round(float(regimes[dominant_regime]), 4),
            "indicators":          indicators,
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
        the FULL sequence, preserving transition-matrix hysteresis.

        Posteriors are floored at posterior_floor and renormalised to
        prevent numerical saturation at 0.0/1.0. Unchanged from v3.
        """
        if observation_date in train_df.index:
            X = train_df[feature_cols].values
        else:
            latest_series = pd.Series(latest_row, name=observation_date)
            extended = pd.concat([train_df[feature_cols], latest_series.to_frame().T])
            X = extended.values.astype(float)

        posteriors = model.predict_proba(X)

        p = posteriors[-1].copy()
        p = np.clip(p, posterior_floor, 1.0 - posterior_floor)
        p = p / p.sum()

        return float(p[high_idx])


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # --- Sanity check 1: v3-compatible mode (raw CFNAI) ---
    print("=== Arm 1a: raw CFNAI (v3-compatible) ===")
    engine_v3 = RegimeHMMEngine(fred, use_cfnai_ma3=False)
    for date in ["2019-12-31", "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]:
        try:
            result = engine_v3.infer(date)
        except ValueError as e:
            print(f"{date}: {e}")
            continue
        if "error" in result:
            print(f"{date}: {result['error']}")
            continue
        ind = result["indicators"]
        print(
            f"{date}  "
            f"P(G_high)={result['growth_high_prob']:.3f}  "
            f"P(I_high)={result['inflation_high_prob']:.3f}  "
            f"dominant={result['dominant_regime']:<35}  "
            f"conf={result['confidence']:.3f}  "
            f"cfnai_z={ind['cfnai_zscore']:+.3f}  "
            f"pce_lvl={ind['pcepilfe_level']:.2%}"
        )

    print()

    # --- Sanity check 2: v4 CFNAI_MA3 mode ---
    # Expected: growth regime should deteriorate faster around March 2020
    # with MA3 (already capturing Feb weakness) vs raw CFNAI (still positive).
    print("=== Arm 1b: CFNAI_MA3 (v4) ===")
    engine_v4 = RegimeHMMEngine(fred, use_cfnai_ma3=True)
    for date in ["2019-12-31", "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]:
        try:
            result = engine_v4.infer(date)
        except ValueError as e:
            print(f"{date}: {e}")
            continue
        if "error" in result:
            print(f"{date}: {result['error']}")
            continue
        ind = result["indicators"]
        print(
            f"{date}  "
            f"P(G_high)={result['growth_high_prob']:.3f}  "
            f"P(I_high)={result['inflation_high_prob']:.3f}  "
            f"dominant={result['dominant_regime']:<35}  "
            f"conf={result['confidence']:.3f}  "
            f"cfnai_ma3_z={ind['cfnai_zscore']:+.3f}  "
            f"pce_lvl={ind['pcepilfe_level']:.2%}"
        )

    print()

    # --- Sanity check 3: current date inflation check ---
    # Expected: pcepilfe_level > 0.025 should trigger Gate 2 if HMM
    # calls Low Inflation, demonstrating the validator gate logic.
    print("=== Sanity check 3: current period indicators ===")
    import datetime
    today = datetime.date.today().replace(day=1) - datetime.timedelta(days=1)
    obs = today.strftime("%Y-%m-%d")
    try:
        result = engine_v4.infer(obs)
        ind = result["indicators"]
        print(f"As of {obs}:")
        print(f"  Dominant regime : {result['dominant_regime']}")
        print(f"  Confidence      : {result['confidence']:.1%}")
        print(f"  PCE level       : {ind['pcepilfe_level']:.2%}  (Fed target: 2.00%)")
        print(f"  PCE z-score     : {ind['pcepilfe_zscore']:+.3f}")
        print(f"  CFNAI_MA3 z     : {ind['cfnai_zscore']:+.3f}")
        gate2 = (
            ind["pcepilfe_level"] > 0.025
            and result["dominant_regime"].endswith("Low Inflation")
        )
        print(f"  Gate 2 would fire: {gate2}")
    except Exception as e:
        print(f"  {obs}: {e}")