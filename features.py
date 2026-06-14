"""
Point-in-time feature engineering for macro-lens v4.

Provides get_pit_zscore(): a rolling z-score of a FRED series, computed
using only data that was actually published as of `observation_date`
(via ALFRED vintage parameters), with no look-ahead bias.

Series used and their transforms:
  CFNAI    — Chicago Fed National Activity Index (already stationary,
              mean-zero by construction); "level" transform, 36m window.
              Preferred over INDPRO because it aggregates 85 monthly
              indicators across production, employment, consumption, and
              orders — a broader growth proxy for an equity-benchmarked
              TAA strategy.
  CFNAI_MA3 — 3-month moving average of CFNAI (Chicago Fed also publishes
              this directly). Same "level" transform, 36m window. Smooths
              hurricane/one-off monthly spikes without introducing a free
              parameter — the 3m window is a published Chicago Fed series,
              not a tuned choice. Used when RegimeHMMEngine is initialised
              with use_cfnai_ma3=True.
  UNRATE   — Unemployment rate (rate series); "diff" transform, 36m window.
              Secondary growth feature: labour market confirmation signal.
  PCEPILFE — Core PCE price index, YoY; "yoy" transform, 120m window.
              Primary inflation feature — the Fed's preferred realized gauge.
              v4 change: window extended from 60m to 120m so that a single
              high-inflation episode (e.g. 2021-23) cannot dominate the
              rolling mean. At 120m the window spans a full decade and
              includes at least one complete low-inflation era, making the
              z-score more stable across regime transitions.
  CPIAUCSL — CPI, all urban consumers, YoY; "yoy" transform, 120m window.
              Secondary inflation feature. Window extended from 60m to 120m
              for the same reason as PCEPILFE.

v4 additions vs v3:
  - CFNAI_MA3 entry in ZSCORE_TRANSFORMS (selectable via use_cfnai_ma3 flag)
  - PCEPILFE and CPIAUCSL window_months: 60 → 120
  - get_pcepilfe_level(): returns raw point-in-time PCE YoY level (not z-scored)
    for use by the LLM validator's Gate 2 check and the blinded input dict
"""

import socket
socket.setdefaulttimeout(10)

import concurrent.futures
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fredapi import Fred
import pandas as pd
from tenacity import retry, retry_if_exception_message, wait_exponential, stop_after_attempt


# Maps each FRED series used as an HMM input to the transform needed
# to make it (approximately) stationary before z-scoring.
#
#   "level"      -> series is already stationary (CFNAI is mean-zero by construction)
#   "pct_change" -> for indices/levels where growth rate is the signal
#   "yoy"        -> 12-month pct_change; smooths monthly noise for slow-moving
#                   price series (PCE, CPI)
#   "diff"       -> for series expressed as rates/spreads (UNRATE)
#
# v4 notes:
#   CFNAI_MA3 uses the same "level" transform as CFNAI — it is already
#   mean-zero by construction (it is just a smoothed version of CFNAI).
#   The fetch logic applies the 3m rolling mean before z-scoring; see
#   get_pit_zscore() for the implementation detail.
#
#   PCEPILFE and CPIAUCSL window extended to 120m. Cache keys are
#   (series_id, observation_date) so existing 60m entries in zscore_cache.json
#   remain valid for CFNAI and UNRATE — only the inflation series entries
#   will be recomputed on first run.
ZSCORE_TRANSFORMS = {
    "CFNAI":     {"transform": "level", "window_months": 36},
    "CFNAI_MA3": {"transform": "level", "window_months": 36},
    "UNRATE":    {"transform": "diff",  "window_months": 36},
    "PCEPILFE":  {"transform": "yoy",   "window_months": 120},
    "CPIAUCSL":  {"transform": "yoy",   "window_months": 120},
}


@retry(
    retry=retry_if_exception_message(match=".*Too Many Requests.*|.*Rate Limit.*|.*timeout.*"),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _fetch_series_with_retry(fred: Fred, series_id: str, start_str: str, observation_date: str):
    """
    ALFRED fetch with a hard 15s timeout (via ThreadPoolExecutor, since
    fredapi/urllib don't reliably honor socket.setdefaulttimeout for this
    code path) and exponential backoff on rate-limit / timeout errors.

    Some series have no ALFRED vintage history — they are daily market
    series that are never revised, so FRED doesn't track real-time versions.
    For these, fall back to a plain (non-vintage) fetch: "today's published
    value" is the only value that has ever existed for a given date, so
    there is no look-ahead concern. CFNAI and CPIAUCSL both have full
    ALFRED vintage history; UNRATE and PCEPILFE also do.

    CFNAI_MA3 is handled by fetching the underlying CFNAI series and
    computing the 3m rolling mean in get_pit_zscore() — not by fetching
    a separate ALFRED series — so this function always receives "CFNAI"
    as the series_id when called for CFNAI_MA3 features.
    """
    def _fetch(use_realtime: bool):
        kwargs = dict(observation_start=start_str, observation_end=observation_date)
        if use_realtime:
            kwargs["realtime_start"] = observation_date
            kwargs["realtime_end"] = observation_date
        return fred.get_series(series_id, **kwargs)

    def _run(use_realtime: bool):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch, use_realtime)
            try:
                return future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"timeout fetching {series_id} as of {observation_date}")

    try:
        return _run(use_realtime=True)
    except ValueError as e:
        if "does not exist in ALFRED" in str(e):
            return _run(use_realtime=False)
        raise


def get_pit_zscore(
    fred: Fred,
    series_id: str,
    observation_date: str,
    transform: str = "pct_change",
    window_months: int = 36,
    clip: float = 3.5,
) -> float:
    """
    Fetch point-in-time ALFRED data and calculate a trailing z-score,
    clipped to [-clip, +clip].

    Args:
        fred: Authenticated fredapi client.
        series_id: FRED series ticker (e.g., 'CFNAI', 'UNRATE', 'CFNAI_MA3').
            For 'CFNAI_MA3', the underlying CFNAI series is fetched and a
            3-month rolling mean is applied before z-scoring. The extra
            smoothing requires 2 additional months of buffer, handled
            internally.
        observation_date: YYYY-MM-DD string representing "today" for
            point-in-time purposes. Only data published on or before
            this date is used — no look-ahead bias.
        transform: How to make the raw series stationary before z-scoring.
            'level'      — use as-is (CFNAI is already mean-zero)
            'pct_change' — month-on-month percentage change
            'yoy'        — 12-month percentage change (PCE, CPI)
            'diff'       — first difference (UNRATE)
        window_months: Number of months for the trailing z-score window.
            The mean and std are computed over this window, so the z-score
            measures how the latest observation compares to its own recent
            history — not to a fixed long-run average.
        clip: Z-scores are clipped to [-clip, +clip] before being returned.
            Extreme tail events are mathematically correct but can dominate
            a 2-state Gaussian HMM's EM fit on a small training set —
            one outlier can pull an entire state's mean toward it, destroying
            the economic anchoring. Clipping preserves "this was an extreme
            month" as a strong signal without letting the magnitude distort
            the fit further. Default 3.5. Set to None to disable.

    Returns:
        The (clipped) z-scored latest observation as a float, or
        float('nan') if insufficient data is available.
    """
    # CFNAI_MA3: fetch the underlying CFNAI series, apply 3m rolling mean,
    # then z-score. The series_id passed to _fetch_series_with_retry is
    # always "CFNAI" in this case.
    fetch_id = "CFNAI" if series_id == "CFNAI_MA3" else series_id
    apply_ma3 = series_id == "CFNAI_MA3"

    obs_dt = datetime.strptime(observation_date, "%Y-%m-%d")

    # Buffer the start date so that after differencing/pct_change we still
    # have `window_months` valid points after the transform drops NaNs.
    # CFNAI_MA3 needs 2 extra months beyond the normal buffer so the rolling
    # mean has enough history at the start of the window.
    buffer = 14 if transform == "yoy" else 2
    if apply_ma3:
        buffer += 2
    start_dt = obs_dt - relativedelta(months=window_months + buffer)
    start_str = start_dt.strftime("%Y-%m-%d")

    try:
        raw_series = _fetch_series_with_retry(fred, fetch_id, start_str, observation_date)
    except Exception as e:
        print(f"Error fetching {fetch_id} for {observation_date}: {e}")
        return float("nan")

    if raw_series is None or raw_series.empty:
        return float("nan")

    raw_series = raw_series.dropna()

    # Apply 3-month rolling mean for CFNAI_MA3 before any other transform.
    # min_periods=3 ensures we only emit values once 3 months are available;
    # earlier NaNs are dropped below via transformed.dropna().
    if apply_ma3:
        raw_series = raw_series.rolling(window=3, min_periods=3).mean()

    if transform == "pct_change":
        transformed = raw_series.pct_change()
    elif transform == "diff":
        transformed = raw_series.diff()
    elif transform == "level":
        transformed = raw_series
    elif transform == "yoy":
        transformed = raw_series.pct_change(12)
    else:
        raise ValueError(
            f"Unknown transform: {transform!r}. "
            "Use 'level', 'pct_change', 'yoy', or 'diff'."
        )

    transformed = transformed.dropna()

    if len(transformed) < window_months:
        # Not fatal — early backtest dates may not have a full window yet.
        # The z-score is still computed on whatever history is available.
        print(
            f"Warning: only {len(transformed)} observations available for "
            f"{series_id} as of {observation_date} (wanted {window_months})"
        )

    transformed = transformed.tail(window_months)

    if len(transformed) < 2:
        return float("nan")

    latest_val  = transformed.iloc[-1]
    window_mean = transformed.mean()
    window_std  = transformed.std()

    if window_std == 0 or pd.isna(window_std):
        return 0.0

    z_score = (latest_val - window_mean) / window_std

    if clip is not None:
        z_score = max(-clip, min(clip, z_score))

    return round(float(z_score), 4)


def get_pcepilfe_level(
    fred: Fred,
    observation_date: str,
) -> float:
    """
    Return the point-in-time core PCE YoY growth rate as a raw decimal
    (e.g. 0.027 for 2.7%), using ALFRED vintage parameters to avoid
    look-ahead bias.

    This is NOT a z-score. It is the raw realized inflation level used by
    the LLM validator's Gate 2 check: if PCE YoY > 2.5% and the HMM
    calls Low Inflation, the validator is invoked regardless of the z-score.

    The z-score can legitimately be negative (PCE is low relative to a
    high-inflation rolling window) while the level is above the Fed's 2%
    target. This function provides the level so the validator can reason
    about absolute inflation independently of the window-relative z-score.

    Returns:
        PCE YoY as a decimal (e.g. 0.027), or float('nan') on fetch failure.
        Returns float('nan') if fewer than 13 months of history are available
        (need 12 months for YoY calculation plus current month).
    """
    obs_dt   = datetime.strptime(observation_date, "%Y-%m-%d")
    # 14 months back: 12 for YoY + 2 buffer for publication lag
    start_dt = obs_dt - relativedelta(months=14)
    start_str = start_dt.strftime("%Y-%m-%d")

    try:
        raw_series = _fetch_series_with_retry(fred, "PCEPILFE", start_str, observation_date)
    except Exception as e:
        print(f"Error fetching PCEPILFE level for {observation_date}: {e}")
        return float("nan")

    if raw_series is None or raw_series.empty:
        return float("nan")

    raw_series = raw_series.dropna()
    yoy = raw_series.pct_change(12).dropna()

    if yoy.empty:
        return float("nan")

    return round(float(yoy.iloc[-1]), 6)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # --- Sanity check 1: z-scores at COVID crash date ---
    # Expected: CFNAI strongly negative, CFNAI_MA3 less extreme (smoothed),
    # PCEPILFE/CPI mildly elevated z-score with 120m window.
    test_date = "2020-03-31"
    print(f"Point-in-time z-scores as of {test_date}:\n")
    for series_id, config in ZSCORE_TRANSFORMS.items():
        z = get_pit_zscore(
            fred, series_id, test_date,
            transform=config["transform"],
            window_months=config["window_months"],
        )
        print(f"  {series_id:<12} ({config['transform']:<10}) -> z = {z}")

    # --- Sanity check 2: raw PCE level ---
    # Expected: ~0.018 (1.8%) in March 2020 — below Fed target, not a
    # Gate 2 trigger. Contrast with mid-2022 where PCE was ~0.052 (5.2%).
    print(f"\nRaw PCE YoY level as of {test_date}:")
    level = get_pcepilfe_level(fred, test_date)
    print(f"  PCEPILFE YoY = {level:.2%}  (Fed target: 2.0%)")

    print(f"\nRaw PCE YoY level as of 2022-06-30:")
    level_2022 = get_pcepilfe_level(fred, "2022-06-30")
    print(f"  PCEPILFE YoY = {level_2022:.2%}  (Fed target: 2.0%)")

    # --- Sanity check 3: CFNAI vs CFNAI_MA3 comparison ---
    # MA3 should be less extreme than raw CFNAI at COVID shock date.
    print(f"\nCFNAI vs CFNAI_MA3 comparison at {test_date}:")
    z_cfnai = get_pit_zscore(fred, "CFNAI", test_date, transform="level", window_months=36)
    z_ma3   = get_pit_zscore(fred, "CFNAI_MA3", test_date, transform="level", window_months=36)
    print(f"  CFNAI     z = {z_cfnai}  (raw, more extreme)")
    print(f"  CFNAI_MA3 z = {z_ma3}   (smoothed, less extreme)")
    print(f"  Difference  = {abs(z_cfnai - z_ma3):.4f}")