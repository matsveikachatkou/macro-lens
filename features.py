"""
Point-in-time feature engineering for macro-lens v3.

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
  UNRATE   — Unemployment rate (rate series); "diff" transform, 36m window.
              Secondary growth feature: labour market confirmation signal.
  PCEPILFE — Core PCE price index, YoY; "yoy" transform, 60m window.
              Primary inflation feature — the Fed's preferred realized gauge.
  CPIAUCSL — CPI, all urban consumers, YoY; "yoy" transform, 60m window.
              Secondary inflation feature. Paired with PCEPILFE rather than
              T10YIE (breakeven inflation) because both are realized series,
              keeping the inflation HMM anchored to what inflation actually
              was rather than mixing realized with market expectations.
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
ZSCORE_TRANSFORMS = {
    "CFNAI":    {"transform": "level", "window_months": 36},
    "UNRATE":   {"transform": "diff",  "window_months": 36},
    "PCEPILFE": {"transform": "yoy",   "window_months": 60},
    "CPIAUCSL": {"transform": "yoy",   "window_months": 60},
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
        series_id: FRED series ticker (e.g., 'CFNAI', 'UNRATE').
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
    obs_dt = datetime.strptime(observation_date, "%Y-%m-%d")

    # Buffer the start date so that after differencing/pct_change we still
    # have `window_months` valid points after the transform drops NaNs.
    buffer = 14 if transform == "yoy" else 2
    start_dt = obs_dt - relativedelta(months=window_months + buffer)
    start_str = start_dt.strftime("%Y-%m-%d")

    try:
        raw_series = _fetch_series_with_retry(fred, series_id, start_str, observation_date)
    except Exception as e:
        print(f"Error fetching {series_id} for {observation_date}: {e}")
        return float("nan")

    if raw_series is None or raw_series.empty:
        return float("nan")

    raw_series = raw_series.dropna()

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


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # Sanity check: compute z-scores for a known historical date.
    # Expected: CFNAI negative (COVID crash), PCEPILFE/CPI mildly elevated.
    test_date = "2020-03-31"
    print(f"Point-in-time z-scores as of {test_date}:\n")

    for series_id, config in ZSCORE_TRANSFORMS.items():
        z = get_pit_zscore(
            fred, series_id, test_date,
            transform=config["transform"],
            window_months=config["window_months"],
        )
        print(f"  {series_id:<10} ({config['transform']:<10}) -> z = {z}")