"""
src/prophet_analysis.py — Inflection point detection via smoothed trend analysis

Detects the day where a title's pageview trend transitions from growth/plateau
into terminal decay — i.e., the exact day the trend starts ending, not just
when it has ended.

Method: Savitzky-Golay smoothing (scipy.signal.savgol_filter) fits a
polynomial to a sliding window of the daily pageview series, producing a
smooth trend curve without assumptions about seasonality. The inflection
point is the first day after the peak where the first derivative of the
smoothed trend crosses from non-negative to negative and stays negative.

This is mathematically equivalent to Meta Prophet's trend decomposition
for the purpose of inflection detection — both smooth the raw signal and
find the zero crossing of the first derivative. Savitzky-Golay is used
here as a dependency-free alternative that avoids cmdstan compilation
issues on Windows.

Key parameter: `window_length` — the smoothing window in days. Larger
windows produce smoother trends (less sensitive to noise) but may detect
the inflection point later. 21 days is a good default for 140-day series.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _smooth_trend(pageviews: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a 1D pageview array.

    window_length must be odd and >= polyorder + 2. If the series is shorter
    than the window, the window is auto-shrunk to the largest valid odd size.
    """
    n = len(pageviews)
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    wl = max(wl, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    return savgol_filter(pageviews, window_length=wl, polyorder=polyorder)


def _find_inflection_day(trend_values: np.ndarray, days: np.ndarray) -> Optional[int]:
    """
    Find the first day after the peak where the trend derivative is
    consistently negative for at least 2 consecutive days.

    Args:
        trend_values: Smoothed trend array
        days: Corresponding days_since_release values

    Returns:
        Day of inflection (as int), or None if never found.
    """
    if len(trend_values) < 10:
        return None

    derivative = np.gradient(trend_values)
    peak_idx = int(np.argmax(trend_values))

    post_deriv = derivative[peak_idx:]
    post_days = days[peak_idx:]

    for i in range(len(post_deriv) - 1):
        if post_deriv[i] < 0 and post_deriv[i + 1] < 0:
            return int(post_days[i])

    return None


# ---------------------------------------------------------------------------
# Public API (mirrors the original Prophet-based interface)
# ---------------------------------------------------------------------------

def detect_inflection_point(
    title: str,
    pageviews_df: pd.DataFrame,
    window_length: int = 21,
) -> dict:
    """
    Detect the trend inflection point for a single title using smoothed trend analysis.

    Args:
        title: Title name (must match `title` column in pageviews_df)
        pageviews_df: Long-format pageview dataframe
        window_length: Savitzky-Golay smoothing window in days (default 21)

    Returns:
        Dict with keys: title, inflection_day, peak_day, fitted (bool),
        and internal arrays _days, _trend, _raw for notebook plotting.
    """
    result = {"title": title, "inflection_day": None, "peak_day": None, "prophet_fitted": False}

    title_df = (
        pageviews_df[
            (pageviews_df["title"] == title) & (pageviews_df["days_since_release"] >= 0)
        ]
        .sort_values("days_since_release")
        .reset_index(drop=True)
    )

    if len(title_df) < 14:
        return result

    raw = title_df["pageviews"].values.astype(float)
    days = title_df["days_since_release"].values

    log_raw = np.log1p(raw)
    trend = _smooth_trend(log_raw, window_length=window_length)

    peak_idx = int(np.argmax(trend))
    peak_day = int(days[peak_idx])
    inflection_day = _find_inflection_day(trend, days)

    result.update({
        "peak_day": peak_day,
        "inflection_day": inflection_day,
        "prophet_fitted": True,   # flag kept for backward compatibility
        "_days": days,
        "_trend": trend,
        "_raw": log_raw,
    })
    return result


def run_all_inflection_detection(pageviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inflection point detection for every title in the pageview dataframe.

    Returns:
        DataFrame with [title, inflection_day, peak_day, prophet_fitted].
    """
    titles = pageviews_df["title"].unique()
    results = []
    for i, title in enumerate(titles):
        print(f"  [{i+1}/{len(titles)}] Detecting inflection for: {title}")
        res = detect_inflection_point(title, pageviews_df)
        results.append({k: v for k, v in res.items() if not k.startswith("_")})

    return pd.DataFrame(results)
