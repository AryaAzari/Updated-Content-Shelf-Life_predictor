"""
src/survival.py — Trend death definition and survival dataset construction

Trend death is defined as the first week where a title's 7-day rolling
average drops below `threshold` * peak_rolling_avg AND remains there for
at least `consecutive_weeks` weeks. This is scale-invariant: the threshold
is relative to each title's own peak, so blockbusters and indie films are
treated fairly.

Titles still above the threshold at week 20 are marked as censored
(event = 0), meaning we ran out of observation window — not that they
never decayed.

The output is a survival dataset with one row per title, ready for
Kaplan-Meier and Cox model fitting.
"""

import numpy as np
import pandas as pd
from typing import Optional
from config import (
    ROLLING_WINDOW_DAYS,
    TREND_DEATH_THRESHOLD,
    TREND_DEATH_CONSECUTIVE_WEEKS,
    OBSERVATION_WINDOW_WEEKS,
    SENSITIVITY_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_rolling_avg(pageviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 7-day rolling average of pageviews for each title.

    Args:
        pageviews_df: Long-format dataframe with [title, days_since_release, pageviews]

    Returns:
        Same dataframe with an additional `rolling_avg` column.
    """
    df = pageviews_df.copy()
    df = df.sort_values(["title", "days_since_release"])
    df["rolling_avg"] = (
        df.groupby("title")["pageviews"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_DAYS, min_periods=1).mean())
    )
    return df


def compute_weekly_avg(pageviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily pageviews into weekly averages (post-release only).

    Week 1 = days 0-6, Week 2 = days 7-13, ..., Week 20 = days 133-139.
    Pre-release days are excluded from survival analysis.

    Returns dataframe with [title, week, weekly_avg] for weeks 1-20.
    """
    df = pageviews_df[pageviews_df["days_since_release"] >= 0].copy()
    df["week"] = (df["days_since_release"] // 7) + 1
    df = df[df["week"] <= OBSERVATION_WINDOW_WEEKS]

    weekly = (
        df.groupby(["title", "week"])["pageviews"]
        .mean()
        .reset_index()
        .rename(columns={"pageviews": "weekly_avg"})
    )
    return weekly


def find_trend_death_week(
    weekly_avg: pd.Series,
    weeks: pd.Series,
    threshold: float = TREND_DEATH_THRESHOLD,
    consecutive: int = TREND_DEATH_CONSECUTIVE_WEEKS,
) -> Optional[int]:
    """
    Find the first week where trend death occurs for a single title.

    Trend death = first week where weekly_avg < threshold * peak
    AND this condition holds for `consecutive` weeks straight.

    Args:
        weekly_avg: Series of weekly average pageviews (ordered by week)
        weeks: Series of week numbers corresponding to weekly_avg
        threshold: Fraction of peak below which the title is "dead"
        consecutive: Minimum consecutive weeks below threshold required

    Returns:
        Week number of trend death, or None if never occurred within window.
    """
    peak = weekly_avg.max()
    if peak == 0:
        return None

    cutoff = threshold * peak
    below = weekly_avg < cutoff

    # Find first run of `consecutive` True values
    streak = 0
    for i, (is_below, week) in enumerate(zip(below, weeks)):
        if is_below:
            streak += 1
            if streak >= consecutive:
                # Return the start of this streak
                return int(weeks.iloc[i - consecutive + 1])
        else:
            streak = 0
    return None


def build_survival_dataset(
    pageviews_df: pd.DataFrame,
    titles_df: pd.DataFrame,
    threshold: float = TREND_DEATH_THRESHOLD,
) -> pd.DataFrame:
    """
    Construct the time-to-event survival dataset.

    One row per title with:
      - duration: weeks until trend death (or 20 if censored)
      - event: 1 = trend death observed, 0 = censored
      - All TMDB covariates
      - early_velocity: slope of linear regression on pageviews days 1-7
      - release_season: Q1/Q2/Q3/Q4 derived from release_date

    Args:
        pageviews_df: Long-format pageview data
        titles_df: Title metadata (from TMDB)
        threshold: Trend death threshold (fraction of peak)

    Returns:
        Survival dataset DataFrame.
    """
    weekly_df = compute_weekly_avg(pageviews_df)

    records = []
    for title, group in weekly_df.groupby("title"):
        group = group.sort_values("week")
        death_week = find_trend_death_week(
            group["weekly_avg"], group["week"], threshold=threshold
        )
        if death_week is not None:
            duration = death_week
            event = 1
        else:
            duration = OBSERVATION_WINDOW_WEEKS
            event = 0
        records.append({"title": title, "duration": duration, "event": event})

    survival_df = pd.DataFrame(records)

    # Compute early_velocity: slope of OLS on pageviews in days 1-7
    velocity_records = []
    for title, group in pageviews_df.groupby("title"):
        early = group[(group["days_since_release"] >= 1) & (group["days_since_release"] <= 7)]
        if len(early) >= 3:
            x = early["days_since_release"].values.astype(float)
            y = early["pageviews"].values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        velocity_records.append({"title": title, "early_velocity": slope})

    velocity_df = pd.DataFrame(velocity_records)
    survival_df = survival_df.merge(velocity_df, on="title", how="left")

    # Merge movie covariates from the movies table
    meta_cols = [
        "title", "genre", "budget_tier", "budget_usd",
        "runtime", "release_date", "is_franchise",
    ]
    available_cols = [c for c in meta_cols if c in titles_df.columns]
    survival_df = survival_df.merge(titles_df[available_cols], on="title", how="left")

    # Derive release_season from release_date
    if "release_date" in survival_df.columns:
        months = pd.to_datetime(survival_df["release_date"]).dt.month
        survival_df["release_season"] = months.map(
            lambda m: "Q1" if m <= 3 else ("Q2" if m <= 6 else ("Q3" if m <= 9 else "Q4"))
        )

    return survival_df.reset_index(drop=True)


def sensitivity_analysis(
    pageviews_df: pd.DataFrame,
    titles_df: pd.DataFrame,
    thresholds: list[float] = SENSITIVITY_THRESHOLDS,
) -> dict[float, pd.DataFrame]:
    """
    Build survival datasets at multiple thresholds to validate stability.

    Returns a dict mapping threshold -> survival DataFrame, used in the
    notebook to overlay KM curves and show they don't move materially.
    """
    return {t: build_survival_dataset(pageviews_df, titles_df, threshold=t) for t in thresholds}
