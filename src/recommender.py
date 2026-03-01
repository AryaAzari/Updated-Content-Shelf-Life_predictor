"""
src/recommender.py — Promotion window recommender

Combines two signals to classify each title into one of three states:

  PROMOTE NOW   — survival probability still above threshold AND title has
                  not yet reached its Prophet-detected inflection point.
                  This is the active promotion window.

  PROMOTE SOON  — title has not yet reached peak (very early post-release),
                  with the window projected to open within 3 weeks based on
                  the Cox-predicted median survival time.

  WINDOW PASSED — post-inflection OR survival probability has dropped below
                  the threshold. Promotion spend here has diminishing returns.

The output is a formatted content calendar DataFrame that a platform
scheduler can act on directly.
"""

import pandas as pd
import numpy as np
from typing import Optional
from config import SURVIVAL_PROMOTE_THRESHOLD, PROMOTE_SOON_WINDOW_WEEKS


def _get_survival_prob_at_day(
    cox_model,
    covariate_row: pd.DataFrame,
    day: int,
) -> float:
    """
    Query the Cox model for survival probability at a specific day (as a week).

    Args:
        cox_model: Fitted lifelines CoxPHFitter instance
        covariate_row: Single-row DataFrame of covariates for this title
        day: Current day since release

    Returns:
        Survival probability (0.0–1.0) at the given week.
    """
    week = max(1, day // 7)
    survival_func = cox_model.predict_survival_function(covariate_row)
    # survival_func index is in the same time units as the training data (weeks)
    timeline = survival_func.index.values
    probs = survival_func.iloc[:, 0].values

    # Interpolate to find prob at `week`
    if week >= timeline[-1]:
        return float(probs[-1])
    if week <= timeline[0]:
        return float(probs[0])
    return float(np.interp(week, timeline, probs))


def classify_title(
    title: str,
    days_since_release: int,
    inflection_day: Optional[int],
    survival_prob: float,
) -> str:
    """
    Classify a single title into a promotion state.

    Args:
        title: Title name (for logging only)
        days_since_release: How many days since the title was released
        inflection_day: Prophet-detected inflection day (None if undetected)
        survival_prob: Cox-predicted survival probability at current day

    Returns:
        One of: "Promote Now", "Promote Soon", "Window Passed"
    """
    post_inflection = (inflection_day is not None) and (days_since_release >= inflection_day)
    survival_below = survival_prob < SURVIVAL_PROMOTE_THRESHOLD
    pre_peak = days_since_release < 7  # still in first week — window not yet open

    if post_inflection or survival_below:
        return "Window Passed"
    if pre_peak:
        return "Promote Soon"
    return "Promote Now"


def build_content_calendar(
    survival_df: pd.DataFrame,
    inflection_df: pd.DataFrame,
    cox_model,
    feature_columns: list[str],
    reference_date: str = "today",
) -> pd.DataFrame:
    """
    Build a formatted promotion calendar for all titles.

    For each title, computes the current promotion state and the predicted
    peak window (Cox median survival time), then formats results as a table
    ready for a platform scheduling team.

    Args:
        survival_df: Survival dataset with covariates (from src/survival.py)
        inflection_df: Inflection point results (from src/prophet_analysis.py)
        cox_model: Fitted lifelines CoxPHFitter
        feature_columns: List of one-hot + numeric feature column names
        reference_date: Date to treat as "today" for state classification.
                        Pass "today" to use the current date.

    Returns:
        DataFrame with columns:
            title, release_date, genre, peak_window_weeks,
            current_survival_prob, inflection_day,
            days_since_release, promotion_state, recommended_action
    """
    from datetime import date, datetime

    if reference_date == "today":
        today = date.today()
    else:
        today = datetime.strptime(reference_date, "%Y-%m-%d").date()

    merged = survival_df.merge(inflection_df[["title", "inflection_day", "peak_day"]], on="title", how="left")
    records = []

    for _, row in merged.iterrows():
        try:
            release_dt = pd.to_datetime(row["release_date"]).date()
            days_since = (today - release_dt).days
        except Exception:
            days_since = 0

        # Build covariate row for Cox prediction
        cov_row = pd.DataFrame([row[feature_columns].to_dict()])

        try:
            surv_prob = _get_survival_prob_at_day(cox_model, cov_row, days_since)
        except Exception:
            surv_prob = float("nan")

        state = classify_title(
            title=row["title"],
            days_since_release=days_since,
            inflection_day=row.get("inflection_day"),
            survival_prob=surv_prob,
        )

        # Predict peak window: weeks where survival prob ≥ threshold
        try:
            sf = cox_model.predict_survival_function(cov_row)
            timeline = sf.index.values
            probs = sf.iloc[:, 0].values
            in_window = timeline[probs >= SURVIVAL_PROMOTE_THRESHOLD]
            peak_window_str = (
                f"Week {int(in_window[0])}–{int(in_window[-1])}"
                if len(in_window) > 0
                else "N/A"
            )
        except Exception:
            peak_window_str = "N/A"

        action_map = {
            "Promote Now":    "Schedule promotion immediately",
            "Promote Soon":   f"Begin pre-promotion; window opens in ~{PROMOTE_SOON_WINDOW_WEEKS} weeks",
            "Window Passed":  "Deprioritize; redirect promotion budget",
        }

        records.append({
            "title": row["title"],
            "release_date": row.get("release_date", ""),
            "genre": row.get("genre", ""),
            "predicted_peak_window": peak_window_str,
            "current_survival_prob": round(surv_prob, 3) if not np.isnan(surv_prob) else None,
            "inflection_day": row.get("inflection_day"),
            "days_since_release": days_since,
            "promotion_state": state,
            "recommended_action": action_map[state],
        })

    calendar_df = pd.DataFrame(records)
    state_order = {"Promote Now": 0, "Promote Soon": 1, "Window Passed": 2}
    calendar_df["_order"] = calendar_df["promotion_state"].map(state_order)
    calendar_df = calendar_df.sort_values(["_order", "current_survival_prob"], ascending=[True, False])
    calendar_df = calendar_df.drop(columns=["_order"]).reset_index(drop=True)
    return calendar_df
