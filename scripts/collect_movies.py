"""
collect_movies.py

Pulls American movies from TMDB with genre-relative budget tiers.
Budget thresholds are calculated per genre using the top 100 most popular
films as a calibration sample — so "high budget" means high for THAT genre.

Rules:
    - No duplicate movies across any genre or tier
    - Each movie is assigned only to its PRIMARY genre (first listed by TMDB)
    - Only inserts movies where ALL fields are known and non-zero
    - Does NOT pad with random movies if targets aren't met

Author: [Your Name]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import sqlite3
import time
import numpy as np
from config import TMDB_API_KEY as API_KEY, DB_PATH

BASE_URL = "https://api.themoviedb.org/3"

VALID_YEARS = [2018, 2019, 2023, 2024]

GENRE_IDS = {
    "action":  28,
    "drama":   18,
    "sci-fi":  878,
    "comedy":  35,
    "horror":  27,
}

# reverse lookup: TMDB genre_id -> our genre name
GENRE_ID_TO_NAME = {v: k for k, v in GENRE_IDS.items()}

TARGETS = {
    "high": 12,
    "mid":  12,
    "low":  6,
}

# ── Database ──────────────────────────────────────────────────────────────────

def create_table(conn):
    conn.execute("DROP TABLE IF EXISTS movies")
    conn.execute("DROP TABLE IF EXISTS genre_budget_thresholds")

    conn.execute("""
        CREATE TABLE movies (
            movie_id        INTEGER PRIMARY KEY,
            title           TEXT NOT NULL,
            genre           TEXT NOT NULL,
            budget_tier     TEXT NOT NULL,
            budget_usd      INTEGER NOT NULL,
            release_date    DATE NOT NULL,
            runtime_min     INTEGER NOT NULL,
            is_franchise    INTEGER NOT NULL,
            vote_average    REAL,
            vote_count      INTEGER,
            popularity      REAL
        )
    """)

    conn.execute("""
        CREATE TABLE genre_budget_thresholds (
            genre           TEXT PRIMARY KEY,
            low_max         INTEGER,
            mid_max         INTEGER,
            high_min        INTEGER,
            sample_size     INTEGER
        )
    """)
    conn.commit()

# ── TMDB Helpers ──────────────────────────────────────────────────────────────

def discover_page(genre_id: int, year: int, page: int) -> dict:
    r = requests.get(f"{BASE_URL}/discover/movie", params={
        "api_key":              API_KEY,
        "with_genres":          genre_id,
        "primary_release_year": year,
        "with_origin_country":  "US",
        "sort_by":              "popularity.desc",
        "vote_count.gte":       50,
        "page":                 page,
    })
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def get_details(movie_id: int) -> dict:
    r = requests.get(f"{BASE_URL}/movie/{movie_id}", params={"api_key": API_KEY})
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def is_complete(details: dict) -> bool:
    """Returns True only if budget, runtime, and release_date are all present."""
    return bool(
        details.get("budget")      and
        details.get("runtime")     and
        details.get("release_date")
    )


def get_primary_genre(details: dict) -> str | None:
    """
    Returns the PRIMARY genre of a movie — defined as the first genre
    listed by TMDB in the genres array — but only if it matches one of
    our five target genres. Returns None if the primary genre is outside
    our target set, disqualifying the movie entirely.
    """
    genres = details.get("genres", [])
    if not genres:
        return None

    primary_id = genres[0]["id"]
    return GENRE_ID_TO_NAME.get(primary_id, None)

# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_genre(genre_name: str, genre_id: int,
                    global_seen: set) -> dict:
    """
    Pulls up to 100 known-budget films for a genre to establish
    genre-specific budget percentiles. Respects global_seen to avoid
    counting duplicates in calibration.
    """
    print(f"\n  Calibrating {genre_name}...")
    budgets = []

    for year in VALID_YEARS:
        page = 1
        while len(budgets) < 100:
            data  = discover_page(genre_id, year, page)
            stubs = data.get("results", [])
            if not stubs:
                break

            for stub in stubs:
                if stub["id"] in global_seen:
                    continue
                details = get_details(stub["id"])
                if not is_complete(details):
                    continue
                primary = get_primary_genre(details)
                if primary != genre_name:
                    continue
                budgets.append(details["budget"])

            if page >= data.get("total_pages", 1) or page >= 5:
                break
            page += 1

    if len(budgets) < 10:
        print(f"  ⚠ Too few samples for {genre_name} — defaulting to absolute tiers")
        return {
            "low_max":     30_000_000,
            "mid_max":     80_000_000,
            "high_min":    80_000_000,
            "sample_size": len(budgets),
        }

    low_max  = int(np.percentile(budgets, 33))
    mid_max  = int(np.percentile(budgets, 67))
    high_min = mid_max

    print(f"    Sample size : {len(budgets)} films")
    print(f"    Low  budget : < ${low_max:,}")
    print(f"    Mid  budget : ${low_max:,} – ${mid_max:,}")
    print(f"    High budget : > ${high_min:,}")

    return {
        "low_max":     low_max,
        "mid_max":     mid_max,
        "high_min":    high_min,
        "sample_size": len(budgets),
    }

# ── Extraction ────────────────────────────────────────────────────────────────

TARGETS_PER_YEAR = {
    "high": 3,
    "mid":  3,
    "low":  2,
}

def fetch_tier(genre_name: str, genre_id: int, tier_name: str,
               thresholds: dict, global_seen: set) -> list:
    """
    Fetches qualifying movies for a specific genre/tier combination.
    Enforces a per-year cap to avoid older years crowding out newer ones.

    Enforces:
        - Per-year target cap (TARGETS_PER_YEAR)
        - No duplicates via global_seen
        - Primary genre must match target genre
        - All fields must be complete
        - Budget must fall within genre-relative tier range
    """
    total_target    = TARGETS[tier_name]
    per_year_target = TARGETS_PER_YEAR[tier_name]
    collected       = []
    year_counts     = {year: 0 for year in VALID_YEARS}

    budget_range = {
        "high": (thresholds["high_min"], float("inf")),
        "mid":  (thresholds["low_max"],  thresholds["mid_max"]),
        "low":  (1,                      thresholds["low_max"]),
    }[tier_name]

    for year in VALID_YEARS:
        if len(collected) >= total_target:
            break

        page = 1
        while year_counts[year] < per_year_target and len(collected) < total_target:
            data  = discover_page(genre_id, year, page)
            stubs = data.get("results", [])
            if not stubs:
                break

            for stub in stubs:
                if year_counts[year] >= per_year_target:
                    break
                if len(collected) >= total_target:
                    break
                if stub["id"] in global_seen:
                    continue

                details = get_details(stub["id"])

                if not is_complete(details):
                    global_seen.add(stub["id"])
                    continue

                primary = get_primary_genre(details)
                if primary != genre_name:
                    continue

                budget = details["budget"]
                if not (budget_range[0] <= budget < budget_range[1]):
                    continue

                global_seen.add(details["id"])
                year_counts[year] += 1
                collected.append((
                    details["id"],
                    details["title"],
                    genre_name,
                    tier_name,
                    budget,
                    details["release_date"],
                    details["runtime"],
                    1 if details.get("belongs_to_collection") else 0,
                    details.get("vote_average"),
                    details.get("vote_count"),
                    details.get("popularity"),
                ))
                print(f"    ✓ {details['title']} ({year}) | ${budget:,} | {details['runtime']} min")

            if page >= data.get("total_pages", 1) or page >= 15:
                break
            page += 1

    if len(collected) < total_target:
        print(f"  ⚠ Only found {len(collected)}/{total_target} — fill remaining manually")

    return collected[:total_target]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # global set shared across ALL genres and tiers
    # any movie_id added here will never be inserted twice
    global_seen = set()

    with sqlite3.connect(DB_PATH) as conn:
        create_table(conn)
        total_inserted = 0

        for genre_name, genre_id in GENRE_IDS.items():
            print(f"\n{'='*50}")
            print(f" GENRE: {genre_name.upper()}")
            print(f"{'='*50}")

            thresholds = calibrate_genre(genre_name, genre_id, global_seen)

            conn.execute("""
                INSERT OR REPLACE INTO genre_budget_thresholds
                VALUES (?, ?, ?, ?, ?)
            """, (
                genre_name,
                thresholds["low_max"],
                thresholds["mid_max"],
                thresholds["high_min"],
                thresholds["sample_size"],
            ))
            conn.commit()

            for tier_name in ["high", "mid", "low"]:
                print(f"\n  ── {tier_name.upper()} BUDGET ──")
                rows = fetch_tier(
                    genre_name, genre_id,
                    tier_name, thresholds,
                    global_seen
                )

                conn.executemany("""
                    INSERT OR IGNORE INTO movies
                    (movie_id, title, genre, budget_tier, budget_usd,
                     release_date, runtime_min, is_franchise,
                     vote_average, vote_count, popularity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)

                conn.commit()
                total_inserted += len(rows)
                print(f"  → {len(rows)}/{TARGETS[tier_name]} inserted")

        print(f"\n✓ Complete. {total_inserted} total movies inserted.")
        print("  Check genre_budget_thresholds table to review per-genre cutoffs.")


if __name__ == "__main__":
    main()
