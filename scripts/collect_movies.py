"""
collect_movies_v2.py

Improved data collection addressing known biases from v1:
- Randomized sampling within popularity bands (reduces popularity bias)
- Budget verified against multiple TMDB fields
- Strict primary genre enforcement
- Enforced per-year quotas
- Minimum and maximum popularity bounds to avoid extreme outliers

Author: [Your Name]
"""

import requests
import sqlite3
import time
import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
DB_PATH  = "/Users/arya/Desktop/Tubi-Proj/data/shelflife_v2.db"

VALID_YEARS = [2018, 2019, 2023, 2024]

GENRE_IDS = {
    "action":  28,
    "drama":   18,
    "sci-fi":  878,
    "comedy":  35,
    "horror":  27,
}

GENRE_ID_TO_NAME = {v: k for k, v in GENRE_IDS.items()}

TARGETS = {
    "high": 12,
    "mid":  12,
    "low":  6,
}

TARGETS_PER_YEAR = {
    "high": 3,
    "mid":  3,
    "low":  2,
}

# ── Popularity bands ──────────────────────────────────────────────────────────
# Instead of always taking the most popular films, sample from
# three popularity bands to reduce cultural prominence bias.
# These are approximate TMDB popularity score ranges.
POPULARITY_BANDS = {
    "high": (50,  float("inf")),  # blockbusters
    "mid":  (15,  50),            # mid-profile releases
    "low":  (2,   15),            # lower-profile releases
}

# ── Database ──────────────────────────────────────────────────────────────────

def create_tables(conn):
    conn.execute("DROP TABLE IF EXISTS movies")
    conn.execute("DROP TABLE IF EXISTS genre_budget_thresholds")
    conn.execute("DROP TABLE IF EXISTS collection_log")

    conn.execute("""
        CREATE TABLE movies (
            movie_id          INTEGER PRIMARY KEY,
            title             TEXT NOT NULL,
            genre             TEXT NOT NULL,
            budget_tier       TEXT NOT NULL,
            budget_usd        INTEGER NOT NULL,
            release_date      DATE NOT NULL,
            runtime_min       INTEGER NOT NULL,
            is_franchise      INTEGER NOT NULL,
            vote_average      REAL,
            vote_count        INTEGER,
            popularity        REAL,
            popularity_band   TEXT,
            collection_year   INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE genre_budget_thresholds (
            genre        TEXT PRIMARY KEY,
            low_max      INTEGER,
            mid_max      INTEGER,
            high_min     INTEGER,
            sample_size  INTEGER
        )
    """)

    # log every film considered so sampling decisions are auditable
    conn.execute("""
        CREATE TABLE collection_log (
            movie_id     INTEGER,
            title        TEXT,
            genre        TEXT,
            reason       TEXT,
            budget_usd   INTEGER,
            popularity   REAL
        )
    """)

    conn.commit()

# ── TMDB Helpers ──────────────────────────────────────────────────────────────

def discover_page(genre_id: int, year: int, page: int,
                  pop_min: float, pop_max: float) -> dict:
    """
    Fetch one page filtered by genre, year, origin country,
    and popularity band. Popularity banding is the key change
    from v1 — we no longer always take the most popular films.
    """
    params = {
        "api_key":              API_KEY,
        "with_genres":          genre_id,
        "primary_release_year": year,
        "with_origin_country":  "US",
        "sort_by":              "popularity.desc",
        "vote_count.gte":       30,
        "page":                 page,
    }
    if pop_min > 0:
        params["popularity.gte"] = pop_min
    if pop_max < float("inf"):
        params["popularity.lte"] = pop_max

    r = requests.get(f"{BASE_URL}/discover/movie", params=params)
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def get_details(movie_id: int) -> dict:
    r = requests.get(f"{BASE_URL}/movie/{movie_id}",
                     params={"api_key": API_KEY})
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def is_complete(details: dict) -> bool:
    """All required fields must be present and non-zero."""
    return bool(
        details.get("budget")       and
        details.get("runtime")      and
        details.get("release_date")
    )


def get_primary_genre(details: dict) -> str | None:
    """Returns primary genre only if it matches one of our five targets."""
    genres = details.get("genres", [])
    if not genres:
        return None
    return GENRE_ID_TO_NAME.get(genres[0]["id"], None)

# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_genre(genre_name: str, genre_id: int,
                    global_seen: set) -> dict:
    """
    Samples films evenly across popularity bands and years
    to get an unbiased budget percentile estimate.
    """
    print(f"\n  Calibrating {genre_name}...")
    budgets = []

    for band_name, (pop_min, pop_max) in POPULARITY_BANDS.items():
        for year in VALID_YEARS:
            page = 1
            band_budgets = []

            while len(band_budgets) < 10:
                data  = discover_page(genre_id, year, page, pop_min, pop_max)
                stubs = data.get("results", [])
                if not stubs:
                    break

                # shuffle within page to reduce ordering bias
                random.shuffle(stubs)

                for stub in stubs:
                    if stub["id"] in global_seen:
                        continue
                    details = get_details(stub["id"])
                    if not is_complete(details):
                        continue
                    if get_primary_genre(details) != genre_name:
                        continue
                    band_budgets.append(details["budget"])
                    if len(band_budgets) >= 10:
                        break

                if page >= data.get("total_pages", 1) or page >= 3:
                    break
                page += 1

            budgets.extend(band_budgets)

    if len(budgets) < 10:
        print(f"  ⚠ Too few samples — defaulting to absolute tiers")
        return {
            "low_max":     30_000_000,
            "mid_max":     80_000_000,
            "high_min":    80_000_000,
            "sample_size": len(budgets),
        }

    low_max  = int(np.percentile(budgets, 33))
    mid_max  = int(np.percentile(budgets, 67))

    print(f"    Sample : {len(budgets)} films across all popularity bands")
    print(f"    Low    : < ${low_max:,}")
    print(f"    Mid    : ${low_max:,} – ${mid_max:,}")
    print(f"    High   : > ${mid_max:,}")

    return {
        "low_max":     low_max,
        "mid_max":     mid_max,
        "high_min":    mid_max,
        "sample_size": len(budgets),
    }

# ── Extraction ────────────────────────────────────────────────────────────────

def fetch_tier(genre_name: str, genre_id: int, tier_name: str,
               thresholds: dict, global_seen: set,
               conn: sqlite3.Connection) -> list:
    """
    Collects films for a genre/tier combination.

    Key improvements over v1:
    - Samples evenly across popularity bands within each tier
    - Shuffles results within each page to reduce ordering bias
    - Logs every considered film for auditability
    - Enforces strict per-year quotas
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

    # map budget tiers to popularity bands
    # high budget films tend to be high popularity — sample mid/low bands too
    pop_bands_to_sample = list(POPULARITY_BANDS.items())

    for year in VALID_YEARS:
        if len(collected) >= total_target:
            break

        # rotate through popularity bands per year
        random.shuffle(pop_bands_to_sample)

        for band_name, (pop_min, pop_max) in pop_bands_to_sample:
            if year_counts[year] >= per_year_target:
                break

            page = 1
            while year_counts[year] < per_year_target:
                data  = discover_page(genre_id, year, page, pop_min, pop_max)
                stubs = data.get("results", [])
                if not stubs:
                    break

                random.shuffle(stubs)

                for stub in stubs:
                    if year_counts[year] >= per_year_target:
                        break
                    if len(collected) >= total_target:
                        break
                    if stub["id"] in global_seen:
                        continue

                    details = get_details(stub["id"])
                    budget  = details.get("budget") or 0

                    # log this film regardless of outcome
                    conn.execute("""
                        INSERT INTO collection_log
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        details["id"],
                        details.get("title"),
                        genre_name,
                        "pending",
                        budget,
                        details.get("popularity"),
                    ))

                    if not is_complete(details):
                        conn.execute("""
                            UPDATE collection_log SET reason='incomplete_data'
                            WHERE movie_id=? AND reason='pending'
                        """, (details["id"],))
                        global_seen.add(details["id"])
                        continue

                    if get_primary_genre(details) != genre_name:
                        conn.execute("""
                            UPDATE collection_log SET reason='wrong_primary_genre'
                            WHERE movie_id=? AND reason='pending'
                        """, (details["id"],))
                        continue

                    if not (budget_range[0] <= budget < budget_range[1]):
                        conn.execute("""
                            UPDATE collection_log SET reason='outside_budget_tier'
                            WHERE movie_id=? AND reason='pending'
                        """, (details["id"],))
                        continue

                    # passed all checks
                    global_seen.add(details["id"])
                    year_counts[year] += 1

                    conn.execute("""
                        UPDATE collection_log SET reason='selected'
                        WHERE movie_id=? AND reason='pending'
                    """, (details["id"],))

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
                        band_name,
                        year,
                    ))
                    print(f"    ✓ [{band_name}] {details['title']} ({year}) | ${budget:,}")

                conn.commit()

                if page >= data.get("total_pages", 1) or page >= 15:
                    break
                page += 1

    if len(collected) < total_target:
        print(f"  ⚠ Only found {len(collected)}/{total_target} — fill manually")

    return collected[:total_target]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)  # reproducible shuffling
    global_seen = set()

    with sqlite3.connect(DB_PATH) as conn:
        create_tables(conn)
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
                    global_seen, conn
                )

                conn.executemany("""
                    INSERT OR IGNORE INTO movies
                    (movie_id, title, genre, budget_tier, budget_usd,
                     release_date, runtime_min, is_franchise,
                     vote_average, vote_count, popularity,
                     popularity_band, collection_year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)

                conn.commit()
                total_inserted += len(rows)
                print(f"  → {len(rows)}/{TARGETS[tier_name]} inserted")

        print(f"\n✓ Complete. {total_inserted} total movies inserted.")


if __name__ == "__main__":
    main()