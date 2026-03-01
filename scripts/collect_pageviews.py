"""
wiki_pageviews_rebuild.py

Completely rebuilds the pageviews table from scratch.
Drops and recreates the table, then pulls fresh data for every
movie currently in the movies table.

Range: 14 days before release → 210 days after release
Restrictions:
    - American movies only (already enforced by movies table)
    - Only inserts days with known pageview counts
    - Tries alternate Wikipedia titles if primary fails
    - Logs all movies that could not be found for manual follow-up

Author: [Your Name]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from config import DB_PATH, WIKIMEDIA_USER_AGENT

BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HEADERS  = {"User-Agent": WIKIMEDIA_USER_AGENT}

# ── Database ──────────────────────────────────────────────────────────────────

def rebuild_pageviews_table(conn):
    """Drops and recreates the pageviews table cleanly."""
    conn.execute("DROP TABLE IF EXISTS pageviews")
    conn.execute("""
        CREATE TABLE pageviews (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id            INTEGER NOT NULL,
            title               TEXT NOT NULL,
            date                DATE NOT NULL,
            pageviews           INTEGER NOT NULL,
            days_from_release   INTEGER NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
            UNIQUE(movie_id, date)
        )
    """)
    conn.commit()
    print("✓ Pageviews table rebuilt\n")

# ── Wikipedia Helpers ─────────────────────────────────────────────────────────

def format_wiki_title(title: str) -> str:
    return title.strip().replace(" ", "_")


def fetch_pageviews(title: str, start_date: datetime,
                    end_date: datetime) -> list:
    """
    Fetches daily pageview counts from Wikimedia API.
    Returns list of (date_str, views) tuples.
    Returns empty list if article not found or request fails.
    """
    wiki_title = format_wiki_title(title)
    start_str  = start_date.strftime("%Y%m%d")
    end_str    = end_date.strftime("%Y%m%d")
    url        = (
        f"{BASE_URL}/en.wikipedia/all-access/all-agents"
        f"/{wiki_title}/daily/{start_str}/{end_str}"
    )

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return [
            (item["timestamp"][:8], item["views"])
            for item in r.json().get("items", [])
        ]
    except requests.exceptions.RequestException as e:
        print(f"    ⚠ Request error: {e}")
        return []


def try_alternate_titles(title: str, year: int, start_date: datetime,
                         end_date: datetime) -> tuple[list, str]:
    """
    Tries common Wikipedia title variations if primary title 404s.
    Returns (results, successful_title) or ([], None) if all fail.
    """
    alternates = [
        f"{title} (film)",
        f"{title} ({year} film)",
        f"{title} (film series)",
    ]

    for alt in alternates:
        print(f"    → Trying: '{alt}'")
        results = fetch_pageviews(alt, start_date, end_date)
        if results:
            print(f"    ✓ Found under: '{alt}'")
            return results, alt

    return [], None

# ── Core Extraction ───────────────────────────────────────────────────────────

def pull_movie_pageviews(movie_id: int, title: str,
                         release_date: datetime) -> tuple[list, bool]:
    """
    Pulls pageview data for a single movie across its full window.
    Caps end date at today for recent releases.

    Returns:
        (rows ready for DB insertion, success bool)
    """
    start_date = release_date - timedelta(days=14)
    end_date   = min(
        release_date + timedelta(days=210),
        datetime.today()
    )

    print(f"\n[{title}]")
    print(f"  Window: {start_date.date()} → {end_date.date()}")

    # primary title attempt
    results      = fetch_pageviews(title, start_date, end_date)
    used_title   = title

    # fallback to alternate titles
    if not results:
        year = release_date.year
        print(f"  ⚠ Not found — trying alternates...")
        results, used_title = try_alternate_titles(title, year, start_date, end_date)

    if not results:
        print(f"  ✗ Could not find Wikipedia article — add to manual list")
        return [], False

    rows = []
    for date_str, views in results:
        date              = datetime.strptime(date_str, "%Y%m%d")
        days_from_release = (date - release_date).days
        rows.append((
            movie_id,
            used_title,
            date.strftime("%Y-%m-%d"),
            views,
            days_from_release,
        ))

    print(f"  ✓ {len(rows)} days pulled via '{used_title}'")
    return rows, True

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Rebuilding pageviews table from scratch...\n", flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        rebuild_pageviews_table(conn)

        movies = pd.read_sql_query("""
            SELECT movie_id, title, release_date
            FROM movies
            ORDER BY release_date
        """, conn)

        print(f"Found {len(movies)} movies to process\n")

        not_found      = []
        total_inserted = 0

        for _, row in movies.iterrows():
            movie_id     = int(row["movie_id"])
            title        = row["title"]
            release_date = datetime.strptime(row["release_date"], "%Y-%m-%d")

            rows, success = pull_movie_pageviews(movie_id, title, release_date)

            if not success:
                not_found.append((movie_id, title))
                continue

            conn.executemany("""
                INSERT OR IGNORE INTO pageviews
                (movie_id, title, date, pageviews, days_from_release)
                VALUES (?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            total_inserted += len(rows)

            time.sleep(0.5)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"✓ Done. {total_inserted} total rows inserted.")
        print(f"  {len(movies) - len(not_found)}/{len(movies)} movies successfully pulled.")

        if not_found:
            print(f"\n⚠ Could not find Wikipedia data for {len(not_found)} movies:")
            print(f"  Use newscrip.py with the correct Wikipedia title for each:\n")
            for mid, t in not_found:
                print(f"  movie_id={mid} | {t}")


if __name__ == "__main__":
    main()