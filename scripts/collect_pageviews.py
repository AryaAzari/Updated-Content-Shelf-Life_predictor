"""
collect_pageviews.py — Wikipedia pageview collection

Pulls daily Wikipedia pageview data for all movies in the database.
Range: 14 days before release → 210 days after release.

Tries primary title first, then common Wikipedia alternates.
Logs all not-found titles for manual follow-up.

Written by Arya Azari with help from Claude Sonnet 4.6
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
HEADERS  = {
    "User-Agent": WIKIMEDIA_USER_AGENT
}

# ── Database ──────────────────────────────────────────────────────────────────

def create_pageviews_table(conn):
    conn.execute("DROP TABLE IF EXISTS pageviews")
    conn.execute("""
        CREATE TABLE pageviews (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id            INTEGER NOT NULL,
            title               TEXT NOT NULL,
            wiki_title          TEXT NOT NULL,
            date                DATE NOT NULL,
            pageviews           INTEGER NOT NULL,
            days_from_release   INTEGER NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
            UNIQUE(movie_id, date)
        )
    """)

    # log movies that could not be found for manual follow-up
    conn.execute("DROP TABLE IF EXISTS pageview_failures")
    conn.execute("""
        CREATE TABLE pageview_failures (
            movie_id     INTEGER PRIMARY KEY,
            title        TEXT,
            release_date TEXT,
            reason       TEXT
        )
    """)
    conn.commit()
    print("✓ Pageviews table created\n")

# ── Wikipedia Helpers ─────────────────────────────────────────────────────────

def format_wiki_title(title: str) -> str:
    return title.strip().replace(" ", "_")


def fetch_pageviews(wiki_title: str, start_date: datetime,
                    end_date: datetime) -> list:
    """
    Fetches daily pageview counts from Wikimedia API.
    Returns list of (date_str, views) tuples.
    Returns empty list if article not found.
    """
    url = (
        f"{BASE_URL}/en.wikipedia/all-access/all-agents"
        f"/{format_wiki_title(wiki_title)}/daily"
        f"/{start_date.strftime('%Y%m%d')}"
        f"/{end_date.strftime('%Y%m%d')}"
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


def try_alternate_titles(title: str, year: int,
                         start_date: datetime,
                         end_date: datetime) -> tuple:
    """
    Tries common Wikipedia title variations if primary title 404s.
    Returns (results, successful_wiki_title) or ([], None).
    """
    alternates = [
        f"{title} (film)",
        f"{title} ({year} film)",
        f"{title} (film series)",
        f"The {title}",
    ]

    for alt in alternates:
        print(f"    → Trying: '{alt}'")
        results = fetch_pageviews(alt, start_date, end_date)
        if results:
            print(f"    ✓ Found under: '{alt}'")
            return results, alt

    return [], None

# ── Core Pull ─────────────────────────────────────────────────────────────────

def pull_movie(movie_id: int, title: str,
               release_date: datetime) -> tuple:
    """
    Pulls full pageview window for a single movie.
    Window: 14 days before release → 210 days after release.
    Caps end date at today for recent releases.

    Returns:
        (rows, wiki_title_used, success_bool)
    """
    start_date = release_date - timedelta(days=14)
    end_date   = min(
        release_date + timedelta(days=210),
        datetime.today()
    )

    print(f"\n[{title}]")
    print(f"  Window: {start_date.date()} → {end_date.date()}")

    # primary attempt
    results    = fetch_pageviews(title, start_date, end_date)
    wiki_title = title

    # fallback to alternates
    if not results:
        print(f"  ⚠ Not found — trying alternates...")
        results, wiki_title = try_alternate_titles(
            title, release_date.year, start_date, end_date
        )

    if not results:
        print(f"  ✗ Could not find Wikipedia article")
        return [], None, False

    rows = []
    for date_str, views in results:
        date              = datetime.strptime(date_str, "%Y%m%d")
        days_from_release = (date - release_date).days
        rows.append((
            movie_id,
            title,
            wiki_title,
            date.strftime("%Y-%m-%d"),
            views,
            days_from_release,
        ))

    print(f"  ✓ {len(rows)} days pulled via '{wiki_title}'")
    return rows, wiki_title, True

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting Wikipedia pageview collection for shelflife_v2.db\n",
          flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        create_pageviews_table(conn)

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

            rows, wiki_title, success = pull_movie(
                movie_id, title, release_date
            )

            if not success:
                not_found.append((movie_id, title))
                conn.execute("""
                    INSERT OR IGNORE INTO pageview_failures
                    VALUES (?, ?, ?, ?)
                """, (
                    movie_id,
                    title,
                    row["release_date"],
                    "wikipedia_article_not_found",
                ))
                conn.commit()
                continue

            conn.executemany("""
                INSERT OR IGNORE INTO pageviews
                (movie_id, title, wiki_title, date,
                 pageviews, days_from_release)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

            total_inserted += len(rows)
            time.sleep(0.5)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"✓ Done. {total_inserted} total rows inserted.")
        print(f"  {len(movies) - len(not_found)}/{len(movies)} movies pulled successfully.")

        if not_found:
            print(f"\n⚠ Could not find Wikipedia data for "
                  f"{len(not_found)} movies:")
            print(f"  Use newscrip.py with the correct Wikipedia "
                  f"title for each:\n")
            for mid, t in not_found:
                print(f"  movie_id={mid} | {t}")
            print(f"\n  These are also logged in the "
                  f"pageview_failures table.")


if __name__ == "__main__":
    main()