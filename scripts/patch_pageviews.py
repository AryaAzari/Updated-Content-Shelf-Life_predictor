"""
patch_pageviews.py — Manual single-movie pageview patch

One-off helper to manually fetch and insert pageview data for a single
movie that was missed or failed during the main collection run.

Edit the three variables at the top, then run:
    python scripts/patch_pageviews.py

Written by Arya Azari with help from Claude Sonnet 4.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import requests
from datetime import datetime, timedelta
from config import DB_PATH, WIKIMEDIA_USER_AGENT

HEADERS = {"User-Agent": WIKIMEDIA_USER_AGENT}

# ── Edit these three values for each failed movie ─────────────────────────────
MOVIE_ID     = 555285
WIKI_TITLE   = "Are You There God? It's Me, Margaret. (film)"
RELEASE_DATE = "2023-03-29"

WIKI_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Are_You_There_God%3F_It%27s_Me%2C_Margaret._(film)/daily/20230414/20231024"
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_insert():
    release_date = datetime.strptime(RELEASE_DATE, "%Y-%m-%d")
    start_date   = release_date - timedelta(days=14)
    end_date     = min(release_date + timedelta(days=210), datetime.today())

    wiki_title_formatted = WIKI_TITLE.replace(" ", "_")

    if WIKI_URL:
        url = WIKI_URL
        print(f"Using hardcoded URL")
    else:
        wiki_title_formatted = WIKI_TITLE.replace(" ", "_")
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
            f"/en.wikipedia/all-access/all-agents/{wiki_title_formatted}/daily"
            f"/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"
        )

    print(f"Fetching: {WIKI_TITLE}")
    print(f"Window:   {start_date.date()} → {end_date.date()}")

    r = requests.get(url, headers=HEADERS, timeout=10)
    print(f"Status:   {r.status_code}")

    if r.status_code != 200:
        print("Failed — check the Wikipedia title is correct")
        print(f"Attempted URL: {url}")
        return

    items = r.json().get("items", [])
    print(f"Found {len(items)} days of data")

    if not items:
        print("No pageview data returned — article may exist but have no views in this window")
        return

    rows = []
    for item in items:
        date              = datetime.strptime(item["timestamp"][:8], "%Y%m%d")
        days_from_release = (date - release_date).days
        rows.append((
            MOVIE_ID,
            WIKI_TITLE,
            date.strftime("%Y-%m-%d"),
            item["views"],
            days_from_release,
        ))

    with sqlite3.connect(DB_PATH) as conn:

        # verify movie_id exists in movies table before inserting
        cursor = conn.execute(
            "SELECT title FROM movies WHERE movie_id = ?", (MOVIE_ID,)
        )
        match = cursor.fetchone()
        if not match:
            print(f"⚠ movie_id {MOVIE_ID} not found in movies table — aborting")
            return

        print(f"  Confirmed movie: '{match[0]}'")

        conn.executemany("""
            INSERT OR IGNORE INTO pageviews
            (movie_id, title, date, pageviews, days_from_release)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

        # confirm how many rows were actually inserted
        cursor = conn.execute(
            "SELECT COUNT(*) FROM pageviews WHERE movie_id = ?", (MOVIE_ID,)
        )
        total = cursor.fetchone()[0]

        print(f"✓ Inserted {len(rows)} rows for '{WIKI_TITLE}'")
        print(f"  Total pageview rows for this movie: {total}")

        # remove from failures table if it was logged there
        conn.execute(
            "DELETE FROM pageview_failures WHERE movie_id = ?", (MOVIE_ID,)
        )
        conn.commit()
        print(f"  Removed from pageview_failures table if present")


if __name__ == "__main__":
    fetch_and_insert()