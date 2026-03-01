"""
patch_pageviews.py

One-off helper to manually fetch and insert pageview data for a single
movie that was missed or failed during the main collection run.

Edit the three variables at the top, then run:
    python scripts/patch_pageviews.py

Author: [Your Name]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import requests
import time
from datetime import datetime, timedelta
from config import DB_PATH, WIKIMEDIA_USER_AGENT

HEADERS = {"User-Agent": WIKIMEDIA_USER_AGENT}

# ── Edit these three values for each failed movie ──────────────────────────
MOVIE_ID     = 123                   # from your movies table
WIKI_TITLE   = "Terrifier"      # exact Wikipedia article title
RELEASE_DATE = "2018-01-25"         # from your movies table
# ──────────────────────────────────────────────────────────────────────────

def fetch_and_insert():
    release_date = datetime.strptime(RELEASE_DATE, "%Y-%m-%d")
    start_date   = release_date - timedelta(days=14)
    end_date     = min(release_date + timedelta(days=210), datetime.today())

    wiki_title = WIKI_TITLE.replace(" ", "_")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
        f"/en.wikipedia/all-access/all-agents/{wiki_title}/daily"
        f"/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"
    )

    r = requests.get(url, headers=HEADERS, timeout=10)
    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print("Failed — check the Wikipedia title is correct")
        return

    items = r.json().get("items", [])
    print(f"Found {len(items)} days of data")

    rows = []
    for item in items:
        date = datetime.strptime(item["timestamp"][:8], "%Y%m%d")
        days_from_release = (date - release_date).days
        rows.append((
            MOVIE_ID,
            WIKI_TITLE,
            date.strftime("%Y-%m-%d"),
            item["views"],
            days_from_release,
        ))

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO pageviews
            (movie_id, title, date, pageviews, days_from_release)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        print(f"✓ Inserted {len(rows)} rows for '{WIKI_TITLE}'")


if __name__ == "__main__":
    fetch_and_insert()
