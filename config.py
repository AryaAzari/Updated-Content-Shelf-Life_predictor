# =============================================================================
# config.py — Project-wide configuration
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# --- API credentials (set in .env) ---
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
WIKIMEDIA_USER_AGENT = os.environ.get(
    "WIKIMEDIA_USER_AGENT", "ShelfLifeProject/1.0 (your@email.com)"
)

# --- Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "shelflife.db")

# --- Trend death definition ---
TREND_DEATH_THRESHOLD = 0.20           # 20% of peak rolling average
TREND_DEATH_CONSECUTIVE_WEEKS = 2      # must stay below threshold for 2 weeks
ROLLING_WINDOW_DAYS = 7
OBSERVATION_WINDOW_WEEKS = 20

# Sensitivity thresholds to test (shows results are stable)
SENSITIVITY_THRESHOLDS = [0.15, 0.20, 0.25]

# --- Cox model ---
# Features used in Cox Proportional Hazards model
# (genre and release_season are one-hot encoded at fit time;
#  budget_usd is log-transformed to log_budget before fitting)
COX_COVARIATES = [
    "budget_usd",
    "runtime",
    "early_velocity",
    "is_franchise",
    # one-hot columns for genre and release_season are added dynamically
]

# --- Promotion recommender ---
SURVIVAL_PROMOTE_THRESHOLD = 0.60      # survival prob must be above this to promote
PROMOTE_SOON_WINDOW_WEEKS = 3          # "Promote Soon" if window opens within 3 weeks
