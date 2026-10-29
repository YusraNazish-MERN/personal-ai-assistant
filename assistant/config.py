"""All settings in one place. Edit the numbers here to tune the assistant."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- Azure settings (from .env) ----
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# ---- Files ----
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "assistant.db"
PROFILE_PATH = DATA_DIR / "profile.json"
PROFILE_EXAMPLE_PATH = DATA_DIR / "profile.example.json"
SEED_PATH = DATA_DIR / "seed_memories.json"
SEED_EXAMPLE_PATH = DATA_DIR / "seed_memories.example.json"

# ---- Model behavior ----
TEMPERATURE = 0.7
MAX_TOKENS = 800

# ---- Memory and history behavior ----
RECENT_MESSAGES = 12        # after summarizing, this many latest messages stay verbatim
SUMMARY_TRIGGER = 30        # when a chat has more unsummarized messages than this, fold the oldest into a summary
MAX_MEMORIES = 8            # max memories injected into one prompt
MAX_ALWAYS_ON = 5           # of those, max identity/preference/learning-style memories
STALE_DAYS = 60             # project/goal memories older than this are flagged "may be outdated"
MAX_PROPOSALS_PER_TURN = 3  # max memory suggestions created from a single message
TIMEZONE = "Asia/Karachi"

CATEGORIES = ["identity", "preference", "learning_style", "project", "goal", "other"]
ALWAYS_ON = ("identity", "preference", "learning_style")
EXPIRING = ("project", "goal")


def missing_settings():
    """Return the names of any .env settings that are empty."""
    required = {
        "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
        "AZURE_OPENAI_API_KEY": API_KEY,
        "AZURE_OPENAI_DEPLOYMENT": DEPLOYMENT,
        "AZURE_OPENAI_API_VERSION": API_VERSION,
    }
    return [name for name, value in required.items() if not value]
