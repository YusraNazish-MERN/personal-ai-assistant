"""Builds exactly what the model sees on each turn.

Context = system prompt + profile + a few RELEVANT memories (with dates) +
          recap of the previous chat + summary of older messages in this chat +
          the recent messages + the new message.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from . import config, db, prompts
from .profile import load_profile, profile_to_text

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "that", "this", "have",
    "has", "was", "were", "what", "when", "where", "which", "who", "how", "why", "can", "could",
    "should", "would", "will", "about", "from", "into", "than", "then", "them", "they", "there",
    "its", "our", "out", "all", "any", "get", "got", "let", "use", "using", "want", "need", "like",
    "just", "some", "more", "also", "give", "make", "help", "please", "tell", "show",
}


@dataclass
class Turn:
    conv_id: int
    user_text: str
    inputs: dict          # variables for the chat prompt
    memory_ids: list      # which memories were injected
    last_assistant: str   # previous assistant message (context for memory extraction)
    system_preview: str   # exact system text, for the debug panel


# ------------------------------------------------------------------ helpers
def stem(word):
    if word.endswith("ss"):
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def keywords(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS}


def age_days(iso, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        then = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return 0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0, (now - then).days)


def is_stale(memory, now=None):
    """Projects and goals change quickly, so old ones must be re-confirmed."""
    return memory["category"] in config.EXPIRING and age_days(memory["updated_at"], now) > config.STALE_DAYS


def today_string():
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(config.TIMEZONE))
    except Exception:  # tzdata missing: fall back to the computer's local time
        now = datetime.now()
    return now.strftime("%A, %d %B %Y")


# ------------------------------------------------------------ memory selection
def select_memories(user_text, previous_user_text, memories):
    """Pick at most MAX_MEMORIES memories that are worth showing the model right now."""
    always = [m for m in memories if m["category"] in config.ALWAYS_ON]
    others = [m for m in memories if m["category"] not in config.ALWAYS_ON]
    always = always[: config.MAX_ALWAYS_ON]  # list is already newest-first

    query = keywords(user_text) | keywords(previous_user_text)
    scored = []
    for m in others:
        overlap = len(query & keywords(m["text"]))
        if overlap:
            scored.append((overlap, m["updated_at"], m["id"], m))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    matched = [t[3] for t in scored]

    if not matched:
        # Nothing matched: show the 2 most recently confirmed projects/goals as background.
        matched = sorted(others, key=lambda m: (m["updated_at"], m["id"]), reverse=True)[:2]

    room = max(0, config.MAX_MEMORIES - len(always))
    return always + matched[:room]


def format_memories(selected, now=None):
    if not selected:
        return "(no saved memories relevant to this message)"
    lines = []
    for m in selected:
        line = f"- [{m['category']}, last confirmed {m['updated_at'][:10]}] {m['text']}"
        if is_stale(m, now):
            days = age_days(m["updated_at"], now)
            line += f"  (MAY BE OUTDATED: {days} days old. Confirm before relying on it.)"
        lines.append(line)
    return "\n".join(lines)


def to_lc_messages(rows):
    return [HumanMessage(content=r["content"]) if r["role"] == "user" else AIMessage(content=r["content"])
            for r in rows]


# ------------------------------------------------------------------- main entry
def build_turn(conv_id, user_text):
    profile = load_profile()
    user_name = profile.get("preferred_name") or "the user"

    conv = db.get_conversation(conv_id)
    rows = db.get_messages(conv_id, after_id=conv["summarized_upto"])
    total = db.count_messages(conv_id)

    previous_user = next((r["content"] for r in reversed(rows) if r["role"] == "user"), "")
    last_assistant = next((r["content"] for r in reversed(rows) if r["role"] == "assistant"), "")

    selected = select_memories(user_text, previous_user, db.list_memories())

    # Continuity across chats: at the start of a new chat, include the recap of the last one.
    recap = "(none)"
    if total < 10:
        other = db.latest_recap_excluding(conv_id)
        if other:
            recap = f"From the chat '{other['title']}' ({other['updated_at'][:10]}):\n{other['recap']}"

    inputs = {
        "user_name": user_name,
        "today": today_string(),
        "profile": profile_to_text(profile),
        "memories": format_memories(selected),
        "recap": recap,
        "earlier": conv["summary"] or "(nothing summarized yet)",
        "history": to_lc_messages(rows),
        "input": user_text,
    }
    return Turn(
        conv_id=conv_id,
        user_text=user_text,
        inputs=inputs,
        memory_ids=[m["id"] for m in selected],
        last_assistant=last_assistant,
        system_preview=prompts.render_system(inputs),
    )
