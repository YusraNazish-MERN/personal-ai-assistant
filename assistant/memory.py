"""Long-term memory: suggestions from chat (never saved without your approval),
manual add/edit/delete, and safety checks against made-up memories."""
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from . import config, db, prompts

MAX_MEMORY_CHARS = 300


# ------------------------------------------------------------ structured output
class MemoryProposal(BaseModel):
    action: str = Field(description="One of: ADD, UPDATE, DELETE")
    text: str = Field(description="The fact as one short standalone sentence. For DELETE, repeat the memory text.")
    category: str = Field(description="One of: identity, preference, learning_style, project, goal, other")
    memory_id: Optional[int] = Field(default=None, description="Id of the existing memory for UPDATE or DELETE. Null for ADD.")
    evidence: str = Field(description="Exact words copied from the user's latest message (max 20 words).")
    reason: str = Field(description="One short sentence: why this is worth saving, updating or deleting.")


class ExtractionResult(BaseModel):
    proposals: List[MemoryProposal] = Field(default_factory=list)


# ------------------------------------------------------------------- validation
def normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def validate_proposals(proposals, user_message, active_memories):
    """Keep only proposals that are safe. This is the anti-hallucination gate.

    A proposal survives only if:
      - the action is ADD / UPDATE / DELETE
      - its 'evidence' is a real quote (3+ words) found inside the USER's message
      - UPDATE/DELETE point at a memory that really exists
      - ADD is not a duplicate of an existing memory
    """
    active = {m["id"]: m for m in active_memories}
    message_norm = normalize(user_message)
    existing_texts = {normalize(m["text"]) for m in active_memories}

    valid = []
    for p in proposals:
        action = (p.action or "").upper().strip()
        text = (p.text or "").strip()
        evidence = normalize(p.evidence)

        if action not in ("ADD", "UPDATE", "DELETE"):
            continue
        if len(evidence.split()) < 3 or evidence not in message_norm:
            continue
        if action in ("UPDATE", "DELETE") and p.memory_id not in active:
            continue
        if action in ("ADD", "UPDATE") and (not text or len(text) > MAX_MEMORY_CHARS):
            continue
        if action == "ADD" and normalize(text) in existing_texts:
            continue

        category = p.category if p.category in config.CATEGORIES else "other"
        if action == "DELETE":
            text = active[p.memory_id]["text"]
            category = active[p.memory_id]["category"]

        valid.append(
            {
                "action": action,
                "text": text,
                "category": category,
                "memory_id": p.memory_id if action != "ADD" else None,
                "evidence": (p.evidence or "").strip(),
                "reason": (p.reason or "").strip(),
            }
        )
        if len(valid) >= config.MAX_PROPOSALS_PER_TURN:
            break
    return valid


# ------------------------------------------------------------------- extraction
def should_extract(user_text):
    text = (user_text or "").strip()
    return len(text) >= 15 and not text.startswith("/")


def extract_proposals(extractor_llm, user_message, last_assistant, user_name):
    """Ask the model what might be worth remembering. Queues suggestions; returns how many."""
    active = db.list_memories()
    existing = "\n".join(f"{m['id']}: [{m['category']}] {m['text']}" for m in active) or "(none)"

    chain = prompts.EXTRACT_PROMPT | extractor_llm.with_structured_output(
        ExtractionResult, method="function_calling"
    )
    result = chain.invoke(
        {
            "user_name": user_name,
            "existing_memories": existing,
            "last_assistant": last_assistant or "(none)",
            "user_message": user_message,
        }
    )
    if result is None:
        return 0

    added = 0
    for item in validate_proposals(result.proposals, user_message, active):
        if db.add_proposal(**item):
            added += 1
    return added


# --------------------------------------------------------- approve / reject / manual
def approve(proposal_id):
    p = db.get_proposal(proposal_id)
    if not p:
        return
    if p["action"] == "ADD":
        db.add_memory(p["text"], p["category"], source="auto")
    elif p["action"] == "UPDATE":
        if db.get_memory(p["memory_id"]):
            db.update_memory(p["memory_id"], text=p["text"], category=p["category"])
        else:
            db.add_memory(p["text"], p["category"], source="auto")
    elif p["action"] == "DELETE":
        db.delete_memory(p["memory_id"])
    db.delete_proposal(proposal_id)


def reject(proposal_id):
    db.delete_proposal(proposal_id)


def add_manual(text, category="other"):
    text = (text or "").strip()
    if not text:
        raise ValueError("Memory text is empty.")
    if len(text) > MAX_MEMORY_CHARS:
        raise ValueError(f"Memory is too long ({len(text)} characters, max {MAX_MEMORY_CHARS}).")
    if category not in config.CATEGORIES:
        category = "other"
    return db.add_memory(text, category, source="manual")


def forget(memory_id):
    return db.delete_memory(memory_id)
