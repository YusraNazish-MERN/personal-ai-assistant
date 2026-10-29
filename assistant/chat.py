"""One conversation turn, in three steps the UI calls in order:

    turn  = prepare_turn(conv_id, text)         # build the context (no model call)
    reply = stream_turn(turn, llm)              # stream the answer
    finish_turn(turn, reply, llm, extractor)    # save messages, summarize, suggest memories

Anything that goes wrong in the background work (summary, memory suggestions)
is swallowed on purpose: it must never break the chat itself.
"""
from langchain_core.output_parsers import StrOutputParser

from . import config, context, db, memory, prompts


def prepare_turn(conv_id, user_text):
    return context.build_turn(conv_id, user_text)


def stream_turn(turn, llm):
    """Yield the reply text piece by piece."""
    chain = prompts.CHAT_PROMPT | llm
    for chunk in chain.stream(turn.inputs):
        if chunk.content:
            yield chunk.content


# ----------------------------------------------------------------- summaries
def _transcript(rows):
    return "\n".join(
        f"{'User' if r['role'] == 'user' else 'Assistant'}: {r['content'][:1500]}" for r in rows
    )


def summarize(llm, existing_summary, rows):
    chain = prompts.SUMMARY_PROMPT | llm | StrOutputParser()
    text = chain.invoke(
        {"existing_summary": existing_summary or "(none)", "transcript": _transcript(rows)}
    )
    return text.strip()


def maybe_summarize(conv_id, llm):
    """If the chat got long, fold the oldest messages into the stored summary."""
    conv = db.get_conversation(conv_id)
    rows = db.get_messages(conv_id, after_id=conv["summarized_upto"])
    if len(rows) <= config.SUMMARY_TRIGGER:
        return False
    to_fold = rows[: -config.RECENT_MESSAGES]
    new_summary = summarize(llm, conv["summary"], to_fold)
    if not new_summary:
        return False
    db.set_summary(conv_id, new_summary, to_fold[-1]["id"])
    return True


def make_recap(conv_id, llm):
    """Store a short recap of a whole chat so the NEXT chat can continue from it."""
    conv = db.get_conversation(conv_id)
    if db.count_messages(conv_id) < 4:
        return False
    all_rows = db.get_messages(conv_id)
    if conv["recap_upto"] == all_rows[-1]["id"]:
        return False  # recap is already up to date
    unsummarized = db.get_messages(conv_id, after_id=conv["summarized_upto"])
    recap = summarize(llm, conv["summary"], unsummarized)
    if not recap:
        return False
    db.set_recap(conv_id, recap, all_rows[-1]["id"])
    return True


def start_new_chat(old_conv_id, llm):
    """Recap the old chat (best effort), then open a fresh one."""
    if db.count_messages(old_conv_id) == 0:
        return old_conv_id  # current chat is empty; reuse it
    try:
        make_recap(old_conv_id, llm)
    except Exception as e:
        print(f"[recap skipped] {e}")
    return db.create_conversation()


# ------------------------------------------------------------------ finishing
def finish_turn(turn, reply, llm, extractor_llm=None, learn=True):
    """Save the exchange, then do background work. Returns what happened."""
    outcome = {"saved": False, "proposals": 0, "summarized": False}
    if not (reply or "").strip():
        return outcome

    db.add_message(turn.conv_id, "user", turn.user_text)
    db.add_message(turn.conv_id, "assistant", reply)
    outcome["saved"] = True

    conv = db.get_conversation(turn.conv_id)
    if conv["title"] == "New chat":
        db.rename_conversation(turn.conv_id, turn.user_text.strip().replace("\n", " ")[:40])

    if learn and memory.should_extract(turn.user_text):
        try:
            outcome["proposals"] = memory.extract_proposals(
                extractor_llm or llm,
                turn.user_text,
                turn.last_assistant,
                turn.inputs["user_name"],
            )
        except Exception as e:
            print(f"[memory suggestions skipped] {e}")

    try:
        outcome["summarized"] = maybe_summarize(turn.conv_id, extractor_llm or llm)
    except Exception as e:
        print(f"[summary skipped] {e}")

    return outcome
