"""Streamlit UI. All the real logic lives in the assistant/ package."""
import json

import streamlit as st

from assistant import chat, config, context, db, memory
from assistant.llm import get_extractor_llm, get_llm
from assistant.profile import load_profile

st.set_page_config(page_title="Personal Assistant", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------- startup checks
missing = config.missing_settings()
if missing:
    st.error(f"Missing settings in .env: {', '.join(missing)}")
    st.stop()

db.init_db()

try:
    profile = load_profile()
except ValueError as e:
    st.error(str(e))
    st.stop()

NAME = profile.get("preferred_name") or "there"


@st.cache_resource
def load_llm():
    return get_llm()


@st.cache_resource
def load_extractor():
    return get_extractor_llm()


llm = load_llm()
extractor = load_extractor()

# ------------------------------------------------------- which conversation is open
if "conv_id" not in st.session_state or db.get_conversation(st.session_state.conv_id) is None:
    st.session_state.conv_id = db.latest_conversation_id() or db.create_conversation()
conv_id = st.session_state.conv_id


def handle_command(text):
    """/remember <text>   and   /forget <memory id>. Returns a message, or None if not a command."""
    t = text.strip()
    low = t.lower()
    if low.startswith("/remember"):
        body = t[len("/remember"):].strip()
        if not body:
            return "Usage: /remember <something to remember>"
        try:
            memory.add_manual(body, "other")
        except ValueError as e:
            return str(e)
        return f"Saved to memory: {body}"
    if low.startswith("/forget"):
        arg = t[len("/forget"):].strip()
        if not arg.isdigit():
            return "Usage: /forget <memory id>  (ids are shown in the sidebar)"
        if memory.forget(int(arg)):
            return f"Forgot memory #{arg}."
        return f"No memory with id {arg}."
    return None


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title(f"🧠 {NAME}'s assistant")

    c1, c2 = st.columns(2)
    if c1.button("➕ New chat", use_container_width=True):
        with st.spinner("Saving a recap of this chat..."):
            st.session_state.conv_id = chat.start_new_chat(conv_id, extractor)
        st.rerun()
    if c2.button("🗑 Delete chat", use_container_width=True):
        db.delete_conversation(conv_id)
        st.session_state.conv_id = db.latest_conversation_id() or db.create_conversation()
        st.rerun()

    convs = db.list_conversations(15)
    labels = {c["id"]: f"{c['title']}  ({c['updated_at'][:10]})" for c in convs}
    ids = list(labels)
    if conv_id in ids:
        picked = st.selectbox("Conversation", ids, index=ids.index(conv_id), format_func=labels.get)
        if picked != conv_id:
            st.session_state.conv_id = picked
            st.rerun()

    learn = st.toggle("Suggest memories from my messages", value=True,
                      help="Costs one extra small model call per message.")
    show_context = st.toggle("Show context sent to the model")

    # ---- pending suggestions
    pending = db.list_proposals()
    st.subheader(f"Suggested memories ({len(pending)})")
    if not pending:
        st.caption("Nothing waiting. Suggestions appear here and are saved only if you approve.")
    for p in pending:
        with st.container(border=True):
            label = {"ADD": "➕ Add", "UPDATE": "✏️ Update", "DELETE": "🗑 Forget"}[p["action"]]
            st.markdown(f"**{label}** · {p['category']}")
            st.write(p["text"])
            if p["action"] in ("UPDATE", "DELETE"):
                old = db.get_memory(p["memory_id"])
                if old:
                    st.caption(f"Current memory #{old['id']}: {old['text']}")
            st.caption(f"You said: “{p['evidence']}”")
            a, b = st.columns(2)
            if a.button("✅ Approve", key=f"ok_{p['id']}", use_container_width=True):
                memory.approve(p["id"])
                st.rerun()
            if b.button("❌ Reject", key=f"no_{p['id']}", use_container_width=True):
                memory.reject(p["id"])
                st.rerun()

    # ---- saved memories
    mems = db.list_memories()
    stale_count = sum(1 for m in mems if context.is_stale(m))
    st.subheader(f"Memories ({len(mems)})")
    if stale_count:
        st.caption(f"⚠️ {stale_count} project/goal memories are older than {config.STALE_DAYS} days.")
    for m in mems:
        with st.container(border=True):
            stale = context.is_stale(m)
            st.markdown(f"**#{m['id']}** · {m['category']} · {m['updated_at'][:10]}" + (" ⚠️ old" if stale else ""))
            st.write(m["text"])
            cols = st.columns(3 if stale else 2)
            with cols[0].popover("Edit"):
                new_text = st.text_area("Text", value=m["text"], key=f"t_{m['id']}")
                new_cat = st.selectbox("Category", config.CATEGORIES,
                                       index=config.CATEGORIES.index(m["category"]) if m["category"] in config.CATEGORIES else 0,
                                       key=f"c_{m['id']}")
                if st.button("Save", key=f"s_{m['id']}"):
                    if new_text.strip():
                        db.update_memory(m["id"], text=new_text.strip(), category=new_cat)
                    st.rerun()
            if cols[1].button("Delete", key=f"d_{m['id']}"):
                memory.forget(m["id"])
                st.rerun()
            if stale and cols[2].button("Still true", key=f"k_{m['id']}"):
                db.touch_memory(m["id"])
                st.rerun()

    with st.form("add_memory", clear_on_submit=True):
        st.markdown("**Add a memory**")
        new_cat = st.selectbox("Category", config.CATEGORIES, index=config.CATEGORIES.index("other"))
        new_text = st.text_input("Fact", placeholder=f"{NAME} prefers ...")
        if st.form_submit_button("Add"):
            try:
                memory.add_manual(new_text, new_cat)
            except ValueError as e:
                st.session_state.notice = str(e)
            st.rerun()

    st.download_button(
        "⬇️ Export memories (JSON)",
        data=json.dumps(db.export_all(), indent=2, ensure_ascii=False),
        file_name="memories_backup.json",
        mime="application/json",
        use_container_width=True,
    )

# --------------------------------------------------------------------- main area
st.title(f"💬 Hi {NAME}")

notice = st.session_state.pop("notice", None)
if notice:
    st.info(notice)

messages = db.get_messages(conv_id)
if not messages:
    st.caption("Ask me anything. I use your profile and saved memories, and I'll say so when I don't know something about you.")
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if show_context and "last_system" in st.session_state:
    with st.expander("Context sent to the model on the last message", expanded=False):
        st.code(st.session_state.last_system, language="text")

prompt = st.chat_input("Message your assistant... (/remember <text>, /forget <id>)")

if prompt:
    command_result = handle_command(prompt)
    if command_result is not None:
        st.session_state.notice = command_result
        st.rerun()

    with st.chat_message("user"):
        st.markdown(prompt)

    turn = chat.prepare_turn(conv_id, prompt)
    st.session_state.last_system = turn.system_preview

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(chat.stream_turn(turn, llm))
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    with st.spinner("Updating memory..."):
        chat.finish_turn(turn, reply, llm, extractor, learn=learn)
    st.rerun()
