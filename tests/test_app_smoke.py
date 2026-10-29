"""Runs the real Streamlit app headlessly with a fake model (no Azure needed).

Run from the project folder:   python -m tests.test_app_smoke
"""
import os
import tempfile
from pathlib import Path

# Fake settings so the app starts. Must be set BEFORE importing assistant.config.
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fake.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "fake-key"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "fake"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import assistant.llm  # noqa: E402
from assistant import config, db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
tmp = tempfile.TemporaryDirectory()
config.DB_PATH = Path(tmp.name) / "smoke.db"
# keep the real profile.json, but never touch the real database

assistant.llm.get_llm = lambda *a, **k: FakeListChatModel(responses=["Hi Yusra! This is a fake reply."] * 10)
assistant.llm.get_extractor_llm = lambda *a, **k: FakeListChatModel(responses=["ok"] * 10)


def check(label, condition):
    print(("PASS  " if condition else "FAIL  ") + label)
    if not condition:
        raise SystemExit(1)


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
at.run()
check("app starts without exceptions", not at.exception)
check("greets by name", any("Yusra" in t.value for t in at.title))

at.chat_input[0].set_value("Hi, this is my first message to test").run()
check("chat turn runs without exceptions", not at.exception)
check("reply is shown", any("fake reply" in m.value for m in at.markdown))
msgs = db.get_messages(db.latest_conversation_id())
check("both messages saved to the database", [m["role"] for m in msgs] == ["user", "assistant"])

at.chat_input[0].set_value("/remember I prefer short answers").run()
check("/remember saves a memory", any("short answers" in m["text"] for m in db.list_memories()))
mid = db.list_memories()[0]["id"]

at.chat_input[0].set_value(f"/forget {mid}").run()
check("/forget removes it", db.list_memories() == [])

# A brand new browser session must still see the old conversation (persistence)
at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
at2.run()
check("new session reloads old messages from the database", any("fake reply" in m.value for m in at2.markdown))

print("\nSMOKE TEST PASSED")
tmp.cleanup()
