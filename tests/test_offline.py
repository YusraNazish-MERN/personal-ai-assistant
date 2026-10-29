"""Offline tests: no Azure, no internet, no API key needed.

Run from the project folder:   python -m unittest tests.test_offline -v
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from assistant import chat, config, context, db, memory, prompts
from assistant.profile import load_profile, profile_to_text


class StubExtractor:
    """Pretends to be a chat model whose structured output is a fixed result."""

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _prompt_value: self.result)


def proposal(action="ADD", text="Yusra prefers learning by building.", category="learning_style",
             memory_id=None, evidence="I learn best by building small projects", reason="stated"):
    return memory.MemoryProposal(action=action, text=text, category=category,
                                 memory_id=memory_id, evidence=evidence, reason=reason)


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        config.DATA_DIR = base
        config.DB_PATH = base / "test.db"
        config.PROFILE_PATH = base / "profile.json"
        config.PROFILE_EXAMPLE_PATH = base / "profile.example.json"
        config.PROFILE_PATH.write_text(
            json.dumps({"preferred_name": "Yusra", "location": "Lahore", "learning_style": [],
                        "communication_preferences": ["Be honest"]}),
            encoding="utf-8",
        )
        db.init_db()
        self.conv = db.create_conversation()

    def tearDown(self):
        self.tmp.cleanup()

    def old_date(self, days):
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


class TestDatabase(BaseCase):
    def test_messages_round_trip_and_after_id(self):
        a = db.add_message(self.conv, "user", "hello")
        db.add_message(self.conv, "assistant", "hi Yusra")
        self.assertEqual([m["content"] for m in db.get_messages(self.conv)], ["hello", "hi Yusra"])
        self.assertEqual(len(db.get_messages(self.conv, after_id=a)), 1)
        self.assertEqual(db.count_messages(self.conv), 2)

    def test_history_survives_a_restart(self):
        db.add_message(self.conv, "user", "remember me")
        # a "restart" = nothing in memory, only the file on disk
        self.assertEqual(db.latest_conversation_id(), self.conv)
        self.assertEqual(db.get_messages(self.conv)[0]["content"], "remember me")

    def test_memory_update_keeps_previous_text_and_delete_removes(self):
        mid = db.add_memory("Yusra likes tea", "preference")
        db.update_memory(mid, text="Yusra likes coffee")
        m = db.get_memory(mid)
        self.assertEqual(m["text"], "Yusra likes coffee")
        self.assertEqual(m["previous_text"], "Yusra likes tea")
        self.assertTrue(db.delete_memory(mid))
        self.assertIsNone(db.get_memory(mid))
        self.assertFalse(db.delete_memory(mid))

    def test_duplicate_proposals_are_not_queued_twice(self):
        first = db.add_proposal("ADD", "Yusra likes tea", "preference", None, "I like tea a lot", "x")
        second = db.add_proposal("ADD", "yusra likes  tea", "preference", None, "I like tea a lot", "x")
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_delete_conversation_removes_messages(self):
        db.add_message(self.conv, "user", "x")
        db.delete_conversation(self.conv)
        self.assertIsNone(db.get_conversation(self.conv))
        self.assertEqual(db.get_messages(self.conv), [])


class TestProfile(BaseCase):
    def test_profile_text_skips_empty_and_private_keys(self):
        text = profile_to_text(load_profile())
        self.assertIn("Location: Lahore", text)
        self.assertIn("Be honest", text)
        self.assertNotIn("Learning style", text)  # empty list skipped

    def test_bad_json_gives_clear_error(self):
        config.PROFILE_PATH.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_profile()


class TestMemorySelection(BaseCase):
    def test_relevant_project_is_chosen_and_irrelevant_one_is_not_forced(self):
        db.add_memory("Yusra prefers short answers", "preference")
        db.add_memory("Yusra is building AuroraClass for visually impaired teachers", "project")
        db.add_memory("Yusra trains an RNN for temperature forecasting", "project")
        db.add_memory("Yusra is making a recipe app", "project")
        selected = context.select_memories("How do I improve my RNN forecasting accuracy?", "", db.list_memories())
        texts = [m["text"] for m in selected]
        self.assertIn("Yusra prefers short answers", texts)           # always-on
        self.assertIn("Yusra trains an RNN for temperature forecasting", texts)
        self.assertNotIn("Yusra is making a recipe app", texts)       # matched something else, so no filler

    def test_cap_is_respected(self):
        for i in range(20):
            db.add_memory(f"Yusra fact number {i} about python", "project")
        selected = context.select_memories("python question", "", db.list_memories())
        self.assertLessEqual(len(selected), config.MAX_MEMORIES)

    def test_fallback_shows_two_recent_projects_when_nothing_matches(self):
        for i in range(4):
            db.add_memory(f"Yusra works on thing{i}", "project")
        selected = context.select_memories("hello there", "", db.list_memories())
        self.assertEqual(len(selected), 2)

    def test_stale_project_is_flagged_but_old_preference_is_not(self):
        db.add_memory("Yusra is finishing the old project", "project", when=self.old_date(90))
        db.add_memory("Yusra prefers tea", "preference", when=self.old_date(400))
        mems = db.list_memories()
        text = context.format_memories(mems)
        self.assertIn("MAY BE OUTDATED", text)
        self.assertEqual(text.count("MAY BE OUTDATED"), 1)

    def test_touch_memory_clears_stale_flag(self):
        mid = db.add_memory("Yusra is finishing the old project", "project", when=self.old_date(90))
        self.assertTrue(context.is_stale(db.get_memory(mid)))
        db.touch_memory(mid)
        self.assertFalse(context.is_stale(db.get_memory(mid)))


class TestValidation(BaseCase):
    def setUp(self):
        super().setUp()
        self.mid = db.add_memory("Yusra likes tea", "preference")
        self.active = db.list_memories()

    def test_valid_add_passes(self):
        out = memory.validate_proposals([proposal()], "I learn best by building small projects first.", self.active)
        self.assertEqual(len(out), 1)

    def test_made_up_evidence_is_dropped(self):
        out = memory.validate_proposals([proposal(evidence="I am a professional chef")],
                                        "I learn best by building small projects first.", self.active)
        self.assertEqual(out, [])

    def test_too_short_evidence_is_dropped(self):
        out = memory.validate_proposals([proposal(evidence="small")],
                                        "I learn best by building small projects first.", self.active)
        self.assertEqual(out, [])

    def test_update_or_delete_of_unknown_memory_is_dropped(self):
        msg = "Actually I like coffee now, not tea."
        bad_update = proposal("UPDATE", "Yusra likes coffee", "preference", memory_id=999, evidence="I like coffee now")
        bad_delete = proposal("DELETE", "x", "preference", memory_id=999, evidence="I like coffee now")
        self.assertEqual(memory.validate_proposals([bad_update, bad_delete], msg, self.active), [])

    def test_valid_update_and_delete_pass(self):
        msg = "Actually I like coffee now, not tea."
        up = proposal("UPDATE", "Yusra likes coffee", "preference", memory_id=self.mid, evidence="I like coffee now")
        out = memory.validate_proposals([up], msg, self.active)
        self.assertEqual(out[0]["action"], "UPDATE")
        de = proposal("DELETE", "ignored text", "other", memory_id=self.mid, evidence="not tea")
        # 'not tea' is only 2 words -> rejected; use a 3+ word quote
        de = proposal("DELETE", "ignored text", "other", memory_id=self.mid, evidence="like coffee now, not tea")
        out = memory.validate_proposals([de], msg, self.active)
        self.assertEqual(out[0]["text"], "Yusra likes tea")  # shows the real memory, not the model's text

    def test_duplicate_add_is_dropped(self):
        dup = proposal(text="Yusra likes tea", evidence="I like tea very much")
        self.assertEqual(memory.validate_proposals([dup], "I like tea very much", self.active), [])

    def test_unknown_action_is_dropped_and_bad_category_becomes_other(self):
        msg = "I learn best by building small projects first."
        self.assertEqual(memory.validate_proposals([proposal("HACK")], msg, self.active), [])
        out = memory.validate_proposals([proposal(category="weird")], msg, self.active)
        self.assertEqual(out[0]["category"], "other")

    def test_max_proposals_per_turn(self):
        msg = "I learn best by building small projects first."
        many = [proposal(text=f"Yusra fact {i}") for i in range(10)]
        self.assertEqual(len(memory.validate_proposals(many, msg, self.active)), config.MAX_PROPOSALS_PER_TURN)

    def test_should_extract_skips_short_and_commands(self):
        self.assertFalse(memory.should_extract("hi"))
        self.assertFalse(memory.should_extract("/remember something long enough here"))
        self.assertTrue(memory.should_extract("I learn best by building small projects"))


class TestApproveFlow(BaseCase):
    def test_extract_then_approve_add_update_delete(self):
        msg = "I learn best by building small projects first."
        stub = StubExtractor(memory.ExtractionResult(proposals=[proposal()]))
        n = memory.extract_proposals(stub, msg, "", "Yusra")
        self.assertEqual(n, 1)
        self.assertEqual(db.list_memories(), [])            # NOT saved yet
        pid = db.list_proposals()[0]["id"]
        memory.approve(pid)
        mems = db.list_memories()
        self.assertEqual(len(mems), 1)
        self.assertEqual(mems[0]["source"], "auto")
        self.assertEqual(db.list_proposals(), [])

        mid = mems[0]["id"]
        db.add_proposal("UPDATE", "Yusra learns by reading docs first", "learning_style", mid, "I now read docs first", "changed")
        memory.approve(db.list_proposals()[0]["id"])
        self.assertEqual(db.get_memory(mid)["text"], "Yusra learns by reading docs first")
        self.assertEqual(db.get_memory(mid)["previous_text"], "Yusra prefers learning by building.")

        db.add_proposal("DELETE", "x", "other", mid, "forget that please", "asked")
        memory.approve(db.list_proposals()[0]["id"])
        self.assertIsNone(db.get_memory(mid))

    def test_reject_saves_nothing(self):
        pid = db.add_proposal("ADD", "Yusra likes tea", "preference", None, "I like tea a lot", "x")
        memory.reject(pid)
        self.assertEqual(db.list_memories(), [])
        self.assertEqual(db.list_proposals(), [])

    def test_manual_add_validation(self):
        with self.assertRaises(ValueError):
            memory.add_manual("   ")
        with self.assertRaises(ValueError):
            memory.add_manual("x" * 400)
        self.assertIsNotNone(memory.add_manual("Yusra likes tea", "weird-category"))
        self.assertEqual(db.list_memories()[0]["category"], "other")


class TestPromptAndTurn(BaseCase):
    def test_system_prompt_renders_with_personal_context(self):
        db.add_memory("Yusra prefers short answers", "preference")
        turn = chat.prepare_turn(self.conv, "What should I work on?")
        sys_text = turn.system_preview
        self.assertIn("Yusra", sys_text)
        self.assertIn("Lahore", sys_text)
        self.assertIn("Yusra prefers short answers", sys_text)
        self.assertIn("Today's date:", sys_text)
        for variable in ("{user_name}", "{profile}", "{memories}", "{recap}", "{earlier}", "{today}"):
            self.assertNotIn(variable, sys_text)  # no unfilled variables
        # the chat prompt itself must format without missing-variable errors
        msgs = prompts.CHAT_PROMPT.format_messages(**turn.inputs)
        self.assertEqual(msgs[0].type, "system")
        self.assertEqual(msgs[-1].content, "What should I work on?")

    def test_deleted_memory_is_never_injected(self):
        mid = db.add_memory("Yusra prefers short answers", "preference")
        db.delete_memory(mid)
        turn = chat.prepare_turn(self.conv, "hello")
        self.assertNotIn("short answers", turn.system_preview)

    def test_full_turn_saves_messages_and_title(self):
        fake = FakeListChatModel(responses=["Hello Yusra, nice to meet you."])
        turn = chat.prepare_turn(self.conv, "Hi there, I'm testing the assistant")
        reply = "".join(chat.stream_turn(turn, fake))
        self.assertEqual(reply, "Hello Yusra, nice to meet you.")
        out = chat.finish_turn(turn, reply, fake, StubExtractor(memory.ExtractionResult()), learn=True)
        self.assertTrue(out["saved"])
        msgs = db.get_messages(self.conv)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(db.get_conversation(self.conv)["title"], "Hi there, I'm testing the assistant")

    def test_next_turn_sees_previous_messages(self):
        fake = FakeListChatModel(responses=["ok"])
        t1 = chat.prepare_turn(self.conv, "My favourite number is 7, remember it")
        chat.finish_turn(t1, "ok", fake, learn=False)
        t2 = chat.prepare_turn(self.conv, "What is my favourite number?")
        self.assertEqual(len(t2.inputs["history"]), 2)
        self.assertIn("favourite number is 7", t2.inputs["history"][0].content)

    def test_failed_reply_saves_nothing(self):
        turn = chat.prepare_turn(self.conv, "this will fail")
        out = chat.finish_turn(turn, "", FakeListChatModel(responses=["x"]), learn=False)
        self.assertFalse(out["saved"])
        self.assertEqual(db.count_messages(self.conv), 0)

    def test_extraction_failure_does_not_break_chat(self):
        class Broken:
            def with_structured_output(self, *a, **k):
                raise RuntimeError("boom")

        fake = FakeListChatModel(responses=["fine"])
        turn = chat.prepare_turn(self.conv, "I learn best by building small projects first.")
        out = chat.finish_turn(turn, "fine", fake, Broken(), learn=True)
        self.assertTrue(out["saved"])
        self.assertEqual(out["proposals"], 0)


class TestSummariesAndContinuity(BaseCase):
    def test_long_chat_is_summarized_and_recent_messages_kept(self):
        for i in range(40):
            db.add_message(self.conv, "user" if i % 2 == 0 else "assistant", f"message {i}")
        summarizer = FakeListChatModel(responses=["- discussed numbers 0 to 27"])
        self.assertTrue(chat.maybe_summarize(self.conv, summarizer))
        conv = db.get_conversation(self.conv)
        self.assertEqual(conv["summary"], "- discussed numbers 0 to 27")
        remaining = db.get_messages(self.conv, after_id=conv["summarized_upto"])
        self.assertEqual(len(remaining), config.RECENT_MESSAGES)
        self.assertEqual(remaining[-1]["content"], "message 39")
        # the model now sees summary + recent messages only
        turn = chat.prepare_turn(self.conv, "what did we cover?")
        self.assertEqual(len(turn.inputs["history"]), config.RECENT_MESSAGES)
        self.assertIn("discussed numbers", turn.system_preview)

    def test_short_chat_is_not_summarized(self):
        for i in range(10):
            db.add_message(self.conv, "user", f"m{i}")
        self.assertFalse(chat.maybe_summarize(self.conv, FakeListChatModel(responses=["x"])))

    def test_new_chat_carries_a_recap_of_the_previous_one(self):
        for i in range(6):
            db.add_message(self.conv, "user" if i % 2 == 0 else "assistant", f"talk about FYP {i}")
        recapper = FakeListChatModel(responses=["- Yusra was planning her FYP demo"])
        new_id = chat.start_new_chat(self.conv, recapper)
        self.assertNotEqual(new_id, self.conv)
        turn = chat.prepare_turn(new_id, "Where did we stop?")
        self.assertIn("planning her FYP demo", turn.system_preview)

    def test_empty_chat_is_reused_by_new_chat(self):
        self.assertEqual(chat.start_new_chat(self.conv, FakeListChatModel(responses=["x"])), self.conv)

    def test_recap_failure_still_opens_new_chat(self):
        for i in range(6):
            db.add_message(self.conv, "user", f"m{i}")

        class Boom(FakeListChatModel):
            def _call(self, *a, **k):
                raise RuntimeError("down")

        new_id = chat.start_new_chat(self.conv, Boom(responses=["x"]))
        self.assertNotEqual(new_id, self.conv)


if __name__ == "__main__":
    unittest.main()
