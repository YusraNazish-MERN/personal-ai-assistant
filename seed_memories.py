"""One-time import of starting memories:  python seed_memories.py

Reads data/seed_memories.json (copy it from data/seed_memories.example.json and edit).
Safe to run again: memories that already exist are skipped.
"""
import json
import shutil

from assistant import config, db

db.init_db()

if not config.SEED_PATH.exists():
    if config.SEED_EXAMPLE_PATH.exists():
        shutil.copyfile(config.SEED_EXAMPLE_PATH, config.SEED_PATH)
        print(f"Created {config.SEED_PATH} from the example. Edit it, then run this script again.")
    else:
        print("No seed file found.")
    raise SystemExit(0)

items = json.loads(config.SEED_PATH.read_text(encoding="utf-8"))
existing = {m["text"].strip().lower() for m in db.list_memories()}

added = skipped = 0
for item in items:
    text = item["text"].strip()
    if text.lower() in existing:
        skipped += 1
        continue
    category = item.get("category", "other")
    if category not in config.CATEGORIES:
        category = "other"
    when = None
    if item.get("as_of"):  # keep the real date so old projects are flagged correctly
        when = f"{item['as_of']}T00:00:00+00:00"
    db.add_memory(text, category, source="seed", when=when)
    added += 1

print(f"Added {added} memories, skipped {skipped} that already existed.")
