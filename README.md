# Personal AI Assistant

A personal assistant built on an Azure OpenAI deployment (gpt-4.1-mini or gpt-4o), LangChain and Streamlit.
It keeps your chat history on disk, learns facts about you **only with your approval**, and builds
a small, relevant context for every message instead of sending everything.

## Folder structure

```
personal-assistant/
├── app.py                  # Streamlit UI only
├── test_connection.py      # run first: checks Azure + LangChain + structured output
├── seed_memories.py        # optional one-time import of starting memories
├── requirements.txt
├── .env.example            # copy to .env
├── assistant/
│   ├── config.py           # settings + tuning numbers
│   ├── llm.py              # LangChain AzureChatOpenAI
│   ├── prompts.py          # ALL prompts (personality, memory extraction, summaries)
│   ├── profile.py          # reads data/profile.json
│   ├── db.py               # SQLite: chats, memories, suggestions
│   ├── context.py          # decides what the model sees each turn
│   ├── memory.py           # suggest / approve / edit / delete memories + safety checks
│   └── chat.py             # one turn: prepare -> stream -> finish
├── data/
│   ├── profile.json        # YOU edit this (private, not uploaded to GitHub)
│   ├── profile.example.json
│   ├── seed_memories.json  # starting memories (private)
│   ├── seed_memories.example.json
│   └── assistant.db        # created automatically (private)
└── tests/                  # offline tests, no Azure needed
```

## Setup (Windows, Git Bash)

```bash
cd personal-assistant
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in your endpoint, key and deployment name.
The endpoint must be the base URL only: `https://your-resource.openai.azure.com/`
(no `/openai/v1`, no `/openai/deployments/...`).

## First run, in this order

1. **Check the connection** (needs your real `.env`):
   ```bash
   python test_connection.py
   ```
   You must see `SUCCESS: all 3 checks passed.` If check 3 fails but 1 and 2 pass, chat still works
   but memory suggestions will not. Send the error message to get it fixed.

2. **Edit your profile**: open `data/profile.json` and correct anything wrong. Fill in `learning_style`
   with your own words. The assistant never edits this file.

3. **(Optional) import starting memories**: edit `data/seed_memories.json`, then:
   ```bash
   python seed_memories.py
   ```

4. **Start the app**:
   ```bash
   streamlit run app.py
   ```

5. **Test it** (each check proves one feature):
   | Do this | Expect |
   |---|---|
   | "What's my name?" | Yusra |
   | "What's my favourite food?" | It says it doesn't know and asks (no invention) |
   | Refresh the browser | Chat is still there |
   | Click **New chat**, ask "Where did we stop?" | It continues from a recap of the last chat |
   | Say "I learn best by building small projects first" | A suggestion appears in the sidebar; nothing is saved until you click Approve |
   | Turn on "Show context sent to the model" and send a message | You see exactly which profile and memories the model received |

Optional offline tests (no Azure needed): `python -m unittest tests.test_offline -v` and `python -m tests.test_app_smoke`.

## How it works

| Concern | Strategy |
|---|---|
| Conversation history | Every message is saved in SQLite. The model sees the messages since the last summary. Once a chat passes 30 unsummarized messages, the oldest are folded into a summary and the latest 12 stay verbatim. |
| Continuity across chats | **New chat** stores a short recap of the old one. The next chat sees the recap of your most recent chat. |
| Profile | `data/profile.json`: facts you wrote. Always included. Never edited by the assistant. |
| Long-term memory | `memories` table: text, category, date, source. Edit and delete in the sidebar. |
| Relevant retrieval | Always-on: identity, preference and learning-style memories (max 5). Others are ranked by keyword overlap with your message, capped at 8 memories total. If nothing matches, the 2 most recent projects/goals are shown as background. |
| Outdated memories | Project/goal memories older than 60 days are marked MAY BE OUTDATED in the prompt, and the model is told to confirm before relying on them. The sidebar has a "Still true" button. |
| Learning new facts | A second small model call proposes memories. Every proposal must quote your exact words, and the code checks the quote really is in your message. Proposals wait in the sidebar until you approve. |
| Updating and removing | Say something that contradicts a memory and it proposes an UPDATE. Say "forget that" and it proposes a DELETE. Or use the sidebar, or the commands `/remember <text>` and `/forget <id>`. |
| No invented personal info | The system prompt only allows personal facts from the profile, memories, recap or the current chat; otherwise it must say it doesn't know. The model also cannot claim it saved anything. |
| Personality | Defined in `assistant/prompts.py` (rules) and `communication_preferences` in `data/profile.json` (your style). |

## Tuning

Edit `assistant/config.py`: `RECENT_MESSAGES`, `SUMMARY_TRIGGER`, `MAX_MEMORIES`, `STALE_DAYS`, `TEMPERATURE`, `MAX_TOKENS`, `TIMEZONE`.
Change the assistant's behaviour by editing the rules in `assistant/prompts.py`.

## Pushing to GitHub

`.gitignore` already keeps these private: `.env`, `venv/`, `data/*.db`, `data/profile.json`, `data/seed_memories.json`.
Only the `*.example.json` files are uploaded.

```bash
git init
git add .
git status        # check: .env, data/profile.json and data/assistant.db must NOT be listed
git commit -m "Personal AI assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you ever pushed a real API key by mistake, regenerate the key in Azure; deleting it from GitHub is not enough.

## Honest limitations

- Retrieval is keyword-based. That is fine for tens to a few hundred memories. If you outgrow it, the next step is embeddings (needs an embedding model deployed in Azure).
- Memory suggestions cost one extra small model call per message (switch it off with the sidebar toggle).
- Summaries are written by the model and can lose detail. The raw messages always stay in the database.
- It is a single-user, local app: no login, no multi-user support.
- Back up `data/assistant.db` (or use the sidebar Export button) before deleting or moving the folder.
