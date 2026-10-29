"""Run this FIRST on your computer:  python test_connection.py

It checks, one by one, the three things the assistant needs from Azure:
  1. a normal reply
  2. streaming
  3. structured output (used for memory suggestions)
"""
from assistant import config
from assistant.llm import get_extractor_llm, get_llm
from assistant.memory import ExtractionResult
from assistant.prompts import EXTRACT_PROMPT

missing = config.missing_settings()
if missing:
    print("MISSING in .env:", missing)
    raise SystemExit(1)

print("ENDPOINT  :", repr(config.AZURE_ENDPOINT))
print("DEPLOYMENT:", repr(config.DEPLOYMENT))
print("VERSION   :", repr(config.API_VERSION))
print()

llm = get_llm()

try:
    print("1. Normal reply  :", llm.invoke("Say hi in one short sentence.").content)

    print("2. Streaming     : ", end="")
    for chunk in llm.stream("Count from 1 to 5."):
        print(chunk.content, end="", flush=True)
    print()

    chain = EXTRACT_PROMPT | get_extractor_llm().with_structured_output(
        ExtractionResult, method="function_calling"
    )
    result = chain.invoke(
        {
            "user_name": "Yusra",
            "existing_memories": "(none)",
            "last_assistant": "(none)",
            "user_message": "I learn best by building small projects first and reading docs later.",
        }
    )
    print("3. Structured out:", result)
    print("\nSUCCESS: all 3 checks passed.")
except Exception as e:
    print("\nFAILED:", e)
    print("If this is a 404, check AZURE_OPENAI_ENDPOINT (no /openai/v1 at the end) and the deployment name.")
