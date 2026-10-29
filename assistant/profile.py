"""The hand-written profile (data/profile.json). The assistant reads it but never edits it."""
import json
import shutil

from . import config


def load_profile():
    """Load data/profile.json. On first run it is copied from profile.example.json."""
    if not config.PROFILE_PATH.exists():
        if config.PROFILE_EXAMPLE_PATH.exists():
            config.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.PROFILE_EXAMPLE_PATH, config.PROFILE_PATH)
        else:
            return {}
    try:
        return json.loads(config.PROFILE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"data/profile.json is not valid JSON: {e}")


def profile_to_text(profile):
    """Turn the profile dict into readable lines for the prompt. Empty fields are skipped."""
    lines = []
    for key, value in profile.items():
        if key.startswith("_") or value in ("", None, [], {}):
            continue
        label = key.replace("_", " ").capitalize()
        if isinstance(value, list):
            lines.append(f"{label}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) or "(profile is empty)"
