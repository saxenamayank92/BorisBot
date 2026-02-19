"""Small persistent chat history store for guide planner chat."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CHAT_HISTORY_DIR = Path.home() / ".borisbot" / "chat_history"
MAX_CHAT_ITEMS = 200


def _agent_key(agent_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (agent_id or "default").strip())
    return cleaned or "default"


def _history_path(agent_id: str) -> Path:
    workspace = str(os.getenv("BORISBOT_WORKSPACE", "")).strip()
    if workspace:
        root = Path(workspace) / ".borisbot" / "chat_history"
    else:
        root = CHAT_HISTORY_DIR
    return root / f"{_agent_key(agent_id)}.json"


def load_chat_history(agent_id: str) -> list[dict[str, str]]:
    path = _history_path(agent_id)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip()
        text = str(row.get("text", "")).strip()
        if role and text:
            items.append({"role": role, "text": text})
    return items[-MAX_CHAT_ITEMS:]


def append_chat_message(agent_id: str, role: str, text: str) -> list[dict[str, str]]:
    role_value = str(role).strip()
    text_value = str(text).strip()
    if not role_value or not text_value:
        raise ValueError("role and text are required")
    items = load_chat_history(agent_id)
    items.append({"role": role_value, "text": text_value})
    items = items[-MAX_CHAT_ITEMS:]
    path = _history_path(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


def clear_chat_history(agent_id: str) -> None:
    path = _history_path(agent_id)
    if path.exists():
        path.unlink()


def clear_chat_roles(agent_id: str, roles: set[str]) -> list[dict[str, str]]:
    """Remove chat items for selected roles and persist remaining history."""
    role_set = {str(role).strip() for role in roles if str(role).strip()}
    if not role_set:
        return load_chat_history(agent_id)
    items = load_chat_history(agent_id)
    remaining = [row for row in items if row.get("role") not in role_set]
    path = _history_path(agent_id)
    if not remaining:
        if path.exists():
            path.unlink()
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    return remaining
