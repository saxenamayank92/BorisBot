"""Per-agent tool permission matrix with prompt/allow/deny decisions."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from borisbot.supervisor.database import DB_PATH, get_db

TOOL_FILESYSTEM = "filesystem"
TOOL_SHELL = "shell"
TOOL_BROWSER = "browser"
TOOL_WEB_FETCH = "web_fetch"
TOOL_SCHEDULER = "scheduler"
TOOL_ASSISTANT = "assistant"
ALLOWED_TOOLS = {
    TOOL_FILESYSTEM,
    TOOL_SHELL,
    TOOL_BROWSER,
    TOOL_WEB_FETCH,
    TOOL_SCHEDULER,
    TOOL_ASSISTANT,
}
ORDERED_TOOLS = sorted(ALLOWED_TOOLS)

DECISION_PROMPT = "prompt"
DECISION_ALLOW = "allow"
DECISION_DENY = "deny"
ALLOWED_DECISIONS = {DECISION_PROMPT, DECISION_ALLOW, DECISION_DENY}


class ToolPermissionError(RuntimeError):
    """Base class for tool permission decision failures."""

    def __init__(self, message: str, *, tool_name: str, agent_id: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.agent_id = agent_id


class ToolPermissionRequired(ToolPermissionError):
    """Raised when tool permission is in prompt state."""


class ToolPermissionDenied(ToolPermissionError):
    """Raised when tool permission is explicitly denied."""


def _ensure_valid_tool(tool_name: str) -> None:
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Unsupported tool '{tool_name}'. Allowed: {sorted(ALLOWED_TOOLS)}")


def _ensure_valid_decision(decision: str) -> None:
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(
            f"Unsupported decision '{decision}'. Allowed: {sorted(ALLOWED_DECISIONS)}"
        )


def _ensure_table_sync(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_tool_permissions (
            agent_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            decision TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, tool_name)
        )
        """
    )


def get_agent_tool_permission_sync(
    agent_id: str,
    tool_name: str,
    *,
    db_path: Path | None = None,
) -> str:
    """Return tool decision for agent, defaulting to prompt."""
    _ensure_valid_tool(tool_name)
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _ensure_table_sync(conn)
        cursor = conn.execute(
            """
            SELECT decision
            FROM agent_tool_permissions
            WHERE agent_id = ? AND tool_name = ?
            """,
            (agent_id, tool_name),
        )
        row = cursor.fetchone()
        return str(row[0]) if row else DECISION_PROMPT
    finally:
        conn.close()


def set_agent_tool_permission_sync(
    agent_id: str,
    tool_name: str,
    decision: str,
    *,
    db_path: Path | None = None,
) -> None:
    """Persist agent tool decision."""
    _ensure_valid_tool(tool_name)
    _ensure_valid_decision(decision)
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _ensure_table_sync(conn)
        conn.execute(
            """
            INSERT INTO agent_tool_permissions (agent_id, tool_name, decision, decided_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(agent_id, tool_name) DO UPDATE SET
                decision = excluded.decision,
                decided_at = excluded.decided_at
            """,
            (agent_id, tool_name, decision),
        )
        conn.commit()
    finally:
        conn.close()


def get_agent_permission_matrix_sync(
    agent_id: str,
    *,
    db_path: Path | None = None,
) -> dict[str, str]:
    """Return all tool decisions for agent, defaulting each tool to prompt."""
    path = db_path or DB_PATH
    matrix: dict[str, str] = {}
    for tool_name in ORDERED_TOOLS:
        matrix[tool_name] = get_agent_tool_permission_sync(agent_id, tool_name, db_path=path)
    return matrix


async def _ensure_table_async() -> None:
    async for db in get_db():
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_tool_permissions (
                agent_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                decision TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, tool_name)
            )
            """
        )
        await db.commit()
        break


async def get_agent_tool_permission(agent_id: str, tool_name: str) -> str:
    """Return async tool decision for agent, defaulting to prompt."""
    _ensure_valid_tool(tool_name)
    await _ensure_table_async()
    async for db in get_db():
        cursor = await db.execute(
            """
            SELECT decision
            FROM agent_tool_permissions
            WHERE agent_id = ? AND tool_name = ?
            """,
            (agent_id, tool_name),
        )
        row = await cursor.fetchone()
        return str(row["decision"]) if row else DECISION_PROMPT
    return DECISION_PROMPT


async def set_agent_tool_permission(agent_id: str, tool_name: str, decision: str) -> None:
    """Persist async tool decision for agent."""
    _ensure_valid_tool(tool_name)
    _ensure_valid_decision(decision)
    await _ensure_table_async()
    async for db in get_db():
        await db.execute(
            """
            INSERT INTO agent_tool_permissions (agent_id, tool_name, decision, decided_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent_id, tool_name) DO UPDATE SET
                decision = excluded.decision,
                decided_at = excluded.decided_at
            """,
            (agent_id, tool_name, decision, datetime.utcnow().isoformat()),
        )
        await db.commit()
        return


async def enforce_tool_permission(agent_id: str, tool_name: str) -> None:
    """Raise explicit permission errors for denied/prompt tool states."""
    decision = await get_agent_tool_permission(agent_id, tool_name)
    if decision == DECISION_ALLOW:
        return
    if decision == DECISION_DENY:
        raise ToolPermissionDenied(
            f"Tool '{tool_name}' denied for agent '{agent_id}'",
            tool_name=tool_name,
            agent_id=agent_id,
        )
    raise ToolPermissionRequired(
        f"Tool '{tool_name}' requires approval for agent '{agent_id}'",
        tool_name=tool_name,
        agent_id=agent_id,
    )
