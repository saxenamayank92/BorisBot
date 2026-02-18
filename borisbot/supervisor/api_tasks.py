"""HTTP API endpoints for deterministic browser task execution."""

import json
import logging
import sqlite3
import hashlib
from typing import Any, Dict
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from borisbot.llm.cost_guard import CostGuard, CostLimitError
from borisbot.supervisor.database import get_db
from borisbot.supervisor.tool_permissions import (
    TOOL_BROWSER,
    ToolPermissionDenied,
    ToolPermissionRequired,
    enforce_tool_permission,
)

logger = logging.getLogger("borisbot.supervisor.api_tasks")
cost_guard = CostGuard()


class TaskRequest(BaseModel):
    """Minimal task request envelope."""

    agent_id: str
    task: Dict[str, Any]


router = APIRouter()


def _extract_idempotency_key(task: Dict[str, Any]) -> str | None:
    """Return optional idempotency key from task payload."""
    key = task.get("idempotency_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def _payload_hash(task: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash for task payload."""
    normalized = json.dumps(task, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_idempotency(existing_row: dict | None, *, agent_id: str, payload_hash: str) -> str:
    """Return idempotency resolution: miss|hit|conflict."""
    if existing_row is None:
        return "miss"
    if str(existing_row.get("agent_id")) != agent_id:
        return "conflict"
    if str(existing_row.get("payload_hash")) != payload_hash:
        return "conflict"
    return "hit"


def _extract_cloud_model_metadata(task: Dict[str, Any]) -> str | None:
    """Return model name when task contains cloud model usage metadata."""
    if isinstance(task.get("cloud"), dict):
        model_name = task["cloud"].get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    if isinstance(task.get("metadata"), dict):
        model_name = task["metadata"].get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    model_name = task.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


@router.post("/tasks", status_code=202)
async def create_task(request: TaskRequest):
    """Create and enqueue a deterministic browser task."""
    agent_id = request.agent_id
    task = request.task

    if "task_id" not in task:
        raise HTTPException(status_code=422, detail="task missing task_id")
    if "commands" not in task or not isinstance(task["commands"], list):
        raise HTTPException(status_code=422, detail="task missing commands")

    task_id = task["task_id"]
    queue_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    payload_json = json.dumps(task)
    idempotency_key = _extract_idempotency_key(task)
    payload_hash = _payload_hash(task) if idempotency_key else None

    try:
        # Deterministic task execution uses browser runtime by contract.
        await enforce_tool_permission(agent_id, TOOL_BROWSER)

        model_name = _extract_cloud_model_metadata(task)
        if model_name:
            try:
                await cost_guard.check_can_execute(agent_id=agent_id, task_id=task_id)
            except CostLimitError as exc:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "cost_limit",
                        "reason": str(exc),
                        "limit_type": exc.limit_type,
                    },
                ) from exc

        async for db in get_db():
            existing_idempotency_row: dict | None = None
            if idempotency_key and payload_hash:
                cursor = await db.execute(
                    """
                    SELECT idempotency_key, agent_id, payload_hash, task_id
                    FROM task_idempotency
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                )
                row = await cursor.fetchone()
                existing_idempotency_row = dict(row) if row else None
                resolution = _resolve_idempotency(
                    existing_idempotency_row,
                    agent_id=agent_id,
                    payload_hash=payload_hash,
                )
                if resolution == "hit":
                    return {
                        "task_id": str(existing_idempotency_row["task_id"]),
                        "status": "deduplicated",
                        "deduplicated": True,
                    }
                if resolution == "conflict":
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "idempotency_conflict",
                            "reason": "idempotency key already exists with different task payload",
                            "idempotency_key": idempotency_key,
                        },
                    )

            try:
                await db.execute(
                    """
                    INSERT INTO tasks (task_id, agent_id, status, created_at, updated_at, payload, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_id, agent_id, "pending", now, now, payload_json, None),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(f"duplicate task_id: {task_id}") from exc

            await db.execute(
                """
                INSERT INTO task_queue (id, task_id, enqueued_at, locked_at, locked_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (queue_id, task_id, now, None, None),
            )
            if idempotency_key and payload_hash:
                try:
                    await db.execute(
                        """
                        INSERT INTO task_idempotency (
                            idempotency_key, agent_id, payload_hash, task_id, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (idempotency_key, agent_id, payload_hash, task_id, now),
                    )
                except sqlite3.IntegrityError:
                    cursor = await db.execute(
                        """
                        SELECT idempotency_key, agent_id, payload_hash, task_id
                        FROM task_idempotency
                        WHERE idempotency_key = ?
                        """,
                        (idempotency_key,),
                    )
                    row = await cursor.fetchone()
                    existing = dict(row) if row else None
                    resolution = _resolve_idempotency(
                        existing,
                        agent_id=agent_id,
                        payload_hash=payload_hash,
                    )
                    if resolution == "hit":
                        await db.rollback()
                        return {
                            "task_id": str(existing["task_id"]) if existing else task_id,
                            "status": "deduplicated",
                            "deduplicated": True,
                        }
                    await db.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "idempotency_conflict",
                            "reason": "idempotency key already exists with different task payload",
                            "idempotency_key": idempotency_key,
                        },
                    )
            await db.commit()
            break
        return {"task_id": task_id, "status": "pending"}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ToolPermissionRequired as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "tool_permission_required",
                "agent_id": exc.agent_id,
                "tool_name": exc.tool_name,
                "reason": str(exc),
            },
        ) from exc
    except ToolPermissionDenied as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tool_permission_denied",
                "agent_id": exc.agent_id,
                "tool_name": exc.tool_name,
                "reason": str(exc),
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled task execution error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Fetch persisted task status and structured result."""
    async for db in get_db():
        async with db.execute(
            "SELECT task_id, status, result FROM tasks WHERE task_id = ?",
            (task_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="task not found")
            return {
                "task_id": row["task_id"],
                "status": row["status"],
                "result": json.loads(row["result"]) if row["result"] else None,
            }


@router.get("/agents/{agent_id}/sessions")
async def get_agent_sessions(agent_id: str):
    """List browser session rows for an agent."""
    async for db in get_db():
        async with db.execute(
            """
            SELECT container_name, status, cdp_port, expires_at
            FROM browser_sessions
            WHERE agent_id = ?
            ORDER BY created_at DESC
            """,
            (agent_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "container_name": row["container_name"],
                    "status": row["status"],
                    "cdp_port": row["cdp_port"],
                    "expires_at": row["expires_at"],
                }
                for row in rows
            ]
