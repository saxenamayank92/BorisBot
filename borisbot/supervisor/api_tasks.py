"""HTTP API endpoints for deterministic browser task execution."""

import json
import logging
import sqlite3
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
