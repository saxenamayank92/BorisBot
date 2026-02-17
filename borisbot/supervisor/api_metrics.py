"""HTTP metrics endpoints for supervisor queue and worker observability."""

import logging

from fastapi import APIRouter

from borisbot.supervisor.database import get_db

logger = logging.getLogger("borisbot.supervisor.api_metrics")

router = APIRouter()


@router.get("/metrics/queue")
async def get_queue_metrics():
    """Return task counts by status for queue visibility."""
    counts = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
    }
    async for db in get_db():
        cursor = await db.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM tasks
            GROUP BY status
            """
        )
        rows = await cursor.fetchall()
    for row in rows:
        status = row["status"]
        if status in counts:
            counts[status] = row["count"]
    return counts


@router.get("/metrics/workers")
async def get_worker_metrics():
    """Return worker heartbeat rows."""
    async for db in get_db():
        cursor = await db.execute(
            """
            SELECT worker_id, last_seen, status
            FROM worker_heartbeats
            ORDER BY worker_id ASC
            """
        )
        rows = await cursor.fetchall()
    return {
        "workers": [
            {
                "worker_id": row["worker_id"],
                "last_seen": row["last_seen"],
                "status": row["status"],
            }
            for row in rows
        ]
    }


@router.get("/metrics/active")
async def get_active_task_metrics():
    """Return actively running task rows."""
    async for db in get_db():
        cursor = await db.execute(
            """
            SELECT task_id, agent_id, status, updated_at
            FROM tasks
            WHERE status = 'running'
            ORDER BY updated_at DESC
            """
        )
        rows = await cursor.fetchall()
    return {
        "active_tasks": [
            {
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "status": row["status"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    }
