"""HTTP metrics endpoints for supervisor queue and worker observability."""

import logging

from fastapi import APIRouter

from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.provider_health import get_provider_health_registry
from borisbot.supervisor.database import get_db
from borisbot.supervisor.heartbeat_runtime import read_heartbeat_snapshot

logger = logging.getLogger("borisbot.supervisor.api_metrics")
cost_guard = CostGuard()
provider_health_registry = get_provider_health_registry()

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


@router.get("/metrics/cost")
async def get_cost_metrics():
    """Return global and per-agent cloud usage cost metrics."""
    snapshot = await cost_guard.get_spend_snapshot()
    agent_spend = await cost_guard.get_agent_spend_today()
    return {
        "daily_spend_usd": snapshot["daily_spend"],
        "monthly_spend_usd": snapshot["monthly_spend"],
        "daily_limit_usd": snapshot["daily_limit"],
        "monthly_limit_usd": snapshot["monthly_limit"],
        "agent_spend_today": agent_spend,
    }


@router.get("/metrics/providers")
async def get_provider_metrics():
    """Return provider-level health and circuit breaker stats."""
    return provider_health_registry.get_snapshot()


@router.get("/metrics/heartbeat")
async def get_runtime_heartbeat():
    """Return latest persisted runtime heartbeat snapshot."""
    snapshot = read_heartbeat_snapshot()
    return snapshot or {"status": "unavailable"}
