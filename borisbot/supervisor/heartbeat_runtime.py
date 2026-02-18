"""Runtime heartbeat supervisor loop with restart-safe snapshot persistence."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.ollama_health import startup_mark_ollama_health
from borisbot.llm.provider_health import ProviderHealthRegistry, get_provider_health_registry
from borisbot.supervisor.database import get_db

HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_PATH = Path.home() / ".borisbot" / "runtime" / "heartbeat.json"


def read_heartbeat_snapshot(path: Path = HEARTBEAT_PATH) -> dict[str, Any] | None:
    """Read persisted heartbeat snapshot if present and valid."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class HeartbeatSupervisor:
    """Collect and persist periodic runtime heartbeat snapshots."""

    def __init__(
        self,
        *,
        cost_guard: CostGuard | None = None,
        health_registry: ProviderHealthRegistry | None = None,
        heartbeat_path: Path = HEARTBEAT_PATH,
        interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
        provider_name: str = "ollama",
        agent_id: str = "default",
        model_name: str = "llama3.2:3b",
        probe_fn: Callable[[ProviderHealthRegistry, str, str], Awaitable[bool]] | None = None,
    ) -> None:
        self.cost_guard = cost_guard or CostGuard()
        self.health_registry = health_registry or get_provider_health_registry()
        self.heartbeat_path = heartbeat_path
        self.interval_seconds = interval_seconds
        self.provider_name = provider_name
        self.agent_id = agent_id
        self.model_name = model_name
        self.probe_fn = probe_fn

    async def _probe_and_heal_provider(self) -> dict[str, object]:
        """Run bounded provider probe and update health state for self-healing."""
        previous_state = self.health_registry.get_provider_state(self.provider_name)
        if self.probe_fn is not None:
            probe_ok = await self.probe_fn(self.health_registry, self.model_name, self.provider_name)
        else:
            probe_ok = await startup_mark_ollama_health(
                self.health_registry,
                model_name=self.model_name,
                provider_name=self.provider_name,
            )
        current_state = self.health_registry.get_provider_state(self.provider_name)
        healed = previous_state == "unhealthy" and current_state == "healthy"
        return {
            "probe_ok": probe_ok,
            "previous_state": previous_state,
            "current_state": current_state,
            "healed": healed,
        }

    async def _get_task_counts(self) -> tuple[int, int]:
        active_tasks = 0
        queue_depth = 0
        async for db in get_db():
            cursor = await db.execute("SELECT COUNT(*) AS count FROM tasks WHERE status = 'running'")
            row = await cursor.fetchone()
            active_tasks = int(row["count"] if row else 0)

            cursor = await db.execute("SELECT COUNT(*) AS count FROM task_queue")
            row = await cursor.fetchone()
            queue_depth = int(row["count"] if row else 0)
            break
        return active_tasks, queue_depth

    async def collect_snapshot(self) -> dict[str, Any]:
        """Build one heartbeat snapshot from live runtime state."""
        heal_info = await self._probe_and_heal_provider()
        budget = await self.cost_guard.get_budget_status(self.agent_id)
        provider_snapshot = self.health_registry.get_snapshot().get(self.provider_name, {})
        provider_health = "unknown"
        if isinstance(provider_snapshot, dict):
            provider_health = str(provider_snapshot.get("state", "unknown"))
        active_tasks, queue_depth = await self._get_task_counts()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "provider_health": provider_health,
            "budget_status": str(budget.get("status", "ok")),
            "daily_cost": float(budget.get("daily_spend", 0.0)),
            "daily_limit": float(budget.get("daily_limit", 0.0)),
            "daily_remaining": float(budget.get("daily_remaining", 0.0)),
            "active_tasks": active_tasks,
            "queue_depth": queue_depth,
            "model_available": provider_health in {"healthy", "degraded", "unknown"},
            "self_heal_probe_ok": bool(heal_info["probe_ok"]),
            "self_heal_healed": bool(heal_info["healed"]),
        }

    async def write_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Persist heartbeat snapshot atomically to runtime file."""
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.heartbeat_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        temp_path.replace(self.heartbeat_path)

    async def initialize_on_startup(self) -> dict[str, Any]:
        """Load prior snapshot and persist reconciled startup snapshot."""
        previous = read_heartbeat_snapshot(self.heartbeat_path)
        current = await self.collect_snapshot()
        current["startup_recovered_from"] = previous.get("timestamp") if isinstance(previous, dict) else None
        if isinstance(previous, dict) and int(previous.get("active_tasks", 0) or 0) > 0:
            current[
                "reconciliation_note"
            ] = "Previous runtime had active tasks; current database state is authoritative."
        await self.write_snapshot(current)
        return current

    async def run_forever(self) -> None:
        """Run heartbeat loop until cancelled."""
        while True:
            snapshot = await self.collect_snapshot()
            await self.write_snapshot(snapshot)
            await asyncio.sleep(self.interval_seconds)
