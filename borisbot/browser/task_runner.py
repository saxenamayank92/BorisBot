"""Deterministic sequential task runner for browser command execution."""

from datetime import datetime
import json
import logging
import sqlite3
import time
from typing import Any, Dict, List
from uuid import uuid4

from .command_router import CommandRouter
from borisbot.contracts import TASK_EVENT_SCHEMA_V1, TASK_RESULT_SCHEMA_V1
from borisbot.failures import classify_failure
from borisbot.supervisor.browser_capability_guard import BrowserCapabilityGuard
from borisbot.supervisor.database import get_db

logger = logging.getLogger("borisbot.browser.task_runner")


class TaskRunner:
    """
    Deterministic sequential task executor.
    """

    def __init__(
        self,
        router: CommandRouter,
        agent_id: str,
        pre_persisted: bool = False,
        worker_id: str | None = None,
    ):
        self._router = router
        self.agent_id = agent_id
        self.pre_persisted = pre_persisted
        self.worker_id = worker_id or "direct"

    async def _insert_task(self, task_id: str, agent_id: str, task: Dict[str, Any]) -> None:
        """Persist initial task checkpoint before execution begins."""
        now = datetime.utcnow().isoformat()
        payload_json = json.dumps(task)
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
            await db.commit()

    async def _mark_task_running(self, task_id: str) -> None:
        """Move persisted task state from pending to running."""
        now = datetime.utcnow().isoformat()
        async for db in get_db():
            await db.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?
                WHERE task_id = ?
                """,
                ("running", now, task_id),
            )
            await db.commit()

    async def _finalize_task(self, task_id: str, status: str, report: Dict[str, Any]) -> None:
        """Persist final task status and structured result payload."""
        now = datetime.utcnow().isoformat()
        result_json = json.dumps(report)
        async for db in get_db():
            await db.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, result = ?
                WHERE task_id = ?
                """,
                (status, now, result_json, task_id),
            )
            await db.commit()

    async def _insert_step_log(
        self,
        task_id: str,
        command_id: str,
        started_at: str,
    ) -> str | None:
        """Insert per-command execution log row without breaking runtime on failure."""
        log_id = str(uuid4())
        try:
            async for db in get_db():
                await db.execute(
                    """
                    INSERT INTO task_execution_logs (
                        id, task_id, command_id, worker_id, status, started_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_id,
                        task_id,
                        command_id,
                        self.worker_id,
                        "running",
                        started_at,
                        started_at,
                    ),
                )
                await db.commit()
            return log_id
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Failed to insert step log for task %s command %s: %s", task_id, command_id, exc)
            return None

    async def _update_step_log(
        self,
        log_id: str | None,
        status: str,
        finished_at: str,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        """Finalize per-command execution log row without breaking runtime on failure."""
        if not log_id:
            return
        try:
            async for db in get_db():
                await db.execute(
                    """
                    UPDATE task_execution_logs
                    SET status = ?, finished_at = ?, duration_ms = ?, error = ?
                    WHERE id = ?
                    """,
                    (status, finished_at, duration_ms, error, log_id),
                )
                await db.commit()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Failed to update step log %s: %s", log_id, exc)

    async def _insert_task_event(
        self,
        task_id: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Insert task event row without interrupting runtime on failure."""
        try:
            now = datetime.utcnow().isoformat()
            event_id = str(uuid4())
            payload_data = dict(payload or {})
            payload_data.setdefault("schema_version", TASK_EVENT_SCHEMA_V1)
            event_payload = json.dumps(payload_data)
            async for db in get_db():
                await db.execute(
                    """
                    INSERT INTO task_events (id, task_id, event_type, payload, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (event_id, task_id, event_type, event_payload, now),
                )
                await db.commit()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Event logging failed: %s", exc)

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a multi-command task.

        Returns:
            {
                "task_id": ...,
                "status": "completed" | "failed",
                "steps": [
                    {
                        "command_id": "...",
                        "status": "ok" | "failed",
                        "result": optional,
                        "error": optional
                    }
                ]
            }
        """

        if "task_id" not in task:
            raise ValueError("Task missing 'task_id'")

        if "commands" not in task:
            raise ValueError("Task missing 'commands'")

        if not isinstance(task["commands"], list):
            raise ValueError("'commands' must be list")

        task_id = task["task_id"]
        agent_id = self.agent_id
        if not agent_id:
            raise ValueError("Task missing 'agent_id'")

        # Prevent re-execution if this task already has a terminal/non-pending status.
        async for db in get_db():
            cursor = await db.execute(
                "SELECT status FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
        if row and row["status"] != "pending":
            logger.info("Task %s already executed", task_id)
            return {
                "schema_version": TASK_RESULT_SCHEMA_V1,
                "task_id": task_id,
                "status": row["status"],
                "duration_ms": 0,
                "steps": [],
            }

        if not self.pre_persisted:
            await self._insert_task(task_id, agent_id, task)
        await self._mark_task_running(task_id)
        await self._insert_task_event(task_id, "task_started", {"task_id": task_id})
        task_started_perf = time.perf_counter()

        final_status = "completed"
        report: Dict[str, Any]
        guard_error: Exception | None = None
        try:
            await BrowserCapabilityGuard().validate_task(agent_id, task)
        except Exception as error:
            guard_error = error

        if guard_error is not None:
            final_status = "rejected"
            total_duration_ms = max(1, int((time.perf_counter() - task_started_perf) * 1000))
            first_url = ""
            for command in task.get("commands", []):
                params = command.get("params", {})
                if isinstance(params, dict) and isinstance(params.get("url"), str):
                    first_url = params.get("url", "")
                    break
            failure = classify_failure(
                error=guard_error,
                step_id="guard",
                action="",
                selector="",
                url=first_url,
            )
            report = {
                "schema_version": TASK_RESULT_SCHEMA_V1,
                "task_id": task["task_id"],
                "status": "rejected",
                "duration_ms": total_duration_ms,
                "reason": str(guard_error),
                "failure": failure,
                "steps": [],
            }
        else:
            results: List[Dict[str, Any]] = []
            for command in task["commands"]:
                command_id = command.get("id", "unknown")
                command_started_perf = time.perf_counter()
                command_started_at = datetime.utcnow().isoformat()
                log_id = await self._insert_step_log(task_id, command_id, command_started_at)
                await self._insert_task_event(
                    task_id,
                    "step_started",
                    {"command_id": command_id},
                )

                try:
                    result = await self._router.execute(command)
                    command_finished_at = datetime.utcnow().isoformat()
                    command_duration_ms = max(1, int((time.perf_counter() - command_started_perf) * 1000))
                    await self._update_step_log(
                        log_id=log_id,
                        status="ok",
                        finished_at=command_finished_at,
                        duration_ms=command_duration_ms,
                    )
                    await self._insert_task_event(
                        task_id,
                        "step_finished",
                        {
                            "command_id": command_id,
                            "status": "ok",
                            "duration_ms": command_duration_ms,
                        },
                    )

                    step_record = {
                        "command_id": command_id,
                        "status": "ok",
                    }

                    if "result" in result:
                        step_record["result"] = result["result"]

                    results.append(step_record)

                except Exception as e:
                    final_status = "failed"
                    params = command.get("params", {}) if isinstance(command, dict) else {}
                    selector = params.get("selector", "") if isinstance(params, dict) else ""
                    url = params.get("url", "") if isinstance(params, dict) else ""
                    failure = classify_failure(
                        error=e,
                        step_id=command_id,
                        action=str(command.get("action", "")),
                        selector=str(selector or ""),
                        url=str(url or ""),
                    )
                    command_finished_at = datetime.utcnow().isoformat()
                    command_duration_ms = max(1, int((time.perf_counter() - command_started_perf) * 1000))
                    await self._update_step_log(
                        log_id=log_id,
                        status="failed",
                        finished_at=command_finished_at,
                        duration_ms=command_duration_ms,
                        error=str(e),
                    )
                    await self._insert_task_event(
                        task_id,
                        "step_finished",
                        {
                            "command_id": command_id,
                            "status": "failed",
                            "duration_ms": command_duration_ms,
                            "error": str(e),
                            "failure": failure,
                        },
                    )
                    results.append(
                        {
                            "command_id": command_id,
                            "status": "failed",
                            "error": str(e),
                            "failure": failure,
                        }
                    )
                    break

            total_duration_ms = max(1, int((time.perf_counter() - task_started_perf) * 1000))
            report = {
                "schema_version": TASK_RESULT_SCHEMA_V1,
                "task_id": task["task_id"],
                "status": final_status,
                "duration_ms": total_duration_ms,
                "steps": results,
            }

        await self._finalize_task(task_id, final_status, report)
        await self._insert_task_event(
            task_id,
            "task_completed" if final_status == "completed" else "task_failed",
            {
                "task_id": task_id,
                "status": final_status,
                "duration_ms": report.get("duration_ms", 0),
            },
        )
        return report
