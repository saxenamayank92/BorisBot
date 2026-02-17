"""Deterministic sequential task runner for browser command execution."""

from datetime import datetime
import json
import logging
import sqlite3
from typing import Any, Dict, List

from .command_router import CommandRouter
from borisbot.supervisor.browser_capability_guard import BrowserCapabilityGuard
from borisbot.supervisor.database import get_db

logger = logging.getLogger("borisbot.browser.task_runner")


class TaskRunner:
    """
    Deterministic sequential task executor.
    """

    def __init__(self, router: CommandRouter, agent_id: str, pre_persisted: bool = False):
        self._router = router
        self.agent_id = agent_id
        self.pre_persisted = pre_persisted

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
            return {"task_id": task_id, "status": row["status"], "steps": []}

        if not self.pre_persisted:
            await self._insert_task(task_id, agent_id, task)
        await self._mark_task_running(task_id)

        final_status = "completed"
        report: Dict[str, Any]
        guard_error: Exception | None = None
        try:
            await BrowserCapabilityGuard().validate_task(agent_id, task)
        except Exception as error:
            guard_error = error

        if guard_error is not None:
            final_status = "rejected"
            report = {
                "task_id": task["task_id"],
                "status": "rejected",
                "reason": str(guard_error),
                "steps": [],
            }
        else:
            results: List[Dict[str, Any]] = []
            for command in task["commands"]:
                command_id = command.get("id", "unknown")

                try:
                    result = await self._router.execute(command)

                    step_record = {
                        "command_id": command_id,
                        "status": "ok",
                    }

                    if "result" in result:
                        step_record["result"] = result["result"]

                    results.append(step_record)

                except Exception as e:
                    final_status = "failed"
                    results.append(
                        {
                            "command_id": command_id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    break

            report = {
                "task_id": task["task_id"],
                "status": final_status,
                "steps": results,
            }

        await self._finalize_task(task_id, final_status, report)
        return report
