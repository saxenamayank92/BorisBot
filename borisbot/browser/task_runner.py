"""Deterministic sequential task runner for browser command execution."""

import logging
from typing import Any, Dict, List

from .command_router import CommandRouter

logger = logging.getLogger("borisbot.browser.task_runner")


class TaskRunner:
    """
    Deterministic sequential task executor.
    """

    def __init__(self, router: CommandRouter):
        self._router = router

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
                results.append(
                    {
                        "command_id": command_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )

                return {
                    "task_id": task["task_id"],
                    "status": "failed",
                    "steps": results,
                }

        return {
            "task_id": task["task_id"],
            "status": "completed",
            "steps": results,
        }
