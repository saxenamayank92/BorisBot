"""Integration-style reliability loop tests for recorder, replay routing, and task persistence."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from borisbot.browser.task_runner import TaskRunner
from borisbot.cli import _ReplayRouter
from borisbot.contracts import TASK_COMMAND_SCHEMA_V1, TASK_RESULT_SCHEMA_V1
from borisbot.recorder.session import RecordingSession
from borisbot.supervisor.capability_manager import CapabilityManager
from borisbot.supervisor.database import get_db, init_db

logger = logging.getLogger(__name__)


class _RouterStub:
    """Deterministic command router stub for TaskRunner integration tests."""

    def __init__(self, failing_command_id: str | None = None):
        self.failing_command_id = failing_command_id
        self.calls: list[dict] = []

    async def execute(self, command: dict) -> dict:
        self.calls.append(command)
        if self.failing_command_id is not None and command.get("id") == self.failing_command_id:
            raise RuntimeError("intentional failure")
        if command.get("action") == "get_title":
            return {"status": "ok", "result": "Example Domain"}
        return {"status": "ok"}


class _BaseReplayRouterStub:
    """Stubbed base router used to validate explicit fallback behavior."""

    def __init__(self):
        self.calls: list[dict] = []

    async def execute(self, command: dict) -> dict:
        self.calls.append(command)
        selector = command.get("params", {}).get("selector")
        if selector == "#good":
            return {"status": "ok"}
        raise RuntimeError(f"bad selector: {selector}")


class RecorderContractTests(unittest.TestCase):
    """Validate recording session normalization and schema version output."""

    def test_recording_session_filters_noise_and_collapses_navigate(self) -> None:
        logger.info("Testing recorder session host filtering and navigate collapsing.")
        session = RecordingSession(task_id="wf_1", start_url="https://www.linkedin.com")
        session.ingest("navigate", {"url": "https://google.com"})
        session.ingest("navigate", {"url": "https://www.linkedin.com/feed"})
        session.ingest("navigate", {"url": "https://www.linkedin.com/login"})
        session.ingest(
            "click",
            {
                "selector": "#submit",
                "fallback_selectors": ["#submit", "#submit", "[name=submit]", "#main button"],
            },
        )

        workflow = session.finalize()
        self.assertEqual(workflow["schema_version"], TASK_COMMAND_SCHEMA_V1)
        self.assertEqual(len(workflow["commands"]), 2)
        self.assertEqual(workflow["commands"][0]["action"], "navigate")
        self.assertEqual(workflow["commands"][0]["params"], {"url": "https://www.linkedin.com/login"})
        self.assertEqual(
            workflow["commands"][1]["params"]["fallback_selectors"],
            ["[name=submit]", "#main button"],
        )


class ReplayRouterFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Validate explicit fallback gating behavior for replay."""

    async def test_replay_router_no_fallback_when_disabled(self) -> None:
        logger.info("Testing replay router with fallback disabled.")
        base_router = _BaseReplayRouterStub()
        router = _ReplayRouter(base_router, allow_fallback=False)
        command = {
            "id": "1",
            "action": "click",
            "params": {"selector": "#bad", "fallback_selectors": ["#good"]},
        }
        with self.assertRaises(RuntimeError):
            await router.execute(command)
        self.assertEqual(len(base_router.calls), 1)

    async def test_replay_router_uses_fallback_when_enabled(self) -> None:
        logger.info("Testing replay router with explicit fallback enabled.")
        base_router = _BaseReplayRouterStub()
        router = _ReplayRouter(base_router, allow_fallback=True)
        command = {
            "id": "1",
            "action": "click",
            "params": {"selector": "#bad", "fallback_selectors": ["#good"]},
        }
        result = await router.execute(command)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(base_router.calls), 2)
        self.assertEqual(base_router.calls[1]["params"]["selector"], "#good")


class TaskRunnerPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Validate deterministic status persistence and event logging in TaskRunner."""

    async def asyncSetUp(self) -> None:
        logger.info("Initializing isolated sqlite DB for task runner persistence tests.")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "borisbot_test.db"
        self.db_patch = mock.patch("borisbot.supervisor.database.DB_PATH", self.db_path)
        self.db_patch.start()
        await init_db()

    async def asyncTearDown(self) -> None:
        self.db_patch.stop()
        self.temp_dir.cleanup()

    async def _fetch_task_row(self, task_id: str) -> dict | None:
        async for db in get_db():
            cursor = await db.execute(
                "SELECT task_id, agent_id, status, result FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None
        return None

    async def _fetch_count(self, sql: str, params: tuple = ()) -> int:
        async for db in get_db():
            cursor = await db.execute(sql, params)
            row = await cursor.fetchone()
            return int(row[0]) if row is not None else 0
        return 0

    async def _fetch_events(self, task_id: str) -> list[str]:
        async for db in get_db():
            cursor = await db.execute(
                "SELECT event_type FROM task_events WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            )
            rows = await cursor.fetchall()
            return [row["event_type"] for row in rows]
        return []

    async def test_task_runner_persists_completed_state(self) -> None:
        logger.info("Testing task runner completed status persistence.")
        agent_id = "agent_completed"
        await CapabilityManager.add_capability(agent_id, "BROWSER", "{}")
        router = _RouterStub()
        runner = TaskRunner(router, agent_id=agent_id)
        task = {
            "task_id": "task_completed",
            "commands": [
                {"id": "1", "action": "navigate", "params": {"url": "https://example.com"}},
                {"id": "2", "action": "get_title", "params": {}},
            ],
        }
        result = await runner.run(task)
        self.assertEqual(result["schema_version"], TASK_RESULT_SCHEMA_V1)
        self.assertEqual(result["status"], "completed")

        row = await self._fetch_task_row("task_completed")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "completed")
        persisted_result = json.loads(row["result"])
        self.assertEqual(persisted_result["status"], "completed")
        self.assertEqual(
            await self._fetch_count(
                "SELECT COUNT(*) FROM task_execution_logs WHERE task_id = ?",
                ("task_completed",),
            ),
            2,
        )
        self.assertIn("task_completed", await self._fetch_events("task_completed"))

    async def test_task_runner_persists_failed_state(self) -> None:
        logger.info("Testing task runner failed status persistence.")
        agent_id = "agent_failed"
        await CapabilityManager.add_capability(agent_id, "BROWSER", "{}")
        router = _RouterStub(failing_command_id="2")
        runner = TaskRunner(router, agent_id=agent_id)
        task = {
            "task_id": "task_failed",
            "commands": [
                {"id": "1", "action": "navigate", "params": {"url": "https://example.com"}},
                {"id": "2", "action": "click", "params": {"selector": "#missing"}},
                {"id": "3", "action": "get_title", "params": {}},
            ],
        }
        result = await runner.run(task)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][-1]["status"], "failed")

        row = await self._fetch_task_row("task_failed")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "failed")
        persisted_result = json.loads(row["result"])
        self.assertEqual(persisted_result["status"], "failed")
        self.assertIn("task_failed", await self._fetch_events("task_failed"))

    async def test_task_runner_persists_rejected_state(self) -> None:
        logger.info("Testing task runner rejected status persistence.")
        router = _RouterStub()
        runner = TaskRunner(router, agent_id="agent_rejected")
        task = {
            "task_id": "task_rejected",
            "commands": [
                {"id": "1", "action": "navigate", "params": {"url": "https://example.com"}},
            ],
        }
        result = await runner.run(task)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("missing capability: BROWSER", result["reason"])

        row = await self._fetch_task_row("task_rejected")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "rejected")
        persisted_result = json.loads(row["result"])
        self.assertEqual(persisted_result["status"], "rejected")
        self.assertEqual(
            await self._fetch_count(
                "SELECT COUNT(*) FROM task_execution_logs WHERE task_id = ?",
                ("task_rejected",),
            ),
            0,
        )

