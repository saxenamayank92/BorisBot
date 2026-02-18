"""Tests for heartbeat supervisor snapshot persistence and restart recovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from borisbot.supervisor.heartbeat_runtime import HeartbeatSupervisor, read_heartbeat_snapshot


class _FakeCostGuard:
    async def get_budget_status(self, agent_id: str) -> dict[str, float | str]:
        return {
            "status": "ok",
            "daily_spend": 1.25,
            "daily_limit": 20.0,
            "daily_remaining": 18.75,
        }


class _FakeHealthRegistry:
    def __init__(self) -> None:
        self._state = "unknown"

    def get_snapshot(self) -> dict:
        return {"ollama": {"state": self._state}}

    def get_provider_state(self, provider_name: str) -> str:
        return self._state

    def mark_probe_success(self, provider_name: str) -> None:
        self._state = "healthy"

    def mark_probe_failure(self, provider_name: str) -> None:
        self._state = "unhealthy"


async def _fake_probe_ok(registry, model_name: str, provider_name: str) -> bool:
    registry.mark_probe_success(provider_name)
    return True


class _TestHeartbeatSupervisor(HeartbeatSupervisor):
    async def _get_task_counts(self) -> tuple[int, int]:
        return (2, 3)


class HeartbeatRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Validate heartbeat snapshot shape and startup reconciliation."""

    async def test_collect_and_persist_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_path = Path(tmpdir) / "runtime" / "heartbeat.json"
            supervisor = _TestHeartbeatSupervisor(
                cost_guard=_FakeCostGuard(),
                health_registry=_FakeHealthRegistry(),
                heartbeat_path=heartbeat_path,
                probe_fn=_fake_probe_ok,
            )
            snapshot = await supervisor.collect_snapshot()
            self.assertEqual(snapshot["provider_health"], "healthy")
            self.assertEqual(snapshot["active_tasks"], 2)
            self.assertTrue(snapshot["self_heal_probe_ok"])
            await supervisor.write_snapshot(snapshot)
            persisted = read_heartbeat_snapshot(heartbeat_path)
            assert persisted is not None
            self.assertEqual(persisted["queue_depth"], 3)

    async def test_initialize_on_startup_includes_recovery_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_path = Path(tmpdir) / "runtime" / "heartbeat.json"
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                '{"timestamp":"2026-02-18T00:00:00","active_tasks":5}',
                encoding="utf-8",
            )
            supervisor = _TestHeartbeatSupervisor(
                cost_guard=_FakeCostGuard(),
                health_registry=_FakeHealthRegistry(),
                heartbeat_path=heartbeat_path,
                probe_fn=_fake_probe_ok,
            )
            snapshot = await supervisor.initialize_on_startup()
            self.assertEqual(snapshot["startup_recovered_from"], "2026-02-18T00:00:00")
            self.assertIn("reconciliation_note", snapshot)


if __name__ == "__main__":
    unittest.main()
