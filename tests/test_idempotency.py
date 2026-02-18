"""Tests for task idempotency helper behavior."""

import unittest

from borisbot.supervisor.api_tasks import _payload_hash, _resolve_idempotency


class IdempotencyHelperTests(unittest.TestCase):
    """Validate deterministic idempotency helper semantics."""

    def test_payload_hash_stable(self) -> None:
        task_a = {"task_id": "t1", "commands": [{"id": "1", "action": "click", "params": {}}]}
        task_b = {"commands": [{"params": {}, "action": "click", "id": "1"}], "task_id": "t1"}
        self.assertEqual(_payload_hash(task_a), _payload_hash(task_b))

    def test_resolve_idempotency_hit(self) -> None:
        row = {"agent_id": "a1", "payload_hash": "abc", "task_id": "t1"}
        self.assertEqual(_resolve_idempotency(row, agent_id="a1", payload_hash="abc"), "hit")

    def test_resolve_idempotency_conflict(self) -> None:
        row = {"agent_id": "a1", "payload_hash": "abc", "task_id": "t1"}
        self.assertEqual(_resolve_idempotency(row, agent_id="a2", payload_hash="abc"), "conflict")
        self.assertEqual(_resolve_idempotency(row, agent_id="a1", payload_hash="xyz"), "conflict")


if __name__ == "__main__":
    unittest.main()

