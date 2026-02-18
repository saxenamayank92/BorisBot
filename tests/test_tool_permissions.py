"""Tests for per-agent tool permission matrix helpers."""

import tempfile
import unittest
from pathlib import Path

from borisbot.supervisor.tool_permissions import (
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_PROMPT,
    TOOL_ASSISTANT,
    TOOL_BROWSER,
    TOOL_FILESYSTEM,
    get_agent_permission_matrix_sync,
    get_agent_tool_permission_sync,
    set_agent_tool_permission_sync,
)


class ToolPermissionTests(unittest.TestCase):
    """Ensure permission matrix defaults and persistence are deterministic."""

    def test_default_permission_is_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "permissions.db"
            decision = get_agent_tool_permission_sync("agent_a", TOOL_BROWSER, db_path=db_path)
            self.assertEqual(decision, DECISION_PROMPT)

    def test_set_and_get_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "permissions.db"
            set_agent_tool_permission_sync(
                "agent_a",
                TOOL_BROWSER,
                DECISION_ALLOW,
                db_path=db_path,
            )
            decision = get_agent_tool_permission_sync("agent_a", TOOL_BROWSER, db_path=db_path)
            self.assertEqual(decision, DECISION_ALLOW)

    def test_permission_matrix_defaults_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "permissions.db"
            before = get_agent_permission_matrix_sync("agent_a", db_path=db_path)
            self.assertEqual(before[TOOL_ASSISTANT], DECISION_PROMPT)
            self.assertEqual(before[TOOL_BROWSER], DECISION_PROMPT)
            self.assertEqual(before[TOOL_FILESYSTEM], DECISION_PROMPT)
            set_agent_tool_permission_sync(
                "agent_a",
                TOOL_FILESYSTEM,
                DECISION_DENY,
                db_path=db_path,
            )
            after = get_agent_permission_matrix_sync("agent_a", db_path=db_path)
            self.assertEqual(after[TOOL_FILESYSTEM], DECISION_DENY)


if __name__ == "__main__":
    unittest.main()
