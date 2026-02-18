"""Tests for per-agent tool permission matrix helpers."""

import tempfile
import unittest
from pathlib import Path

from borisbot.supervisor.tool_permissions import (
    DECISION_ALLOW,
    DECISION_PROMPT,
    TOOL_BROWSER,
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


if __name__ == "__main__":
    unittest.main()

