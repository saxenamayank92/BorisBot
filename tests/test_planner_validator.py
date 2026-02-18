"""Tests for planner validator safety boundaries and conversion."""

import unittest

from borisbot.llm.errors import LLMInvalidOutputError
from borisbot.llm.planner_validator import validate_and_convert_plan


class PlannerValidatorTests(unittest.TestCase):
    """Ensure planner action conversion is deterministic and safe."""

    def test_convert_supported_actions(self) -> None:
        plan = {
            "planner_schema_version": "planner.v1",
            "intent": "x",
            "proposed_actions": [
                {"action": "navigate", "target": "https://example.com", "input": ""},
                {"action": "type", "target": "#email", "input": "user@example.com"},
            ],
        }
        commands = validate_and_convert_plan(plan)
        self.assertEqual(commands[0]["action"], "navigate")
        self.assertEqual(commands[1]["action"], "type")

    def test_blocked_action_rejected(self) -> None:
        plan = {
            "planner_schema_version": "planner.v1",
            "intent": "x",
            "proposed_actions": [
                {"action": "execute_js", "target": "window.x=1", "input": ""},
            ],
        }
        with self.assertRaises(LLMInvalidOutputError):
            validate_and_convert_plan(plan)


if __name__ == "__main__":
    unittest.main()

