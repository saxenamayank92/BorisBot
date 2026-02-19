"""Tests for strict planner.v1 contract parsing and repair limits."""

import unittest
from unittest import mock

from borisbot.llm.errors import LLMInvalidOutputError
from borisbot.llm import planner_contract
from borisbot.llm.planner_contract import parse_planner_output


class PlannerContractTests(unittest.TestCase):
    """Validate strict schema and one-pass repair behavior."""

    def test_parse_valid_payload(self) -> None:
        payload = """
        {
          "planner_schema_version": "planner.v1",
          "intent": "open home and click login",
          "proposed_actions": [
            {"action": "navigate", "target": "https://example.com", "input": ""},
            {"action": "click", "target": "#login", "input": ""}
          ]
        }
        """
        parsed = parse_planner_output(payload)
        self.assertEqual(parsed["planner_schema_version"], "planner.v1")
        self.assertEqual(len(parsed["proposed_actions"]), 2)

    def test_single_repair_pass_extracts_json_once(self) -> None:
        raw = 'text prefix {"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[]}'
        parsed = parse_planner_output(raw)
        self.assertEqual(parsed["intent"], "x")

    def test_rejects_extra_fields(self) -> None:
        raw = (
            '{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[],"extra":"nope"}'
        )
        with self.assertRaises(LLMInvalidOutputError):
            parse_planner_output(raw)

    def test_rejects_invalid_after_single_repair(self) -> None:
        raw = "not json and no braces"
        with self.assertRaises(LLMInvalidOutputError):
            parse_planner_output(raw)

    def test_json_repair_fallback_used_when_available(self) -> None:
        raw = "not-quite-json"
        repaired = {
            "planner_schema_version": "planner.v1",
            "intent": "x",
            "proposed_actions": [],
        }
        with mock.patch.object(planner_contract, "_repair_json_fn", return_value=repaired):
            parsed = parse_planner_output(raw)
        self.assertEqual(parsed["intent"], "x")


if __name__ == "__main__":
    unittest.main()
