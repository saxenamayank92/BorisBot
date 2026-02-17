"""Tests for static selector analyzer scoring and reporting."""

import unittest

from borisbot.recorder.analyzer import analyze_workflow_payload


class SelectorAnalyzerTests(unittest.TestCase):
    """Validate deterministic selector scoring and risk classification."""

    def test_scores_high_quality_selectors_as_stable(self) -> None:
        workflow = {
            "task_id": "wf_stable",
            "schema_version": "task_command.v1",
            "commands": [
                {
                    "id": "1",
                    "action": "click",
                    "params": {"selector": "[data-testid=\"submit-btn\"]", "fallback_selectors": ["button"]},
                }
            ],
        }
        report = analyze_workflow_payload(workflow)
        command = report["commands"][0]
        self.assertEqual(command["band"], "stable")
        self.assertGreaterEqual(command["score"], 90)

    def test_penalizes_broad_and_reused_selectors(self) -> None:
        workflow = {
            "task_id": "wf_fragile",
            "schema_version": "task_command.v1",
            "commands": [
                {"id": "1", "action": "click", "params": {"selector": "button"}},
                {"id": "2", "action": "type", "params": {"selector": "button", "text": "x"}},
            ],
        }
        report = analyze_workflow_payload(workflow)
        self.assertEqual(report["summary"]["selector_commands"], 2)
        self.assertEqual(report["summary"]["high_risk"], 2)
        self.assertTrue(all(item["score"] <= 10 for item in report["commands"]))

    def test_flags_dynamic_tokens(self) -> None:
        workflow = {
            "task_id": "wf_dynamic",
            "schema_version": "task_command.v1",
            "commands": [
                {"id": "1", "action": "click", "params": {"selector": "#react-123456"}},
            ],
        }
        report = analyze_workflow_payload(workflow)
        reasons = report["commands"][0]["reasons"]
        self.assertIn("contains long numeric token", reasons)
        self.assertLess(report["commands"][0]["score"], 90)


if __name__ == "__main__":
    unittest.main()
