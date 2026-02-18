"""Golden regression tests for planner.v1 parse + validator conversion."""

import json
import unittest
from pathlib import Path

from borisbot.llm.planner_contract import parse_planner_output
from borisbot.llm.planner_validator import validate_and_convert_plan


class GoldenPlannerRegressionTests(unittest.TestCase):
    """Ensure planner parse/validator output remains deterministic over releases."""

    def test_golden_cases(self) -> None:
        fixture_path = Path(__file__).parent / "golden" / "planner_v1_cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case.get("name")):
                parsed = parse_planner_output(case["raw_output"])
                commands = validate_and_convert_plan(parsed)
                self.assertEqual(commands, case["expected_commands"])


if __name__ == "__main__":
    unittest.main()

