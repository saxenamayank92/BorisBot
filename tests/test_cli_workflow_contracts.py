"""Tests for CLI workflow contract validation and lint gating semantics."""

import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
import io

import typer

from borisbot.cli import _load_and_validate_workflow, lint_workflow
from borisbot.cli import _compute_lint_violations


class CliWorkflowContractTests(unittest.TestCase):
    """Validate workflow schema checks and deterministic lint exit behavior."""

    def _write_workflow(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with tmp:
            json.dump(payload, tmp)
        return Path(tmp.name)

    def test_load_and_validate_rejects_unknown_schema(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v99",
                "task_id": "wf_invalid",
                "commands": [],
            }
        )
        with self.assertRaises(ValueError):
            _load_and_validate_workflow(workflow_path)

    def test_lint_workflow_exits_nonzero_for_high_risk(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v1",
                "task_id": "wf_risky",
                "commands": [
                    {"id": "1", "action": "click", "params": {"selector": "button"}},
                    {"id": "2", "action": "type", "params": {"selector": "button", "text": "x"}},
                ],
            }
        )
        with self.assertRaises(typer.Exit) as cm:
            with redirect_stdout(io.StringIO()):
                lint_workflow(workflow_path, min_average_score=95.0, max_fragile=0, max_high_risk=0)
        self.assertEqual(cm.exception.exit_code, 1)

    def test_lint_workflow_passes_for_stable_selector(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v1",
                "task_id": "wf_stable",
                "commands": [
                    {
                        "id": "1",
                        "action": "click",
                        "params": {"selector": "[data-testid=\"submit\"]"},
                    }
                ],
            }
        )
        try:
            with redirect_stdout(io.StringIO()):
                lint_workflow(workflow_path, min_average_score=85.0, max_fragile=0, max_high_risk=0)
        except typer.Exit as exc:
            self.fail(f"lint_workflow unexpectedly exited: {exc.exit_code}")

    def test_compute_lint_violations_thresholds(self) -> None:
        report = {
            "summary": {
                "average_score": 60.0,
                "fragile": 6,
                "high_risk": 1,
            }
        }
        violations = _compute_lint_violations(
            report,
            min_average_score=70.0,
            max_fragile=5,
            max_high_risk=0,
        )
        self.assertEqual(len(violations), 3)


if __name__ == "__main__":
    unittest.main()
