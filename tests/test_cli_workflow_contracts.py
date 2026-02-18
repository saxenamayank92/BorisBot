"""Tests for CLI workflow contract validation and lint gating semantics."""

import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
import io
from unittest import mock

import typer

from borisbot.cli import (
    _build_doctor_report,
    _load_and_validate_workflow,
    assistant_chat,
    chat_clear,
    chat_history,
    llm_setup,
    lint_workflow,
    policy_apply,
    policy_list,
    plan_preview,
    permissions,
    provider_status,
    provider_test,
    release_check,
    doctor,
    session_status,
    setup,
    set_permission,
)
from borisbot.cli import _compute_lint_violations
from borisbot.cli import _format_record_runtime_error


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

    def test_release_check_includes_failure_summary(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v1",
                "task_id": "wf_risky_release",
                "commands": [
                    {"id": "1", "action": "click", "params": {"selector": "button"}},
                ],
            }
        )
        with mock.patch(
            "borisbot.cli._run_verify_suite",
            return_value={"returncode": 0, "stdout": "", "stderr": ""},
        ), mock.patch(
            "borisbot.cli._run_golden_suite",
            return_value={"returncode": 0, "stdout": "Ran 1 tests\n\nOK\n", "stderr": ""},
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    release_check(
                        [workflow_path],
                        min_average_score=90.0,
                        max_fragile=0,
                        max_high_risk=0,
                        json_output=True,
                    )
            self.assertEqual(cm.exception.exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["workflows"][0]["status"], "failed")
            self.assertIn("failure", payload["workflows"][0])
            self.assertEqual(payload["workflows"][0]["failure"]["error_schema_version"], "error.v1")

    def test_release_check_human_output_compact(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v1",
                "task_id": "wf_human",
                "commands": [
                    {"id": "1", "action": "click", "params": {"selector": "button"}},
                ],
            }
        )
        with mock.patch(
            "borisbot.cli._run_verify_suite",
            return_value={"returncode": 0, "stdout": "Ran 17 tests\n\nOK\n", "stderr": ""},
        ), mock.patch(
            "borisbot.cli._run_golden_suite",
            return_value={"returncode": 0, "stdout": "Ran 1 tests\n\nOK\n", "stderr": ""},
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit):
                with redirect_stdout(output):
                    release_check(
                        [workflow_path],
                        min_average_score=90.0,
                        max_fragile=0,
                        max_high_risk=0,
                    )
            text = output.getvalue()
            self.assertIn("RELEASE CHECK: FAIL", text)
            self.assertIn("Workflow:", text)

    def test_release_check_json_includes_golden_status(self) -> None:
        workflow_path = self._write_workflow(
            {
                "schema_version": "task_command.v1",
                "task_id": "wf_golden",
                "commands": [
                    {
                        "id": "1",
                        "action": "click",
                        "params": {"selector": "[data-testid=\"submit\"]"},
                    }
                ],
            }
        )
        with mock.patch(
            "borisbot.cli._run_verify_suite",
            return_value={"returncode": 0, "stdout": "Ran 17 tests\n\nOK\n", "stderr": ""},
        ), mock.patch(
            "borisbot.cli._run_golden_suite",
            return_value={"returncode": 0, "stdout": "Ran 1 tests\n\nOK\n", "stderr": ""},
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                release_check(
                    [workflow_path],
                    min_average_score=0.0,
                    max_fragile=999,
                    max_high_risk=999,
                    json_output=True,
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["golden_status"], "ok")
            self.assertIn("golden", payload)

    def test_format_record_runtime_error_docker_down(self) -> None:
        msg = _format_record_runtime_error(
            RuntimeError("Cannot connect to the Docker daemon at unix:///tmp/docker.sock")
        )
        self.assertIn("Docker is not running", msg)

    def test_format_record_runtime_error_session_limit(self) -> None:
        msg = _format_record_runtime_error(RuntimeError("Maximum browser sessions reached"))
        self.assertIn("cleanup-browsers", msg)

    def test_plan_preview_human_ok(self) -> None:
        with mock.patch(
            "borisbot.cli._build_dry_run_preview",
            return_value={
                "status": "ok",
                "provider_name": "ollama",
                "validated_commands": [{"id": "1", "action": "get_title", "params": {}}],
                "token_estimate": {"total_tokens": 42},
                "cost_estimate_usd": 0.0,
            },
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                plan_preview("test prompt", json_output=False)
            text = output.getvalue()
            self.assertIn("PLAN PREVIEW: OK", text)
            self.assertIn("provider: ollama", text)

    def test_plan_preview_json_fail_exits_nonzero(self) -> None:
        with mock.patch(
            "borisbot.cli._build_dry_run_preview",
            return_value={"status": "failed", "error_code": "LLM_PROVIDER_UNHEALTHY"},
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    plan_preview("test prompt", json_output=True)
            self.assertEqual(cm.exception.exit_code, 1)

    def test_provider_status_human(self) -> None:
        with mock.patch(
            "borisbot.cli._collect_runtime_status",
            return_value={
                "provider_name": "ollama",
                "provider_matrix": {
                    "ollama": {"enabled": True, "configured": True, "usable": True, "reason": ""},
                    "openai": {"enabled": True, "configured": False, "usable": False, "reason": "api_key_missing"},
                },
            },
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                provider_status(json_output=False)
            text = output.getvalue()
            self.assertIn("PROVIDER STATUS", text)
            self.assertIn("openai", text)

    def test_session_status_json_output(self) -> None:
        with mock.patch(
            "borisbot.cli._build_session_status",
            return_value={
                "provider_name": "ollama",
                "provider_state": "healthy",
                "model_name": "llama3.2:3b",
                "session_tokens": 12,
                "session_cost_usd": 0.0,
                "today_cost_usd": 0.0,
                "daily_limit_usd": 1.0,
                "daily_remaining_usd": 1.0,
                "budget_status": "OK",
                "active_tasks": 0,
                "queue_depth": 0,
                "heartbeat_timestamp": "",
                "heartbeat_age_seconds": -1,
            },
        ), mock.patch("borisbot.cli.read_heartbeat_snapshot", return_value={}):
            output = io.StringIO()
            with redirect_stdout(output):
                session_status(json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["provider_name"], "ollama")
            self.assertEqual(payload["budget_status"], "OK")

    def test_provider_test_fail_exits_nonzero(self) -> None:
        with mock.patch("borisbot.cli._probe_provider_connection", return_value=(False, "missing key")):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    provider_test(provider_name="openai", model_name="gpt-4o-mini")
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("PROVIDER TEST: FAIL", output.getvalue())

    def test_assistant_chat_human_ok(self) -> None:
        with mock.patch("borisbot.cli.get_agent_tool_permission_sync", return_value="allow"), mock.patch(
            "borisbot.cli._build_assistant_response",
            return_value={
                "status": "ok",
                "provider_name": "ollama",
                "message": "hello",
                "token_estimate": {"total_tokens": 12},
                "cost_estimate_usd": 0.0,
            },
        ), mock.patch("borisbot.cli.append_chat_message"):
            output = io.StringIO()
            with redirect_stdout(output):
                assistant_chat("say hi", json_output=False)
            text = output.getvalue()
            self.assertIn("ASSISTANT CHAT: OK", text)
            self.assertIn("provider: ollama", text)
            self.assertIn("hello", text)

    def test_assistant_chat_json_fail_exits_nonzero(self) -> None:
        with mock.patch("borisbot.cli.get_agent_tool_permission_sync", return_value="allow"), mock.patch(
            "borisbot.cli._build_assistant_response",
            return_value={"status": "failed", "error_code": "LLM_PROVIDER_UNHEALTHY"},
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    assistant_chat("say hi", json_output=True)
            self.assertEqual(cm.exception.exit_code, 1)

    def test_assistant_chat_requires_permission_without_approve_flag(self) -> None:
        with mock.patch("borisbot.cli.get_agent_tool_permission_sync", return_value="prompt"):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    assistant_chat("say hi", json_output=False)
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("ASSISTANT_PERMISSION_REQUIRED", output.getvalue())

    def test_assistant_chat_approve_permission_sets_allow(self) -> None:
        with mock.patch("borisbot.cli.get_agent_tool_permission_sync", return_value="prompt"), mock.patch(
            "borisbot.cli.set_agent_tool_permission_sync"
        ) as set_perm, mock.patch(
            "borisbot.cli._build_assistant_response",
            return_value={
                "status": "ok",
                "provider_name": "ollama",
                "message": "hello",
                "token_estimate": {"total_tokens": 1},
                "cost_estimate_usd": 0.0,
            },
        ), mock.patch("borisbot.cli.append_chat_message"):
            output = io.StringIO()
            with redirect_stdout(output):
                assistant_chat("say hi", approve_permission=True, json_output=False)
            set_perm.assert_called_once()
            self.assertIn("ASSISTANT CHAT: OK", output.getvalue())

    def test_assistant_chat_no_save_history(self) -> None:
        with mock.patch("borisbot.cli.get_agent_tool_permission_sync", return_value="allow"), mock.patch(
            "borisbot.cli._build_assistant_response",
            return_value={
                "status": "ok",
                "provider_name": "ollama",
                "message": "hello",
                "token_estimate": {"total_tokens": 1},
                "cost_estimate_usd": 0.0,
            },
        ), mock.patch("borisbot.cli.append_chat_message") as append_mock:
            output = io.StringIO()
            with redirect_stdout(output):
                assistant_chat("say hi", save_history=False, json_output=False)
            append_mock.assert_not_called()

    def test_chat_history_json(self) -> None:
        with mock.patch(
            "borisbot.cli.load_chat_history",
            return_value=[{"role": "assistant", "text": "hello"}],
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                chat_history(agent_id="default", json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["agent_id"], "default")
            self.assertEqual(len(payload["items"]), 1)

    def test_chat_clear_assistant_only(self) -> None:
        with mock.patch("borisbot.cli.clear_chat_roles") as clear_roles:
            output = io.StringIO()
            with redirect_stdout(output):
                chat_clear(agent_id="default", assistant_only=True)
            clear_roles.assert_called_once()
            self.assertIn("assistant_only=true", output.getvalue())

    def test_permissions_human_output(self) -> None:
        with mock.patch(
            "borisbot.cli.get_agent_permission_matrix_sync",
            return_value={"assistant": "allow", "browser": "prompt"},
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                permissions(agent_id="default", json_output=False)
            text = output.getvalue()
            self.assertIn("PERMISSIONS (default)", text)
            self.assertIn("assistant: allow", text)

    def test_permissions_json_output(self) -> None:
        with mock.patch(
            "borisbot.cli.get_agent_permission_matrix_sync",
            return_value={"assistant": "allow"},
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                permissions(agent_id="default", json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["agent_id"], "default")
            self.assertEqual(payload["permissions"]["assistant"], "allow")

    def test_set_permission_ok(self) -> None:
        with mock.patch("borisbot.cli.set_agent_tool_permission_sync") as set_perm:
            output = io.StringIO()
            with redirect_stdout(output):
                set_permission(tool_name="assistant", decision="allow", agent_id="agent-1")
            set_perm.assert_called_once_with("agent-1", "assistant", "allow")
            self.assertIn("SET PERMISSION: OK", output.getvalue())

    def test_set_permission_rejects_unknown_tool(self) -> None:
        output = io.StringIO()
        with self.assertRaises(typer.Exit) as cm:
            with redirect_stdout(output):
                set_permission(tool_name="unknown", decision="allow", agent_id="default")
        self.assertEqual(cm.exception.exit_code, 1)
        self.assertIn("unsupported tool", output.getvalue())

    def test_set_permission_rejects_unknown_decision(self) -> None:
        output = io.StringIO()
        with self.assertRaises(typer.Exit) as cm:
            with redirect_stdout(output):
                set_permission(tool_name="assistant", decision="maybe", agent_id="default")
        self.assertEqual(cm.exception.exit_code, 1)
        self.assertIn("unsupported decision", output.getvalue())

    def test_policy_apply_unknown_policy_fails(self) -> None:
        output = io.StringIO()
        with self.assertRaises(typer.Exit) as cm:
            with redirect_stdout(output):
                policy_apply(policy_name="nope", agent_id="default", json_output=False)
        self.assertEqual(cm.exception.exit_code, 1)
        self.assertIn("UNKNOWN_POLICY", output.getvalue())

    def test_policy_apply_updates_profile_and_permissions(self) -> None:
        base_profile = {
            "schema_version": "profile.v2",
            "agent_name": "default",
            "primary_provider": "ollama",
            "provider_chain": ["ollama"],
            "model_name": "llama3.2:3b",
            "provider_settings": {
                "ollama": {"enabled": True, "model_name": ""},
                "openai": {"enabled": False, "model_name": ""},
                "anthropic": {"enabled": False, "model_name": ""},
                "google": {"enabled": False, "model_name": ""},
                "azure": {"enabled": False, "model_name": ""},
            },
        }
        with mock.patch("borisbot.cli.load_profile", return_value=base_profile.copy()), mock.patch(
            "borisbot.cli.save_profile",
            side_effect=lambda profile: profile,
        ) as save_profile_mock, mock.patch(
            "borisbot.cli.set_agent_tool_permission_sync"
        ) as set_perm:
            output = io.StringIO()
            with redirect_stdout(output):
                policy_apply(policy_name="automation", agent_id="default", json_output=False)
            self.assertIn("POLICY APPLY: OK", output.getvalue())
            save_profile_mock.assert_called_once()
            self.assertGreaterEqual(set_perm.call_count, 1)

    def test_policy_apply_json_output(self) -> None:
        base_profile = {
            "schema_version": "profile.v2",
            "agent_name": "default",
            "primary_provider": "ollama",
            "provider_chain": ["ollama"],
            "model_name": "llama3.2:3b",
            "provider_settings": {
                "ollama": {"enabled": True, "model_name": ""},
                "openai": {"enabled": False, "model_name": ""},
                "anthropic": {"enabled": False, "model_name": ""},
                "google": {"enabled": False, "model_name": ""},
                "azure": {"enabled": False, "model_name": ""},
            },
        }
        with mock.patch("borisbot.cli.load_profile", return_value=base_profile.copy()), mock.patch(
            "borisbot.cli.save_profile",
            side_effect=lambda profile: profile,
        ), mock.patch("borisbot.cli.set_agent_tool_permission_sync"):
            output = io.StringIO()
            with redirect_stdout(output):
                policy_apply(policy_name="safe-local", agent_id="default", json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["policy"], "safe-local")

    def test_policy_list_human_output(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            policy_list(json_output=False)
        text = output.getvalue()
        self.assertIn("POLICY PACKS", text)
        self.assertIn("safe-local", text)

    def test_policy_list_json_output(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            policy_list(json_output=True)
        payload = json.loads(output.getvalue())
        self.assertTrue(isinstance(payload.get("items"), list))
        self.assertGreaterEqual(len(payload["items"]), 1)

    def test_llm_setup_fails_when_missing_without_auto_install(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value=None):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    llm_setup(model_name="llama3.2:3b", auto_install=False)
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("LLM SETUP: FAIL", output.getvalue())

    def test_llm_setup_success_path(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value="/usr/bin/ollama"), mock.patch(
            "borisbot.cli._resolve_ollama_start_command",
            return_value=["ollama", "serve"],
        ), mock.patch(
            "borisbot.cli._run_setup_command",
            side_effect=[(0, "started"), (0, "pulled")],
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                llm_setup(model_name="llama3.2:3b", auto_install=True)
            text = output.getvalue()
            self.assertIn("LLM SETUP: OK", text)

    def test_llm_setup_install_command_unavailable(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value=None), mock.patch(
            "borisbot.cli._resolve_ollama_install_command",
            side_effect=ValueError("unsupported platform"),
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    llm_setup(model_name="llama3.2:3b", auto_install=True)
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("unsupported platform", output.getvalue())

    def test_llm_setup_json_success_payload(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value="/usr/bin/ollama"), mock.patch(
            "borisbot.cli._resolve_ollama_start_command",
            return_value=["ollama", "serve"],
        ), mock.patch(
            "borisbot.cli._run_setup_command",
            side_effect=[(0, "started"), (0, "pulled")],
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                llm_setup(model_name="llama3.2:3b", auto_install=True, json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["model"], "llama3.2:3b")
            self.assertEqual(len(payload["steps"]), 2)

    def test_llm_setup_json_fail_payload(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value=None):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    llm_setup(model_name="llama3.2:3b", auto_install=False, json_output=True)
            self.assertEqual(cm.exception.exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"], "OLLAMA_NOT_INSTALLED")

    def test_setup_json_payload_without_guide_launch(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value="/usr/bin/docker"), mock.patch(
            "borisbot.cli._run_setup_command",
            return_value=(0, "docker ok"),
        ), mock.patch(
            "borisbot.cli.llm_setup",
            return_value=None,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                setup(launch_guide=False, json_output=True)
            payload = json.loads(output.getvalue())
            self.assertIn("docker", payload)
            self.assertIn("llm_setup", payload)
            self.assertEqual(payload["guide"]["launch_requested"], False)

    def test_setup_launches_guide_when_requested(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value="/usr/bin/docker"), mock.patch(
            "borisbot.cli._run_setup_command",
            return_value=(0, "docker ok"),
        ), mock.patch(
            "borisbot.cli.llm_setup",
            return_value=None,
        ), mock.patch("borisbot.cli.run_guide_server") as run_guide:
            output = io.StringIO()
            with redirect_stdout(output):
                setup(launch_guide=True, json_output=False, guide_host="127.0.0.1", guide_port=7799)
            run_guide.assert_called_once()

    def test_setup_nonzero_when_guide_disabled_and_setup_fails(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", return_value=None), mock.patch(
            "borisbot.cli.llm_setup",
            side_effect=typer.Exit(code=1),
        ):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    setup(launch_guide=False, json_output=False)
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("SETUP: WARN", output.getvalue())

    def test_build_doctor_report_ok(self) -> None:
        with mock.patch("borisbot.cli.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), mock.patch(
            "borisbot.cli._run_setup_command",
            return_value=(0, "docker ok"),
        ), mock.patch("borisbot.cli.httpx.get") as http_get:
            http_get.return_value = mock.Mock(
                status_code=200,
                json=lambda: {"models": [{"name": "llama3.2:3b"}]},
            )
            payload = _build_doctor_report("llama3.2:3b")
        self.assertEqual(payload["status"], "ok")

    def test_doctor_strict_exits_nonzero_on_warn(self) -> None:
        with mock.patch("borisbot.cli._build_doctor_report", return_value={"status": "warn", "docker": {}, "ollama": {}, "model": "llama3.2:3b"}):
            output = io.StringIO()
            with self.assertRaises(typer.Exit) as cm:
                with redirect_stdout(output):
                    doctor(model_name="llama3.2:3b", strict=True, json_output=False)
            self.assertEqual(cm.exception.exit_code, 1)
            self.assertIn("DOCTOR: WARN", output.getvalue())

    def test_doctor_json_output(self) -> None:
        with mock.patch(
            "borisbot.cli._build_doctor_report",
            return_value={"status": "ok", "model": "llama3.2:3b", "docker": {}, "ollama": {}},
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                doctor(model_name="llama3.2:3b", strict=False, json_output=True)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
