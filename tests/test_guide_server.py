"""Tests for guided UI command mapping and validation."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from borisbot.guide.server import (
    GuideState,
    _resolve_ollama_install_command,
    _resolve_ollama_start_command,
    _estimate_tokens,
    _extract_required_tools_from_plan,
    build_action_command,
    extract_browser_ui_url,
    _render_html,
    required_tool_for_action,
)


class GuideServerCommandTests(unittest.TestCase):
    """Ensure guide actions map to stable CLI commands."""

    def test_release_check_json_command(self) -> None:
        cmd = build_action_command(
            "release_check_json",
            {"workflow_path": "workflows/sample.json"},
            workspace=Path.cwd(),
            python_bin=sys.executable,
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "borisbot.cli",
                "release-check",
                "workflows/sample.json",
                "--json",
            ],
        )

    def test_cleanup_sessions_command(self) -> None:
        cmd = build_action_command(
            "cleanup_sessions",
            {},
            workspace=Path.cwd(),
            python_bin=sys.executable,
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "borisbot.cli",
                "cleanup-browsers",
            ],
        )

    def test_ollama_pull_command_uses_model_param(self) -> None:
        cmd = build_action_command(
            "ollama_pull",
            {"model_name": "llama3.2:3b"},
            workspace=Path.cwd(),
            python_bin=sys.executable,
        )
        self.assertEqual(cmd, ["ollama", "pull", "llama3.2:3b"])

    def test_session_status_command(self) -> None:
        cmd = build_action_command(
            "session_status",
            {},
            workspace=Path.cwd(),
            python_bin=sys.executable,
        )
        self.assertEqual(cmd, [sys.executable, "-m", "borisbot.cli", "session-status"])

    def test_ollama_install_command(self) -> None:
        cmd = _resolve_ollama_install_command(
            "darwin",
            which=lambda _: "/opt/homebrew/bin/brew",
        )
        self.assertEqual(cmd, ["brew", "install", "ollama"])

    def test_ollama_install_linux_command(self) -> None:
        cmd = _resolve_ollama_install_command("linux")
        self.assertEqual(cmd, ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])

    def test_ollama_start_linux_prefers_systemctl(self) -> None:
        cmd = _resolve_ollama_start_command(
            "linux",
            which=lambda name: "/bin/systemctl" if name == "systemctl" else None,
        )
        self.assertEqual(cmd, ["systemctl", "--user", "start", "ollama"])

    def test_required_tool_mapping(self) -> None:
        self.assertEqual(required_tool_for_action("record"), "browser")
        self.assertEqual(required_tool_for_action("verify"), "shell")
        self.assertIsNone(required_tool_for_action("unknown_action"))

    def test_extract_required_tools_from_plan(self) -> None:
        plan = {
            "planner_schema_version": "planner.v1",
            "intent": "x",
            "proposed_actions": [
                {"action": "navigate", "target": "https://example.com", "input": ""},
                {"action": "web_fetch", "target": "https://example.com", "input": ""},
                {"action": "navigate", "target": "https://example.com", "input": ""},
            ],
        }
        self.assertEqual(_extract_required_tools_from_plan(plan), ["browser", "web_fetch"])

    def test_estimate_tokens_non_empty(self) -> None:
        self.assertGreaterEqual(_estimate_tokens("hello world"), 1)

    def test_record_rejects_invalid_url(self) -> None:
        with self.assertRaises(ValueError):
            build_action_command(
                "record",
                {"task_id": "wf_demo", "start_url": "linkedin.com"},
                workspace=Path.cwd(),
                python_bin=sys.executable,
            )

    def test_record_uses_defaults(self) -> None:
        cmd = build_action_command(
            "record",
            {},
            workspace=Path.cwd(),
            python_bin=sys.executable,
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "borisbot.cli",
                "record",
                "wf_new",
                "--start-url",
                "https://example.com",
            ],
        )

    def test_extract_browser_ui_url_uses_latest_match(self) -> None:
        output = (
            "line1\n"
            "Open browser UI at: http://localhost:49000\n"
            "other\n"
            "Open browser UI at: http://localhost:49001\n"
        )
        self.assertEqual(extract_browser_ui_url(output), "http://localhost:49001")

    def test_create_job_rejects_parallel_browser_jobs(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        with mock.patch.object(state, "_start_job", return_value=None):
            first = state.create_job(
                "record",
                {"task_id": "wf_demo", "start_url": "https://example.com"},
            )
        first.status = "running"
        with self.assertRaises(ValueError):
            state.create_job(
                "replay",
                {"workflow_path": "workflows/wf_demo.json"},
            )

    def test_create_job_rejects_duplicate_running_action(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        with mock.patch.object(state, "_start_job", return_value=None):
            first = state.create_job(
                "verify",
                {"agent_id": "default"},
            )
        first.status = "running"
        with self.assertRaises(ValueError):
            state.create_job(
                "verify",
                {"agent_id": "default"},
            )

    def test_add_plan_trace_exposed_in_list(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        trace = state.add_plan_trace(
            agent_id="default",
            model_name="llama3.2:3b",
            intent="do x",
            preview={"status": "ok"},
        )
        traces = state.list_traces()
        self.assertTrue(traces)
        self.assertEqual(traces[0]["trace_id"], trace["trace_id"])

    def test_get_trace_and_append_stage(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        trace = state.add_plan_trace(
            agent_id="default",
            model_name="llama3.2:3b",
            intent="do x",
            preview={"status": "ok", "validated_commands": [{"id": "1", "action": "get_title", "params": {}}]},
        )
        trace_id = trace["trace_id"]
        state.append_trace_stage(trace_id, {"event": "approved_execute_requested", "task_id": "t1"})
        got = state.get_trace(trace_id)
        self.assertIsNotNone(got)
        assert got is not None
        stages = got.get("stages", [])
        self.assertGreaterEqual(len(stages), 2)

    def test_list_trace_summaries_reports_stage_count(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        trace = state.add_plan_trace(
            agent_id="default",
            model_name="llama3.2:3b",
            intent="do y",
            preview={"status": "ok"},
        )
        state.append_trace_stage(trace["trace_id"], {"event": "planner_validated"})
        summaries = state.list_trace_summaries()
        self.assertTrue(summaries)
        latest = summaries[0]
        self.assertEqual(latest["trace_id"], trace["trace_id"])
        self.assertEqual(latest["stage_count"], 2)
        self.assertEqual(latest["last_event"], "planner_validated")

    def test_render_html_includes_one_touch_setup(self) -> None:
        html = _render_html(["workflows/sample.json"])
        self.assertIn("One-Touch LLM Setup", html)
        self.assertIn("runOneTouchLlmSetup()", html)
        self.assertIn("Planner Chat", html)
        self.assertIn("sendChatPrompt()", html)
        self.assertIn("Provider Onboarding", html)
        self.assertIn("refreshProviderSecrets()", html)
        self.assertIn("clearChatHistory()", html)


if __name__ == "__main__":
    unittest.main()
