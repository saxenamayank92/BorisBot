"""Tests for guided UI command mapping and validation."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from borisbot.guide.server import (
    GuideState,
    _collect_runtime_status,
    _build_ollama_setup_plan,
    _resolve_ollama_install_command,
    _resolve_ollama_start_command,
    _resolve_model_for_provider,
    _build_assistant_response,
    _build_dry_run_preview,
    _estimate_preview_cost_usd,
    _generate_plan_raw_with_provider,
    _probe_provider_connection,
    _provider_is_usable,
    _trace_already_executed,
    _normalize_browser_ui_url,
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

    def test_build_ollama_setup_plan_includes_commands(self) -> None:
        plan = _build_ollama_setup_plan(
            "llama3.2:3b",
            platform_name="linux",
            which=lambda _: "/usr/bin/fake",
        )
        self.assertEqual(plan["platform"], "linux")
        self.assertEqual(plan["pull_command"], ["ollama", "pull", "llama3.2:3b"])
        self.assertTrue(plan["install_command"])
        self.assertEqual(plan["start_command"], ["systemctl", "--user", "start", "ollama"])

    def test_build_ollama_setup_plan_handles_missing_installer(self) -> None:
        plan = _build_ollama_setup_plan(
            "llama3.2:3b",
            platform_name="darwin",
            which=lambda _: None,
        )
        self.assertIsNone(plan["install_command"])
        self.assertIn("Homebrew not found", plan["install_error"])

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

    def test_resolve_model_for_provider_prefers_requested(self) -> None:
        model = _resolve_model_for_provider("openai", "gpt-4o-mini")
        self.assertEqual(model, "gpt-4o-mini")

    def test_resolve_model_for_provider_uses_profile_setting(self) -> None:
        with mock.patch(
            "borisbot.guide.server.load_profile",
            return_value={
                "model_name": "llama3.2:3b",
                "provider_settings": {"anthropic": {"model_name": "claude-3-5-sonnet-latest"}},
            },
        ):
            model = _resolve_model_for_provider("anthropic", "")
        self.assertEqual(model, "claude-3-5-sonnet-latest")

    def test_estimate_preview_cost_non_ollama(self) -> None:
        cost = _estimate_preview_cost_usd("openai", 1000, 1000)
        self.assertGreater(cost, 0.0)

    def test_build_dry_run_preview_budget_blocked(self) -> None:
        with mock.patch("borisbot.guide.server._load_budget_snapshot", return_value={"blocked": True}):
            preview = _build_dry_run_preview(
                "open page",
                agent_id="default",
                model_name="llama3.2:3b",
                provider_name="openai",
            )
        self.assertEqual(preview["status"], "failed")
        self.assertEqual(preview["error_code"], "BUDGET_BLOCKED")

    def test_build_dry_run_preview_falls_back_to_ollama(self) -> None:
        with mock.patch("borisbot.guide.server._load_budget_snapshot", return_value={"blocked": False}), mock.patch(
            "borisbot.guide.server._resolve_provider_chain", return_value=["openai", "ollama"]
        ), mock.patch(
            "borisbot.guide.server._provider_is_usable",
            side_effect=[(False, "api_key_missing"), (True, "")],
        ), mock.patch(
            "borisbot.guide.server._generate_plan_raw_with_provider",
            return_value='{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[{"action":"get_title","target":"","input":""}]}',
        ):
            preview = _build_dry_run_preview(
                "open page",
                agent_id="default",
                model_name="llama3.2:3b",
                provider_name="openai",
            )
        self.assertEqual(preview["status"], "ok")
        self.assertEqual(preview["provider_name"], "ollama")
        self.assertEqual(preview["provider_attempts"][0]["provider"], "openai")

    def test_build_dry_run_preview_fails_when_no_provider_usable(self) -> None:
        with mock.patch("borisbot.guide.server._load_budget_snapshot", return_value={"blocked": False}), mock.patch(
            "borisbot.guide.server._resolve_provider_chain", return_value=["openai"]
        ), mock.patch(
            "borisbot.guide.server._provider_is_usable",
            return_value=(False, "api_key_missing"),
        ):
            preview = _build_dry_run_preview(
                "open page",
                agent_id="default",
                model_name="llama3.2:3b",
                provider_name="openai",
            )
        self.assertEqual(preview["status"], "failed")
        self.assertEqual(preview["error_code"], "LLM_PROVIDER_UNHEALTHY")

    def test_build_assistant_response_budget_blocked(self) -> None:
        with mock.patch("borisbot.guide.server._load_budget_snapshot", return_value={"blocked": True}):
            payload = _build_assistant_response(
                "hello",
                agent_id="default",
                model_name="llama3.2:3b",
                provider_name="ollama",
            )
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_code"], "BUDGET_BLOCKED")

    def test_build_assistant_response_falls_back_to_ollama(self) -> None:
        with mock.patch("borisbot.guide.server._load_budget_snapshot", return_value={"blocked": False}), mock.patch(
            "borisbot.guide.server._resolve_provider_chain",
            return_value=["openai", "ollama"],
        ), mock.patch(
            "borisbot.guide.server._provider_is_usable",
            side_effect=[(True, ""), (True, "")],
        ), mock.patch(
            "borisbot.guide.server._generate_chat_raw_with_provider",
            side_effect=[ValueError("openai failed"), "hello from local"],
        ):
            payload = _build_assistant_response(
                "say hi",
                agent_id="default",
                model_name="llama3.2:3b",
                provider_name="openai",
            )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["provider_name"], "ollama")
        self.assertEqual(payload["provider_attempts"][0]["provider"], "openai")
        self.assertEqual(payload["provider_attempts"][1]["provider"], "ollama")

    def test_provider_is_usable_unimplemented_transport(self) -> None:
        with mock.patch("borisbot.guide.server.get_secret_status", return_value={"azure": {"configured": True}}), mock.patch.dict(
            "os.environ",
            {},
            clear=False,
        ):
            ok, reason = _provider_is_usable("azure")
        self.assertFalse(ok)
        self.assertEqual(reason, "azure_endpoint_missing")

    def test_generate_plan_raw_with_provider_openai(self) -> None:
        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[]}'
                            }
                        }
                    ]
                }

        with mock.patch("borisbot.guide.server.get_provider_secret", return_value="sk-test"), mock.patch(
            "borisbot.guide.server.httpx.post",
            return_value=_Resp(),
        ):
            raw = _generate_plan_raw_with_provider("openai", "x", "gpt-4o-mini")
        self.assertIn('"planner_schema_version":"planner.v1"', raw)

    def test_generate_plan_raw_with_provider_anthropic(self) -> None:
        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "content": [
                        {"type": "text", "text": '{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[]}'}
                    ]
                }

        with mock.patch("borisbot.guide.server.get_provider_secret", return_value="sk-ant-test"), mock.patch(
            "borisbot.guide.server.httpx.post",
            return_value=_Resp(),
        ):
            raw = _generate_plan_raw_with_provider("anthropic", "x", "claude-3-5-sonnet-latest")
        self.assertIn('"planner_schema_version":"planner.v1"', raw)

    def test_generate_plan_raw_with_provider_google(self) -> None:
        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[]}'}
                                ]
                            }
                        }
                    ]
                }

        with mock.patch("borisbot.guide.server.get_provider_secret", return_value="google-key"), mock.patch(
            "borisbot.guide.server.httpx.post",
            return_value=_Resp(),
        ):
            raw = _generate_plan_raw_with_provider("google", "x", "gemini-1.5-flash")
        self.assertIn('"planner_schema_version":"planner.v1"', raw)

    def test_generate_plan_raw_with_provider_azure(self) -> None:
        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"planner_schema_version":"planner.v1","intent":"x","proposed_actions":[]}'
                            }
                        }
                    ]
                }

        with mock.patch("borisbot.guide.server.get_provider_secret", return_value="azure-key"), mock.patch.dict(
            "os.environ",
            {"BORISBOT_AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com"},
            clear=False,
        ), mock.patch(
            "borisbot.guide.server.httpx.post",
            return_value=_Resp(),
        ):
            raw = _generate_plan_raw_with_provider("azure", "x", "gpt-4o-mini")
        self.assertIn('"planner_schema_version":"planner.v1"', raw)

    def test_probe_provider_connection_openai_missing_key(self) -> None:
        with mock.patch("borisbot.guide.server.get_provider_secret", return_value=""):
            ok, message = _probe_provider_connection("openai", "gpt-4o-mini")
        self.assertFalse(ok)
        self.assertIn("missing", message.lower())

    def test_probe_provider_connection_anthropic_missing_key(self) -> None:
        with mock.patch("borisbot.guide.server.get_provider_secret", return_value=""):
            ok, message = _probe_provider_connection("anthropic", "claude-3-5-sonnet-latest")
        self.assertFalse(ok)
        self.assertIn("missing", message.lower())

    def test_probe_provider_connection_google_missing_key(self) -> None:
        with mock.patch("borisbot.guide.server.get_provider_secret", return_value=""):
            ok, message = _probe_provider_connection("google", "gemini-1.5-flash")
        self.assertFalse(ok)
        self.assertIn("missing", message.lower())

    def test_probe_provider_connection_azure_missing_endpoint(self) -> None:
        with mock.patch("borisbot.guide.server.get_provider_secret", return_value="azure-key"), mock.patch.dict(
            "os.environ",
            {},
            clear=False,
        ):
            ok, message = _probe_provider_connection("azure", "gpt-4o-mini")
        self.assertFalse(ok)
        self.assertIn("endpoint", message.lower())

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
        self.assertEqual(
            extract_browser_ui_url(output),
            "http://localhost:49001/vnc.html?autoconnect=1&resize=remote&reconnect=1",
        )

    def test_normalize_browser_ui_url_maps_root_to_vnc_html(self) -> None:
        self.assertEqual(
            _normalize_browser_ui_url("http://localhost:6080"),
            "http://localhost:6080/vnc.html?autoconnect=1&resize=remote&reconnect=1",
        )

    def test_normalize_browser_ui_url_preserves_existing_query(self) -> None:
        self.assertEqual(
            _normalize_browser_ui_url("http://localhost:6080/vnc.html?resize=scale"),
            "http://localhost:6080/vnc.html?resize=scale&autoconnect=1&reconnect=1",
        )

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

    def test_add_assistant_trace_exposed_in_list(self) -> None:
        state = GuideState(workspace=Path.cwd(), python_bin=sys.executable)
        trace = state.add_assistant_trace(
            agent_id="default",
            model_name="llama3.2:3b",
            prompt="hello",
            response={"status": "ok", "message": "hi"},
        )
        traces = state.list_traces()
        self.assertTrue(traces)
        self.assertEqual(traces[0]["trace_id"], trace["trace_id"])
        self.assertEqual(traces[0]["type"], "assistant_chat")

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
        self.assertIn("testPrimaryProvider()", html)
        self.assertIn("Show Setup Plan", html)
        self.assertIn("showOllamaSetupPlan()", html)
        self.assertIn("clearChatHistory()", html)
        self.assertIn("Assistant Chat", html)
        self.assertIn("sendAssistantPrompt()", html)
        self.assertIn("provider-cards", html)

    def test_collect_runtime_status_includes_provider_matrix(self) -> None:
        with mock.patch("borisbot.guide.server.load_profile", return_value={"primary_provider": "ollama", "model_name": "llama3.2:3b", "provider_settings": {}}), mock.patch(
            "borisbot.guide.server.get_secret_status",
            return_value={"openai": {"configured": False}},
        ):
            status = _collect_runtime_status(sys.executable)
        self.assertIn("provider_matrix", status)
        self.assertIn("ollama", status["provider_matrix"])

    def test_trace_already_executed_detection(self) -> None:
        trace = {
            "stages": [
                {"event": "created", "data": {}},
                {"event": "approved_execute_submitted", "data": {"job_id": "job_1"}},
            ]
        }
        self.assertTrue(_trace_already_executed(trace))


if __name__ == "__main__":
    unittest.main()
