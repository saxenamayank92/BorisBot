"""Local guided web UI for common BorisBot reliability workflows."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse, urlsplit

import httpx

from borisbot.llm.errors import LLMInvalidOutputError
from borisbot.guide.chat_history_store import (
    append_chat_message,
    clear_chat_history,
    load_chat_history,
)
from borisbot.llm.planner_contract import parse_planner_output
from borisbot.llm.planner_validator import validate_and_convert_plan
from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.provider_health import get_provider_health_registry
from borisbot.supervisor.database import get_db
from borisbot.supervisor.heartbeat_runtime import read_heartbeat_snapshot
from borisbot.supervisor.profile_config import load_profile, save_profile
from borisbot.supervisor.provider_secrets import (
    get_provider_secret,
    get_secret_status,
    set_provider_secret,
)
from borisbot.supervisor.tool_permissions import (
    ALLOWED_TOOLS,
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_PROMPT,
    TOOL_BROWSER,
    TOOL_FILESYSTEM,
    TOOL_SCHEDULER,
    TOOL_SHELL,
    TOOL_WEB_FETCH,
    get_agent_permission_matrix_sync,
    get_agent_tool_permission_sync,
    set_agent_tool_permission_sync,
)


def _resolve_ollama_install_command(
    platform_name: str,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return OS-aware install command for Ollama."""
    if platform_name.startswith("darwin"):
        if which("brew"):
            return ["brew", "install", "ollama"]
        raise ValueError("Homebrew not found. Install Homebrew first: https://brew.sh")
    if platform_name.startswith("linux"):
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if platform_name.startswith("win") or os.name == "nt":
        if which("winget"):
            return ["winget", "install", "-e", "--id", "Ollama.Ollama"]
        raise ValueError(
            "winget not found. Install Ollama manually from https://ollama.com/download/windows"
        )
    raise ValueError(f"Unsupported platform for automated Ollama install: {platform_name}")


def _resolve_ollama_start_command(
    platform_name: str,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return OS-aware command to start Ollama service/runtime."""
    if platform_name.startswith("darwin"):
        if which("brew"):
            return ["brew", "services", "start", "ollama"]
        return ["ollama", "serve"]
    if platform_name.startswith("linux"):
        if which("systemctl"):
            return ["systemctl", "--user", "start", "ollama"]
        return ["ollama", "serve"]
    if platform_name.startswith("win") or os.name == "nt":
        return ["ollama", "serve"]
    return ["ollama", "serve"]


ACTION_TOOL_MAP = {
    "docker_info": TOOL_SHELL,
    "cleanup_sessions": TOOL_BROWSER,
    "session_status": TOOL_SHELL,
    "ollama_check": TOOL_SHELL,
    "ollama_install": TOOL_SHELL,
    "ollama_start": TOOL_SHELL,
    "ollama_pull": TOOL_SHELL,
    "verify": TOOL_SHELL,
    "analyze": TOOL_FILESYSTEM,
    "lint": TOOL_FILESYSTEM,
    "replay": TOOL_BROWSER,
    "release_check": TOOL_SHELL,
    "release_check_json": TOOL_SHELL,
    "record": TOOL_BROWSER,
}


def required_tool_for_action(action: str) -> str | None:
    """Return required tool gate for guide action if any."""
    return ACTION_TOOL_MAP.get(action)


PLANNER_ACTION_TOOL_MAP = {
    "navigate": TOOL_BROWSER,
    "click": TOOL_BROWSER,
    "type": TOOL_BROWSER,
    "wait_for_url": TOOL_BROWSER,
    "get_text": TOOL_BROWSER,
    "get_title": TOOL_BROWSER,
    "read_file": TOOL_FILESYSTEM,
    "write_file": TOOL_FILESYSTEM,
    "list_dir": TOOL_FILESYSTEM,
    "run_shell": TOOL_SHELL,
    "web_fetch": TOOL_WEB_FETCH,
    "web_search": TOOL_WEB_FETCH,
    "schedule": TOOL_SCHEDULER,
}

ESTIMATED_PRICING_USD_PER_1K = {
    "openai": {"input": 0.005, "output": 0.015},
    "anthropic": {"input": 0.008, "output": 0.024},
    "google": {"input": 0.0035, "output": 0.0105},
    "azure": {"input": 0.005, "output": 0.015},
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count deterministically from raw text length."""
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))


def _extract_required_tools_from_plan(plan: dict) -> list[str]:
    """Derive required tool set from planner proposed actions."""
    tools: set[str] = set()
    actions = plan.get("proposed_actions", [])
    if not isinstance(actions, list):
        return []
    for action_obj in actions:
        if not isinstance(action_obj, dict):
            continue
        action_name = action_obj.get("action")
        if not isinstance(action_name, str):
            continue
        tool = PLANNER_ACTION_TOOL_MAP.get(action_name.strip())
        if tool:
            tools.add(tool)
    return sorted(tools)


def _build_planner_prompt(user_intent: str) -> str:
    """Build strict planner prompt that must return planner.v1 JSON only."""
    return (
        "You are a planning engine.\n"
        "Return strict JSON only with exact schema:\n"
        "{"
        "\"planner_schema_version\":\"planner.v1\","
        "\"intent\":\"...\","
        "\"proposed_actions\":[{\"action\":\"...\",\"target\":\"...\",\"input\":\"...\"}]"
        "}\n"
        "No markdown. No extra keys. No commentary.\n"
        f"User request: {user_intent.strip()}"
    )


def _estimate_preview_cost_usd(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        return 0.0
    rates = ESTIMATED_PRICING_USD_PER_1K.get(provider)
    if not rates:
        return 0.0
    input_cost = (input_tokens / 1000.0) * float(rates["input"])
    output_cost = (output_tokens / 1000.0) * float(rates["output"])
    return round(input_cost + output_cost, 6)


def _load_budget_snapshot(agent_id: str) -> dict:
    guard = CostGuard()
    return asyncio.run(guard.get_budget_status(agent_id))


def _resolve_provider_chain(requested_provider: str) -> list[str]:
    profile = load_profile()
    raw_chain = profile.get("provider_chain", ["ollama"])
    chain: list[str] = []
    if isinstance(raw_chain, list):
        for item in raw_chain:
            name = str(item).strip().lower()
            if name and name not in chain:
                chain.append(name)
    preferred = str(requested_provider).strip().lower()
    if preferred and preferred in chain:
        chain.remove(preferred)
        chain.insert(0, preferred)
    elif preferred:
        chain.insert(0, preferred)
    return chain[:5] or ["ollama"]


def _provider_is_usable(provider_name: str) -> tuple[bool, str]:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        if shutil.which("ollama") is None:
            return False, "ollama_not_installed"
        return True, ""
    if provider in {"anthropic", "google", "azure"}:
        return False, "transport_unimplemented"
    status = get_secret_status().get(provider, {})
    if not bool(status.get("configured", False)):
        return False, "api_key_missing"
    return True, ""


def _generate_plan_raw_with_ollama(user_intent: str, model_name: str) -> str:
    """Call Ollama generate endpoint and return raw planner output text."""
    if shutil.which("ollama") is None:
        raise ValueError("Ollama is not installed. Install it from Step 1 first.")
    prompt = _build_planner_prompt(user_intent)
    response = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=20.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Ollama generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Ollama response payload invalid")
    output = payload.get("response")
    if not isinstance(output, str):
        raise ValueError("Ollama response missing text output")
    return output


def _generate_plan_raw_with_openai(user_intent: str, model_name: str) -> str:
    """Call OpenAI chat completions endpoint and return text output."""
    api_key = get_provider_secret("openai")
    if not api_key:
        raise ValueError("OpenAI API key missing. Configure it in Provider Onboarding.")
    prompt = _build_planner_prompt(user_intent)
    endpoint = os.getenv("BORISBOT_OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"OpenAI generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response payload invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI response choice invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI response message invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI response content invalid")
    return content


def _generate_plan_raw_with_provider(provider_name: str, user_intent: str, model_name: str) -> str:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        return _generate_plan_raw_with_ollama(user_intent, model_name=model_name)
    if provider == "openai":
        return _generate_plan_raw_with_openai(user_intent, model_name=model_name)
    raise ValueError(f"Provider '{provider}' planner transport is not enabled in this build")


def _build_dry_run_preview(intent: str, agent_id: str, model_name: str, provider_name: str) -> dict:
    """Build dry-run planner preview with strict schema + permission requirements."""
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Plan prompt cannot be empty.")
    budget_status = _load_budget_snapshot(agent_id)
    if bool(budget_status.get("blocked", False)):
        return {
            "status": "failed",
            "error_class": "cost_guard",
            "error_code": "BUDGET_BLOCKED",
            "message": "Planner calls are blocked by budget limits.",
            "budget": budget_status,
        }
    provider_chain = _resolve_provider_chain(provider_name)
    attempts: list[dict[str, str]] = []
    raw_output = ""
    selected_provider = ""
    for provider in provider_chain:
        usable, reason = _provider_is_usable(provider)
        if not usable:
            attempts.append({"provider": provider, "status": "skipped", "reason": reason})
            continue
        try:
            raw_output = _generate_plan_raw_with_provider(provider, intent, model_name=model_name)
            selected_provider = provider
            attempts.append({"provider": provider, "status": "ok"})
            break
        except ValueError as exc:
            attempts.append({"provider": provider, "status": "failed", "reason": str(exc)})
            continue
    if not raw_output:
        return {
            "status": "failed",
            "error_class": "llm_provider",
            "error_code": "LLM_PROVIDER_UNHEALTHY",
            "message": "No usable provider available for planner dry-run.",
            "provider_attempts": attempts,
            "provider_chain": provider_chain,
            "budget": budget_status,
        }
    try:
        parsed = parse_planner_output(raw_output)
    except LLMInvalidOutputError as exc:
        return {
            "status": "failed",
            "error_class": exc.error_class,
            "error_code": exc.error_code,
            "message": str(exc),
            "raw_output": raw_output,
        }

    required_tools = _extract_required_tools_from_plan(parsed)
    try:
        preview_commands = validate_and_convert_plan(parsed)
    except LLMInvalidOutputError as exc:
        return {
            "status": "failed",
            "error_class": exc.error_class,
            "error_code": exc.error_code,
            "message": str(exc),
            "planner_output": parsed,
            "raw_output": raw_output,
        }
    required_permissions = [
        {
            "tool_name": tool,
            "decision": get_agent_tool_permission_sync(agent_id, tool),
        }
        for tool in required_tools
    ]

    prompt_tokens = _estimate_tokens(_build_planner_prompt(intent))
    completion_tokens = _estimate_tokens(raw_output)
    estimated_cost = _estimate_preview_cost_usd(
        selected_provider,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )
    return {
        "status": "ok",
        "planner_output": parsed,
        "validated_commands": preview_commands,
        "required_permissions": required_permissions,
        "token_estimate": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_estimate_usd": estimated_cost,
        "provider_name": selected_provider,
        "provider_attempts": attempts,
        "provider_chain": provider_chain,
        "raw_output": raw_output,
        "budget": budget_status,
    }


def extract_browser_ui_url(output: str) -> str:
    """Extract latest noVNC/browser URL printed by recorder output."""
    matches = re.findall(r"Open browser UI at:\s*(https?://\S+)", output or "")
    if not matches:
        return ""
    return matches[-1].strip()


def build_action_command(
    action: str,
    params: dict[str, str],
    workspace: Path,
    python_bin: str,
) -> list[str]:
    """Return a whitelisted CLI command for supported guide actions."""
    workflow_path = params.get("workflow_path", "workflows/real_login_test.json").strip()
    if not workflow_path:
        workflow_path = "workflows/real_login_test.json"

    if action == "docker_info":
        return ["docker", "info"]
    if action == "cleanup_sessions":
        return [python_bin, "-m", "borisbot.cli", "cleanup-browsers"]
    if action == "session_status":
        return [python_bin, "-m", "borisbot.cli", "session-status"]
    if action == "ollama_check":
        return ["ollama", "--version"]
    if action == "ollama_install":
        return _resolve_ollama_install_command(sys.platform)
    if action == "ollama_start":
        return _resolve_ollama_start_command(sys.platform)
    if action == "ollama_pull":
        model = params.get("model_name", "").strip() or "llama3.2:3b"
        return ["ollama", "pull", model]
    if action == "verify":
        return [python_bin, "-m", "borisbot.cli", "verify"]
    if action == "analyze":
        return [python_bin, "-m", "borisbot.cli", "analyze-workflow", workflow_path]
    if action == "lint":
        return [python_bin, "-m", "borisbot.cli", "lint-workflow", workflow_path]
    if action == "replay":
        return [python_bin, "-m", "borisbot.cli", "replay", workflow_path]
    if action == "release_check":
        return [python_bin, "-m", "borisbot.cli", "release-check", workflow_path]
    if action == "release_check_json":
        return [python_bin, "-m", "borisbot.cli", "release-check", workflow_path, "--json"]
    if action == "record":
        task_id = params.get("task_id", "").strip() or "wf_new"
        start_url = params.get("start_url", "").strip() or "https://example.com"
        parsed = urlparse(start_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("start_url must include protocol and host, e.g. https://example.com")
        return [python_bin, "-m", "borisbot.cli", "record", task_id, "--start-url", start_url]

    raise ValueError(f"Unsupported action: {action}")


@dataclass
class GuideJob:
    """Background process metadata and output buffer."""

    job_id: str
    action: str
    command: list[str]
    params: dict[str, str] = field(default_factory=dict)
    status: str = "running"
    output: list[str] = field(default_factory=list)
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    process: subprocess.Popen[str] | None = None
    trace_id: str | None = None

    def append(self, line: str) -> None:
        self.output.append(line)
        if len(self.output) > 800:
            self.output = self.output[-800:]

    def to_dict(self) -> dict:
        output_text = "".join(self.output)
        return {
            "job_id": self.job_id,
            "action": self.action,
            "params": self.params,
            "command": " ".join(shlex.quote(part) for part in self.command),
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output": output_text,
            "browser_ui_url": extract_browser_ui_url(output_text),
        }


class GuideState:
    """Shared mutable state for the guide web app."""

    def __init__(self, workspace: Path, python_bin: str):
        self.workspace = workspace
        self.python_bin = python_bin
        self._jobs: dict[str, GuideJob] = {}
        self._traces: list[dict] = []
        self._lock = threading.Lock()
        self._counter = 0
        self._trace_counter = 0

    def workflows(self) -> list[str]:
        workflow_dir = self.workspace / "workflows"
        if not workflow_dir.exists():
            return []
        return sorted(str(p.relative_to(self.workspace)) for p in workflow_dir.glob("*.json"))

    def create_job(self, action: str, params: dict[str, str]) -> GuideJob:
        request_fingerprint = self._request_fingerprint(action, params)
        with self._lock:
            for existing in self._jobs.values():
                if existing.status != "running":
                    continue
                if existing.action != action:
                    continue
                if self._request_fingerprint(existing.action, existing.params) == request_fingerprint:
                    raise ValueError(
                        "An identical job is already running. Wait for it to finish before retrying."
                    )
        if action in {"record", "replay"}:
            running_browser_job = self._find_running_browser_job()
            if running_browser_job is not None:
                raise ValueError(
                    "A browser job is already running. Stop it before starting another record/replay."
                )
        command = build_action_command(action, params, self.workspace, self.python_bin)
        with self._lock:
            self._counter += 1
            job_id = f"job_{self._counter:04d}"
            job = GuideJob(job_id=job_id, action=action, command=command, params=params)
            self._jobs[job_id] = job
            trace = self._create_trace_locked(
                trace_type="action_run",
                data={
                    "action": action,
                    "params": params,
                    "command": " ".join(shlex.quote(part) for part in command),
                },
            )
            job.trace_id = trace["trace_id"]
        self._start_job(job)
        return job

    def _request_fingerprint(self, action: str, params: dict[str, str]) -> str:
        raw = json.dumps({"action": action, "params": params}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = [job.to_dict() for job in self._jobs.values()]
        jobs.sort(key=lambda row: row["started_at"], reverse=True)
        return jobs

    def runtime_status(self) -> dict:
        """Return runtime status snapshot for GUI display."""
        return _collect_runtime_status(self.python_bin)

    def list_traces(self) -> list[dict]:
        """Return recent traces sorted newest first."""
        with self._lock:
            return list(reversed(self._traces[-30:]))

    def list_trace_summaries(self) -> list[dict]:
        """Return compact summaries for recent traces."""
        traces = self.list_traces()
        summaries: list[dict] = []
        for trace in traces:
            stages = trace.get("stages", [])
            last_event = "unknown"
            if isinstance(stages, list) and stages:
                last = stages[-1]
                if isinstance(last, dict):
                    last_event = str(last.get("event", "unknown"))
            summaries.append(
                {
                    "trace_id": str(trace.get("trace_id", "")),
                    "type": str(trace.get("type", "")),
                    "created_at": str(trace.get("created_at", "")),
                    "stage_count": len(stages) if isinstance(stages, list) else 0,
                    "last_event": last_event,
                }
            )
        return summaries

    def add_plan_trace(self, *, agent_id: str, model_name: str, intent: str, preview: dict) -> dict:
        """Append dry-run planner trace entry."""
        with self._lock:
            return self._create_trace_locked(
                trace_type="plan_preview",
                data={
                    "agent_id": agent_id,
                    "model_name": model_name,
                    "intent": intent,
                    "preview": preview,
                },
            )

    def get_trace(self, trace_id: str) -> dict | None:
        """Fetch trace by id."""
        with self._lock:
            for trace in self._traces:
                if trace.get("trace_id") == trace_id:
                    return trace
        return None

    def append_trace_stage(self, trace_id: str, stage_data: dict) -> None:
        """Append stage to an existing trace."""
        self._append_trace_stage(trace_id, stage_data)

    def get_job(self, job_id: str) -> GuideJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None or job.process is None or job.status != "running":
            return False
        try:
            if os.name == "nt":
                job.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(job.process.pid, signal.SIGINT)
            return True
        except Exception:
            return False

    def _find_running_browser_job(self) -> GuideJob | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running" and job.action in {"record", "replay"}:
                    return job
        return None

    def _start_job(self, job: GuideJob) -> None:
        env = os.environ.copy()
        # Ensure child Python CLI output streams immediately into the guide UI.
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict = {
            "cwd": str(self.workspace),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["preexec_fn"] = os.setsid
        process = subprocess.Popen(job.command, **kwargs)
        job.process = process

        def _reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                job.append(line)
            process.wait()
            job.returncode = process.returncode
            job.finished_at = time.time()
            job.status = "completed" if process.returncode == 0 else "failed"
            if job.trace_id:
                self._append_trace_stage(
                    job.trace_id,
                    {
                        "event": "job_finished",
                        "status": job.status,
                        "returncode": job.returncode,
                    },
                )

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

    def _create_trace_locked(self, *, trace_type: str, data: dict) -> dict:
        self._trace_counter += 1
        trace = {
            "trace_id": f"trace_{self._trace_counter:05d}",
            "type": trace_type,
            "created_at": datetime.utcnow().isoformat(),
            "stages": [{"event": "created", "at": datetime.utcnow().isoformat(), "data": data}],
        }
        self._traces.append(trace)
        if len(self._traces) > 200:
            self._traces = self._traces[-200:]
        return trace

    def _append_trace_stage(self, trace_id: str, stage_data: dict) -> None:
        with self._lock:
            for trace in reversed(self._traces):
                if trace.get("trace_id") == trace_id:
                    trace.setdefault("stages", []).append(
                        {
                            "event": str(stage_data.get("event", "stage")),
                            "at": datetime.utcnow().isoformat(),
                            "data": stage_data,
                        }
                    )
                    return


def _make_handler(state: GuideState) -> Callable[..., BaseHTTPRequestHandler]:
    class GuideHandler(BaseHTTPRequestHandler):
        def _json_response(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                page = _render_html(state.workflows()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/api/workflows":
                self._json_response({"items": state.workflows()})
                return
            if self.path == "/api/jobs":
                self._json_response({"items": state.list_jobs()})
                return
            if self.path == "/api/runtime-status":
                self._json_response(state.runtime_status())
                return
            if self.path == "/api/profile":
                self._json_response(load_profile())
                return
            if self.path.startswith("/api/chat-history"):
                query = parse_qs(urlsplit(self.path).query)
                agent_id = str(query.get("agent_id", ["default"])[0]).strip() or "default"
                self._json_response({"agent_id": agent_id, "items": load_chat_history(agent_id)})
                return
            if self.path == "/api/provider-secrets":
                self._json_response({"providers": get_secret_status()})
                return
            if self.path.startswith("/api/permissions"):
                query = parse_qs(urlsplit(self.path).query)
                agent_id = str(query.get("agent_id", ["default"])[0]).strip() or "default"
                matrix = get_agent_permission_matrix_sync(agent_id)
                self._json_response({"agent_id": agent_id, "permissions": matrix})
                return
            if self.path == "/api/traces":
                self._json_response({"items": state.list_trace_summaries()})
                return
            if self.path.startswith("/api/traces/"):
                trace_id = self.path.split("/")[-1]
                trace = state.get_trace(trace_id)
                if not isinstance(trace, dict):
                    self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response(trace)
                return
            if self.path.startswith("/api/jobs/"):
                job_id = self.path.split("/")[-1]
                job = state.get_job(job_id)
                if job is None:
                    self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response(job.to_dict())
                return
            self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/run":
                payload = self._read_json()
                action = str(payload.get("action", "")).strip()
                params = payload.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                agent_id = str(params.get("agent_id", "default")).strip() or "default"
                required_tool = required_tool_for_action(action)
                if required_tool:
                    decision = get_agent_tool_permission_sync(agent_id, required_tool)
                    approve_permission = bool(payload.get("approve_permission", False))
                    if decision == DECISION_DENY:
                        self._json_response(
                            {
                                "error": "permission_denied",
                                "agent_id": agent_id,
                                "tool_name": required_tool,
                                "message": f"Tool '{required_tool}' is denied for agent '{agent_id}'.",
                            },
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return
                    if decision == DECISION_PROMPT and not approve_permission:
                        self._json_response(
                            {
                                "error": "permission_required",
                                "agent_id": agent_id,
                                "tool_name": required_tool,
                                "message": (
                                    f"Agent '{agent_id}' requires approval for tool '{required_tool}'."
                                ),
                            },
                            status=HTTPStatus.CONFLICT,
                        )
                        return
                    if decision == DECISION_PROMPT and approve_permission:
                        set_agent_tool_permission_sync(agent_id, required_tool, DECISION_ALLOW)
                try:
                    job = state.create_job(action, {k: str(v) for k, v in params.items()})
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(job.to_dict(), status=HTTPStatus.CREATED)
                return
            if self.path == "/api/profile":
                payload = self._read_json()
                try:
                    profile = save_profile(payload)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(profile, status=HTTPStatus.OK)
                return
            if self.path == "/api/provider-secrets":
                payload = self._read_json()
                provider = str(payload.get("provider", "")).strip().lower()
                api_key = str(payload.get("api_key", "")).strip()
                try:
                    providers = set_provider_secret(provider, api_key)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"providers": providers}, status=HTTPStatus.OK)
                return
            if self.path == "/api/chat-history":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                role = str(payload.get("role", "")).strip()
                text = str(payload.get("text", "")).strip()
                try:
                    items = append_chat_message(agent_id, role, text)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"agent_id": agent_id, "items": items}, status=HTTPStatus.OK)
                return
            if self.path == "/api/chat-clear":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                clear_chat_history(agent_id)
                self._json_response({"agent_id": agent_id, "items": []}, status=HTTPStatus.OK)
                return
            if self.path == "/api/permissions":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                tool_name = str(payload.get("tool_name", "")).strip()
                decision = str(payload.get("decision", "")).strip()
                if tool_name not in ALLOWED_TOOLS:
                    self._json_response(
                        {"error": f"unsupported tool_name '{tool_name}'"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    set_agent_tool_permission_sync(agent_id, tool_name, decision)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                matrix = get_agent_permission_matrix_sync(agent_id)
                self._json_response(
                    {"agent_id": agent_id, "tool_name": tool_name, "decision": decision, "permissions": matrix},
                    status=HTTPStatus.OK,
                )
                return
            if self.path == "/api/plan-preview":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                intent = str(payload.get("intent", "")).strip()
                model_name = str(payload.get("model_name", "llama3.2:3b")).strip() or "llama3.2:3b"
                provider_name = str(payload.get("provider_name", "ollama")).strip() or "ollama"
                try:
                    preview = _build_dry_run_preview(
                        intent,
                        agent_id=agent_id,
                        model_name=model_name,
                        provider_name=provider_name,
                    )
                except ValueError as exc:
                    self._json_response({"status": "failed", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.add_plan_trace(
                    agent_id=agent_id,
                    model_name=model_name,
                    intent=intent,
                    preview=preview,
                )
                preview["trace_id"] = trace["trace_id"]
                self._json_response(preview, status=HTTPStatus.OK)
                return
            if self.path == "/api/execute-plan":
                payload = self._read_json()
                trace_id = str(payload.get("trace_id", "")).strip()
                if not trace_id:
                    self._json_response({"error": "trace_id required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.get_trace(trace_id)
                if not isinstance(trace, dict):
                    self._json_response({"error": "trace_not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                stages = trace.get("stages", [])
                if not isinstance(stages, list) or not stages:
                    self._json_response({"error": "trace_has_no_stages"}, status=HTTPStatus.BAD_REQUEST)
                    return
                latest_preview = {}
                for stage in reversed(stages):
                    if isinstance(stage, dict):
                        data = stage.get("data", {})
                        if isinstance(data, dict) and isinstance(data.get("preview"), dict):
                            latest_preview = data["preview"]
                            break
                if not isinstance(latest_preview, dict) or latest_preview.get("status") != "ok":
                    self._json_response(
                        {"error": "trace_preview_not_executable"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                validated_commands = latest_preview.get("validated_commands", [])
                if not isinstance(validated_commands, list) or not validated_commands:
                    self._json_response(
                        {"error": "trace_has_no_validated_commands"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                required_tool = TOOL_BROWSER
                decision = get_agent_tool_permission_sync(agent_id, required_tool)
                approve_permission = bool(payload.get("approve_permission", False))
                if decision == DECISION_DENY:
                    self._json_response(
                        {
                            "error": "permission_denied",
                            "agent_id": agent_id,
                            "tool_name": required_tool,
                            "message": f"Tool '{required_tool}' is denied for agent '{agent_id}'.",
                        },
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                if decision == DECISION_PROMPT and not approve_permission:
                    self._json_response(
                        {
                            "error": "permission_required",
                            "agent_id": agent_id,
                            "tool_name": required_tool,
                            "message": (
                                f"Agent '{agent_id}' requires approval for tool '{required_tool}'."
                            ),
                        },
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                if decision == DECISION_PROMPT and approve_permission:
                    set_agent_tool_permission_sync(agent_id, required_tool, DECISION_ALLOW)

                generated_dir = state.workspace / "workflows" / "generated"
                generated_dir.mkdir(parents=True, exist_ok=True)
                task_id = f"plan_{trace_id}_{int(time.time())}"
                workflow_path = generated_dir / f"{task_id}.json"
                workflow_payload = {
                    "schema_version": "task_command.v1",
                    "task_id": task_id,
                    "commands": validated_commands,
                }
                workflow_path.write_text(json.dumps(workflow_payload, indent=2), encoding="utf-8")
                state.append_trace_stage(
                    trace_id,
                    {
                        "event": "approved_execute_requested",
                        "task_id": task_id,
                        "workflow_path": str(workflow_path),
                    },
                )
                try:
                    job = state.create_job("replay", {"workflow_path": str(workflow_path)})
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                state.append_trace_stage(
                    trace_id,
                    {
                        "event": "approved_execute_submitted",
                        "job_id": job.job_id,
                        "task_id": task_id,
                    },
                )
                self._json_response(
                    {
                        "status": "submitted",
                        "job_id": job.job_id,
                        "task_id": task_id,
                        "workflow_path": str(workflow_path),
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
                parts = self.path.split("/")
                if len(parts) < 4:
                    self._json_response({"error": "invalid_path"}, status=HTTPStatus.BAD_REQUEST)
                    return
                job_id = parts[3]
                stopped = state.stop_job(job_id)
                if not stopped:
                    self._json_response({"error": "job_not_running"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"status": "stopping", "job_id": job_id})
                return
            self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return GuideHandler


def _render_html(workflows: list[str]) -> str:
    options = "\n".join(
        f'<option value="{wf}">{wf}</option>' for wf in workflows
    ) or '<option value="workflows/real_login_test.json">workflows/real_login_test.json</option>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BorisBot Guide</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf1;
      --ink: #1e2a2f;
      --muted: #5f6a6d;
      --brand: #0d6e6e;
      --brand-2: #d67443;
      --border: #dacfbf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 10%, #fffdf6 0%, var(--bg) 55%);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0.2px; }}
    p {{ margin: 0; color: var(--muted); }}
    .layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; align-items: start; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 5px 20px rgba(30, 42, 47, 0.06);
    }}
    .step {{ margin-top: 12px; padding-top: 12px; border-top: 1px dashed var(--border); }}
    .step:first-child {{ border-top: 0; margin-top: 0; padding-top: 0; }}
    .step h3 {{ margin: 0 0 6px; font-size: 17px; letter-spacing: 0.02em; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    button {{
      border: 0;
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      cursor: pointer;
      background: var(--brand);
      color: #fff;
    }}
    button.secondary {{ background: var(--brand-2); }}
    label {{ display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }}
    input, select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px;
      font-size: 14px;
      background: #fff;
      color: var(--ink);
      margin-bottom: 10px;
    }}
    pre {{
      margin: 0;
      background: #111a1f;
      color: #d9f5eb;
      border-radius: 12px;
      padding: 12px;
      min-height: 300px;
      max-height: 500px;
      overflow: auto;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .viewer-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .viewer-toolbar button {{
      background: #365564;
    }}
    .viewer-toolbar button.mode {{
      background: #5d7f8e;
    }}
    .viewer-grid {{
      display: grid;
      gap: 10px;
      height: 560px;
    }}
    .viewer-grid.mode-terminal {{ grid-template-rows: 1fr; }}
    .viewer-grid.mode-browser {{ grid-template-rows: 1fr; }}
    .viewer-grid.mode-split {{ grid-template-rows: 1fr 1fr; }}
    .pane {{
      min-height: 0;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
      display: flex;
      flex-direction: column;
    }}
    .pane-header {{
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      background: #fff7ea;
    }}
    .pane-body {{
      flex: 1;
      min-height: 0;
    }}
    .pane.hidden {{ display: none; }}
    iframe {{
      border: 0;
      width: 100%;
      height: 100%;
      background: #0f1519;
    }}
    .browser-link {{
      font-size: 12px;
      color: var(--brand);
      margin-bottom: 8px;
      display: block;
      word-break: break-all;
    }}
    .hover-card {{
      transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}
    .hover-card:hover {{
      transform: translateY(-1px);
      box-shadow: 0 10px 18px rgba(54, 85, 100, 0.16);
      border-color: #89aeb9;
    }}
    .provider-cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .provider-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
    }}
    .job-meta {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
    @media (max-width: 920px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>BorisBot Guided Validation</h1>
    <p>Use this checklist UI to run reliability actions without memorizing CLI commands.</p>

    <div class="layout">
      <section class="card">
        <label for="workflow">Workflow file</label>
        <select id="workflow">{options}</select>
        <label for="agent_name">Agent name</label>
        <input id="agent_name" value="default" />
        <label for="provider_chain">Provider chain (comma-separated, max 5)</label>
        <input id="provider_chain" value="ollama" />
        <label for="primary_provider">Primary provider</label>
        <input id="primary_provider" value="ollama" />
        <div class="step">
          <h3>Provider Onboarding</h3>
          <p>Configure primary/fallback providers and API credentials (stored locally).</p>
          <div id="provider-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;"></div>
          <div class="actions">
            <button class="secondary" onclick="refreshProviderSecrets()">Refresh Provider Status</button>
          </div>
        </div>
        <label for="task">New task id for recording</label>
        <input id="task" value="wf_demo" />
        <label for="agent">Agent id</label>
        <input id="agent" value="default" onchange="refreshPermissions();loadChatHistory();" />
        <label for="start">Start URL for recording</label>
        <input id="start" value="https://example.com" />
        <label for="model">Ollama model</label>
        <input id="model" value="llama3.2:3b" />
        <label for="prompt">Dry-run planner prompt</label>
        <textarea id="prompt" rows="4" style="width:100%;border:1px solid var(--border);border-radius:10px;padding:9px;font-size:14px;background:#fff;color:var(--ink);margin-bottom:10px;">Open LinkedIn feed, scroll a few posts, and like one relevant post.</textarea>

        <div class="step">
          <h3>1. Environment</h3>
          <p>Confirm Docker and Ollama are ready, then validate baseline.</p>
          <div class="actions">
            <button onclick="runAction('docker_info')">Check Docker</button>
            <button onclick="runAction('ollama_install')">Install Ollama</button>
            <button onclick="runAction('ollama_check')">Check Ollama</button>
            <button onclick="runAction('ollama_start')">Start Ollama</button>
            <button onclick="runAction('ollama_pull')">Pull Model</button>
            <button onclick="runOneTouchLlmSetup()">One-Touch LLM Setup</button>
            <button onclick="runAction('cleanup_sessions')">Reset Browser Sessions</button>
            <button onclick="runAction('verify')">Run Verify</button>
            <button onclick="runAction('session_status')">Session Status</button>
            <button onclick="runPlanPreview()">Dry-Run Planner</button>
            <button onclick="executeApprovedPlan()">Execute Approved Plan</button>
            <button onclick="saveProfile()">Save Profile</button>
          </div>
        </div>

        <div class="step">
          <h3>2. Record Workflow</h3>
          <p>Click start, perform actions in the opened noVNC browser, then click stop.</p>
          <div class="actions">
            <button class="secondary" onclick="startRecord()">Start Recording</button>
            <button onclick="stopCurrent()">Stop Recording</button>
          </div>
        </div>

        <div class="step">
          <h3>Permission Matrix</h3>
          <p>Set per-agent tool decisions used by dry-run and execution gates.</p>
          <div id="permission-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;"></div>
          <div class="actions">
            <button class="secondary" onclick="refreshPermissions()">Refresh Permissions</button>
          </div>
        </div>

        <div class="step">
          <h3>3. Analyze and Lint</h3>
          <p>Score selectors and catch fragile workflows early.</p>
          <div class="actions">
            <button onclick="runAction('analyze')">Analyze Workflow</button>
            <button onclick="runAction('lint')">Lint Workflow</button>
          </div>
        </div>

        <div class="step">
          <h3>4. Replay and Gate</h3>
          <p>Replay deterministically and run release-check in both modes.</p>
          <div class="actions">
            <button onclick="runAction('replay')">Replay Workflow</button>
            <button onclick="runAction('release_check')">Release Check</button>
            <button onclick="runAction('release_check_json')">Release Check JSON</button>
          </div>
        </div>

        <div class="step">
          <h3>Planner Chat</h3>
          <p>Ask for a plan in natural language. Response is validated dry-run JSON.</p>
          <textarea id="chat-input" rows="3" style="width:100%;border:1px solid var(--border);border-radius:10px;padding:9px;font-size:14px;background:#fff;color:var(--ink);margin-bottom:8px;" placeholder="Example: Open LinkedIn, scan posts, and like one post about AI tooling."></textarea>
          <div class="actions">
            <button class="secondary" onclick="sendChatPrompt()">Send To Planner</button>
            <button onclick="clearChatHistory()">Clear Chat</button>
          </div>
          <pre id="chat-history" style="margin-top:8px;min-height:120px;max-height:220px;"></pre>
        </div>
      </section>

      <section class="card" id="viewer-card">
        <div class="step" style="margin-top:0;padding-top:0;border-top:0;">
          <h3>Runtime Status</h3>
          <p id="runtime-line">Loading...</p>
          <div id="provider-cards" class="provider-cards"></div>
        </div>
        <div class="job-meta" id="meta">No command running.</div>
        <pre id="plan-output" style="margin-bottom:10px;min-height:120px;max-height:220px;"></pre>
        <div class="actions" style="margin-bottom:8px;">
          <select id="trace-select" style="flex:1;min-width:240px;" onchange="loadSelectedTrace()"></select>
          <button class="secondary" onclick="loadSelectedTrace()">View Trace</button>
          <button onclick="exportSelectedTrace()">Export Trace JSON</button>
        </div>
        <pre id="trace-output" style="margin-bottom:10px;min-height:120px;max-height:220px;"></pre>
        <a class="browser-link" id="browser-link" href="#" target="_blank" style="display:none;"></a>
        <div class="viewer-toolbar">
          <button class="mode" onclick="setViewMode('terminal')">Terminal View</button>
          <button class="mode" onclick="setViewMode('browser')">Browser View</button>
          <button class="mode" onclick="setViewMode('split')">Split View</button>
          <button onclick="maximizePane('terminal-pane')">Maximize Terminal</button>
          <button onclick="maximizePane('browser-pane')">Maximize Browser</button>
        </div>
        <div class="viewer-grid mode-split" id="viewer-grid">
          <div class="pane" id="terminal-pane">
            <div class="pane-header">Terminal Output</div>
            <div class="pane-body">
              <pre id="output"></pre>
            </div>
          </div>
          <div class="pane" id="browser-pane">
            <div class="pane-header">Live Browser (noVNC)</div>
            <div class="pane-body">
              <iframe id="browser-frame" title="Browser view"></iframe>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    let currentJobId = null;
    let pollHandle = null;
    let viewMode = 'split';
    let lastPlanTraceId = null;
    let chatHistory = [];
    const providerNames = ['ollama', 'openai', 'anthropic', 'google', 'azure'];

    function currentParams() {{
      return {{
        workflow_path: document.getElementById('workflow').value,
        task_id: document.getElementById('task').value,
        agent_id: document.getElementById('agent').value,
        start_url: document.getElementById('start').value,
        model_name: document.getElementById('model').value
      }};
    }}

    async function runAction(action, approvePermission=false) {{
      const response = await fetch('/api/run', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action, params: currentParams(), approve_permission: approvePermission}})
      }});
      const data = await response.json();
      if (!response.ok) {{
        if (data.error === 'permission_required') {{
          const ok = window.confirm(`${{data.message}} Approve now?`);
          if (ok) {{
            const result = await runAction(action, true);
            refreshPermissions();
            return result;
          }}
          document.getElementById('meta').textContent = 'Permission not granted: ' + (data.tool_name || '');
          return;
        }}
        document.getElementById('meta').textContent = 'Failed: ' + (data.error || 'unknown error');
        return;
      }}
      currentJobId = data.job_id;
      startPolling();
      return data;
    }}

    async function waitForJobCompletion(jobId, timeoutMs=180000) {{
      const started = Date.now();
      while ((Date.now() - started) < timeoutMs) {{
        const response = await fetch(`/api/jobs/${{jobId}}`);
        if (!response.ok) {{
          throw new Error('job_status_fetch_failed');
        }}
        const data = await response.json();
        document.getElementById('meta').textContent = `[${{data.status}}] ${{data.command}}`;
        document.getElementById('output').textContent = data.output || '';
        updateBrowserLink(data.browser_ui_url || '');
        if (data.status !== 'running') {{
          return data;
        }}
        await new Promise(resolve => setTimeout(resolve, 900));
      }}
      throw new Error('job_timeout');
    }}

    async function runActionAndWait(action) {{
      const data = await runAction(action);
      if (!data || !data.job_id) {{
        throw new Error('job_submit_failed');
      }}
      return waitForJobCompletion(data.job_id);
    }}

    async function runOneTouchLlmSetup() {{
      document.getElementById('meta').textContent = 'Starting one-touch LLM setup...';
      try {{
        const statusResp = await fetch('/api/runtime-status');
        const status = statusResp.ok ? await statusResp.json() : {{}};
        if (!status.ollama_installed) {{
          document.getElementById('meta').textContent = 'Installing Ollama...';
          const installJob = await runActionAndWait('ollama_install');
          if (installJob.status !== 'completed') {{
            document.getElementById('meta').textContent = 'Ollama install failed. Review terminal output.';
            return;
          }}
        }}
        document.getElementById('meta').textContent = 'Starting Ollama runtime...';
        const startJob = await runActionAndWait('ollama_start');
        if (startJob.status !== 'completed') {{
          document.getElementById('meta').textContent = 'Ollama start failed. Review terminal output.';
          return;
        }}
        document.getElementById('meta').textContent = 'Pulling selected model...';
        const pullJob = await runActionAndWait('ollama_pull');
        if (pullJob.status !== 'completed') {{
          document.getElementById('meta').textContent = 'Model pull failed. Review terminal output.';
          return;
        }}
        await runActionAndWait('session_status');
        document.getElementById('meta').textContent = 'One-touch LLM setup completed.';
        refreshRuntimeStatus();
      }} catch (e) {{
        document.getElementById('meta').textContent = 'One-touch setup failed unexpectedly.';
      }}
    }}

    async function loadProfile() {{
      try {{
        const response = await fetch('/api/profile');
        if (!response.ok) return;
        const profile = await response.json();
        document.getElementById('agent_name').value = profile.agent_name || 'default';
        if (!document.getElementById('agent').value) {{
          document.getElementById('agent').value = profile.agent_name || 'default';
        }}
        document.getElementById('primary_provider').value = profile.primary_provider || 'ollama';
        document.getElementById('provider_chain').value = Array.isArray(profile.provider_chain)
          ? profile.provider_chain.join(',')
          : 'ollama';
        renderProviderGrid(profile.provider_settings || {{}});
        if (profile.model_name) {{
          document.getElementById('model').value = profile.model_name;
        }}
        refreshProviderSecrets();
      }} catch (e) {{
        // ignore profile load failures
      }}
    }}

    function renderProviderGrid(providerSettings) {{
      const rows = providerNames.map(name => {{
        const settings = providerSettings[name] || {{}};
        const enabled = !!settings.enabled;
        const modelName = settings.model_name || '';
        const checked = enabled ? 'checked' : '';
        const secretField = name === 'ollama'
          ? ''
          : `<label style="margin:0;">${{name}} API key</label>
             <input id="provider_key_${{name}}" type="password" placeholder="Paste key (blank keeps existing)" />`;
        return `
          <label style="margin:0;">${{name}} enabled</label>
          <input id="provider_enabled_${{name}}" type="checkbox" ${{checked}} />
          <label style="margin:0;">${{name}} model</label>
          <input id="provider_model_${{name}}" value="${{modelName}}" />
          ${{secretField}}
          ${{name === 'ollama' ? '' : `<div id="provider_secret_status_${{name}}" style="font-size:12px;color:var(--muted);">status: unknown</div>`}}
        `;
      }}).join('');
      document.getElementById('provider-grid').innerHTML = rows;
    }}

    async function saveProfile() {{
      const providerChain = document.getElementById('provider_chain').value
        .split(',')
        .map(x => x.trim())
        .filter(Boolean);
      const providerSettings = {{}};
      for (const provider of providerNames) {{
        const enabledEl = document.getElementById(`provider_enabled_${{provider}}`);
        const modelEl = document.getElementById(`provider_model_${{provider}}`);
        providerSettings[provider] = {{
          enabled: !!(enabledEl && enabledEl.checked),
          model_name: modelEl ? (modelEl.value || '').trim() : ''
        }};
      }}
      const payload = {{
        schema_version: 'profile.v2',
        agent_name: document.getElementById('agent_name').value || 'default',
        primary_provider: document.getElementById('primary_provider').value || 'ollama',
        provider_chain: providerChain.length ? providerChain : ['ollama'],
        model_name: document.getElementById('model').value || 'llama3.2:3b',
        provider_settings: providerSettings
      }};
      const response = await fetch('/api/profile', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      const data = await response.json();
      if (!response.ok) {{
        document.getElementById('meta').textContent = 'Profile save failed: ' + (data.error || 'unknown error');
        return;
      }}
      document.getElementById('meta').textContent = 'Profile saved.';
      document.getElementById('agent').value = payload.agent_name;
      await saveProviderSecrets();
      refreshPermissions();
      refreshRuntimeStatus();
    }}

    async function refreshProviderSecrets() {{
      try {{
        const response = await fetch('/api/provider-secrets');
        if (!response.ok) return;
        const data = await response.json();
        const providers = data.providers || {{}};
        for (const provider of providerNames) {{
          if (provider === 'ollama') continue;
          const statusEl = document.getElementById(`provider_secret_status_${{provider}}`);
          if (!statusEl) continue;
          const row = providers[provider] || {{}};
          const configured = !!row.configured;
          const masked = row.masked || '';
          statusEl.textContent = configured ? `status: configured (${{masked}})` : 'status: not configured';
        }}
      }} catch (e) {{
        // ignore provider secret refresh failures
      }}
    }}

    async function saveProviderSecrets() {{
      for (const provider of providerNames) {{
        if (provider === 'ollama') continue;
        const keyEl = document.getElementById(`provider_key_${{provider}}`);
        if (!keyEl) continue;
        const key = (keyEl.value || '').trim();
        if (!key) continue;
        const response = await fetch('/api/provider-secrets', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ provider, api_key: key }})
        }});
        if (response.ok) {{
          keyEl.value = '';
        }}
      }}
      await refreshProviderSecrets();
    }}

    function permissionOptionMarkup(selected) {{
      const values = ['prompt', 'allow', 'deny'];
      return values.map(v => `<option value="${{v}}" ${{selected === v ? 'selected' : ''}}>${{v}}</option>`).join('');
    }}

    async function refreshPermissions() {{
      const agentId = document.getElementById('agent').value || 'default';
      try {{
        const response = await fetch(`/api/permissions?agent_id=${{encodeURIComponent(agentId)}}`);
        if (!response.ok) return;
        const data = await response.json();
        const permissions = data.permissions || {{}};
        const tools = Object.keys(permissions).sort();
        const html = tools.map(tool => `
          <label style="margin:0;">${{tool}}</label>
          <select id="perm_${{tool}}" onchange="savePermission('${{tool}}')">
            ${{permissionOptionMarkup(permissions[tool])}}
          </select>
        `).join('');
        document.getElementById('permission-grid').innerHTML = html;
      }} catch (e) {{
        // ignore transient permission fetch failures
      }}
    }}

    async function savePermission(toolName) {{
      const agentId = document.getElementById('agent').value || 'default';
      const select = document.getElementById(`perm_${{toolName}}`);
      if (!select) return;
      const decision = select.value;
      const response = await fetch('/api/permissions', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{agent_id: agentId, tool_name: toolName, decision}})
      }});
      const data = await response.json();
      if (!response.ok) {{
        document.getElementById('meta').textContent = 'Permission update failed: ' + (data.error || 'unknown error');
        return;
      }}
      document.getElementById('meta').textContent = `Permission updated: ${{toolName}}=${{decision}}`;
    }}

    async function runPlanPreview() {{
      const response = await fetch('/api/plan-preview', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          intent: document.getElementById('prompt').value,
          agent_id: document.getElementById('agent').value,
          model_name: document.getElementById('model').value,
          provider_name: document.getElementById('primary_provider').value || 'ollama'
        }})
      }});
      const data = await response.json();
      if (!response.ok) {{
        document.getElementById('plan-output').textContent = 'Dry-run failed: ' + (data.error || 'unknown error');
        lastPlanTraceId = null;
        return;
      }}
      lastPlanTraceId = data.trace_id || null;
      document.getElementById('plan-output').textContent = JSON.stringify(data, null, 2);
      refreshTraces();
    }}

    function renderChatHistory() {{
      const text = chatHistory.map(item => `[${{item.role}}] ${{item.text}}`).join('\n\n');
      document.getElementById('chat-history').textContent = text || 'No planner messages yet.';
    }}

    async function loadChatHistory() {{
      const agentId = document.getElementById('agent').value || 'default';
      try {{
        const response = await fetch(`/api/chat-history?agent_id=${{encodeURIComponent(agentId)}}`);
        if (!response.ok) return;
        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];
        chatHistory = items;
        renderChatHistory();
      }} catch (e) {{
        // ignore chat load failures
      }}
    }}

    async function appendChatHistory(role, text) {{
      const agentId = document.getElementById('agent').value || 'default';
      try {{
        await fetch('/api/chat-history', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{agent_id: agentId, role, text}})
        }});
      }} catch (e) {{
        // ignore chat save failures
      }}
    }}

    async function clearChatHistory() {{
      const agentId = document.getElementById('agent').value || 'default';
      try {{
        await fetch('/api/chat-clear', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{agent_id: agentId}})
        }});
      }} catch (e) {{
        // ignore chat clear failures
      }}
      chatHistory = [];
      renderChatHistory();
    }}

    async function sendChatPrompt() {{
      const input = document.getElementById('chat-input');
      const prompt = (input.value || '').trim();
      if (!prompt) return;
      chatHistory.push({{ role: 'user', text: prompt }});
      renderChatHistory();
      appendChatHistory('user', prompt);
      const response = await fetch('/api/plan-preview', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          intent: prompt,
          agent_id: document.getElementById('agent').value,
          model_name: document.getElementById('model').value,
          provider_name: document.getElementById('primary_provider').value || 'ollama'
        }})
      }});
      const data = await response.json();
      if (!response.ok) {{
        const msg = 'Dry-run failed: ' + (data.error || 'unknown error');
        chatHistory.push({{ role: 'planner', text: msg }});
        appendChatHistory('planner', msg);
        renderChatHistory();
        return;
      }}
      lastPlanTraceId = data.trace_id || null;
      document.getElementById('plan-output').textContent = JSON.stringify(data, null, 2);
      const summary = {{
        status: data.status,
        trace_id: data.trace_id,
        token_estimate: data.token_estimate || null,
        required_permissions: data.required_permissions || [],
        commands: (data.validated_commands || []).map(c => c.action),
      }};
      const plannerText = JSON.stringify(summary, null, 2);
      chatHistory.push({{ role: 'planner', text: plannerText }});
      appendChatHistory('planner', plannerText);
      renderChatHistory();
      input.value = '';
      refreshTraces();
    }}

    async function executeApprovedPlan(approvePermission=false) {{
      if (!lastPlanTraceId) {{
        document.getElementById('meta').textContent = 'Run Dry-Run Planner first.';
        return;
      }}
      const response = await fetch('/api/execute-plan', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          trace_id: lastPlanTraceId,
          agent_id: document.getElementById('agent').value,
          approve_permission: approvePermission
        }})
      }});
      const data = await response.json();
      if (!response.ok) {{
        if (data.error === 'permission_required') {{
          const ok = window.confirm(`${{data.message}} Approve now?`);
          if (ok) {{
            const result = await executeApprovedPlan(true);
            refreshPermissions();
            return result;
          }}
          document.getElementById('meta').textContent = 'Permission not granted: ' + (data.tool_name || '');
          return;
        }}
        document.getElementById('meta').textContent = 'Execute failed: ' + (data.error || 'unknown error');
        return;
      }}
      if (data.job_id) {{
        currentJobId = data.job_id;
        startPolling();
      }}
      refreshTraces();
    }}

    function startRecord() {{
      runAction('record');
    }}

    async function stopCurrent() {{
      if (!currentJobId) return;
      await fetch(`/api/jobs/${{currentJobId}}/stop`, {{ method: 'POST' }});
    }}

    function startPolling() {{
      if (pollHandle) clearInterval(pollHandle);
      pollHandle = setInterval(refreshJob, 900);
      refreshJob();
    }}

    function setViewMode(mode) {{
      viewMode = mode;
      const grid = document.getElementById('viewer-grid');
      grid.classList.remove('mode-terminal', 'mode-browser', 'mode-split');
      grid.classList.add('mode-' + mode);
      const terminalPane = document.getElementById('terminal-pane');
      const browserPane = document.getElementById('browser-pane');
      terminalPane.classList.toggle('hidden', mode === 'browser');
      browserPane.classList.toggle('hidden', mode === 'terminal');
    }}

    function maximizePane(paneId) {{
      const pane = document.getElementById(paneId);
      if (!pane) return;
      if (pane.requestFullscreen) {{
        pane.requestFullscreen();
      }}
    }}

    function updateBrowserLink(url) {{
      const link = document.getElementById('browser-link');
      const frame = document.getElementById('browser-frame');
      if (!url) {{
        link.style.display = 'none';
        return;
      }}
      link.style.display = 'block';
      link.href = url;
      link.textContent = 'Browser UI URL: ' + url;
      if (frame.src !== url) {{
        frame.src = url;
      }}
    }}

    async function refreshJob() {{
      if (!currentJobId) return;
      const response = await fetch(`/api/jobs/${{currentJobId}}`);
      if (!response.ok) return;
      const data = await response.json();
      document.getElementById('meta').textContent = `[${{data.status}}] ${{data.command}}`;
      document.getElementById('output').textContent = data.output || '';
      updateBrowserLink(data.browser_ui_url || '');
      if (data.status !== 'running' && pollHandle) {{
        clearInterval(pollHandle);
        pollHandle = null;
      }}
    }}

    async function refreshRuntimeStatus() {{
      try {{
        const response = await fetch('/api/runtime-status');
        if (!response.ok) return;
        const data = await response.json();
        const ollamaNote = data.ollama_installed ? '' : ' | Ollama=missing (click Install Ollama)';
        const heal = data.self_heal_healed ? 'recovered' : (data.self_heal_probe_ok ? 'ok' : 'failed');
        const line = `Provider=${{data.provider_name}}:${{data.model_name}} | Health=${{data.provider_state}} | Budget=${{data.budget_status}} | Today=$${{Number(data.today_cost_usd || 0).toFixed(2)}}/${{Number(data.daily_limit_usd || 0).toFixed(2)}} | Queue=${{data.queue_depth}} | Heartbeat=${{data.heartbeat_age_seconds >= 0 ? data.heartbeat_age_seconds + 's' : 'unknown'}} | Heal=${{heal}}${{ollamaNote}}`;
        document.getElementById('runtime-line').textContent = line;
        const matrix = data.provider_matrix || {{}};
        const cards = Object.keys(matrix).sort().map(name => {{
          const row = matrix[name] || {{}};
          const usable = !!row.usable;
          const reason = row.reason ? ` | reason=${{row.reason}}` : '';
          const model = row.model_name ? ` | model=${{row.model_name}}` : '';
          const bg = usable ? '#e7f6ef' : '#fff3ea';
          return `<div class="provider-card hover-card" title="enabled=${{!!row.enabled}}, configured=${{!!row.configured}}${{reason}}${{model}}" style="background:${{bg}};">
            <strong>${{name}}</strong><br/>
            <span style="font-size:12px;color:var(--muted);">enabled=${{!!row.enabled}} | configured=${{!!row.configured}} | usable=${{usable}}</span>
          </div>`;
        }}).join('');
        document.getElementById('provider-cards').innerHTML = cards || '';
      }} catch (e) {{
        // keep previous UI state on transient fetch failure
      }}
    }}

    async function refreshTraces() {{
      try {{
        const response = await fetch('/api/traces');
        if (!response.ok) return;
        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];
        const select = document.getElementById('trace-select');
        if (!items.length) {{
          select.innerHTML = '<option value="">No traces yet</option>';
          document.getElementById('trace-output').textContent = 'No traces yet.';
          return;
        }}
        const previous = select.value;
        select.innerHTML = items.map(item => {{
          const label = `${{item.trace_id}} | ${{item.type}} | stages=${{item.stage_count}} | last=${{item.last_event}}`;
          const selected = (previous && previous === item.trace_id) || (!previous && item === items[0]);
          return `<option value="${{item.trace_id}}" ${{selected ? 'selected' : ''}}>${{label}}</option>`;
        }}).join('');
        await loadSelectedTrace();
      }} catch (e) {{
        // keep existing trace view on fetch failures
      }}
    }}

    async function loadSelectedTrace() {{
      const select = document.getElementById('trace-select');
      const traceId = select.value;
      if (!traceId) return;
      try {{
        const response = await fetch(`/api/traces/${{encodeURIComponent(traceId)}}`);
        if (!response.ok) {{
          document.getElementById('trace-output').textContent = 'Failed to load trace details.';
          return;
        }}
        const trace = await response.json();
        document.getElementById('trace-output').textContent = JSON.stringify(trace, null, 2);
      }} catch (e) {{
        document.getElementById('trace-output').textContent = 'Failed to load trace details.';
      }}
    }}

    async function exportSelectedTrace() {{
      const select = document.getElementById('trace-select');
      const traceId = select.value;
      if (!traceId) return;
      try {{
        const response = await fetch(`/api/traces/${{encodeURIComponent(traceId)}}`);
        if (!response.ok) {{
          document.getElementById('meta').textContent = 'Trace export failed.';
          return;
        }}
        const trace = await response.json();
        const blob = new Blob([JSON.stringify(trace, null, 2)], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${{traceId}}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        document.getElementById('meta').textContent = `Trace exported: ${{traceId}}.json`;
      }} catch (e) {{
        document.getElementById('meta').textContent = 'Trace export failed.';
      }}
    }}

    setViewMode('split');
    loadProfile();
    refreshPermissions();
    refreshRuntimeStatus();
    refreshTraces();
    loadChatHistory();
    setInterval(refreshRuntimeStatus, 5000);
    setInterval(refreshTraces, 5000);
  </script>
</body>
</html>
"""


def run_guide_server(
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 7788,
    open_browser: bool = True,
) -> None:
    """Run the guided web server until interrupted."""
    state = GuideState(workspace=workspace, python_bin=sys.executable)
    handler = _make_handler(state)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"BorisBot guide available at: {url}")
    print("Press Ctrl+C to stop the guide server.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _collect_runtime_status(python_bin: str) -> dict:
    """Collect runtime status for GUI without requiring CLI invocation."""
    profile = load_profile()
    model_name = str(profile.get("model_name", os.getenv("BORISBOT_OLLAMA_MODEL", "llama3.2:3b")))
    provider_name = str(profile.get("primary_provider", "ollama"))
    provider_state = "unknown"
    budget_status = "ok"
    today_cost = 0.0
    daily_limit = 0.0
    daily_remaining = 0.0
    session_tokens = 0
    session_cost = 0.0
    active_tasks = 0
    queue_depth = 0
    heartbeat_age = -1
    self_heal_probe_ok = False
    self_heal_healed = False
    ollama_installed = shutil.which("ollama") is not None
    provider_matrix: dict[str, dict[str, object]] = {}

    try:
        response = httpx.get("http://127.0.0.1:7777/metrics/providers", timeout=1.5)
        payload = response.json() if response.status_code == 200 else {}
        if isinstance(payload, dict):
            row = payload.get(provider_name, {})
            if isinstance(row, dict):
                provider_state = str(row.get("state", provider_state))
    except Exception:
        row = get_provider_health_registry().get_snapshot().get(provider_name, {})
        if isinstance(row, dict):
            provider_state = str(row.get("state", provider_state))

    try:
        budget = asyncio.run(CostGuard().get_budget_status("default"))
        budget_status = str(budget.get("status", "ok")).upper()
        today_cost = float(budget.get("daily_spend", 0.0))
        daily_limit = float(budget.get("daily_limit", 0.0))
        daily_remaining = float(budget.get("daily_remaining", 0.0))
    except Exception:
        pass

    try:
        cost_guard = CostGuard()
        session_start = asyncio.run(cost_guard.get_runtime_session_started_at())
        usage = asyncio.run(cost_guard.get_usage_window(start_iso=session_start))
        session_tokens = int(usage.get("total_tokens", 0))
        session_cost = float(usage.get("cost_usd", 0.0))
    except Exception:
        pass

    async def _task_counts() -> tuple[int, int]:
        async for db in get_db():
            cursor = await db.execute("SELECT COUNT(*) AS count FROM tasks WHERE status = 'running'")
            row = await cursor.fetchone()
            running = int(row["count"] if row else 0)
            cursor = await db.execute("SELECT COUNT(*) AS count FROM task_queue")
            row = await cursor.fetchone()
            depth = int(row["count"] if row else 0)
            return running, depth
        return 0, 0

    try:
        active_tasks, queue_depth = asyncio.run(_task_counts())
    except Exception:
        pass

    heartbeat = read_heartbeat_snapshot()
    if isinstance(heartbeat, dict):
        ts = heartbeat.get("timestamp")
        if isinstance(ts, str):
            try:
                heartbeat_age = int((datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds())
            except Exception:
                heartbeat_age = -1
        self_heal_probe_ok = bool(heartbeat.get("self_heal_probe_ok", False))
        self_heal_healed = bool(heartbeat.get("self_heal_healed", False))

    provider_settings = profile.get("provider_settings", {})
    secret_status = get_secret_status()
    for provider in ["ollama", "openai", "anthropic", "google", "azure"]:
        settings = provider_settings.get(provider, {}) if isinstance(provider_settings, dict) else {}
        enabled = bool(settings.get("enabled", provider == "ollama")) if isinstance(settings, dict) else (provider == "ollama")
        model = str(settings.get("model_name", "")).strip() if isinstance(settings, dict) else ""
        if provider == "ollama":
            configured = ollama_installed
            usable = enabled and ollama_installed
            reason = "" if usable else "ollama_not_installed"
        elif provider in {"anthropic", "google", "azure"}:
            configured = bool(secret_status.get(provider, {}).get("configured", False))
            usable = False
            reason = "transport_unimplemented" if enabled else "disabled"
        else:
            configured = bool(secret_status.get(provider, {}).get("configured", False))
            usable = enabled and configured
            reason = "" if usable else ("api_key_missing" if enabled else "disabled")
        provider_matrix[provider] = {
            "enabled": enabled,
            "configured": configured,
            "usable": usable,
            "reason": reason,
            "model_name": model,
        }

    return {
        "provider_name": provider_name,
        "provider_state": provider_state,
        "provider_matrix": provider_matrix,
        "model_name": model_name,
        "session_tokens": session_tokens,
        "session_cost_usd": session_cost,
        "today_cost_usd": today_cost,
        "daily_limit_usd": daily_limit,
        "daily_remaining_usd": daily_remaining,
        "budget_status": budget_status,
        "active_tasks": active_tasks,
        "queue_depth": queue_depth,
        "heartbeat_age_seconds": heartbeat_age,
        "self_heal_probe_ok": self_heal_probe_ok,
        "self_heal_healed": self_heal_healed,
        "ollama_installed": ollama_installed,
    }
