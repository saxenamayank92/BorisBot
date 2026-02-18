import asyncio
import json
import typer
import uvicorn
import subprocess
import shutil
import os
import sys
import time
import re
import httpx
import socket
import io
from urllib.parse import quote
from contextlib import redirect_stdout
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Optional
from uuid import uuid4

from borisbot.contracts import SUPPORTED_TASK_COMMAND_SCHEMAS, TASK_COMMAND_SCHEMA_V1
from borisbot.failures import build_failure
from borisbot.recorder.analyzer import analyze_workflow_file
from borisbot.recorder.runner import run_record
from borisbot.guide.server import (
    _build_assistant_response,
    _build_dry_run_preview,
    _collect_runtime_status,
    _probe_provider_connection,
    _resolve_ollama_install_command,
    _resolve_ollama_start_command,
    run_guide_server,
)
from borisbot.guide.chat_history_store import (
    append_chat_message,
    clear_chat_history,
    clear_chat_roles,
    load_chat_history,
)
from borisbot.browser.actions import BrowserActions
from borisbot.browser.command_router import CommandRouter
from borisbot.browser.executor import BrowserExecutor
from borisbot.browser.task_runner import TaskRunner
from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.provider_health import get_provider_health_registry
from borisbot.supervisor.browser_manager import BrowserManager
from borisbot.supervisor.capability_manager import CapabilityManager
from borisbot.supervisor.database import get_db
from borisbot.supervisor.heartbeat_runtime import read_heartbeat_snapshot
from borisbot.supervisor.profile_config import load_profile, save_profile
from borisbot.supervisor.tool_permissions import (
    ALLOWED_DECISIONS,
    ALLOWED_TOOLS,
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_PROMPT,
    TOOL_ASSISTANT,
    TOOL_BROWSER,
    TOOL_FILESYSTEM,
    TOOL_SCHEDULER,
    TOOL_SHELL,
    TOOL_WEB_FETCH,
    get_agent_permission_matrix_sync,
    get_agent_tool_permission_sync,
    set_agent_tool_permission_sync,
)
from borisbot.supervisor.worker import Worker

app = typer.Typer()
worker_app = typer.Typer()
app.add_typer(worker_app, name="worker")

BORISBOT_DIR = Path.home() / ".borisbot"
PID_FILE = BORISBOT_DIR / "supervisor.pid"
LOG_DIR = BORISBOT_DIR / "logs"
SUPERVISOR_HOST = "127.0.0.1"
SUPERVISOR_PORT = 7777
SUPERVISOR_URL = f"http://{SUPERVISOR_HOST}:{SUPERVISOR_PORT}"

POLICY_PACKS: dict[str, dict[str, object]] = {
    "safe-local": {
        "profile": {
            "primary_provider": "ollama",
            "provider_chain": ["ollama"],
            "provider_settings_enabled": {"ollama": True},
        },
        "permissions": {
            TOOL_ASSISTANT: DECISION_ALLOW,
            TOOL_BROWSER: DECISION_PROMPT,
            TOOL_FILESYSTEM: DECISION_PROMPT,
            TOOL_SHELL: DECISION_PROMPT,
            TOOL_WEB_FETCH: DECISION_DENY,
            TOOL_SCHEDULER: DECISION_DENY,
        },
    },
    "web-readonly": {
        "profile": {
            "primary_provider": "ollama",
            "provider_chain": ["ollama", "openai"],
            "provider_settings_enabled": {"ollama": True, "openai": True},
        },
        "permissions": {
            TOOL_ASSISTANT: DECISION_ALLOW,
            TOOL_BROWSER: DECISION_ALLOW,
            TOOL_FILESYSTEM: DECISION_PROMPT,
            TOOL_SHELL: DECISION_DENY,
            TOOL_WEB_FETCH: DECISION_ALLOW,
            TOOL_SCHEDULER: DECISION_DENY,
        },
    },
    "automation": {
        "profile": {
            "primary_provider": "ollama",
            "provider_chain": ["ollama", "openai"],
            "provider_settings_enabled": {"ollama": True, "openai": True},
        },
        "permissions": {
            TOOL_ASSISTANT: DECISION_ALLOW,
            TOOL_BROWSER: DECISION_ALLOW,
            TOOL_FILESYSTEM: DECISION_ALLOW,
            TOOL_SHELL: DECISION_ALLOW,
            TOOL_WEB_FETCH: DECISION_ALLOW,
            TOOL_SCHEDULER: DECISION_ALLOW,
        },
    },
}

POLICY_PACK_DESCRIPTIONS = {
    "safe-local": "Local-first with conservative tool prompts and no scheduler/web-fetch.",
    "web-readonly": "Browser + web read access with shell denied by default.",
    "automation": "High-autonomy defaults with all core tools allowed.",
}

def ensure_dirs():
    BORISBOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((SUPERVISOR_HOST, port)) == 0

@app.command()
def start():
    """Start the supervisor in the background."""
    ensure_dirs()
    
    # 1. Check PID file first
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text())
            if psutil_pid_exists(pid):
                typer.echo(f"Supervisor already running (PID: {pid})")
                return
            else:
                typer.echo("Stale PID file found. Removing...")
                PID_FILE.unlink()
        except Exception:
            PID_FILE.unlink()

    # 2. Check Port
    if is_port_in_use(SUPERVISOR_PORT):
        typer.echo(f"Error: Port {SUPERVISOR_PORT} is already in use by another process.")
        typer.echo("Please stop the existing process or check for zombies.")
        raise typer.Exit(code=1)

    typer.echo("Starting supervisor...")
    
    # Launch uvicorn as a subprocess
    # We use sys.executable to ensure we use the same python environment
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "borisbot.supervisor.app:app", 
        "--host", SUPERVISOR_HOST, 
        "--port", str(SUPERVISOR_PORT)
    ]
    
    log_file = open(LOG_DIR / "supervisor.log", "a")
    
    # Detach process - complicated on Windows vs Unix
    # On Windows, creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS might needed
    # but strictly specific. 
    # For now, we will just use Popen. On a real daemon setup this needs more work.
    
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        **kwargs
    )
    
    PID_FILE.write_text(str(process.pid))
    typer.echo(f"Supervisor started (PID: {process.pid})")

@app.command()
def stop():
    """Stop the supervisor."""
    if not PID_FILE.exists():
        typer.echo("Supervisor not running (no PID file)")
        return
        
    try:
        pid = int(PID_FILE.read_text())
        
        # 1. Attempt Graceful Shutdown via API
        try:
            typer.echo("Attempting graceful shutdown...")
            response = httpx.post(f"{SUPERVISOR_URL}/shutdown", timeout=5.0)
            if response.status_code == 200:
                typer.echo(f"Supervisor shutting down gracefully (PID: {pid})...")
                PID_FILE.unlink()
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            typer.echo("Graceful shutdown failed (API unreachable).")

        # 2. Fallback to OS Kill
        import signal
        typer.echo(f"Forcing stop (PID: {pid})...")
        os.kill(pid, signal.SIGTERM)
        typer.echo("Supervisor stopped.")
        PID_FILE.unlink()
    except ProcessLookupError:
        typer.echo("Supervisor process not found. Cleaning up PID file.")
        PID_FILE.unlink()
    except Exception as e:
        typer.echo(f"Failed to stop supervisor: {e}")

@app.command()
def status():
    """Check supervisor status."""
    if not PID_FILE.exists():
        typer.echo("Supervisor: STOPPED")
        return

    try:
        response = httpx.get(f"{SUPERVISOR_URL}/health")
        if response.status_code == 200:
            typer.echo("Supervisor: RUNNING")
            
            # Get agents
            agents_resp = httpx.get(f"{SUPERVISOR_URL}/agents")
            if agents_resp.status_code == 200:
                agents = agents_resp.json()
                typer.echo(f"Active Agents: {len(agents)}")
                for agent in agents:
                    typer.echo(f" - {agent['name']} ({agent['id']}) [{agent['status']}]")
        else:
            typer.echo("Supervisor: UNHEALTHY (API not responding correctly)")
    except httpx.ConnectError:
        typer.echo("Supervisor: NOT RESPONDING (Connection refused)")

@app.command()
def spawn(name: str):
    """Spawn a new agent."""
    try:
        response = httpx.post(f"{SUPERVISOR_URL}/agents", json={"name": name})
        if response.status_code == 200:
            data = response.json()
            typer.echo(f"Agent spawned: {data['name']} (ID: {data['id']})")
        else:
            typer.echo(f"Failed to spawn agent: {response.text}")
    except httpx.ConnectError:
        typer.echo("Supervisor is not running.")


@app.command()
def record(
    task_id: str,
    start_url: str = typer.Option(..., "--start-url", help="Initial URL to open before recording"),
):
    """Record workflow actions and replay immediately for validation."""
    try:
        asyncio.run(run_record(task_id, start_url=start_url))
    except RuntimeError as exc:
        typer.echo(_format_record_runtime_error(exc))
        raise typer.Exit(code=1)


def _format_record_runtime_error(exc: RuntimeError) -> str:
    """Return compact user-facing guidance for common recorder runtime failures."""
    message = str(exc).strip()
    lower = message.lower()
    if "docker daemon" in lower or "cannot connect to the docker daemon" in lower:
        return (
            "Recording failed: Docker is not running.\n"
            "Start Docker Desktop (or docker service) and retry."
        )
    if "maximum browser sessions reached" in lower:
        return (
            "Recording failed: Maximum browser sessions reached.\n"
            "Run `borisbot cleanup-browsers` and retry."
        )
    if "address already in use" in lower:
        return (
            "Recording failed: Recorder port is already in use.\n"
            "Stop stale borisbot processes and retry."
        )
    return f"Recording failed: {message}"


@app.command()
def guide(
    host: str = typer.Option("127.0.0.1", "--host", help="Guide server host"),
    port: int = typer.Option(7788, "--port", help="Guide server port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
):
    """Launch guided local web UI for record/replay/release-check actions."""
    run_guide_server(Path.cwd(), host=host, port=port, open_browser=open_browser)


@app.command("plan-preview")
def plan_preview(
    prompt: str,
    agent_id: str = typer.Option("default", "--agent-id"),
    model_name: str = typer.Option("llama3.2:3b", "--model"),
    provider_name: str = typer.Option("ollama", "--provider"),
    json_output: bool = typer.Option(False, "--json", help="Print full JSON preview payload"),
):
    """Run planner dry-run preview from CLI with provider fallback metadata."""
    preview = _build_dry_run_preview(
        prompt,
        agent_id=agent_id,
        model_name=model_name,
        provider_name=provider_name,
    )
    if json_output:
        typer.echo(json.dumps(preview, indent=2))
        if preview.get("status") != "ok":
            raise typer.Exit(code=1)
        return

    if preview.get("status") != "ok":
        typer.echo("PLAN PREVIEW: FAIL")
        typer.echo(f"  error: {preview.get('error_code', 'UNKNOWN')}")
        attempts = preview.get("provider_attempts", [])
        if isinstance(attempts, list) and attempts:
            first = attempts[0]
            if isinstance(first, dict):
                typer.echo(
                    f"  provider_attempt: {first.get('provider', 'unknown')} ({first.get('status', 'unknown')})"
                )
        raise typer.Exit(code=1)

    token_est = preview.get("token_estimate", {})
    total_tokens = int(token_est.get("total_tokens", 0)) if isinstance(token_est, dict) else 0
    cost = float(preview.get("cost_estimate_usd", 0.0))
    provider_selected = str(preview.get("provider_name", "unknown"))
    commands = preview.get("validated_commands", [])
    command_count = len(commands) if isinstance(commands, list) else 0
    required_permissions = preview.get("required_permissions", [])
    if not isinstance(required_permissions, list):
        required_permissions = []
    budget_snapshot = preview.get("budget", {})
    budget_status = (
        str(budget_snapshot.get("status", "unknown")).upper()
        if isinstance(budget_snapshot, dict)
        else "UNKNOWN"
    )
    typer.echo("PLAN PREVIEW: OK")
    typer.echo(f"  provider: {provider_selected}")
    typer.echo(f"  commands: {command_count}")
    typer.echo(f"  tokens_est: {total_tokens}")
    typer.echo(f"  cost_est_usd: ${cost:.4f}")
    typer.echo(f"  budget_status: {budget_status}")
    typer.echo(f"  required_permissions: {len(required_permissions)}")
    for row in required_permissions:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("tool_name", "")).strip() or "unknown"
        decision = str(row.get("decision", "")).strip() or "unknown"
        typer.echo(f"    - {tool_name}: {decision}")


def _run_setup_command(command: list[str]) -> tuple[int, str]:
    """Run setup command and return (returncode, combined output)."""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, f"command not found: {command[0] if command else 'unknown'}"
    output = (result.stdout or "") + (result.stderr or "")
    return int(result.returncode), output.strip()


def _build_doctor_report(model_name: str) -> dict[str, object]:
    """Build deterministic local prerequisite diagnostics snapshot."""
    model = (model_name or "").strip() or "llama3.2:3b"
    docker_installed = shutil.which("docker") is not None
    docker_ready = False
    docker_message = ""
    if docker_installed:
        docker_rc, docker_output = _run_setup_command(["docker", "info"])
        docker_ready = docker_rc == 0
        docker_message = docker_output or ("ok" if docker_ready else "docker info failed")
    else:
        docker_message = "docker command not found"

    ollama_installed = shutil.which("ollama") is not None
    ollama_reachable = False
    ollama_message = ""
    model_pulled = False
    if ollama_installed:
        try:
            response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                ollama_reachable = True
                payload = response.json()
                models = payload.get("models", []) if isinstance(payload, dict) else []
                if isinstance(models, list):
                    for row in models:
                        if not isinstance(row, dict):
                            continue
                        name = str(row.get("name", "")).strip()
                        if name == model:
                            model_pulled = True
                            break
                ollama_message = "reachable"
            else:
                ollama_message = f"tags probe failed: HTTP {response.status_code}"
        except Exception as exc:
            ollama_message = str(exc)
    else:
        ollama_message = "ollama command not found"

    status = "ok" if (docker_ready and ollama_reachable and model_pulled) else "warn"
    return {
        "status": status,
        "model": model,
        "docker": {
            "installed": docker_installed,
            "ready": docker_ready,
            "message": docker_message,
        },
        "ollama": {
            "installed": ollama_installed,
            "reachable": ollama_reachable,
            "message": ollama_message,
            "model_pulled": model_pulled,
        },
    }


@app.command("llm-setup")
def llm_setup(
    model_name: str = typer.Option("llama3.2:3b", "--model"),
    auto_install: bool = typer.Option(
        True,
        "--auto-install/--no-auto-install",
        help="Install Ollama automatically if missing",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print structured setup result"),
):
    """Install/start Ollama and pull selected model in one flow."""
    if not isinstance(model_name, str):
        model_name = "llama3.2:3b"
    if not isinstance(auto_install, bool):
        auto_install = True
    if not isinstance(json_output, bool):
        json_output = False

    model = model_name.strip() or "llama3.2:3b"
    payload: dict[str, object] = {
        "status": "failed",
        "model": model,
        "steps": [],
    }

    def _add_step(step: str, command: list[str] | None, status: str, output: str = "") -> None:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return
        steps.append(
            {
                "step": step,
                "command": " ".join(command) if isinstance(command, list) else "",
                "status": status,
                "output": output.strip(),
            }
        )

    def _fail(exit_code: int = 1) -> None:
        if json_output:
            typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=exit_code)

    ollama_installed = shutil.which("ollama") is not None
    if not ollama_installed and not auto_install:
        payload["error"] = "OLLAMA_NOT_INSTALLED"
        payload["message"] = "rerun with --auto-install or install from https://ollama.com/download"
        if not json_output:
            typer.echo("LLM SETUP: FAIL")
            typer.echo("  error: OLLAMA_NOT_INSTALLED")
            typer.echo("  hint: rerun with --auto-install or install from https://ollama.com/download")
        _fail(1)

    if not ollama_installed:
        try:
            install_cmd = _resolve_ollama_install_command(sys.platform)
        except ValueError as exc:
            payload["error"] = "INSTALL_COMMAND_UNAVAILABLE"
            payload["message"] = str(exc)
            _add_step("install", None, "failed", str(exc))
            if not json_output:
                typer.echo("LLM SETUP: FAIL")
                typer.echo("  step: install")
                typer.echo(f"  error: {exc}")
            _fail(1)
        rc, output = _run_setup_command(install_cmd)
        _add_step("install", install_cmd, "completed" if rc == 0 else "failed", output)
        if rc != 0:
            payload["error"] = "INSTALL_FAILED"
            if not json_output:
                typer.echo("LLM SETUP: FAIL")
                typer.echo("  step: install")
                typer.echo(f"  command: {' '.join(install_cmd)}")
                typer.echo(f"  output: {output or 'no output'}")
            _fail(1)

    start_cmd = _resolve_ollama_start_command(sys.platform)
    start_rc, start_output = _run_setup_command(start_cmd)
    _add_step("start", start_cmd, "completed" if start_rc == 0 else "failed", start_output)
    if start_rc != 0:
        payload["error"] = "START_FAILED"
        if not json_output:
            typer.echo("LLM SETUP: FAIL")
            typer.echo("  step: start")
            typer.echo(f"  command: {' '.join(start_cmd)}")
            typer.echo(f"  output: {start_output or 'no output'}")
        _fail(1)

    pull_cmd = ["ollama", "pull", model]
    pull_rc, pull_output = _run_setup_command(pull_cmd)
    _add_step("pull", pull_cmd, "completed" if pull_rc == 0 else "failed", pull_output)
    if pull_rc != 0:
        payload["error"] = "PULL_FAILED"
        if not json_output:
            typer.echo("LLM SETUP: FAIL")
            typer.echo("  step: pull")
            typer.echo(f"  command: {' '.join(pull_cmd)}")
            typer.echo(f"  output: {pull_output or 'no output'}")
        _fail(1)

    payload["status"] = "ok"
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("LLM SETUP: OK")
    typer.echo(f"  model: {model}")
    typer.echo(f"  start: {' '.join(start_cmd)}")
    typer.echo(f"  pull: {' '.join(pull_cmd)}")


@app.command("setup")
def setup(
    model_name: str = typer.Option("llama3.2:3b", "--model"),
    auto_install: bool = typer.Option(True, "--auto-install/--no-auto-install"),
    launch_guide: bool = typer.Option(True, "--launch-guide/--no-launch-guide"),
    guide_host: str = typer.Option("127.0.0.1", "--guide-host"),
    guide_port: int = typer.Option(7788, "--guide-port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Bootstrap local prerequisites and optionally launch the guided UI."""
    if not isinstance(model_name, str):
        model_name = "llama3.2:3b"
    if not isinstance(auto_install, bool):
        auto_install = True
    if not isinstance(launch_guide, bool):
        launch_guide = True
    if not isinstance(guide_host, str):
        guide_host = "127.0.0.1"
    if not isinstance(guide_port, int):
        guide_port = 7788
    if not isinstance(open_browser, bool):
        open_browser = True
    if not isinstance(json_output, bool):
        json_output = False

    model = model_name.strip() or "llama3.2:3b"
    docker_installed = shutil.which("docker") is not None
    docker_ok = False
    docker_message = ""
    if docker_installed:
        docker_rc, docker_output = _run_setup_command(["docker", "info"])
        docker_ok = docker_rc == 0
        docker_message = docker_output or ("ok" if docker_ok else "docker info failed")
    else:
        docker_message = "docker command not found"

    llm_payload: dict[str, object] = {}
    llm_ok = False
    llm_out = io.StringIO()
    llm_exit_code = 0
    try:
        with redirect_stdout(llm_out):
            llm_setup(model_name=model, auto_install=auto_install, json_output=True)
        llm_ok = True
    except typer.Exit as exc:
        llm_exit_code = int(exc.exit_code or 1)
    llm_text = llm_out.getvalue().strip()
    if llm_text:
        try:
            parsed = json.loads(llm_text)
            if isinstance(parsed, dict):
                llm_payload = parsed
                llm_ok = str(parsed.get("status", "")).lower() == "ok"
        except json.JSONDecodeError:
            llm_payload = {"status": "failed", "error": "INVALID_SETUP_PAYLOAD", "raw": llm_text}
            llm_ok = False
    if not llm_payload:
        llm_payload = {"status": "failed", "error": "LLM_SETUP_NO_OUTPUT"}
        llm_ok = False

    overall_ok = docker_ok and llm_ok
    guide_url = f"http://{guide_host}:{guide_port}"
    payload = {
        "status": "ok" if overall_ok else "warn",
        "docker": {
            "installed": docker_installed,
            "ready": docker_ok,
            "message": docker_message,
        },
        "llm_setup": llm_payload,
        "guide": {
            "launch_requested": launch_guide,
            "url": guide_url,
        },
    }

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        if launch_guide:
            run_guide_server(Path.cwd(), host=guide_host, port=guide_port, open_browser=open_browser)
        if llm_exit_code != 0 and not launch_guide:
            raise typer.Exit(code=llm_exit_code)
        return

    typer.echo("SETUP: OK" if overall_ok else "SETUP: WARN")
    typer.echo(f"  docker: {'ready' if docker_ok else 'not_ready'}")
    if docker_message:
        typer.echo(f"  docker_detail: {docker_message}")
    llm_status = str(llm_payload.get("status", "failed"))
    typer.echo(f"  llm_setup: {llm_status}")
    if llm_status != "ok":
        err = str(llm_payload.get("error", "UNKNOWN")).strip()
        msg = str(llm_payload.get("message", "")).strip()
        typer.echo(f"  llm_error: {err}")
        if msg:
            typer.echo(f"  llm_message: {msg}")
    typer.echo(f"  guide_url: {guide_url}")
    if launch_guide:
        run_guide_server(Path.cwd(), host=guide_host, port=guide_port, open_browser=open_browser)
        return
    if not overall_ok:
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor(
    model_name: str = typer.Option("llama3.2:3b", "--model"),
    strict: bool = typer.Option(False, "--strict", help="Exit nonzero when status is not OK"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Run local runtime diagnostics with actionable status hints."""
    if not isinstance(model_name, str):
        model_name = "llama3.2:3b"
    if not isinstance(strict, bool):
        strict = False
    if not isinstance(json_output, bool):
        json_output = False
    report = _build_doctor_report(model_name)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        if strict and report.get("status") != "ok":
            raise typer.Exit(code=1)
        return

    typer.echo("DOCTOR: " + str(report.get("status", "unknown")).upper())
    docker = report.get("docker", {})
    ollama = report.get("ollama", {})
    typer.echo(
        f"  docker: installed={str(bool(docker.get('installed', False))).lower()} ready={str(bool(docker.get('ready', False))).lower()}"
    )
    typer.echo(
        f"  ollama: installed={str(bool(ollama.get('installed', False))).lower()} reachable={str(bool(ollama.get('reachable', False))).lower()} model_pulled={str(bool(ollama.get('model_pulled', False))).lower()}"
    )
    if report.get("status") != "ok":
        typer.echo("  hint: run `borisbot setup --model " + str(report.get("model", "llama3.2:3b")) + "`")
    if strict and report.get("status") != "ok":
        raise typer.Exit(code=1)


@app.command("assistant-chat")
def assistant_chat(
    prompt: str,
    agent_id: str = typer.Option("default", "--agent-id"),
    model_name: str = typer.Option("llama3.2:3b", "--model"),
    provider_name: str = typer.Option("ollama", "--provider"),
    approve_permission: bool = typer.Option(
        False,
        "--approve-permission",
        help="Approve assistant tool permission when agent policy is prompt",
    ),
    save_history: bool = typer.Option(
        True,
        "--save-history/--no-save-history",
        help="Persist user/assistant chat messages to local history",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print full JSON assistant payload"),
):
    """Run general LLM assistant chat using provider fallback and budget checks."""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(model_name, str):
        model_name = "llama3.2:3b"
    if not isinstance(provider_name, str):
        provider_name = "ollama"
    if not isinstance(approve_permission, bool):
        approve_permission = False
    if not isinstance(save_history, bool):
        save_history = True
    if not isinstance(json_output, bool):
        json_output = False

    agent_id = agent_id.strip() or "default"
    model_name = model_name.strip() or "llama3.2:3b"
    provider_name = provider_name.strip() or "ollama"

    decision = get_agent_tool_permission_sync(agent_id, TOOL_ASSISTANT)
    if decision != DECISION_ALLOW:
        if decision == DECISION_PROMPT and approve_permission:
            set_agent_tool_permission_sync(agent_id, TOOL_ASSISTANT, DECISION_ALLOW)
        else:
            typer.echo("ASSISTANT CHAT: FAIL")
            if decision == DECISION_PROMPT:
                typer.echo("  error: ASSISTANT_PERMISSION_REQUIRED")
                typer.echo("  hint: rerun with --approve-permission")
            else:
                typer.echo("  error: ASSISTANT_PERMISSION_DENIED")
            raise typer.Exit(code=1)

    payload = _build_assistant_response(
        prompt,
        agent_id=agent_id,
        model_name=model_name,
        provider_name=provider_name,
    )
    if payload.get("status") == "ok" and save_history:
        assistant_message = str(payload.get("message", "")).strip()
        if prompt.strip():
            append_chat_message(agent_id, "assistant_user", prompt.strip())
        if assistant_message:
            append_chat_message(agent_id, "assistant", assistant_message)

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        if payload.get("status") != "ok":
            raise typer.Exit(code=1)
        return

    if payload.get("status") != "ok":
        typer.echo("ASSISTANT CHAT: FAIL")
        typer.echo(f"  error: {payload.get('error_code', 'UNKNOWN')}")
        attempts = payload.get("provider_attempts", [])
        if isinstance(attempts, list) and attempts:
            first = attempts[0]
            if isinstance(first, dict):
                typer.echo(
                    f"  provider_attempt: {first.get('provider', 'unknown')} ({first.get('status', 'unknown')})"
                )
        raise typer.Exit(code=1)

    token_est = payload.get("token_estimate", {})
    total_tokens = int(token_est.get("total_tokens", 0)) if isinstance(token_est, dict) else 0
    cost = float(payload.get("cost_estimate_usd", 0.0))
    provider_selected = str(payload.get("provider_name", "unknown"))
    message = str(payload.get("message", "")).strip()
    typer.echo("ASSISTANT CHAT: OK")
    typer.echo(f"  provider: {provider_selected}")
    typer.echo(f"  tokens_est: {total_tokens}")
    typer.echo(f"  cost_est_usd: ${cost:.4f}")
    typer.echo("")
    typer.echo(message)


@app.command("chat-history")
def chat_history(
    agent_id: str = typer.Option("default", "--agent-id"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Print persisted chat history for an agent."""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(json_output, bool):
        json_output = False
    agent = agent_id.strip() or "default"
    items = load_chat_history(agent)
    if json_output:
        typer.echo(json.dumps({"agent_id": agent, "items": items}, indent=2))
        return
    typer.echo(f"CHAT HISTORY ({agent})")
    if not items:
        typer.echo("  (empty)")
        return
    for row in items:
        role = str(row.get("role", "")).strip() or "unknown"
        text = str(row.get("text", "")).strip()
        typer.echo(f"  [{role}] {text}")


@app.command("chat-clear")
def chat_clear(
    agent_id: str = typer.Option("default", "--agent-id"),
    assistant_only: bool = typer.Option(False, "--assistant-only"),
):
    """Clear persisted chat history for an agent."""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(assistant_only, bool):
        assistant_only = False
    agent = agent_id.strip() or "default"
    if assistant_only:
        clear_chat_roles(agent, {"assistant_user", "assistant"})
        typer.echo(f"CHAT CLEAR: OK ({agent}, assistant_only=true)")
        return
    clear_chat_history(agent)
    typer.echo(f"CHAT CLEAR: OK ({agent}, assistant_only=false)")


@app.command("permissions")
def permissions(
    agent_id: str = typer.Option("default", "--agent-id"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Print per-agent tool permission matrix."""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(json_output, bool):
        json_output = False
    agent = agent_id.strip() or "default"
    matrix = get_agent_permission_matrix_sync(agent)
    if json_output:
        typer.echo(json.dumps({"agent_id": agent, "permissions": matrix}, indent=2))
        return
    typer.echo(f"PERMISSIONS ({agent})")
    for tool_name in sorted(matrix.keys()):
        typer.echo(f"  - {tool_name}: {matrix[tool_name]}")


@app.command("set-permission")
def set_permission(
    tool_name: str = typer.Option(..., "--tool"),
    decision: str = typer.Option(..., "--decision"),
    agent_id: str = typer.Option("default", "--agent-id"),
):
    """Set per-agent tool permission decision."""
    if not isinstance(tool_name, str):
        tool_name = ""
    if not isinstance(decision, str):
        decision = ""
    if not isinstance(agent_id, str):
        agent_id = "default"
    tool = tool_name.strip().lower()
    value = decision.strip().lower()
    agent = agent_id.strip() or "default"

    if tool not in ALLOWED_TOOLS:
        typer.echo("SET PERMISSION: FAIL")
        typer.echo(f"  error: unsupported tool '{tool}'")
        typer.echo(f"  allowed_tools: {', '.join(sorted(ALLOWED_TOOLS))}")
        raise typer.Exit(code=1)
    if value not in ALLOWED_DECISIONS:
        typer.echo("SET PERMISSION: FAIL")
        typer.echo(f"  error: unsupported decision '{value}'")
        typer.echo(f"  allowed_decisions: {', '.join(sorted(ALLOWED_DECISIONS))}")
        raise typer.Exit(code=1)

    set_agent_tool_permission_sync(agent, tool, value)
    typer.echo("SET PERMISSION: OK")
    typer.echo(f"  agent: {agent}")
    typer.echo(f"  tool: {tool}")
    typer.echo(f"  decision: {value}")


@app.command("policy-apply")
def policy_apply(
    policy_name: str = typer.Option(..., "--policy"),
    agent_id: str = typer.Option("default", "--agent-id"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Apply preset profile + permission policy pack."""
    if not isinstance(policy_name, str):
        policy_name = ""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(json_output, bool):
        json_output = False

    policy = policy_name.strip().lower()
    agent = agent_id.strip() or "default"
    pack = POLICY_PACKS.get(policy)
    if not isinstance(pack, dict):
        allowed = ", ".join(sorted(POLICY_PACKS.keys()))
        if json_output:
            typer.echo(
                json.dumps(
                    {"status": "failed", "error": "UNKNOWN_POLICY", "allowed_policies": sorted(POLICY_PACKS.keys())},
                    indent=2,
                )
            )
        else:
            typer.echo("POLICY APPLY: FAIL")
            typer.echo(f"  error: UNKNOWN_POLICY ({policy})")
            typer.echo(f"  allowed: {allowed}")
        raise typer.Exit(code=1)

    profile = load_profile()
    profile_config = pack.get("profile", {})
    if not isinstance(profile_config, dict):
        profile_config = {}
    provider_chain = profile_config.get("provider_chain", ["ollama"])
    primary_provider = str(profile_config.get("primary_provider", "ollama")).strip().lower() or "ollama"
    provider_settings_enabled = profile_config.get("provider_settings_enabled", {})
    if not isinstance(provider_settings_enabled, dict):
        provider_settings_enabled = {}

    provider_settings = profile.get("provider_settings", {})
    if not isinstance(provider_settings, dict):
        provider_settings = {}
    for name, row in provider_settings.items():
        if not isinstance(row, dict):
            continue
        row["enabled"] = bool(provider_settings_enabled.get(name, False))
    profile["provider_chain"] = provider_chain
    profile["primary_provider"] = primary_provider
    profile["provider_settings"] = provider_settings
    saved_profile = save_profile(profile)

    permissions = pack.get("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
    applied_permissions: dict[str, str] = {}
    for tool_name, decision in permissions.items():
        if tool_name not in ALLOWED_TOOLS:
            continue
        value = str(decision).strip().lower()
        if value not in ALLOWED_DECISIONS:
            continue
        set_agent_tool_permission_sync(agent, tool_name, value)
        applied_permissions[tool_name] = value

    payload = {
        "status": "ok",
        "policy": policy,
        "agent_id": agent,
        "provider_chain": saved_profile.get("provider_chain", []),
        "primary_provider": saved_profile.get("primary_provider", "ollama"),
        "permissions": applied_permissions,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo("POLICY APPLY: OK")
    typer.echo(f"  policy: {policy}")
    typer.echo(f"  agent: {agent}")
    typer.echo(f"  primary_provider: {payload['primary_provider']}")
    typer.echo(f"  provider_chain: {', '.join(payload['provider_chain'])}")
    typer.echo(f"  permissions_applied: {len(applied_permissions)}")


@app.command("policy-list")
def policy_list(
    json_output: bool = typer.Option(False, "--json"),
):
    """List available policy packs."""
    if not isinstance(json_output, bool):
        json_output = False
    rows = []
    for name in sorted(POLICY_PACKS.keys()):
        rows.append(
            {
                "policy": name,
                "description": POLICY_PACK_DESCRIPTIONS.get(name, ""),
            }
        )
    if json_output:
        typer.echo(json.dumps({"items": rows}, indent=2))
        return
    typer.echo("POLICY PACKS")
    for row in rows:
        typer.echo(f"  - {row['policy']}: {row['description']}")


@app.command("provider-status")
def provider_status(
    json_output: bool = typer.Option(False, "--json", help="Print provider matrix as JSON"),
):
    """Print provider readiness matrix from runtime status snapshot."""
    status = _collect_runtime_status(sys.executable)
    matrix = status.get("provider_matrix", {})
    if json_output:
        typer.echo(json.dumps({"provider_name": status.get("provider_name"), "provider_matrix": matrix}, indent=2))
        return
    typer.echo("PROVIDER STATUS")
    primary = str(status.get("provider_name", "unknown"))
    typer.echo(f"  primary: {primary}")
    if not isinstance(matrix, dict) or not matrix:
        typer.echo("  providers: none")
        return
    for name in sorted(matrix.keys()):
        row = matrix.get(name, {})
        if not isinstance(row, dict):
            continue
        enabled = bool(row.get("enabled", False))
        configured = bool(row.get("configured", False))
        usable = bool(row.get("usable", False))
        reason = str(row.get("reason", "")).strip()
        suffix = f" ({reason})" if reason else ""
        typer.echo(
            f"  - {name}: enabled={str(enabled).lower()} configured={str(configured).lower()} usable={str(usable).lower()}{suffix}"
        )


@app.command("provider-test")
def provider_test(
    provider_name: str = typer.Option("ollama", "--provider"),
    model_name: str = typer.Option("llama3.2:3b", "--model"),
):
    """Probe provider connectivity for the selected provider/model."""
    ok, message = _probe_provider_connection(provider_name, model_name)
    if ok:
        typer.echo("PROVIDER TEST: OK")
        typer.echo(f"  provider: {provider_name}")
        typer.echo(f"  message: {message}")
        return
    typer.echo("PROVIDER TEST: FAIL")
    typer.echo(f"  provider: {provider_name}")
    typer.echo(f"  message: {message}")
    raise typer.Exit(code=1)


@app.command("cleanup-browsers")
def cleanup_browsers():
    """Force-remove orphan browser containers and reset running session rows."""
    asyncio.run(BrowserManager().cleanup_orphan_containers())
    typer.echo("Browser session cleanup complete.")


def _resolve_session_provider_model(provider_name: str, model_name: str) -> tuple[str, str]:
    """Resolve provider/model defaults from profile when args are blank."""
    provider = str(provider_name).strip().lower()
    model = str(model_name).strip()
    profile = load_profile()
    if not provider:
        candidate = str(profile.get("primary_provider", "")).strip().lower()
        provider = candidate or "ollama"
    if not model:
        settings = profile.get("provider_settings", {})
        if isinstance(settings, dict):
            row = settings.get(provider, {})
            if isinstance(row, dict):
                candidate = str(row.get("model_name", "")).strip()
                if candidate:
                    model = candidate
        if not model:
            candidate = str(profile.get("model_name", "")).strip()
            model = candidate or "llama3.2:3b"
    return provider, model


async def _build_session_status(agent_id: str, provider_name: str, model_name: str) -> dict:
    """Assemble deterministic runtime budget and health snapshot."""
    provider_name, model_name = _resolve_session_provider_model(provider_name, model_name)
    cost_guard = CostGuard()
    budget = await cost_guard.get_budget_status(agent_id)
    session_start = await cost_guard.get_runtime_session_started_at()
    session_usage = await cost_guard.get_usage_window(start_iso=session_start)
    day_start, day_end = cost_guard._utc_day_bounds()
    today_usage = await cost_guard.get_usage_window(start_iso=day_start, end_iso=day_end)

    active_tasks = 0
    queue_depth = 0
    async for db in get_db():
        cursor = await db.execute("SELECT COUNT(*) AS count FROM tasks WHERE status = 'running'")
        row = await cursor.fetchone()
        active_tasks = int(row["count"] if row else 0)

        cursor = await db.execute("SELECT COUNT(*) AS count FROM task_queue")
        row = await cursor.fetchone()
        queue_depth = int(row["count"] if row else 0)
        break

    provider_state = "unknown"
    try:
        response = httpx.get(f"{SUPERVISOR_URL}/metrics/providers", timeout=2.0)
        payload = response.json() if response.status_code == 200 else {}
        if isinstance(payload, dict):
            provider_snapshot = payload.get(provider_name, {})
            if isinstance(provider_snapshot, dict):
                provider_state = str(provider_snapshot.get("state", provider_state))
    except Exception:
        provider_snapshot = get_provider_health_registry().get_snapshot().get(provider_name, {})
        if isinstance(provider_snapshot, dict):
            provider_state = str(provider_snapshot.get("state", provider_state))

    return {
        "provider_name": provider_name,
        "provider_state": provider_state,
        "model_name": model_name,
        "session_tokens": int(session_usage.get("total_tokens", 0)),
        "session_cost_usd": float(session_usage.get("cost_usd", 0.0)),
        "today_cost_usd": float(today_usage.get("cost_usd", 0.0)),
        "daily_limit_usd": float(budget["daily_limit"]),
        "daily_remaining_usd": float(budget["daily_remaining"]),
        "budget_status": str(budget["status"]).upper(),
        "active_tasks": active_tasks,
        "queue_depth": queue_depth,
        "heartbeat_timestamp": "",
        "heartbeat_age_seconds": -1,
    }


async def _build_budget_status_snapshot(agent_id: str) -> dict:
    """Assemble deterministic budget + spend snapshot for CLI rendering."""
    guard = CostGuard()
    combined = await guard.get_budget_status(agent_id)
    global_spend = await guard.get_spend_snapshot()
    per_agent = await guard.get_agent_spend_today()
    return {
        "agent_id": agent_id,
        "status": str(combined.get("status", "ok")).upper(),
        "global_status": str(combined.get("global_status", "ok")).upper(),
        "agent_status": str(combined.get("agent_status", "ok")).upper(),
        "daily_spend_usd": float(combined.get("daily_spend", 0.0)),
        "daily_limit_usd": float(combined.get("daily_limit", 0.0)),
        "daily_remaining_usd": float(combined.get("daily_remaining", 0.0)),
        "agent_daily_spend_usd": float(combined.get("agent_daily_spend", 0.0)),
        "agent_daily_limit_usd": float(combined.get("agent_daily_limit", 0.0)),
        "monthly_spend_usd": float(global_spend.get("monthly_spend", 0.0)),
        "monthly_limit_usd": float(global_spend.get("monthly_limit", 0.0)),
        "agent_spend_today": per_agent,
    }


async def _set_budget_limits(
    *,
    system_daily_limit_usd: float | None,
    agent_daily_limit_usd: float | None,
    monthly_limit_usd: float | None,
) -> dict[str, float]:
    """Persist one or more budget limits and return applied values."""
    updates: dict[str, float] = {}
    if system_daily_limit_usd is not None:
        updates["system_daily_limit_usd"] = round(float(system_daily_limit_usd), 6)
    if agent_daily_limit_usd is not None:
        updates["agent_daily_limit_usd"] = round(float(agent_daily_limit_usd), 6)
    if monthly_limit_usd is not None:
        updates["monthly_budget_usd"] = round(float(monthly_limit_usd), 6)
    if not updates:
        return {}

    async for db in get_db():
        for key, value in updates.items():
            await db.execute(
                """
                INSERT INTO system_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
        await db.commit()
        break
    return updates


@app.command("session-status")
def session_status(
    agent_id: str = typer.Option("default", "--agent-id", help="Agent id for budget status"),
    model_name: str = typer.Option(
        "",
        "--model-name",
        help="Model label override (defaults to profile provider model)",
    ),
    provider_name: str = typer.Option(
        "",
        "--provider",
        help="Provider override (defaults to profile primary provider)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print full status snapshot as JSON"),
):
    """Print deterministic provider, token, and budget status snapshot."""
    if not isinstance(json_output, bool):
        json_output = False
    snapshot = asyncio.run(
        _build_session_status(
            agent_id=agent_id,
            provider_name=provider_name,
            model_name=model_name,
        )
    )
    heartbeat = read_heartbeat_snapshot()
    if isinstance(heartbeat, dict):
        ts = heartbeat.get("timestamp")
        if isinstance(ts, str):
            snapshot["heartbeat_timestamp"] = ts
            try:
                age = int((datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds())
                snapshot["heartbeat_age_seconds"] = max(age, 0)
            except Exception:
                snapshot["heartbeat_age_seconds"] = -1
        snapshot["self_heal_probe_ok"] = bool(heartbeat.get("self_heal_probe_ok", False))
        snapshot["self_heal_healed"] = bool(heartbeat.get("self_heal_healed", False))
    if json_output:
        typer.echo(json.dumps(snapshot, indent=2))
        return
    typer.echo(f"Provider: {snapshot['provider_name']}:{snapshot['model_name']}")
    typer.echo(f"Health: {snapshot['provider_state']}")
    typer.echo(f"Session tokens: {snapshot['session_tokens']:,}")
    typer.echo(f"Session cost: ${snapshot['session_cost_usd']:.2f}")
    typer.echo(f"Today cost: ${snapshot['today_cost_usd']:.2f} / ${snapshot['daily_limit_usd']:.2f}")
    typer.echo(f"Budget remaining: ${snapshot['daily_remaining_usd']:.2f}")
    typer.echo(f"Budget status: {snapshot['budget_status']}")
    typer.echo(f"Active tasks: {snapshot['active_tasks']}")
    typer.echo(f"Queue depth: {snapshot['queue_depth']}")
    if snapshot["heartbeat_age_seconds"] >= 0:
        typer.echo(f"Heartbeat: alive (last seen {snapshot['heartbeat_age_seconds']}s ago)")
    else:
        typer.echo("Heartbeat: unknown")
    if snapshot.get("self_heal_healed"):
        typer.echo("Self-heal: recovered provider in latest heartbeat")
    elif snapshot.get("self_heal_probe_ok"):
        typer.echo("Self-heal: probe OK")
    else:
        typer.echo("Self-heal: probe failed or unavailable")


@app.command("budget-status")
def budget_status(
    agent_id: str = typer.Option("default", "--agent-id", help="Agent id for per-agent budget status"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON payload"),
):
    """Print deterministic spend + budget snapshot including per-agent view."""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(json_output, bool):
        json_output = False
    snapshot = asyncio.run(_build_budget_status_snapshot(agent_id.strip() or "default"))
    if json_output:
        typer.echo(json.dumps(snapshot, indent=2))
        return
    typer.echo("BUDGET STATUS")
    typer.echo(f"  agent_id: {snapshot['agent_id']}")
    typer.echo(f"  status: {snapshot['status']} (global={snapshot['global_status']} agent={snapshot['agent_status']})")
    typer.echo(f"  daily: ${snapshot['daily_spend_usd']:.2f} / ${snapshot['daily_limit_usd']:.2f}")
    typer.echo(f"  daily_remaining: ${snapshot['daily_remaining_usd']:.2f}")
    typer.echo(
        f"  agent_daily: ${snapshot['agent_daily_spend_usd']:.2f} / ${snapshot['agent_daily_limit_usd']:.2f}"
    )
    typer.echo(f"  monthly: ${snapshot['monthly_spend_usd']:.2f} / ${snapshot['monthly_limit_usd']:.2f}")


@app.command("budget-set")
def budget_set(
    system_daily_limit_usd: Optional[float] = typer.Option(None, "--system-daily-limit-usd"),
    agent_daily_limit_usd: Optional[float] = typer.Option(None, "--agent-daily-limit-usd"),
    monthly_limit_usd: Optional[float] = typer.Option(None, "--monthly-limit-usd"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON payload"),
):
    """Update runtime budget limits in system settings."""
    if not isinstance(system_daily_limit_usd, (int, float)):
        system_daily_limit_usd = None
    if not isinstance(agent_daily_limit_usd, (int, float)):
        agent_daily_limit_usd = None
    if not isinstance(monthly_limit_usd, (int, float)):
        monthly_limit_usd = None
    if not isinstance(json_output, bool):
        json_output = False
    provided = [value for value in (system_daily_limit_usd, agent_daily_limit_usd, monthly_limit_usd) if value is not None]
    if not provided:
        typer.echo("BUDGET SET: FAIL")
        typer.echo("  error: provide at least one limit flag")
        raise typer.Exit(code=1)
    for value in provided:
        if float(value) <= 0:
            typer.echo("BUDGET SET: FAIL")
            typer.echo("  error: limits must be > 0")
            raise typer.Exit(code=1)
    applied = asyncio.run(
        _set_budget_limits(
            system_daily_limit_usd=system_daily_limit_usd,
            agent_daily_limit_usd=agent_daily_limit_usd,
            monthly_limit_usd=monthly_limit_usd,
        )
    )
    payload = {"status": "ok", "applied": applied}
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo("BUDGET SET: OK")
    for key in sorted(applied.keys()):
        typer.echo(f"  {key}: {applied[key]:.2f}")


def _fetch_guide_json(host: str, port: int, path: str) -> dict:
    """Fetch JSON payload from the guide server and validate response status."""
    url = f"http://{host}:{port}{path}"
    response = httpx.get(url, timeout=2.0)
    if response.status_code != 200:
        raise RuntimeError(f"guide request failed: HTTP {response.status_code} for {path}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"guide request failed: non-object payload for {path}")
    return payload


@app.command("trace-list")
def trace_list(
    host: str = typer.Option("127.0.0.1", "--host", help="Guide server host"),
    port: int = typer.Option(7788, "--port", help="Guide server port"),
    trace_type: str = typer.Option("all", "--type", help="Trace type filter (all|plan_preview|assistant_chat|action_run)"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON payload"),
):
    """List recent planner/assistant/action traces from guide server."""
    if not isinstance(trace_type, str):
        trace_type = "all"
    if not isinstance(json_output, bool):
        json_output = False
    try:
        payload = _fetch_guide_json(host, port, "/api/traces")
    except Exception as exc:
        typer.echo("TRACE LIST: FAIL")
        typer.echo(f"  error: {exc}")
        raise typer.Exit(code=1)
    items = payload.get("items", [])
    if not isinstance(items, list):
        items = []
    selected_type = str(trace_type).strip() or "all"
    if selected_type != "all":
        items = [row for row in items if isinstance(row, dict) and str(row.get("type", "")) == selected_type]
    if json_output:
        typer.echo(json.dumps({"items": items}, indent=2))
        return
    typer.echo("TRACE LIST: OK")
    typer.echo(f"  count: {len(items)}")
    for row in items:
        if not isinstance(row, dict):
            continue
        trace_id = str(row.get("trace_id", "unknown"))
        kind = str(row.get("type", "unknown"))
        stage_count = int(row.get("stage_count", 0))
        last_event = str(row.get("last_event", "n/a"))
        typer.echo(f"  {trace_id} | {kind} | stages={stage_count} | last={last_event}")


@app.command("trace-show")
def trace_show(
    trace_id: str,
    host: str = typer.Option("127.0.0.1", "--host", help="Guide server host"),
    port: int = typer.Option(7788, "--port", help="Guide server port"),
    json_output: bool = typer.Option(False, "--json", help="Print full trace JSON payload"),
):
    """Show full trace details by trace id from guide server."""
    if not isinstance(trace_id, str):
        trace_id = ""
    if not isinstance(json_output, bool):
        json_output = False
    trace_key = str(trace_id).strip()
    if not trace_key:
        typer.echo("TRACE SHOW: FAIL")
        typer.echo("  error: trace_id is required")
        raise typer.Exit(code=1)
    try:
        payload = _fetch_guide_json(host, port, f"/api/traces/{quote(trace_key, safe='')}")
    except Exception as exc:
        typer.echo("TRACE SHOW: FAIL")
        typer.echo(f"  error: {exc}")
        raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    stages = payload.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    typer.echo("TRACE SHOW: OK")
    typer.echo(f"  trace_id: {payload.get('trace_id', trace_key)}")
    typer.echo(f"  type: {payload.get('type', 'unknown')}")
    typer.echo(f"  created_at: {payload.get('created_at', 'unknown')}")
    typer.echo(f"  stages: {len(stages)}")
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            continue
        event = str(stage.get("event", "unknown"))
        ts = str(stage.get("timestamp", ""))
        typer.echo(f"    {index}. {event} {ts}".rstrip())


@app.command("support-bundle")
def support_bundle(
    agent_id: str = typer.Option("default", "--agent-id", help="Agent id for permission + budget snapshot"),
    host: str = typer.Option("127.0.0.1", "--host", help="Guide server host"),
    port: int = typer.Option(7788, "--port", help="Guide server port"),
    output: Path | None = typer.Option(None, "--output", help="Optional output path for JSON bundle"),
    json_output: bool = typer.Option(False, "--json", help="Print bundle JSON to stdout"),
):
    """Export support bundle from guide runtime for diagnostics."""
    if not isinstance(agent_id, str):
        agent_id = "default"
    if not isinstance(json_output, bool):
        json_output = False
    agent = agent_id.strip() or "default"
    try:
        payload = _fetch_guide_json(host, port, f"/api/support-bundle?agent_id={quote(agent, safe='')}")
    except Exception as exc:
        typer.echo("SUPPORT BUNDLE: FAIL")
        typer.echo(f"  error: {exc}")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return

    target = output
    if target is None:
        runtime_dir = Path("runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        target = runtime_dir / f"support_bundle_{timestamp}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trace_summaries = payload.get("trace_summaries", [])
    if not isinstance(trace_summaries, list):
        trace_summaries = []
    typer.echo("SUPPORT BUNDLE: OK")
    typer.echo(f"  file: {target}")
    typer.echo(f"  trace_summaries: {len(trace_summaries)}")


class _ReplayRouter:
    """Replay router with optional explicit selector fallback attempts."""

    def __init__(self, base_router: CommandRouter, allow_fallback: bool):
        self._base_router = base_router
        self._allow_fallback = allow_fallback

    async def execute(self, command: dict) -> dict:
        if not self._allow_fallback:
            return await self._base_router.execute(command)

        action = command.get("action")
        params = command.get("params", {})
        fallback_selectors = params.get("fallback_selectors", [])
        if action not in {"click", "type", "get_text"} or not isinstance(fallback_selectors, list):
            return await self._base_router.execute(command)

        try:
            return await self._base_router.execute(command)
        except Exception:
            primary_exc = None
            for fallback_selector in fallback_selectors:
                if not isinstance(fallback_selector, str) or not fallback_selector.strip():
                    continue
                fallback_command = deepcopy(command)
                fallback_command.setdefault("params", {})
                fallback_command["params"]["selector"] = fallback_selector.strip()
                try:
                    return await self._base_router.execute(fallback_command)
                except Exception as exc:
                    primary_exc = exc
                    continue
            if primary_exc is not None:
                raise primary_exc
            raise


async def _run_replay_with_options(workflow_path: Path, from_step: int, allow_fallback: bool) -> dict:
    workflow = _load_and_validate_workflow(workflow_path)
    if "task_id" not in workflow or "commands" not in workflow:
        raise ValueError("Workflow must include 'task_id' and 'commands'")
    if not isinstance(workflow["commands"], list):
        raise ValueError("'commands' must be list")
    if from_step < 1:
        raise ValueError("--from-step must be >= 1")

    replay_task_id = f"{workflow['task_id']}_replay_{uuid4().hex[:8]}"
    replay_commands = workflow["commands"][from_step - 1 :]
    replay_workflow = {"task_id": replay_task_id, "commands": replay_commands}

    agent_id = f"replay_{workflow['task_id']}"
    capabilities = await CapabilityManager.get_capabilities(agent_id) or []
    has_browser_cap = any(row["capability_type"] == "BROWSER" for row in capabilities)
    if not has_browser_cap:
        await CapabilityManager.add_capability(agent_id, "BROWSER", "{}")

    manager = BrowserManager()
    executor: BrowserExecutor | None = None
    try:
        browser_session = await manager.request_session(agent_id)
        executor = BrowserExecutor(browser_session["cdp_port"])
        await executor.connect()
        actions = BrowserActions(executor)
        base_router = CommandRouter(actions)
        router = _ReplayRouter(base_router, allow_fallback=allow_fallback)
        runner = TaskRunner(router, agent_id=agent_id, pre_persisted=False, worker_id="direct")
        return await runner.run(replay_workflow)
    finally:
        if executor is not None:
            await executor.close()
        try:
            await manager.stop_session(agent_id)
        except Exception:
            pass


async def _inspect_task(task_id: str) -> dict:
    task_row = None
    step_rows = []
    event_rows = []
    async for db in get_db():
        cursor = await db.execute(
            """
            SELECT task_id, agent_id, status, created_at, updated_at, payload, result
            FROM tasks
            WHERE task_id = ?
            """,
            (task_id,),
        )
        task_row = await cursor.fetchone()
        if task_row is None:
            return {"task_id": task_id, "status": "not_found"}

        cursor = await db.execute(
            """
            SELECT command_id, status, duration_ms, error, started_at, finished_at, worker_id
            FROM task_execution_logs
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        )
        step_rows = [dict(row) for row in await cursor.fetchall()]

        cursor = await db.execute(
            """
            SELECT event_type, payload, created_at
            FROM task_events
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        )
        event_rows = [dict(row) for row in await cursor.fetchall()]

    return {
        "task_id": task_row["task_id"],
        "agent_id": task_row["agent_id"],
        "status": task_row["status"],
        "created_at": task_row["created_at"],
        "updated_at": task_row["updated_at"],
        "payload": json.loads(task_row["payload"]) if task_row["payload"] else None,
        "result": json.loads(task_row["result"]) if task_row["result"] else None,
        "step_logs": step_rows,
        "events": [
            {
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]) if row["payload"] else {},
                "created_at": row["created_at"],
            }
            for row in event_rows
        ],
    }


@app.command()
def replay(
    workflow_path: Path,
    from_step: int = typer.Option(1, "--from-step", help="1-based command index to start replay from"),
    allow_fallback: bool = typer.Option(False, "--allow-fallback", help="Allow explicit selector fallback candidates during replay"),
):
    """Replay a workflow JSON through the deterministic runtime."""
    result = asyncio.run(_run_replay_with_options(workflow_path, from_step, allow_fallback))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def inspect(task_id: str):
    """Inspect persisted task result, step logs, and task events."""
    details = asyncio.run(_inspect_task(task_id))
    typer.echo(json.dumps(details, indent=2))


@app.command("analyze-workflow")
def analyze_workflow(workflow_path: Path):
    """Score selector robustness from a recorded workflow (no browser execution)."""
    _load_and_validate_workflow(workflow_path)
    report = analyze_workflow_file(workflow_path)
    typer.echo(json.dumps(report, indent=2))


@app.command("lint-workflow")
def lint_workflow(
    workflow_path: Path,
    min_average_score: float = typer.Option(70.0, "--min-average-score"),
    max_fragile: int = typer.Option(5, "--max-fragile"),
    max_high_risk: int = typer.Option(0, "--max-high-risk"),
):
    """Lint a recorded workflow for CI gating using selector robustness thresholds."""
    _load_and_validate_workflow(workflow_path)
    report = analyze_workflow_file(workflow_path)
    violations = _compute_lint_violations(report, min_average_score, max_fragile, max_high_risk)

    output = {"status": "ok" if not violations else "failed", "summary": report.get("summary", {})}
    if violations:
        output["violations"] = violations
    typer.echo(json.dumps(output, indent=2))
    if violations:
        raise typer.Exit(code=1)


@app.command("release-check")
def release_check(
    workflow_paths: list[Path] = typer.Argument(..., help="Workflow JSON files to lint"),
    min_average_score: float = typer.Option(70.0, "--min-average-score"),
    max_fragile: int = typer.Option(5, "--max-fragile"),
    max_high_risk: int = typer.Option(0, "--max-high-risk"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON only"),
):
    """Run full release gate: test suite + workflow lint checks."""
    if not isinstance(json_output, bool):
        json_output = False
    verify_result = _run_verify_suite()
    verify_summary = _summarize_verify_output(
        verify_result["stdout"], verify_result["stderr"], verify_result["returncode"]
    )
    golden_result = _run_golden_suite()
    golden_summary = _summarize_verify_output(
        golden_result["stdout"], golden_result["stderr"], golden_result["returncode"]
    )

    workflow_outputs = []
    has_lint_failure = False
    for workflow_path in workflow_paths:
        _load_and_validate_workflow(workflow_path)
        report = analyze_workflow_file(workflow_path)
        violations = _compute_lint_violations(
            report, min_average_score, max_fragile, max_high_risk
        )
        status = "ok" if not violations else "failed"
        has_lint_failure = has_lint_failure or bool(violations)
        item = {
            "workflow_path": str(workflow_path),
            "status": status,
            "summary": report.get("summary", {}),
        }
        if violations:
            item["violations"] = violations
            item["failure"] = build_failure(
                error_class="interaction_failed",
                error_code="WORKFLOW_LINT_FAILED",
                step_id="lint",
                selector="",
                url="",
                message="; ".join(violations),
            )
        workflow_outputs.append(item)

    verify_failed = verify_result["returncode"] != 0
    golden_failed = golden_result["returncode"] != 0
    release_output = {
        "verify_status": "ok" if not verify_failed else "failed",
        "verify": verify_summary,
        "golden_status": "ok" if not golden_failed else "failed",
        "golden": golden_summary,
        "workflows": workflow_outputs,
    }
    if has_lint_failure:
        failed_items = [item for item in workflow_outputs if item.get("status") == "failed"]
        release_output["failure"] = failed_items[0].get("failure") if failed_items else None
    if verify_failed and not release_output.get("failure"):
        release_output["failure"] = build_failure(
            error_class="interaction_failed",
            error_code="VERIFY_SUITE_FAILED",
            step_id="verify",
            selector="",
            url="",
            message=verify_summary.get("status", "verify suite failed"),
        )
    if golden_failed and not release_output.get("failure"):
        release_output["failure"] = build_failure(
            error_class="interaction_failed",
            error_code="GOLDEN_SUITE_FAILED",
            step_id="golden",
            selector="",
            url="",
            message=golden_summary.get("status", "golden suite failed"),
        )

    if json_output:
        typer.echo(json.dumps(release_output, indent=2))
    else:
        _print_release_check_human(release_output)

    if has_lint_failure or verify_failed or golden_failed:
        raise typer.Exit(code=1)


@app.command()
def verify():
    """Run deterministic reliability checks used as release gate."""
    result = _run_verify_suite()
    if result["stdout"]:
        typer.echo(result["stdout"].rstrip())
    if result["stderr"]:
        typer.echo(result["stderr"].rstrip(), err=True)
    if result["returncode"] != 0:
        raise typer.Exit(code=result["returncode"])


@app.command("golden-check")
def golden_check():
    """Run golden planner regression suite."""
    cmd = [sys.executable, "-m", "unittest", "-v", "tests.test_golden_planner_regression"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        typer.echo(result.stdout.rstrip())
    if result.stderr:
        typer.echo(result.stderr.rstrip(), err=True)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _load_and_validate_workflow(workflow_path: Path) -> dict:
    """Load a workflow JSON file and enforce supported schema contract."""
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    schema_version = workflow.get("schema_version", TASK_COMMAND_SCHEMA_V1)
    if schema_version not in SUPPORTED_TASK_COMMAND_SCHEMAS:
        raise ValueError(
            f"Unsupported workflow schema_version '{schema_version}'. "
            f"Supported: {sorted(SUPPORTED_TASK_COMMAND_SCHEMAS)}"
        )
    return workflow


def _compute_lint_violations(
    report: dict,
    min_average_score: float,
    max_fragile: int,
    max_high_risk: int,
) -> list[str]:
    """Compute deterministic lint violations from analyzer report summary."""
    summary = report.get("summary", {})
    average_score = float(summary.get("average_score", 0))
    fragile = int(summary.get("fragile", 0))
    high_risk = int(summary.get("high_risk", 0))

    violations: list[str] = []
    if average_score < min_average_score:
        violations.append(
            f"average_score {average_score} below minimum {min_average_score}"
        )
    if fragile > max_fragile:
        violations.append(f"fragile {fragile} exceeds maximum {max_fragile}")
    if high_risk > max_high_risk:
        violations.append(f"high_risk {high_risk} exceeds maximum {max_high_risk}")
    return violations


def _run_verify_suite() -> dict:
    """Run unittest discovery and return structured subprocess result."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_golden_suite() -> dict:
    """Run golden regression suite and return structured subprocess result."""
    cmd = [sys.executable, "-m", "unittest", "-v", "tests.test_golden_planner_regression"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _summarize_verify_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Extract concise verify summary from unittest output."""
    combined = f"{stdout or ''}\n{stderr or ''}"
    ran_match = re.search(r"Ran (\d+) tests", combined)
    ran = int(ran_match.group(1)) if ran_match else 0
    passed = ran if returncode == 0 else 0
    status = "ok" if returncode == 0 else "failed"
    return {"status": status, "passed": passed, "total": ran}


def _print_release_check_human(release_output: dict) -> None:
    """Print compact release-check summary for humans."""
    verify = release_output.get("verify", {})
    golden = release_output.get("golden", {})
    workflows = release_output.get("workflows", [])
    failed_workflows = [item for item in workflows if item.get("status") == "failed"]
    verify_failed = release_output.get("verify_status") != "ok"
    golden_failed = release_output.get("golden_status") != "ok"
    overall_failed = verify_failed or golden_failed or bool(failed_workflows)

    if not overall_failed:
        typer.echo("RELEASE CHECK: PASS")
        typer.echo(f"  tests: {verify.get('passed', 0)}/{verify.get('total', 0)}")
        typer.echo(f"  golden: {golden.get('passed', 0)}/{golden.get('total', 0)}")
        typer.echo(f"  workflows: {len(workflows)}/{len(workflows)}")
        return

    typer.echo("RELEASE CHECK: FAIL")
    if verify_failed:
        typer.echo(
            f"  tests: {verify.get('passed', 0)}/{verify.get('total', 0)} "
            "(verify suite failed)"
        )
    if golden_failed:
        typer.echo(
            f"  golden: {golden.get('passed', 0)}/{golden.get('total', 0)} "
            "(golden suite failed)"
        )
    for item in failed_workflows:
        workflow_name = Path(item.get("workflow_path", "")).name or item.get("workflow_path", "")
        failure = item.get("failure") or {}
        fingerprint = str(failure.get("fingerprint", ""))
        short_fingerprint = fingerprint[:8] if fingerprint else ""
        typer.echo("")
        typer.echo(f"  Workflow: {workflow_name}")
        typer.echo(
            "    - "
            f"{failure.get('error_class', 'interaction_failed')} "
            f"({failure.get('error_code', 'UNKNOWN')})"
        )
        if failure.get("step_id"):
            typer.echo(f"      step={failure['step_id']}")
        if short_fingerprint:
            typer.echo(f"      fingerprint={short_fingerprint}")


def psutil_pid_exists(pid):
    """Check whether pid exists in the current process table."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        process = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if process != 0:
            kernel32.CloseHandle(process)
            return True
        else:
            return False
    else:
        if pid < 0: return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError: 
            return False
        except PermissionError: 
            return True
        except OSError: 
            return False
        return True

@worker_app.command("start")
def worker_start():
    """Start queue worker process."""
    asyncio.run(Worker(str(uuid4())).run_forever())


if __name__ == "__main__":
    app()
