import asyncio
import json
import typer
import uvicorn
import subprocess
import os
import sys
import time
import httpx
import socket
from copy import deepcopy
from pathlib import Path
from typing import Optional
from uuid import uuid4

from borisbot.contracts import SUPPORTED_TASK_COMMAND_SCHEMAS, TASK_COMMAND_SCHEMA_V1
from borisbot.failures import build_failure
from borisbot.recorder.analyzer import analyze_workflow_file
from borisbot.recorder.runner import run_record
from borisbot.browser.actions import BrowserActions
from borisbot.browser.command_router import CommandRouter
from borisbot.browser.executor import BrowserExecutor
from borisbot.browser.task_runner import TaskRunner
from borisbot.supervisor.browser_manager import BrowserManager
from borisbot.supervisor.capability_manager import CapabilityManager
from borisbot.supervisor.database import get_db
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
    asyncio.run(run_record(task_id, start_url=start_url))


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
):
    """Run full release gate: test suite + workflow lint checks."""
    verify_result = _run_verify_suite()
    if verify_result["stdout"]:
        typer.echo(verify_result["stdout"].rstrip())
    if verify_result["stderr"]:
        typer.echo(verify_result["stderr"].rstrip(), err=True)
    if verify_result["returncode"] != 0:
        raise typer.Exit(code=verify_result["returncode"])

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

    release_output = {"verify_status": "ok", "workflows": workflow_outputs}
    if has_lint_failure:
        failed_items = [item for item in workflow_outputs if item.get("status") == "failed"]
        release_output["failure"] = failed_items[0].get("failure") if failed_items else None
    typer.echo(json.dumps(release_output, indent=2))
    if has_lint_failure:
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
