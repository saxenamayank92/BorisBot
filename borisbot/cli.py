import asyncio
import json
import typer
import uvicorn
import subprocess
import os
import sys
import time
import re
import httpx
import socket
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Optional
from uuid import uuid4

from borisbot.contracts import SUPPORTED_TASK_COMMAND_SCHEMAS, TASK_COMMAND_SCHEMA_V1
from borisbot.failures import build_failure
from borisbot.recorder.analyzer import analyze_workflow_file
from borisbot.recorder.runner import run_record
from borisbot.guide.server import run_guide_server
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


@app.command()
def guide(
    host: str = typer.Option("127.0.0.1", "--host", help="Guide server host"),
    port: int = typer.Option(7788, "--port", help="Guide server port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
):
    """Launch guided local web UI for record/replay/release-check actions."""
    run_guide_server(Path.cwd(), host=host, port=port, open_browser=open_browser)


@app.command("cleanup-browsers")
def cleanup_browsers():
    """Force-remove orphan browser containers and reset running session rows."""
    asyncio.run(BrowserManager().cleanup_orphan_containers())
    typer.echo("Browser session cleanup complete.")


async def _build_session_status(agent_id: str, model_name: str) -> dict:
    """Assemble deterministic runtime budget and health snapshot."""
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

    provider_name = "ollama"
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


@app.command("session-status")
def session_status(
    agent_id: str = typer.Option("default", "--agent-id", help="Agent id for budget status"),
    model_name: str = typer.Option(
        os.getenv("BORISBOT_OLLAMA_MODEL", "llama3.2:3b"),
        "--model-name",
        help="Primary model label to display",
    ),
):
    """Print deterministic provider, token, and budget status snapshot."""
    snapshot = asyncio.run(_build_session_status(agent_id=agent_id, model_name=model_name))
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
