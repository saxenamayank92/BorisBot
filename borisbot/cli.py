import asyncio
import typer
import uvicorn
import subprocess
import os
import sys
import time
import httpx
import socket
from pathlib import Path
from typing import Optional
from uuid import uuid4

from borisbot.recorder.runner import run_record
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
