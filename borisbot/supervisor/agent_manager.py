import asyncio
import subprocess
import sys
import uuid
import os
import signal
from pathlib import Path
from datetime import datetime, timedelta
from .database import get_db
from .models import AgentCreate, AgentResponse, AgentStatus
from .state import ProcessState
import aiosqlite

# We need to find the python executable to spawn the worker
PYTHON_EXEC = sys.executable

class AgentManager:
    @staticmethod
    async def spawn_agent(agent_data: AgentCreate, supervisor_url: str):
        agent_id = str(uuid.uuid4())
        
        # 1. Start the subprocess
        # We use Popen to start it in the background.
        # It's important that the agent knows its ID and where to report back.
        cmd = [
            PYTHON_EXEC, "-m", "borisbot.agent.worker",
            "--agent-id", agent_id,
            "--supervisor-url", supervisor_url
        ]
        
        # Redirect stdout/stderr to log file for debugging
        log_dir = Path.home() / ".borisbot" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f"agent_{agent_id}.log", "a")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                text=True
            )
            
            # 2. Register in-memory process handle
            ProcessState.register(agent_id, process)
            
            # 3. Persist to DB
            async for db in get_db():
                await db.execute(
                    """
                    INSERT INTO agents (id, name, status, parent_id, autonomy_mode, pid, created_at, last_heartbeat)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (agent_id, agent_data.name, AgentStatus.STARTING, 
                     agent_data.parent_id, agent_data.autonomy_mode, process.pid,
                     datetime.utcnow(), datetime.utcnow())
                )
                await db.commit()
                
            return AgentResponse(
                id=agent_id,
                name=agent_data.name,
                status=AgentStatus.STARTING,
                parent_id=agent_data.parent_id,
                autonomy_mode=agent_data.autonomy_mode,
                created_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                pid=process.pid
            )
            
        except Exception as e:
            # TODO: Add logging here
            print(f"Failed to spawn agent: {e}")
            raise e

    @staticmethod
    async def stop_agent(agent_id: str):
        process = ProcessState.get(agent_id)
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            ProcessState.remove(agent_id)
        
        # Update DB
        async for db in get_db():
            await db.execute(
                "UPDATE agents SET status = ? WHERE id = ?",
                (AgentStatus.STOPPED, agent_id)
            )
            await db.commit()
            
    @staticmethod
    async def list_agents():
        async for db in get_db():
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM agents") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    @staticmethod
    async def handle_heartbeat(agent_id: str):
        """Update the last_heartbeat timestamp for an agent."""
        # Also, if the agent was marked as STARTING, we can move it to RUNNING
        async for db in get_db():
            await db.execute(
                """
                UPDATE agents 
                SET last_heartbeat = ?, status = CASE WHEN status = ? THEN ? ELSE status END
                WHERE id = ?
                """,
                (datetime.utcnow(), AgentStatus.STARTING, AgentStatus.RUNNING, agent_id)
            )
            await db.commit()

    @staticmethod
    async def cleanup_orphans():
        """Clean up agents that were running/starting when the supervisor last died."""
        async for db in get_db():
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id, pid, status FROM agents WHERE status IN ('running', 'starting')") as cursor:
                orphans = await cursor.fetchall()
                
            for orphan in orphans:
                agent_id = orphan['id']
                pid = orphan['pid']
                print(f"Cleaning up orphan agent {agent_id} (PID: {pid})")
                
                # Attempt to kill the process
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass # Process likely already gone
                        
                # Mark as crashed
                await db.execute(
                    "UPDATE agents SET status = ? WHERE id = ?",
                    (AgentStatus.CRASHED, agent_id)
                )
            await db.commit()

    @staticmethod
    async def shutdown():
        """Gracefully stop all managed agents on supervisor shutdown."""
        active_agent_ids = list(ProcessState._processes.keys())
        print(f"Shutting down {len(active_agent_ids)} active agents...")
        
        for agent_id in active_agent_ids:
            process = ProcessState.get(agent_id)
            if process:
                process.terminate()
        
        # Wait a bit
        # This is a bit blocking but acceptable for shutdown
        await asyncio.sleep(2) 
        
        for agent_id in active_agent_ids:
            process = ProcessState.get(agent_id)
            if process and process.poll() is None:
                process.kill()
            
            # Update DB to STOPPED
            ProcessState.remove(agent_id)
            
        async for db in get_db():
            for agent_id in active_agent_ids:
                 await db.execute(
                    "UPDATE agents SET status = ? WHERE id = ?",
                    (AgentStatus.STOPPED, agent_id)
                )
            await db.commit()

    @staticmethod
    async def monitor_agents():
        """Background task to check for crashed or unresponsive agents."""
        while True:
            try:
                # 1. Check for process exits
                active_agent_ids = list(ProcessState._processes.keys())
                
                for agent_id in active_agent_ids:
                    process = ProcessState.get(agent_id)
                    if process and process.poll() is not None:
                        exit_code = process.returncode
                        print(f"Agent {agent_id} exited with code {exit_code}")
                        new_status = AgentStatus.STOPPED if exit_code == 0 else AgentStatus.CRASHED
                        
                        async for db in get_db():
                            await db.execute(
                                "UPDATE agents SET status = ? WHERE id = ?",
                                (new_status, agent_id)
                            )
                            await db.commit()
                        
                        ProcessState.remove(agent_id)

                # 2. Check for unresponsive agents (Heartbeat Watchdog)
                cutoff = datetime.utcnow() - timedelta(seconds=15)
                async for db in get_db():
                    # We need to manually filter or ensure datetime storage is compatible.
                    # aiosqlite/sqlite stores datetimes as strings usually.
                    # We should rely on sqlite's datetime function if possible or ensure python comparison works.
                    # Let's try python side filtering for safety if DB size is small, or trusted SQL.
                    # Default storage is string ISO format. Python's datetime.utcnow() -> string match?
                    # Safer to use sqlite datetime function: datetime(last_heartbeat) < datetime(?)
                    
                    await db.execute(
                        """
                        UPDATE agents 
                        SET status = ? 
                        WHERE status = 'running' 
                        AND last_heartbeat < ?
                        """,
                        (AgentStatus.CRASHED, cutoff)
                    )
                    await db.commit()
                        
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                
            await asyncio.sleep(2)
