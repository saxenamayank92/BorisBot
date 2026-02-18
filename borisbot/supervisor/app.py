from fastapi import FastAPI, HTTPException
import asyncio
import logging
import os
from datetime import datetime
from .database import get_db, init_db
from .models import AgentCreate, AgentResponse
from .agent_manager import AgentManager
from .browser_manager import BrowserManager
from .heartbeat_runtime import HeartbeatSupervisor
from borisbot.llm.ollama_health import startup_mark_ollama_health
from borisbot.llm.provider_health import get_provider_health_registry
from .api_admin import router as admin_router
from .api_metrics import router as metrics_router
from .api_stream import router as stream_router
from .api_tasks import router as task_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # TODO: Add file handler to ~/.borisbot/logs/supervisor.log
    ]
)
logger = logging.getLogger("borisbot.supervisor")
ttl_logger = logging.getLogger("borisbot.supervisor.ttl_worker")

TTL_INTERVAL_SECONDS = 30
_ttl_task: asyncio.Task | None = None
_heartbeat_task: asyncio.Task | None = None

app = FastAPI(title="BorisBot Supervisor")
app.include_router(task_router)
app.include_router(admin_router)
app.include_router(metrics_router)
app.include_router(stream_router)


async def _ttl_enforcement_loop():
    manager = BrowserManager()
    while True:
        try:
            await manager.expire_stale_sessions()
        except Exception as e:
            ttl_logger.error("TTL enforcement error: %s", e)
        await asyncio.sleep(TTL_INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    global _ttl_task, _heartbeat_task
    logger.info("Initializing database...")
    await init_db()

    logger.info("Cleaning up orphans...")
    await AgentManager.cleanup_orphans()
    browser_manager = BrowserManager()
    await browser_manager.cleanup_orphan_containers()
    async for db in get_db():
        await db.execute("UPDATE tasks SET status = 'pending' WHERE status = 'running'")
        await db.commit()
    async for db in get_db():
        await db.execute(
            """
            UPDATE task_queue
            SET locked_at = NULL, locked_by = NULL, lock_expires_at = NULL
            WHERE lock_expires_at IS NOT NULL
            """
        )
        await db.commit()
    await browser_manager.expire_stale_sessions()
    session_started_at = datetime.utcnow().isoformat()
    async for db in get_db():
        await db.execute(
            """
            INSERT INTO system_settings (key, value)
            VALUES ('runtime_session_started_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (session_started_at,),
        )
        await db.commit()
        break
    try:
        ollama_model = os.getenv("BORISBOT_OLLAMA_MODEL", "llama3.2:3b")
        healthy = await startup_mark_ollama_health(
            get_provider_health_registry(),
            model_name=ollama_model,
            provider_name="ollama",
        )
        logger.info("Ollama startup probe status=%s model=%s", "healthy" if healthy else "unhealthy", ollama_model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Ollama startup probe failed: %s", exc)

    _ttl_task = asyncio.create_task(_ttl_enforcement_loop())
    heartbeat_supervisor = HeartbeatSupervisor()
    await heartbeat_supervisor.initialize_on_startup()
    _heartbeat_task = asyncio.create_task(heartbeat_supervisor.run_forever())
    
    # Start the agent monitoring loop
    asyncio.create_task(AgentManager.monitor_agents())
    
    logger.info("Database initialized and monitoring started.")

@app.on_event("shutdown")
async def shutdown_event():
    global _ttl_task, _heartbeat_task
    if _heartbeat_task:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
        _heartbeat_task = None
    if _ttl_task:
        _ttl_task.cancel()
        try:
            await _ttl_task
        except asyncio.CancelledError:
            pass
        _ttl_task = None
    logger.info("Shutting down agents...")
    await AgentManager.shutdown()
    logger.info("Agents stopped.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.post("/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str):
    try:
        await AgentManager.handle_heartbeat(agent_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to handle heartbeat for {agent_id}: {e}")
        # Return 200 even on error to prevent agent crash loops if DB is locked etc, 
        # but 500 is technically correct. Let's stick to 200 ok for now or 500?
        # Standard practice: 500
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents", response_model=AgentResponse)
async def spawn_agent(agent: AgentCreate):
    try:
        # For now, hardcode supervisor URL or get from config
        # Localhost 7777 is the requirement
        supervisor_url = "http://localhost:7777"
        return await AgentManager.spawn_agent(agent, supervisor_url)
    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    return await AgentManager.list_agents()

@app.post("/agents/{agent_id}/kill")
async def kill_agent(agent_id: str):
    try:
        await AgentManager.stop_agent(agent_id)
        return {"status": "stopped", "id": agent_id}
    except Exception as e:
        logger.error(f"Failed to stop agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown requested via API.")
    await AgentManager.shutdown()
    # Schedule process exit to allow response to be sent
    import asyncio
    import os
    loop = asyncio.get_running_loop()
    loop.call_later(1, lambda: os._exit(0))
    return {"status": "shutting_down"}
