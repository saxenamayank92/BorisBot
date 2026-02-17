from fastapi import FastAPI, HTTPException
import asyncio
import logging
from .database import init_db, reconcile_running_tasks_after_crash
from .models import AgentCreate, AgentResponse
from .agent_manager import AgentManager
from .browser_manager import BrowserManager

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

app = FastAPI(title="BorisBot Supervisor")


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
    global _ttl_task
    logger.info("Initializing database...")
    await init_db()

    logger.info("Reconciling stale running tasks...")
    await reconcile_running_tasks_after_crash()
    
    logger.info("Cleaning up orphans...")
    await AgentManager.cleanup_orphans()
    browser_manager = BrowserManager()
    await browser_manager.cleanup_orphan_containers()
    await browser_manager.expire_stale_sessions()

    _ttl_task = asyncio.create_task(_ttl_enforcement_loop())
    
    # Start the agent monitoring loop
    asyncio.create_task(AgentManager.monitor_agents())
    
    logger.info("Database initialized and monitoring started.")

@app.on_event("shutdown")
async def shutdown_event():
    global _ttl_task
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
