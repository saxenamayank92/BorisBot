from fastapi import FastAPI, HTTPException
import logging
from .database import init_db
from .models import AgentCreate, AgentResponse
from .agent_manager import AgentManager

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

app = FastAPI(title="BorisBot Supervisor")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    await init_db()
    
    logger.info("Cleaning up orphans...")
    await AgentManager.cleanup_orphans()
    
    # Start the agent monitoring loop
    import asyncio
    asyncio.create_task(AgentManager.monitor_agents())
    
    logger.info("Database initialized and monitoring started.")

@app.on_event("shutdown")
async def shutdown_event():
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
