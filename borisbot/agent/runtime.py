import asyncio
import httpx
import logging
import signal
import sys
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # TODO: Add file handler to ~/.borisbot/logs/agent_<id>.log
    ]
)
logger = logging.getLogger("borisbot.agent")

class Runtime:
    def __init__(self, agent_id: str, supervisor_url: str):
        self.agent_id = agent_id
        self.supervisor_url = supervisor_url
        self.running = False
        self._shutdown_event = asyncio.Event()
        self.last_successful_heartbeat = time.time()

    async def start(self):
        """Main agent loop."""
        self.running = True
        logger.info(f"Agent {self.agent_id} starting...")

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        except NotImplementedError:
            # Windows ProactorEventLoop does not support add_signal_handler
            logger.warning("Signal handlers not supported on this platform (likely Windows).")

        try:
            while self.running:
                await self.send_heartbeat()
                
                # Check for lease timeout
                if time.time() - self.last_successful_heartbeat > 30:
                    logger.critical("Supervisor unreachable for >30s. Shutting down.")
                    await self.stop()
                    break

                # Wait for next tick or shutdown
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass  # Just a tick
        except asyncio.CancelledError:
            logger.info("Agent run loop cancelled.")
        finally:
            logger.info(f"Agent {self.agent_id} stopped.")

    async def stop(self):
        """Stop the agent loop."""
        logger.info("Stopping agent...")
        self.running = False
        self._shutdown_event.set()

    async def send_heartbeat(self):
        """Send heartbeat to supervisor."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.supervisor_url}/agents/{self.agent_id}/heartbeat"
                response = await client.post(url)
                if response.status_code == 200:
                    logger.info(f"Heartbeat sent for {self.agent_id}")
                    self.last_successful_heartbeat = time.time()
                else:
                    logger.warning(f"Heartbeat failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
