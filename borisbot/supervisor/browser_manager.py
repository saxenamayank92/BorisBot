"""Browser session lifecycle scaffolding for supervisor-managed agents."""

import logging
from pathlib import Path

from platformdirs import user_data_dir

logger = logging.getLogger("borisbot.supervisor.browser_manager")


class BrowserManager:
    """Coordinates browser session allocation and lifecycle for agents."""

    def _profile_path_for_agent(self, agent_id: str) -> Path:
        """Return the per-agent browser profile directory path."""
        base_dir = Path(user_data_dir("borisbot")) / "browser_profiles"
        profile_path = base_dir / agent_id
        logger.debug("Computed profile path for agent %s: %s", agent_id, profile_path)
        return profile_path

    async def request_session(self, agent_id: str) -> dict:
        """Request or initialize a browser session for an agent.

        This scaffold records lifecycle intent only; Docker/session startup is
        intentionally not implemented in this phase.
        """
        profile_path = self._profile_path_for_agent(agent_id)
        logger.info(
            "Browser session requested for agent %s with profile path %s",
            agent_id,
            profile_path,
        )
        return {
            "agent_id": agent_id,
            "profile_path": str(profile_path),
            "status": "scaffolded",
        }

    async def stop_session(self, agent_id: str) -> None:
        """Stop an active browser session for an agent.

        Docker/container stop behavior is intentionally deferred.
        """
        logger.info("Stop browser session requested for agent %s", agent_id)

    async def cleanup_orphan_containers(self) -> None:
        """Clean up orphaned browser containers from previous supervisor runs.

        Actual container interaction is intentionally deferred.
        """
        logger.info("Cleanup of orphan browser containers requested (scaffold only).")

    async def health_check_sessions(self) -> None:
        """Run health checks for active browser sessions.

        Runtime health probing is intentionally deferred in this scaffold.
        """
        logger.info("Health check of browser sessions requested (scaffold only).")
