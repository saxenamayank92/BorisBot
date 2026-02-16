"""Browser session lifecycle scaffolding for supervisor-managed agents."""

import asyncio
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import socket
import subprocess
import uuid

from platformdirs import user_data_dir

from .database import get_db

logger = logging.getLogger("borisbot.supervisor.browser_manager")

MAX_BROWSER_SESSIONS = 3
IMAGE_NAME = "borisbot-browser"
CDP_CONTAINER_PORT = 9222
VNC_CONTAINER_PORT = 5900
NOVNC_CONTAINER_PORT = 6080


class BrowserManager:
    """Coordinates browser session allocation and lifecycle for agents."""

    def _repo_root(self) -> Path:
        """Resolve the repository root from this module location."""
        return Path(__file__).resolve().parents[2]

    def _docker_context_dir(self) -> Path:
        """Return the Docker build context path for browser image assets."""
        return self._repo_root() / "borisbot" / "docker" / "browser"

    def _compute_dockerfile_hash(self) -> str:
        """Compute deterministic SHA256 hash of browser image scaffold files."""
        docker_context = self._docker_context_dir()
        files_to_hash = [
            docker_context / "Dockerfile",
            docker_context / "vnc-start.sh",
        ]
        digest = hashlib.sha256()
        for file_path in files_to_hash:
            logger.debug("Hashing browser image input file: %s", file_path)
            digest.update(file_path.read_bytes())
        return digest.hexdigest()

    async def _run_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a subprocess command and capture stdout/stderr."""
        logger.info("Running command: %s", " ".join(args))
        return await asyncio.to_thread(
            subprocess.run,
            args,
            capture_output=True,
            text=True,
        )

    async def _get_latest_dockerfile_hash(self) -> str | None:
        """Fetch the most recent browser session dockerfile hash from DB."""
        async for db in get_db():
            async with db.execute(
                """
                SELECT dockerfile_hash
                FROM browser_sessions
                ORDER BY created_at DESC
                LIMIT 1
                """
            ) as cursor:
                row = await cursor.fetchone()
                return row["dockerfile_hash"] if row else None
        return None

    async def _count_running_sessions(self) -> int:
        """Count currently running browser sessions."""
        async for db in get_db():
            async with db.execute(
                "SELECT COUNT(*) AS count FROM browser_sessions WHERE status = 'running'"
            ) as cursor:
                row = await cursor.fetchone()
                return int(row["count"])
        return 0

    def _parse_port_mappings(self, output: str) -> dict[int, int]:
        """Parse `docker port` output into {container_port: host_port} mapping."""
        mappings: dict[int, int] = {}
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or "->" not in line or "/tcp" not in line:
                continue
            left, right = line.split("->", maxsplit=1)
            container_port_str = left.strip().split("/")[0]
            host_port_str = right.strip().rsplit(":", maxsplit=1)[-1]
            try:
                mappings[int(container_port_str)] = int(host_port_str)
            except ValueError:
                logger.warning("Unable to parse docker port mapping line: %s", line)
        return mappings

    async def _wait_for_tcp_readiness(self, port: int, timeout_seconds: int = 30) -> bool:
        """Wait for localhost TCP port readiness using 1-second retries."""
        for _ in range(timeout_seconds):
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    return True
            except OSError:
                await asyncio.sleep(1)
        return False

    def _profile_path_for_agent(self, agent_id: str) -> Path:
        """Return the per-agent browser profile directory path."""
        base_dir = Path(user_data_dir("borisbot")) / "browser_profiles"
        profile_path = base_dir / agent_id
        logger.debug("Computed profile path for agent %s: %s", agent_id, profile_path)
        return profile_path

    async def request_session(self, agent_id: str) -> dict:
        """Request or initialize a browser session for an agent."""
        profile_path = self._profile_path_for_agent(agent_id)
        profile_path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured browser profile directory exists: %s", profile_path)

        dockerfile_hash = self._compute_dockerfile_hash()
        latest_hash = await self._get_latest_dockerfile_hash()

        image_check = await self._run_command(["docker", "images", "-q", IMAGE_NAME])
        if image_check.stdout:
            logger.info("Docker image lookup output: %s", image_check.stdout.strip())
        if image_check.stderr:
            logger.info("Docker image lookup stderr: %s", image_check.stderr.strip())
        if image_check.returncode != 0:
            raise RuntimeError(
                f"Failed to check Docker image '{IMAGE_NAME}': {image_check.stderr.strip()}"
            )

        image_exists = bool(image_check.stdout.strip())
        build_required = (not image_exists) or (latest_hash != dockerfile_hash)
        logger.info(
            "Image build decision for %s: image_exists=%s latest_hash=%s build_required=%s",
            agent_id,
            image_exists,
            latest_hash,
            build_required,
        )

        if build_required:
            build_result = await self._run_command(
                ["docker", "build", "-t", IMAGE_NAME, str(self._docker_context_dir())]
            )
            if build_result.stdout:
                logger.info("Docker build stdout:\n%s", build_result.stdout)
            if build_result.stderr:
                logger.info("Docker build stderr:\n%s", build_result.stderr)
            if build_result.returncode != 0:
                raise RuntimeError(
                    f"Docker build failed for image '{IMAGE_NAME}': {build_result.stderr.strip()}"
                )

        running_sessions = await self._count_running_sessions()
        if running_sessions >= MAX_BROWSER_SESSIONS:
            raise RuntimeError("Maximum browser sessions reached")

        container_name = f"borisbot_browser_{agent_id}"
        run_result = await self._run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{profile_path}:/browser-profile",
                "-P",
                IMAGE_NAME,
            ]
        )
        if run_result.stdout:
            logger.info("Docker run stdout: %s", run_result.stdout.strip())
        if run_result.stderr:
            logger.info("Docker run stderr: %s", run_result.stderr.strip())
        if run_result.returncode != 0:
            raise RuntimeError(f"Failed to start browser container: {run_result.stderr.strip()}")

        container_id = run_result.stdout.strip()
        logger.info("Started browser container %s (%s)", container_name, container_id)

        port_result = await self._run_command(["docker", "port", container_name])
        if port_result.stdout:
            logger.info("Docker port output:\n%s", port_result.stdout.strip())
        if port_result.stderr:
            logger.info("Docker port stderr: %s", port_result.stderr.strip())
        if port_result.returncode != 0:
            raise RuntimeError(f"Failed to inspect container ports: {port_result.stderr.strip()}")

        mappings = self._parse_port_mappings(port_result.stdout)
        if (
            CDP_CONTAINER_PORT not in mappings
            or VNC_CONTAINER_PORT not in mappings
            or NOVNC_CONTAINER_PORT not in mappings
        ):
            raise RuntimeError(f"Missing required port mappings: {mappings}")

        cdp_host_port = mappings[CDP_CONTAINER_PORT]
        vnc_host_port = mappings[VNC_CONTAINER_PORT]
        novnc_host_port = mappings[NOVNC_CONTAINER_PORT]

        cdp_ready = await self._wait_for_tcp_readiness(cdp_host_port, timeout_seconds=30)
        if not cdp_ready:
            stop_result = await self._run_command(["docker", "stop", container_name])
            if stop_result.returncode != 0:
                logger.warning(
                    "Failed to stop unready container %s: %s",
                    container_name,
                    stop_result.stderr.strip(),
                )
            raise RuntimeError("Timed out waiting for CDP port readiness")

        now = datetime.utcnow().isoformat()
        session_id = str(uuid.uuid4())
        async for db in get_db():
            await db.execute(
                """
                INSERT INTO browser_sessions (
                    id,
                    agent_id,
                    container_name,
                    cdp_port,
                    vnc_port,
                    profile_path,
                    status,
                    dockerfile_hash,
                    created_at,
                    last_health_check
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent_id,
                    container_name,
                    cdp_host_port,
                    vnc_host_port,
                    str(profile_path),
                    "running",
                    dockerfile_hash,
                    now,
                    now,
                ),
            )
            await db.commit()

        logger.info(
            "Browser session ready for agent %s (container=%s cdp=%s vnc=%s novnc=%s)",
            agent_id,
            container_name,
            cdp_host_port,
            vnc_host_port,
            novnc_host_port,
        )
        return {
            "agent_id": agent_id,
            "container_name": container_name,
            "cdp_port": cdp_host_port,
            "vnc_port": vnc_host_port,
            "novnc_port": novnc_host_port,
            "vnc_url": f"http://localhost:{novnc_host_port}",
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
