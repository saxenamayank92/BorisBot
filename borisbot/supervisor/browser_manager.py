"""Browser session lifecycle scaffolding for supervisor-managed agents."""

import asyncio
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
import socket
import subprocess
import time
import uuid

from platformdirs import user_data_dir

from .database import get_db

logger = logging.getLogger("borisbot.supervisor.browser_manager")

MAX_BROWSER_SESSIONS = 3
SESSION_TTL_MINUTES = 15
IMAGE_NAME = "borisbot-browser"
CDP_CONTAINER_PORT = 9222
VNC_CONTAINER_PORT = 5900
NOVNC_CONTAINER_PORT = 6080


def build_novnc_url(host_port: int) -> str:
    """Return direct noVNC URL that opens the client and auto-connects."""
    return f"http://localhost:{host_port}/vnc.html?autoconnect=1&resize=remote&reconnect=1"


def _is_missing_container_error(stderr: str) -> bool:
    """Return True when docker stderr indicates the container does not exist."""
    return "no such container" in (stderr or "").strip().lower()


class BrowserManager:
    """Coordinates browser session allocation and lifecycle for agents."""

    _shared_expire_lock: asyncio.Lock | None = None

    def __init__(self) -> None:
        if BrowserManager._shared_expire_lock is None:
            BrowserManager._shared_expire_lock = asyncio.Lock()
        self._expire_lock = BrowserManager._shared_expire_lock

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

    async def _is_container_running(self, container_name: str) -> bool:
        """Return True if Docker reports the container is currently running."""
        result = await self._run_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
        )
        if result.returncode != 0:
            logger.info(
                "Container inspect failed for %s (likely missing): %s",
                container_name,
                result.stderr.strip(),
            )
            return False
        return result.stdout.strip().lower() == "true"

    async def _is_tcp_port_open(self, host: str, port: int, timeout: int = 2) -> bool:
        """Return True when a TCP connection can be established."""
        def _probe() -> bool:
            with socket.create_connection((host, port), timeout=timeout):
                return True

        try:
            return await asyncio.to_thread(_probe)
        except OSError:
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

        await self.expire_stale_sessions()
        # Reconcile crashed/dead sessions before enforcing max running limit.
        await self.health_check_sessions()

        running_sessions = await self._count_running_sessions()
        if running_sessions >= MAX_BROWSER_SESSIONS:
            raise RuntimeError("Maximum browser sessions reached")

        async for db in get_db():
            async with db.execute(
                """
                SELECT id FROM browser_sessions
                WHERE agent_id = ? AND status = 'expired'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id,),
            ) as cursor:
                expired_row = await cursor.fetchone()
                if expired_row:
                    logger.info(
                        "Found expired browser session for agent %s; creating fresh session.",
                        agent_id,
                    )
            break

        container_name = f"borisbot_browser_{agent_id}_{uuid.uuid4().hex[:8]}"
        preclean_result = await self._run_command(["docker", "rm", "-f", container_name])
        if preclean_result.stdout:
            logger.info("Docker pre-clean stdout: %s", preclean_result.stdout.strip())
        if preclean_result.stderr:
            logger.info("Docker pre-clean stderr: %s", preclean_result.stderr.strip())
        if preclean_result.returncode != 0 and not _is_missing_container_error(preclean_result.stderr):
            raise RuntimeError(
                f"Failed pre-clean for browser container {container_name}: "
                f"{preclean_result.stderr.strip()}"
            )

        run_result = await self._run_command(
            [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-v",
        f"{profile_path}:/browser-profile",
        "--dns", "8.8.8.8",
        "--dns", "1.1.1.1",
        "-p", "9222",
        "-p", "5900",
        "-p", "6080",
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

        now_dt = datetime.utcnow()
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(minutes=SESSION_TTL_MINUTES)).isoformat()
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
                    last_health_check,
                    expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    expires_at,
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
            "vnc_url": build_novnc_url(novnc_host_port),
        }

    async def expire_stale_sessions(self) -> None:
        """Expire running sessions whose TTL has elapsed."""
        started = time.perf_counter()
        async with self._expire_lock:
            now = datetime.utcnow().isoformat()
            async for db in get_db():
                async with db.execute(
                    """
                    SELECT id, container_name
                    FROM browser_sessions
                    WHERE status = 'running'
                      AND expires_at IS NOT NULL
                      AND expires_at < ?
                    """,
                    (now,),
                ) as cursor:
                    stale_rows = await cursor.fetchall()

                logger.info("Expiring %d stale browser sessions", len(stale_rows))

                for row in stale_rows:
                    container_name = row["container_name"]
                    stop_result = await self._run_command(["docker", "stop", container_name])
                    if stop_result.returncode != 0 and "No such container" not in stop_result.stderr:
                        raise RuntimeError(
                            f"Failed to stop stale container {container_name}: "
                            f"{stop_result.stderr.strip()}"
                        )
                    rm_result = await self._run_command(["docker", "rm", "-f", container_name])
                    if rm_result.returncode != 0 and "No such container" not in rm_result.stderr:
                        raise RuntimeError(
                            f"Failed to remove stale container {container_name}: "
                            f"{rm_result.stderr.strip()}"
                        )
                    await db.execute(
                        """
                        UPDATE browser_sessions
                        SET status = ?, last_health_check = ?
                        WHERE id = ?
                        """,
                        ("expired", now, row["id"]),
                    )
                    logger.info("Expired browser container: %s", container_name)
                await db.commit()
        elapsed = time.perf_counter() - started
        logger.info("expire_stale_sessions completed in %.3fs", elapsed)

    async def stop_session(self, agent_id: str) -> None:
        """Stop an active browser session for an agent."""
        logger.info("Stopping browser session for agent %s", agent_id)
        now = datetime.utcnow().isoformat()
        async for db in get_db():
            async with db.execute(
                """
                SELECT id, container_name
                FROM browser_sessions
                WHERE agent_id = ? AND status = 'running'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id,),
            ) as cursor:
                session_row = await cursor.fetchone()

            if not session_row:
                logger.info("No running browser session found for agent %s", agent_id)
                return

            container_name = session_row["container_name"]
            stop_result = await self._run_command(["docker", "stop", container_name])
            if stop_result.stdout:
                logger.info("Docker stop stdout: %s", stop_result.stdout.strip())
            if stop_result.stderr:
                logger.info("Docker stop stderr: %s", stop_result.stderr.strip())
            stderr = (stop_result.stderr or "").strip()
            if stop_result.returncode != 0 and not _is_missing_container_error(stderr):
                raise RuntimeError(
                    f"Failed to stop browser container {container_name}: {stderr}"
                )
            session_status = "stopped"
            if stop_result.returncode != 0 and _is_missing_container_error(stderr):
                session_status = "crashed"
            rm_result = await self._run_command(["docker", "rm", "-f", container_name])
            rm_stderr = (rm_result.stderr or "").strip()
            if rm_result.returncode != 0 and not _is_missing_container_error(rm_stderr):
                raise RuntimeError(
                    f"Failed to remove browser container {container_name}: {rm_stderr}"
                )

            await db.execute(
                """
                UPDATE browser_sessions
                SET status = ?, last_health_check = ?
                WHERE id = ?
                """,
                (session_status, now, session_row["id"]),
            )
            await db.commit()
            logger.info(
                "Stopped browser session for agent %s (container=%s)",
                agent_id,
                container_name,
            )
            return

    async def cleanup_orphan_containers(self) -> None:
        """Apply strict startup reset by force-removing all browser containers."""
        logger.info("Cleaning up orphaned browser containers...")
        now = datetime.utcnow().isoformat()
        list_result = await self._run_command(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=borisbot_browser_",
                "--format",
                "{{.Names}}",
            ]
        )
        if list_result.stdout:
            logger.info("Docker container listing output:\n%s", list_result.stdout.strip())
        if list_result.stderr:
            logger.info("Docker container listing stderr: %s", list_result.stderr.strip())
        if list_result.returncode != 0:
            raise RuntimeError(
                "Failed to list browser containers for cleanup: "
                f"{list_result.stderr.strip()}"
            )

        container_names = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
        for container_name in container_names:
            logger.info("Force removing browser container %s", container_name)
            rm_result = await self._run_command(["docker", "rm", "-f", container_name])
            if rm_result.stdout:
                logger.info("Docker rm stdout: %s", rm_result.stdout.strip())
            if rm_result.stderr:
                logger.info("Docker rm stderr: %s", rm_result.stderr.strip())
            if rm_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to force remove browser container {container_name}: "
                    f"{rm_result.stderr.strip()}"
                )

        async for db in get_db():
            await db.execute(
                """
                UPDATE browser_sessions
                SET status = ?, last_health_check = ?
                WHERE status = 'running'
                """,
                ("crashed", now),
            )
            await db.commit()
        logger.info("Orphan browser container cleanup completed.")

    async def health_check_sessions(self) -> None:
        """Run one-shot health checks for running browser sessions."""
        logger.info("Running browser session health checks...")
        now = datetime.utcnow().isoformat()
        async for db in get_db():
            async with db.execute(
                """
                SELECT id, container_name, cdp_port
                FROM browser_sessions
                WHERE status = 'running'
                """
            ) as cursor:
                running_rows = await cursor.fetchall()

            for row in running_rows:
                session_id = row["id"]
                container_name = row["container_name"]
                cdp_port = int(row["cdp_port"])
                status = "running"

                container_running = await self._is_container_running(container_name)
                if not container_running:
                    logger.warning(
                        "Browser container is not running during health check: %s",
                        container_name,
                    )
                    status = "crashed"
                else:
                    cdp_ready = await self._is_tcp_port_open("localhost", cdp_port, timeout=2)
                    if not cdp_ready:
                        logger.warning(
                            "CDP port health check failed for container %s on port %s",
                            container_name,
                            cdp_port,
                        )
                        stop_result = await self._run_command(["docker", "stop", container_name])
                        if stop_result.returncode != 0:
                            logger.warning(
                                "Failed to stop unhealthy browser container %s: %s",
                                container_name,
                                stop_result.stderr.strip(),
                            )
                        status = "crashed"

                await db.execute(
                    """
                    UPDATE browser_sessions
                    SET status = ?, last_health_check = ?
                    WHERE id = ?
                    """,
                    (status, now, session_id),
                )

            await db.commit()
        logger.info("Browser session health checks complete.")
