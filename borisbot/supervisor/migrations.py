"""Database migrations for supervisor tables."""

import logging

import aiosqlite

logger = logging.getLogger("borisbot.supervisor.migrations")

MIGRATIONS: list[tuple[str, str]] = [
    (
        "20260216_create_browser_sessions",
        """
        CREATE TABLE IF NOT EXISTS browser_sessions (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            container_name TEXT,
            cdp_port INTEGER,
            vnc_port INTEGER,
            profile_path TEXT,
            status TEXT,
            dockerfile_hash TEXT,
            created_at TEXT,
            last_health_check TEXT
        )
        """,
    ),
    (
        "20260217_create_tasks",
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            result TEXT
        )
        """,
    ),
    (
        "20260217_create_task_queue",
        """
        CREATE TABLE IF NOT EXISTS task_queue (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            enqueued_at TEXT NOT NULL,
            locked_at TEXT,
            locked_by TEXT,
            FOREIGN KEY(task_id) REFERENCES tasks(task_id)
        )
        """,
    ),
    (
        "20260217_create_task_queue_locked_index",
        """
        CREATE INDEX IF NOT EXISTS idx_task_queue_locked ON task_queue(locked_at)
        """,
    ),
    (
        "20260217_add_task_queue_lock_expires_at",
        """
        ALTER TABLE task_queue ADD COLUMN lock_expires_at TEXT
        """,
    ),
    (
        "20260217_create_task_queue_lock_expires_index",
        """
        CREATE INDEX IF NOT EXISTS idx_task_queue_lock_expires ON task_queue(lock_expires_at)
        """,
    ),
    (
        "20260217_create_task_execution_logs",
        """
        CREATE TABLE IF NOT EXISTS task_execution_logs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            command_id TEXT,
            worker_id TEXT,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            duration_ms INTEGER,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """,
    ),
    (
        "20260217_create_task_execution_logs_task_index",
        """
        CREATE INDEX IF NOT EXISTS idx_task_execution_logs_task ON task_execution_logs(task_id)
        """,
    ),
    (
        "20260217_create_worker_heartbeats",
        """
        CREATE TABLE IF NOT EXISTS worker_heartbeats (
            worker_id TEXT PRIMARY KEY,
            last_seen TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """,
    ),
]


async def run_migrations(db: aiosqlite.Connection) -> None:
    """Apply one-time database migrations in order."""
    logger.info("Running supervisor DB migrations...")
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    async with db.execute("SELECT id FROM schema_migrations") as cursor:
        rows = await cursor.fetchall()
    applied = {row[0] for row in rows}

    for migration_id, sql in MIGRATIONS:
        if migration_id in applied:
            logger.debug("Migration already applied: %s", migration_id)
            continue
        logger.info("Applying migration: %s", migration_id)
        await db.execute(sql)
        await db.execute(
            "INSERT INTO schema_migrations (id) VALUES (?)",
            (migration_id,),
        )

    await db.commit()
    logger.info("Supervisor DB migrations complete.")
