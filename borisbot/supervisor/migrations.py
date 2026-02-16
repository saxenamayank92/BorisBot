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
    )
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
