"""Database initialization and connection helpers for the supervisor."""

import json
import logging
from pathlib import Path
from datetime import datetime

import aiosqlite

from .migrations import run_migrations

logger = logging.getLogger("borisbot.supervisor.database")

DB_PATH = Path.home() / ".borisbot" / "borisbot.db"

async def get_db_path() -> Path:
    """Ensure the directory exists and return the DB path."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


async def init_db():
    """Initialize the database with required tables."""
    db_path = await get_db_path()
    logger.info("Initializing supervisor database at %s", db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                parent_id TEXT,
                autonomy_mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_heartbeat TIMESTAMP,
                pid INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                capability_type TEXT NOT NULL,
                capability_value TEXT NOT NULL,
                FOREIGN KEY(agent_id) REFERENCES agents(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                result TEXT
            )
        """)
        await run_migrations(db)
        async with db.execute("PRAGMA table_info(browser_sessions)") as cursor:
            browser_session_columns = {row[1] for row in await cursor.fetchall()}
        if "expires_at" not in browser_session_columns:
            await db.execute("ALTER TABLE browser_sessions ADD COLUMN expires_at DATETIME")
        await db.execute(
            """
            UPDATE browser_sessions
            SET expires_at = created_at
            WHERE expires_at IS NULL
            """
        )
        await db.commit()


async def get_db():
    """Dependency for getting a DB cursor."""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def reconcile_running_tasks_after_crash() -> None:
    """Mark stale running tasks as failed on supervisor startup."""
    now = datetime.utcnow().isoformat()
    failure_result = json.dumps(
        {"status": "failed", "reason": "supervisor crash", "steps": []}
    )
    async for db in get_db():
        await db.execute(
            """
            UPDATE tasks
            SET status = ?, updated_at = ?, result = ?
            WHERE status = 'running'
            """,
            ("failed", now, failure_result),
        )
        await db.commit()
    logger.info("Reconciled stale running tasks after crash recovery.")
