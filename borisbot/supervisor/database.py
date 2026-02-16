import aiosqlite
import os
from pathlib import Path

DB_PATH = Path.home() / ".borisbot" / "borisbot.db"

async def get_db_path() -> Path:
    """Ensure the directory exists and return the DB path."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH

async def init_db():
    """Initialize the database with required tables."""
    db_path = await get_db_path()
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
        await db.commit()

async def get_db():
    """Dependency for getting a DB cursor."""
    db_path = await get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db
