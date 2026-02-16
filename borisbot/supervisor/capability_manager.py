from .database import get_db
import aiosqlite

class CapabilityManager:
    @staticmethod
    async def add_capability(agent_id: str, cap_type: str, cap_value: str):
        async for db in get_db():
            await db.execute(
                "INSERT INTO capabilities (agent_id, capability_type, capability_value) VALUES (?, ?, ?)",
                (agent_id, cap_type, cap_value)
            )
            await db.commit()

    @staticmethod
    async def get_capabilities(agent_id: str):
        async for db in get_db():
            async with db.execute(
                "SELECT capability_type, capability_value FROM capabilities WHERE agent_id = ?",
                (agent_id,)
            ) as cursor:
                return await cursor.fetchall()
