"""Server-sent event streaming endpoints for task execution events."""

import asyncio
import json

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from borisbot.supervisor.database import get_db

router = APIRouter()


@router.get("/tasks/{task_id}/stream")
async def stream_task_events(request: Request, task_id: str, after: str | None = None):
    """Stream task events from DB in SSE format with optional resume cursor."""

    async def event_generator():
        last_seen = after

        while True:
            if await request.is_disconnected():
                break

            query = """
            SELECT created_at, event_type, payload
            FROM task_events
            WHERE task_id = ?
            {after_clause}
            ORDER BY created_at ASC
            """

            after_clause = ""
            params: list[str] = [task_id]
            if last_seen:
                after_clause = "AND created_at > ?"
                params.append(last_seen)

            async for db in get_db():
                cursor = await db.execute(
                    query.format(after_clause=after_clause),
                    tuple(params),
                )
                rows = await cursor.fetchall()

            for row in rows:
                data = {
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload"] or "{}"),
                    "created_at": row["created_at"],
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_seen = row["created_at"]

            await asyncio.sleep(0.5)

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), headers=headers)
