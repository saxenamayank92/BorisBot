"""DB-backed task queue worker for deterministic browser task execution."""

import asyncio
from datetime import datetime, timedelta
import json
import logging

from borisbot.browser.actions import BrowserActions
from borisbot.browser.command_router import CommandRouter
from borisbot.browser.executor import BrowserExecutor
from borisbot.browser.task_runner import TaskRunner
from borisbot.supervisor.browser_manager import BrowserManager
from borisbot.supervisor.database import get_db

logger = logging.getLogger("borisbot.supervisor.worker")

LOCK_TIMEOUT_SECONDS = 60


class Worker:
    """Deterministic single-task-at-a-time queue worker."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    async def run_forever(self):
        """Continuously claim and execute queued tasks."""
        while True:
            await self._reclaim_stale_tasks()
            claimed = await self._claim_task()
            if not claimed:
                await asyncio.sleep(1)
                continue
            task_id = claimed["task_id"]
            logger.info("Worker %s claimed task %s", self.worker_id, task_id)
            await self._execute_task(task_id)

    async def _claim_task(self) -> dict[str, str] | None:
        """Atomically claim one unlocked queue row."""
        now = datetime.utcnow().isoformat()
        lock_expires_at = (
            datetime.utcnow() + timedelta(seconds=LOCK_TIMEOUT_SECONDS)
        ).isoformat()
        query = """
            UPDATE task_queue
            SET locked_at = ?,
                locked_by = ?,
                lock_expires_at = ?
            WHERE id = (
              SELECT id FROM task_queue
              WHERE locked_at IS NULL
              ORDER BY enqueued_at ASC
              LIMIT 1
            )
            RETURNING task_id
        """
        async for db in get_db():
            cursor = await db.execute(query, (now, self.worker_id, lock_expires_at))
            row = await cursor.fetchone()
            await db.commit()
            if not row:
                return None
            return {"task_id": row["task_id"]}

    async def _reclaim_stale_tasks(self) -> None:
        """Reclaim expired queue locks and reset task status to pending."""
        now = datetime.utcnow().isoformat()
        reclaim_query = """
            SELECT id, task_id FROM task_queue
            WHERE locked_at IS NOT NULL
            AND lock_expires_at < ?
        """
        async for db in get_db():
            cursor = await db.execute(reclaim_query, (now,))
            stale = await cursor.fetchall()
            for row in stale:
                task_id = row["task_id"]
                logger.info("Reclaiming stale lock for task %s", task_id)
                await db.execute(
                    """
                    UPDATE task_queue
                    SET locked_at = NULL,
                        locked_by = NULL,
                        lock_expires_at = NULL
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                await db.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending'
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
            if stale:
                await db.commit()

    async def _execute_task(self, task_id: str) -> None:
        """Execute claimed task exactly once and delete queue row."""
        async for db in get_db():
            cursor = await db.execute(
                """
                SELECT q.id AS queue_id, t.agent_id, t.payload
                FROM task_queue q
                JOIN tasks t ON t.task_id = q.task_id
                WHERE q.task_id = ?
                """,
                (task_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            logger.error("Worker %s missing task row for task_id=%s", self.worker_id, task_id)
            async for db in get_db():
                await db.execute("DELETE FROM task_queue WHERE task_id = ?", (task_id,))
                await db.commit()
            return

        queue_id = row["queue_id"]
        agent_id = row["agent_id"]
        task = json.loads(row["payload"])

        bm = BrowserManager()
        executor: BrowserExecutor | None = None
        logger.info("Worker %s starting execution for task %s", self.worker_id, task_id)
        try:
            session = await bm.request_session(agent_id)
            executor = BrowserExecutor(session["cdp_port"])
            await executor.connect()

            actions = BrowserActions(executor)
            router_obj = CommandRouter(actions)
            runner = TaskRunner(router_obj, agent_id=agent_id, pre_persisted=True)
            await runner.run(task)
            logger.info("Worker %s finished execution for task %s", self.worker_id, task_id)
        finally:
            if executor is not None:
                await executor.close()
            async for db in get_db():
                await db.execute("DELETE FROM task_queue WHERE id = ?", (queue_id,))
                await db.commit()
