"""Local HTTP receiver for recorder browser events."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from borisbot.recorder.session import RecordingSession

logger = logging.getLogger("borisbot.recorder.server")


class RecordingServer:
    """Aiohttp server that receives recorder events and updates session state."""

    def __init__(self, session: RecordingSession, host: str = "127.0.0.1", port: int = 7331):
        self.session = session
        self.host = host
        self.port = port
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def _handle_event(self, request: web.Request) -> web.Response:
        """Accept and process recorder event payload."""
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return web.json_response({"status": "bad_request"}, status=400)

        event_type = body.get("event_type")
        payload = body.get("payload", {})
        if not isinstance(event_type, str):
            return web.json_response({"status": "bad_request"}, status=400)
        if not isinstance(payload, dict):
            payload = {}

        self.session.ingest(event_type, payload)
        return web.json_response({"status": "ok"})

    async def start(self) -> None:
        """Start aiohttp receiver."""
        self._app = web.Application()
        self._app.router.add_post("/event", self._handle_event)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=self.host, port=self.port)
        await self._site.start()
        logger.info("Recording server started at http://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        """Stop aiohttp receiver."""
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("Recording server stopped")
