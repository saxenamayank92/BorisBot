"""Deterministic command router for structured browser actions."""

import logging
from typing import Any, Dict

from .actions import BrowserActions

logger = logging.getLogger("borisbot.browser.command_router")


class CommandRouter:
    """
    Deterministic command dispatcher for browser automation.
    """

    def __init__(self, actions: BrowserActions):
        self._actions = actions

    async def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a structured browser command.

        Raises:
            ValueError for invalid schema
            RuntimeError for execution errors
        """
        self._validate(command)

        action = command["action"]
        params = command["params"]

        logger.info("Executing browser command: %s", action)

        if action == "navigate":
            await self._actions.safe_navigate(params["url"])
            return {"status": "ok"}

        elif action == "click":
            await self._actions.safe_click(params["selector"])
            return {"status": "ok"}

        elif action == "type":
            await self._actions.safe_type(params["selector"], params["text"])
            return {"status": "ok"}

        elif action == "wait_for_url":
            await self._actions.safe_wait_for_url_contains(params["contains"])
            return {"status": "ok"}

        elif action == "get_text":
            text = await self._actions.safe_get_text(params["selector"])
            return {"status": "ok", "result": text}

        elif action == "get_title":
            title = await self._actions.executor.get_title()
            return {"status": "ok", "result": title}

        else:
            raise ValueError(f"Unsupported action: {action}")

    def _validate(self, command: Dict[str, Any]) -> None:
        if not isinstance(command, dict):
            raise ValueError("Command must be dict")

        if "id" not in command:
            raise ValueError("Command missing 'id'")

        if "action" not in command:
            raise ValueError("Command missing 'action'")

        if "params" not in command:
            raise ValueError("Command missing 'params'")

        if not isinstance(command["params"], dict):
            raise ValueError("'params' must be dict")
