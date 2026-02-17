"""In-memory deterministic recording session for browser workflow capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordingSession:
    """Collect and normalize browser interaction events into task commands."""

    task_id: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    _last_navigate_url: str | None = None

    def _append(self, action: str, params: dict[str, Any]) -> None:
        command = {
            "id": str(self._next_id),
            "action": action,
            "params": params,
        }
        self.commands.append(command)
        self._next_id += 1

    def ingest(self, event_type: str, payload: dict[str, Any] | None) -> None:
        """Ingest a normalized browser event and append deterministic command."""
        payload = payload or {}
        if event_type == "navigate":
            url = str(payload.get("url", "")).strip()
            if not url:
                return
            if url == self._last_navigate_url:
                return
            self._last_navigate_url = url
            self._append("navigate", {"url": url})
            return

        if event_type == "click":
            selector = str(payload.get("selector", "")).strip()
            if not selector:
                return
            self._append("click", {"selector": selector})
            return

        if event_type == "type":
            selector = str(payload.get("selector", "")).strip()
            if not selector:
                return
            text = str(payload.get("text", ""))
            self._append("type", {"selector": selector, "text": text})

    def finalize(self) -> dict[str, Any]:
        """Return finalized workflow payload in TaskRunner-compatible schema."""
        return {
            "task_id": self.task_id,
            "commands": list(self.commands),
        }
