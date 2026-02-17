"""In-memory deterministic recording session for browser workflow capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class RecordingSession:
    """Collect and normalize browser interaction events into task commands."""

    task_id: str
    start_url: str | None = None
    commands: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    _last_navigate_url: str | None = None
    _base_host: str | None = None

    def __post_init__(self) -> None:
        if not self.start_url:
            return
        parsed = urlparse(self.start_url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return
        self._base_host = host[4:] if host.startswith("www.") else host

    def _append(self, action: str, params: dict[str, Any]) -> None:
        command = {
            "id": str(self._next_id),
            "action": action,
            "params": params,
        }
        self.commands.append(command)
        self._next_id += 1

    def _replace_last_navigate(self, url: str) -> None:
        if not self.commands:
            self._append("navigate", {"url": url})
            return
        last = self.commands[-1]
        if last.get("action") != "navigate":
            self._append("navigate", {"url": url})
            return
        last["params"] = {"url": url}
        self._last_navigate_url = url

    def _allow_navigation(self, host: str) -> bool:
        if not self._base_host:
            return True
        if host == self._base_host:
            return True
        return host.endswith(f".{self._base_host}")

    def ingest(self, event_type: str, payload: dict[str, Any] | None) -> None:
        """Ingest a normalized browser event and append deterministic command."""
        payload = payload or {}
        if event_type == "navigate":
            url = str(payload.get("url", "")).strip()
            if not url:
                return
            parsed = urlparse(url)
            host = (parsed.hostname or "").strip().lower()
            if not host:
                return
            if not self._allow_navigation(host):
                return
            if url == self._last_navigate_url:
                return
            self._replace_last_navigate(url)
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
