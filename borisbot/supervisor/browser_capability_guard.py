"""Capability guard for deterministic browser task enforcement."""

import json
from typing import Any
from urllib.parse import urlparse

from .capability_manager import CapabilityManager


class BrowserCapabilityGuard:
    """Validates browser task permissions and domain scopes for an agent."""

    async def validate_task(self, agent_id: str, task: dict) -> None:
        """Raise when agent/browser task violates capability policy."""
        capabilities = await CapabilityManager.get_capabilities(agent_id)
        browser_capability = self._find_browser_capability(capabilities)
        if browser_capability is None:
            raise RuntimeError("missing capability: BROWSER")

        allowed_domains = self._extract_allowed_domains(browser_capability)
        self._validate_commands(task, allowed_domains)

    def _find_browser_capability(self, capabilities: Any) -> Any:
        """Return the first BROWSER capability record, else None."""
        for record in capabilities or []:
            cap_type = record["capability_type"] if "capability_type" in record.keys() else None
            if cap_type == "BROWSER":
                return record
        return None

    def _extract_allowed_domains(self, capability_record: Any) -> list[str] | None:
        """Parse allowed_domains from capability JSON payload."""
        raw_value = capability_record["capability_value"]
        try:
            parsed = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return None
        allowed = parsed.get("allowed_domains")
        if not isinstance(allowed, list):
            return None
        normalized = [str(domain).strip().lower() for domain in allowed if str(domain).strip()]
        return normalized or None

    def _validate_commands(self, task: dict, allowed_domains: list[str] | None) -> None:
        """Validate command ordering and optional domain allowlist."""
        has_valid_navigation = False
        for command in task.get("commands", []):
            action = command.get("action")
            params = command.get("params", {})
            if action == "navigate":
                url = params.get("url")
                hostname = self._extract_hostname(url)
                if allowed_domains and not self._is_domain_allowed(hostname, allowed_domains):
                    raise RuntimeError(f"domain not allowed: {hostname}")
                has_valid_navigation = True
            else:
                if not has_valid_navigation:
                    raise RuntimeError("navigate required before interaction")

    def _extract_hostname(self, url: str | None) -> str:
        """Extract normalized hostname from URL."""
        if not isinstance(url, str) or not url.strip():
            raise RuntimeError("navigate required before interaction")
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().strip()
        if not hostname:
            raise RuntimeError("navigate required before interaction")
        return hostname

    def _is_domain_allowed(self, hostname: str, allowed_domains: list[str]) -> bool:
        """Return True when hostname equals or is a subdomain of allowed domain."""
        for allowed in allowed_domains:
            if hostname == allowed or hostname.endswith(f".{allowed}"):
                return True
        return False
