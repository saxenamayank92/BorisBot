"""Deterministic model-to-provider routing with circuit-aware filtering."""

from __future__ import annotations

import sqlite3

from borisbot.llm.provider_health import ProviderHealthRegistry
from borisbot.supervisor.database import DB_PATH

DEFAULT_PROVIDER_ORDER: dict[str, list[str]] = {
    "gpt-4o": ["openai", "anthropic"],
    "local-llama": ["ollama"],
}


class RoutingPolicy:
    """Deterministic provider order selection with readiness filtering."""

    def __init__(self, health_registry: ProviderHealthRegistry):
        self._health_registry = health_registry

    def _provider_has_pricing(self, provider_name: str, model_name: str) -> bool:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT 1
                FROM model_pricing
                WHERE model_name = ?
                LIMIT 1
                """,
                (model_name,),
            )
            return cursor.fetchone() is not None
        except sqlite3.Error:
            return False
        finally:
            conn.close()

    def choose_providers(self, model_name: str) -> list[str]:
        """Return deterministic ordered providers for model, filtered by policy."""
        ordered = list(DEFAULT_PROVIDER_ORDER.get(model_name, []))
        chosen: list[str] = []
        for provider in ordered:
            if self._health_registry.is_circuit_open(provider):
                continue
            if not self._health_registry.is_provider_usable(provider):
                continue
            if not self._provider_has_pricing(provider, model_name):
                continue
            chosen.append(provider)
        return chosen
