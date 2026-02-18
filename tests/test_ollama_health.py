"""Tests for Ollama startup health probe integration."""

import unittest
from unittest import mock

from borisbot.llm.ollama_health import startup_mark_ollama_health
from borisbot.llm.provider_health import ProviderHealthRegistry


class OllamaHealthTests(unittest.IsolatedAsyncioTestCase):
    """Validate startup probe drives provider state machine."""

    async def test_startup_probe_marks_healthy(self) -> None:
        registry = ProviderHealthRegistry()
        with mock.patch("borisbot.llm.ollama_health.probe_ollama", return_value=True):
            healthy = await startup_mark_ollama_health(registry, provider_name="ollama")
        self.assertTrue(healthy)
        self.assertEqual(registry.get_provider_state("ollama"), "healthy")

    async def test_startup_probe_marks_unhealthy(self) -> None:
        registry = ProviderHealthRegistry()
        with mock.patch("borisbot.llm.ollama_health.probe_ollama", return_value=False):
            healthy = await startup_mark_ollama_health(registry, provider_name="ollama")
        self.assertFalse(healthy)
        self.assertEqual(registry.get_provider_state("ollama"), "unhealthy")


if __name__ == "__main__":
    unittest.main()

