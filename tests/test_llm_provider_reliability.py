"""Tests for provider state-machine transitions and retry-to-unhealthy behavior."""

import unittest

from borisbot.llm.engine import LLMEngine
from borisbot.llm.errors import LLMAllProvidersFailed
from borisbot.llm.provider_health import (
    PROVIDER_STATE_DEGRADED,
    PROVIDER_STATE_HEALTHY,
    PROVIDER_STATE_UNHEALTHY,
    ProviderHealthRegistry,
)


class _AlwaysFailProvider:
    async def complete(self, *, model_name: str, prompt: str, **kwargs):
        raise TimeoutError("timeout")

    async def stream(self, *, model_name: str, prompt: str, **kwargs):  # pragma: no cover
        raise TimeoutError("timeout")
        yield ""


class _NoopCostGuard:
    async def check_can_execute(self, agent_id: str, task_id: str | None = None) -> None:
        return

    async def record_usage(self, agent_id: str, model_name: str, input_tokens: int, output_tokens: int) -> float:
        return 0.0


class _FixedRouting:
    def choose_providers(self, model_name: str) -> list[str]:
        return ["ollama"]


class ProviderReliabilityTests(unittest.IsolatedAsyncioTestCase):
    """Verify deterministic provider state transitions."""

    async def test_runtime_failures_degrade_then_unhealthy(self) -> None:
        registry = ProviderHealthRegistry()
        engine = LLMEngine(
            providers={"ollama": _AlwaysFailProvider()},
            health_registry=registry,
            routing_policy=_FixedRouting(),
            cost_guard=_NoopCostGuard(),
        )

        with self.assertRaises(LLMAllProvidersFailed):
            await engine.complete(agent_id="a1", model_name="local-llama", prompt="hi")
        self.assertEqual(registry.get_provider_state("ollama"), PROVIDER_STATE_UNHEALTHY)

        with self.assertRaises(LLMAllProvidersFailed):
            await engine.complete(agent_id="a1", model_name="local-llama", prompt="hi")
        self.assertEqual(registry.get_provider_state("ollama"), PROVIDER_STATE_UNHEALTHY)

    async def test_probe_transitions(self) -> None:
        registry = ProviderHealthRegistry()
        self.assertEqual(registry.get_provider_state("ollama"), "unknown")
        registry.mark_probe_failure("ollama")
        self.assertEqual(registry.get_provider_state("ollama"), PROVIDER_STATE_UNHEALTHY)
        registry.mark_probe_success("ollama")
        self.assertEqual(registry.get_provider_state("ollama"), PROVIDER_STATE_HEALTHY)
        registry.mark_runtime_failure("ollama")
        self.assertEqual(registry.get_provider_state("ollama"), PROVIDER_STATE_DEGRADED)


if __name__ == "__main__":
    unittest.main()

