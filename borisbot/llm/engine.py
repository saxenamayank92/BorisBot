"""Deterministic multi-provider LLM engine with failover and circuit support."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.errors import (
    LLMAllProvidersFailed,
    LLMError,
    LLMProviderUnhealthyError,
    LLMRetryableError,
    LLMTimeoutStructuredError,
    LLMValidationError,
)
from borisbot.llm.middleware_stack import (
    CircuitBreakerMiddleware,
    CostGuardMiddleware,
    LoggingMiddleware,
    MiddlewareStack,
    RetryMiddleware,
    TimeoutMiddleware,
)
from borisbot.llm.provider_health import ProviderHealthRegistry, get_provider_health_registry
from borisbot.llm.routing import RoutingPolicy

logger = logging.getLogger("borisbot.llm.engine")


class LLMProvider(Protocol):
    """Provider contract for completion and streaming operations."""

    async def complete(self, *, model_name: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Return response payload with optional token usage."""

    async def stream(self, *, model_name: str, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield streaming token chunks."""


class LLMEngine:
    """Deterministic provider failover engine for completion and streaming."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        health_registry: ProviderHealthRegistry | None = None,
        routing_policy: RoutingPolicy | None = None,
        cost_guard: CostGuard | None = None,
    ) -> None:
        self.providers = providers
        self.health_registry = health_registry or get_provider_health_registry()
        self.routing_policy = routing_policy or RoutingPolicy(self.health_registry)
        self.cost_guard = cost_guard or CostGuard()

    @staticmethod
    def _map_provider_error(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, TimeoutError):
            return LLMTimeoutStructuredError("provider timeout")
        message = str(exc).lower()
        if "timeout" in message:
            return LLMTimeoutStructuredError(str(exc))
        if "rate limit" in message or "temporar" in message or "unavailable" in message:
            return LLMRetryableError(str(exc))
        return LLMRetryableError(str(exc))

    def _build_stack(self, provider_name: str, agent_id: str, task_id: str | None) -> MiddlewareStack:
        return MiddlewareStack(
            [
                LoggingMiddleware(provider_name),
                CostGuardMiddleware(self.cost_guard, agent_id=agent_id, task_id=task_id),
                CircuitBreakerMiddleware(self.health_registry, provider_name=provider_name),
                RetryMiddleware(retries=0),
                TimeoutMiddleware(timeout_seconds=30.0),
            ]
        )

    async def complete(
        self,
        *,
        agent_id: str,
        model_name: str,
        prompt: str,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute non-streaming completion with deterministic provider failover."""
        if not model_name or not isinstance(model_name, str):
            raise LLMValidationError("model_name is required")
        if not isinstance(prompt, str):
            raise LLMValidationError("prompt must be string")

        providers = self.routing_policy.choose_providers(model_name)
        if not providers:
            raise LLMAllProvidersFailed(f"no available providers for model: {model_name}")

        last_error: LLMError | None = None
        for provider_name in providers:
            if self.health_registry.is_circuit_open(provider_name):
                continue
            if not self.health_registry.is_provider_usable(provider_name):
                last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                continue
            provider = self.providers.get(provider_name)
            if provider is None:
                continue

            stack = self._build_stack(provider_name, agent_id=agent_id, task_id=task_id)
            start = time.perf_counter()
            for attempt_idx in range(2):
                try:
                    response = await stack.execute(
                        lambda: provider.complete(model_name=model_name, prompt=prompt, **kwargs)
                    )
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self.health_registry.mark_success(provider_name, latency_ms)

                    usage = response.get("usage") if isinstance(response, dict) else None
                    if isinstance(usage, dict):
                        input_tokens = int(usage.get("input_tokens", 0))
                        output_tokens = int(usage.get("output_tokens", 0))
                        try:
                            await self.cost_guard.record_usage(
                                agent_id=agent_id,
                                model_name=model_name,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )
                        except Exception as exc:  # pragma: no cover - defensive path
                            logger.warning("record_usage failed: %s", exc)
                    return response
                except Exception as exc:
                    mapped = self._map_provider_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
                    self.health_registry.mark_runtime_failure(provider_name)
                    if attempt_idx == 0:
                        if not self.health_registry.is_provider_usable(provider_name):
                            last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                            break
                        continue
                    if not self.health_registry.is_provider_usable(provider_name):
                        last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                    else:
                        last_error = mapped
                    break

        raise LLMAllProvidersFailed(str(last_error) if last_error else "all providers failed")

    async def stream(
        self,
        *,
        agent_id: str,
        model_name: str,
        prompt: str,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completions with failover before first chunk only."""
        if not model_name or not isinstance(model_name, str):
            raise LLMValidationError("model_name is required")
        if not isinstance(prompt, str):
            raise LLMValidationError("prompt must be string")

        providers = self.routing_policy.choose_providers(model_name)
        if not providers:
            raise LLMAllProvidersFailed(f"no available providers for model: {model_name}")

        last_error: LLMError | None = None
        for provider_name in providers:
            if self.health_registry.is_circuit_open(provider_name):
                continue
            if not self.health_registry.is_provider_usable(provider_name):
                last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                continue
            provider = self.providers.get(provider_name)
            if provider is None:
                continue

            stack = self._build_stack(provider_name, agent_id=agent_id, task_id=task_id)
            start = time.perf_counter()
            emitted = False
            for attempt_idx in range(2):
                try:
                    async def start_stream() -> AsyncIterator[str]:
                        result = provider.stream(model_name=model_name, prompt=prompt, **kwargs)
                        if inspect.isawaitable(result):
                            return await result  # type: ignore[return-value]
                        return result  # type: ignore[return-value]

                    stream_iter = await stack.execute(start_stream)

                    first_chunk = await anext(stream_iter)
                    emitted = True
                    yield first_chunk

                    async for chunk in stream_iter:
                        yield chunk

                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self.health_registry.mark_success(provider_name, latency_ms)
                    return
                except StopAsyncIteration:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self.health_registry.mark_success(provider_name, latency_ms)
                    return
                except Exception as exc:
                    mapped = self._map_provider_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
                    if emitted:
                        raise mapped
                    self.health_registry.mark_runtime_failure(provider_name)
                    if attempt_idx == 0:
                        if not self.health_registry.is_provider_usable(provider_name):
                            last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                            break
                        continue
                    if not self.health_registry.is_provider_usable(provider_name):
                        last_error = LLMProviderUnhealthyError(f"provider unavailable: {provider_name}")
                    else:
                        last_error = mapped
                    break

        raise LLMAllProvidersFailed(str(last_error) if last_error else "all providers failed")
