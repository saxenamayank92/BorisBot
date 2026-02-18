"""Composable middleware wrappers for deterministic provider execution."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.errors import (
    LLMCircuitOpenError,
    LLMRetryableError,
    LLMTimeoutError,
)
from borisbot.llm.provider_health import ProviderHealthRegistry

logger = logging.getLogger("borisbot.llm.middleware")

ProviderCallable = Callable[[], Awaitable[Any]]


class TimeoutMiddleware:
    """Enforce bounded provider call duration."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    async def __call__(self, call: ProviderCallable) -> Any:
        try:
            return await asyncio.wait_for(call(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError("provider timeout") from exc


class RetryMiddleware:
    """Retry retryable errors without modifying failover semantics."""

    def __init__(self, retries: int = 2):
        self.retries = retries

    async def __call__(self, call: ProviderCallable) -> Any:
        attempts = 0
        while True:
            try:
                return await call()
            except LLMRetryableError:
                if attempts >= self.retries:
                    raise
                attempts += 1


class CircuitBreakerMiddleware:
    """Block calls when provider circuit is open."""

    def __init__(self, health_registry: ProviderHealthRegistry, provider_name: str):
        self.health_registry = health_registry
        self.provider_name = provider_name

    async def __call__(self, call: ProviderCallable) -> Any:
        if self.health_registry.is_circuit_open(self.provider_name):
            raise LLMCircuitOpenError(f"circuit open: {self.provider_name}")
        return await call()


class CostGuardMiddleware:
    """Enforce cloud budget limits before provider request."""

    def __init__(self, cost_guard: CostGuard, agent_id: str, task_id: str | None = None):
        self.cost_guard = cost_guard
        self.agent_id = agent_id
        self.task_id = task_id

    async def __call__(self, call: ProviderCallable) -> Any:
        await self.cost_guard.check_can_execute(agent_id=self.agent_id, task_id=self.task_id)
        return await call()


class LoggingMiddleware:
    """Emit structured provider-call timing logs."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    async def __call__(self, call: ProviderCallable) -> Any:
        start = time.perf_counter()
        try:
            result = await call()
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.info("Provider call success provider=%s latency_ms=%s", self.provider_name, latency_ms)
            return result
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.info("Provider call failure provider=%s latency_ms=%s", self.provider_name, latency_ms)
            raise


class MiddlewareStack:
    """Apply middleware sequence to a provider callable."""

    def __init__(self, middlewares: list[Callable[[ProviderCallable], Awaitable[Any]]]):
        self.middlewares = middlewares

    async def execute(self, call: ProviderCallable) -> Any:
        wrapped = call
        for middleware in reversed(self.middlewares):
            current = wrapped

            async def wrapped_fn(
                mw: Callable[[ProviderCallable], Awaitable[Any]] = middleware,
                fn: ProviderCallable = current,
            ) -> Any:
                return await mw(fn)

            wrapped = wrapped_fn
        return await wrapped()
