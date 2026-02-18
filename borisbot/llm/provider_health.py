"""In-memory provider health state with circuit breaker windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque

FAILURE_WINDOW_SECONDS = 60
FAILURE_THRESHOLD = 5
CIRCUIT_OPEN_SECONDS = 30
PROVIDER_STATE_UNKNOWN = "unknown"
PROVIDER_STATE_HEALTHY = "healthy"
PROVIDER_STATE_UNHEALTHY = "unhealthy"
PROVIDER_STATE_DEGRADED = "degraded"
RUNTIME_UNHEALTHY_FAILURE_THRESHOLD = 2


@dataclass
class ProviderHealth:
    """Mutable provider health state used by circuit and metrics systems."""

    success_count: int = 0
    failure_count: int = 0
    last_failure_at: str | None = None
    last_success_at: str | None = None
    circuit_open_until: str | None = None
    average_latency_ms: int = 0
    _latency_samples: int = 0
    failure_window: Deque[datetime] = field(default_factory=deque)
    half_open_trial_allowed: bool = True
    state: str = PROVIDER_STATE_UNKNOWN
    consecutive_runtime_failures: int = 0
    last_state_reason: str | None = None


class ProviderHealthRegistry:
    """Registry of per-provider health and deterministic circuit state."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderHealth] = {}

    def _get(self, provider_name: str) -> ProviderHealth:
        state = self._providers.get(provider_name)
        if state is None:
            state = ProviderHealth()
            self._providers[provider_name] = state
        return state

    def mark_success(self, provider_name: str, latency_ms: int) -> None:
        state = self._get(provider_name)
        now = datetime.utcnow()
        state.success_count += 1
        state.last_success_at = now.isoformat()
        state.failure_window.clear()
        state.circuit_open_until = None
        state.half_open_trial_allowed = True
        state.consecutive_runtime_failures = 0
        state.state = PROVIDER_STATE_HEALTHY
        state.last_state_reason = "runtime_success"

        state._latency_samples += 1
        if state._latency_samples == 1:
            state.average_latency_ms = int(latency_ms)
        else:
            total = state.average_latency_ms * (state._latency_samples - 1) + int(latency_ms)
            state.average_latency_ms = int(round(total / state._latency_samples))

    def mark_failure(self, provider_name: str) -> None:
        self.mark_runtime_failure(provider_name)

    def mark_probe_success(self, provider_name: str) -> None:
        """Transition provider state to healthy after explicit probe success."""
        state = self._get(provider_name)
        state.state = PROVIDER_STATE_HEALTHY
        state.consecutive_runtime_failures = 0
        state.last_state_reason = "startup_probe_success"

    def mark_probe_failure(self, provider_name: str) -> None:
        """Transition provider state to unhealthy after explicit probe failure."""
        state = self._get(provider_name)
        now = datetime.utcnow()
        state.failure_count += 1
        state.last_failure_at = now.isoformat()
        state.state = PROVIDER_STATE_UNHEALTHY
        state.last_state_reason = "startup_probe_failure"
        state.consecutive_runtime_failures = max(state.consecutive_runtime_failures, 1)

    def mark_runtime_failure(self, provider_name: str) -> None:
        """Track runtime failure with deterministic state transitions."""
        state = self._get(provider_name)
        now = datetime.utcnow()
        state.failure_count += 1
        state.last_failure_at = now.isoformat()
        state.consecutive_runtime_failures += 1

        cutoff = now - timedelta(seconds=FAILURE_WINDOW_SECONDS)
        state.failure_window.append(now)
        while state.failure_window and state.failure_window[0] < cutoff:
            state.failure_window.popleft()

        if len(state.failure_window) >= FAILURE_THRESHOLD:
            state.circuit_open_until = (now + timedelta(seconds=CIRCUIT_OPEN_SECONDS)).isoformat()
            state.half_open_trial_allowed = True

        if state.consecutive_runtime_failures >= RUNTIME_UNHEALTHY_FAILURE_THRESHOLD:
            state.state = PROVIDER_STATE_UNHEALTHY
            state.last_state_reason = "runtime_failure_threshold"
        else:
            state.state = PROVIDER_STATE_DEGRADED
            state.last_state_reason = "runtime_failure"

    def is_circuit_open(self, provider_name: str) -> bool:
        state = self._get(provider_name)
        if not state.circuit_open_until:
            return False

        now = datetime.utcnow()
        open_until = datetime.fromisoformat(state.circuit_open_until)
        if now < open_until:
            return True

        if state.half_open_trial_allowed:
            state.half_open_trial_allowed = False
            return False
        return True

    def get_provider_state(self, provider_name: str) -> str:
        """Return explicit provider health state string."""
        state = self._get(provider_name)
        return state.state

    def is_provider_usable(self, provider_name: str) -> bool:
        """Return True when provider can be selected for normal runtime calls."""
        state = self._get(provider_name)
        return state.state in {
            PROVIDER_STATE_UNKNOWN,
            PROVIDER_STATE_HEALTHY,
            PROVIDER_STATE_DEGRADED,
        }

    def get_snapshot(self) -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for provider_name, state in self._providers.items():
            snapshot[provider_name] = {
                "state": state.state,
                "circuit_open": self.is_circuit_open(provider_name),
                "successes": state.success_count,
                "failures": state.failure_count,
                "avg_latency_ms": state.average_latency_ms,
                "last_success_at": state.last_success_at,
                "last_failure_at": state.last_failure_at,
                "circuit_open_until": state.circuit_open_until,
                "consecutive_runtime_failures": state.consecutive_runtime_failures,
                "last_state_reason": state.last_state_reason,
            }
        return snapshot


_GLOBAL_PROVIDER_HEALTH_REGISTRY = ProviderHealthRegistry()


def get_provider_health_registry() -> ProviderHealthRegistry:
    """Return singleton provider health registry used across engine/API."""
    return _GLOBAL_PROVIDER_HEALTH_REGISTRY
