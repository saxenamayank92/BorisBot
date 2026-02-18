"""Ollama health probing helpers for startup/runtime supervision."""

from __future__ import annotations

import httpx

from borisbot.llm.provider_health import ProviderHealthRegistry


async def probe_ollama(
    *,
    base_url: str = "http://127.0.0.1:11434",
    model_name: str = "llama3.2:3b",
    timeout_seconds: float = 3.0,
) -> bool:
    """Return True when Ollama tags+tiny-generate probes both succeed."""
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tags_resp = await client.get(f"{base_url}/api/tags")
        if tags_resp.status_code != 200:
            return False

        generate_resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1},
            },
        )
        if generate_resp.status_code != 200:
            return False
        payload = generate_resp.json()
        return isinstance(payload, dict) and "response" in payload


async def startup_mark_ollama_health(
    registry: ProviderHealthRegistry,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model_name: str = "llama3.2:3b",
    provider_name: str = "ollama",
) -> bool:
    """Probe Ollama and update provider state machine deterministically."""
    try:
        healthy = await probe_ollama(base_url=base_url, model_name=model_name)
    except Exception:
        healthy = False

    if healthy:
        registry.mark_probe_success(provider_name)
    else:
        registry.mark_probe_failure(provider_name)
    return healthy

