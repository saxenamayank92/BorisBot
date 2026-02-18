"""Persistent runtime profile configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROFILE_PATH = Path.home() / ".borisbot" / "profile.json"
ALLOWED_PROVIDERS = {"ollama", "openai", "anthropic", "google", "azure"}
MAX_PROVIDER_CHAIN = 5
PROFILE_SCHEMA_VERSION = "profile.v2"


def _default_provider_settings() -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for provider in sorted(ALLOWED_PROVIDERS):
        settings[provider] = {"enabled": provider == "ollama", "model_name": ""}
    return settings


def default_profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "agent_name": "default",
        "primary_provider": "ollama",
        "provider_chain": ["ollama"],
        "model_name": "llama3.2:3b",
        "provider_settings": _default_provider_settings(),
    }


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate profile schema and provider chain constraints."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be object")
    schema_version = profile.get("schema_version", PROFILE_SCHEMA_VERSION)
    if schema_version not in {"profile.v1", PROFILE_SCHEMA_VERSION}:
        raise ValueError("unsupported profile schema_version")
    agent_name = str(profile.get("agent_name", "default")).strip() or "default"
    model_name = str(profile.get("model_name", "llama3.2:3b")).strip() or "llama3.2:3b"
    provider_chain = profile.get("provider_chain", ["ollama"])
    if not isinstance(provider_chain, list) or not provider_chain:
        raise ValueError("provider_chain must be non-empty list")
    normalized_chain: list[str] = []
    for provider in provider_chain:
        provider_name = str(provider).strip().lower()
        if provider_name not in ALLOWED_PROVIDERS:
            raise ValueError(f"unsupported provider: {provider_name}")
        if provider_name not in normalized_chain:
            normalized_chain.append(provider_name)
    if len(normalized_chain) > MAX_PROVIDER_CHAIN:
        raise ValueError(f"provider_chain exceeds max {MAX_PROVIDER_CHAIN}")
    primary_provider = str(profile.get("primary_provider", normalized_chain[0])).strip().lower()
    if primary_provider not in normalized_chain:
        raise ValueError("primary_provider must exist in provider_chain")
    provider_settings_raw = profile.get("provider_settings", {})
    if provider_settings_raw is None:
        provider_settings_raw = {}
    if not isinstance(provider_settings_raw, dict):
        raise ValueError("provider_settings must be object")
    provider_settings = _default_provider_settings()
    for provider_name, setting in provider_settings_raw.items():
        name = str(provider_name).strip().lower()
        if name not in ALLOWED_PROVIDERS:
            continue
        if not isinstance(setting, dict):
            continue
        enabled = bool(setting.get("enabled", provider_settings[name]["enabled"]))
        model_name_value = str(setting.get("model_name", "")).strip()
        provider_settings[name] = {
            "enabled": enabled,
            "model_name": model_name_value,
        }
    for provider in normalized_chain:
        if provider != "ollama":
            provider_settings[provider]["enabled"] = True
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "agent_name": agent_name,
        "primary_provider": primary_provider,
        "provider_chain": normalized_chain,
        "model_name": model_name,
        "provider_settings": provider_settings,
    }


def load_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    """Load profile from disk or return defaults."""
    if not path.exists():
        return default_profile()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_profile()
    if not isinstance(raw, dict):
        return default_profile()
    try:
        return validate_profile(raw)
    except ValueError:
        return default_profile()


def save_profile(profile: dict[str, Any], path: Path = PROFILE_PATH) -> dict[str, Any]:
    """Validate and persist profile to disk."""
    validated = validate_profile(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated
