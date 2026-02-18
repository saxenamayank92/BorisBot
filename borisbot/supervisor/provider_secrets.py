"""Persistent provider API secret helpers with masked responses."""

from __future__ import annotations

import json
from pathlib import Path

from borisbot.supervisor.profile_config import ALLOWED_PROVIDERS

SECRETS_PATH = Path.home() / ".borisbot" / "provider_secrets.json"
ALLOWED_SECRET_PROVIDERS = {p for p in ALLOWED_PROVIDERS if p != "ollama"}


def _load_raw(path: Path = SECRETS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for provider, key in raw.items():
        name = str(provider).strip().lower()
        if name not in ALLOWED_SECRET_PROVIDERS:
            continue
        key_value = str(key).strip()
        if key_value:
            out[name] = key_value
    return out


def get_secret_status(path: Path = SECRETS_PATH) -> dict[str, dict[str, str | bool]]:
    raw = _load_raw(path)
    status: dict[str, dict[str, str | bool]] = {}
    for provider in sorted(ALLOWED_SECRET_PROVIDERS):
        key = raw.get(provider, "")
        status[provider] = {
            "configured": bool(key),
            "masked": _mask_key(key),
        }
    return status


def get_provider_secret(provider: str, path: Path = SECRETS_PATH) -> str:
    """Return raw API key for provider or empty string when missing."""
    name = str(provider).strip().lower()
    if name not in ALLOWED_SECRET_PROVIDERS:
        return ""
    return _load_raw(path).get(name, "")


def set_provider_secret(provider: str, api_key: str, path: Path = SECRETS_PATH) -> dict[str, dict[str, str | bool]]:
    name = str(provider).strip().lower()
    if name not in ALLOWED_SECRET_PROVIDERS:
        raise ValueError(f"unsupported provider: {name}")
    key = str(api_key).strip()
    raw = _load_raw(path)
    if key:
        raw[name] = key
    else:
        raw.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return get_secret_status(path=path)


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"
