"""Local guided web UI for common BorisBot reliability workflows."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Any
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit

import httpx

from borisbot.llm.errors import LLMInvalidOutputError
from borisbot.guide.chat_history_store import (
    append_chat_message,
    clear_chat_roles,
    clear_chat_history,
    load_chat_history,
)
from borisbot.llm.planner_contract import parse_planner_output
from borisbot.llm.planner_validator import validate_and_convert_plan
from borisbot.llm.cost_guard import CostGuard
from borisbot.llm.provider_health import get_provider_health_registry
from borisbot.supervisor.database import get_db
from borisbot.supervisor.heartbeat_runtime import read_heartbeat_snapshot
from borisbot.supervisor.profile_config import load_profile, save_profile
from borisbot.supervisor.provider_secrets import (
    get_provider_secret,
    get_secret_status,
    set_provider_secret,
)
from borisbot.supervisor.tool_permissions import (
    ALLOWED_DECISIONS,
    ALLOWED_TOOLS,
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_PROMPT,
    TOOL_ASSISTANT,
    TOOL_BROWSER,
    TOOL_FILESYSTEM,
    TOOL_SCHEDULER,
    TOOL_SHELL,
    TOOL_WEB_FETCH,
    get_agent_permission_matrix_sync,
    get_agent_tool_permission_sync,
    set_agent_tool_permission_sync,
)


def _resolve_ollama_install_command(
    platform_name: str,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return OS-aware install command for Ollama."""
    if platform_name.startswith("darwin"):
        if which("brew"):
            return ["brew", "install", "ollama"]
        raise ValueError("Homebrew not found. Install Homebrew first: https://brew.sh")
    if platform_name.startswith("linux"):
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if platform_name.startswith("win") or os.name == "nt":
        if which("winget"):
            return ["winget", "install", "-e", "--id", "Ollama.Ollama"]
        raise ValueError(
            "winget not found. Install Ollama manually from https://ollama.com/download/windows"
        )
    raise ValueError(f"Unsupported platform for automated Ollama install: {platform_name}")


def _resolve_ollama_start_command(
    platform_name: str,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return OS-aware command to start Ollama service/runtime."""
    if platform_name.startswith("darwin"):
        if which("brew"):
            return ["brew", "services", "start", "ollama"]
        return ["ollama", "serve"]
    if platform_name.startswith("linux"):
        if which("systemctl"):
            return ["systemctl", "--user", "start", "ollama"]
        return ["ollama", "serve"]
    if platform_name.startswith("win") or os.name == "nt":
        return ["ollama", "serve"]
    return ["ollama", "serve"]


def _build_ollama_setup_plan(
    model_name: str,
    platform_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> dict:
    """Return cross-platform Ollama setup plan for GUI onboarding."""
    model = (model_name or "").strip() or "llama3.2:3b"
    platform_value = (platform_name or sys.platform).strip().lower()
    install_command: list[str] | None = None
    install_error = ""
    try:
        install_command = _resolve_ollama_install_command(platform_value, which=which)
    except ValueError as exc:
        install_error = str(exc)
    start_command = _resolve_ollama_start_command(platform_value, which=which)
    plan = {
        "platform": platform_value,
        "model_name": model,
        "install_command": install_command,
        "start_command": start_command,
        "pull_command": ["ollama", "pull", model],
        "manual_download_url": "https://ollama.com/download",
        "install_error": install_error,
    }
    return plan


GUIDE_STATE_DIR = Path.home() / ".borisbot"
GUIDE_WIZARD_STATE_FILE = GUIDE_STATE_DIR / "guide_wizard_state.json"

WIZARD_STEP_DEFS = [
    {"step_id": "docker_ready", "label": "Docker Running"},
    {"step_id": "llm_ready", "label": "LLM Ready"},
    {"step_id": "provider_ready", "label": "Primary Provider Reachable"},
    {"step_id": "permissions_reviewed", "label": "Permissions Reviewed"},
    {"step_id": "first_preview", "label": "First Dry-Run Preview"},
]


ACTION_TOOL_MAP = {
    "docker_info": TOOL_SHELL,
    "cleanup_sessions": TOOL_BROWSER,
    "session_status": TOOL_SHELL,
    "budget_status": TOOL_SHELL,
    "budget_set": TOOL_SHELL,
    "doctor": TOOL_SHELL,
    "ollama_check": TOOL_SHELL,
    "ollama_install": TOOL_SHELL,
    "ollama_start": TOOL_SHELL,
    "ollama_pull": TOOL_SHELL,
    "llm_setup": TOOL_SHELL,
    "bootstrap_setup": TOOL_SHELL,
    "verify": TOOL_SHELL,
    "analyze": TOOL_FILESYSTEM,
    "lint": TOOL_FILESYSTEM,
    "replay": TOOL_BROWSER,
    "release_check": TOOL_SHELL,
    "release_check_json": TOOL_SHELL,
    "record": TOOL_BROWSER,
    "policy_safe_local": TOOL_SHELL,
    "policy_web_readonly": TOOL_SHELL,
    "policy_automation": TOOL_SHELL,
    "policy_apply": TOOL_SHELL,
    "verify_agent_logic": TOOL_SHELL,
}


def required_tool_for_action(action: str) -> str | None:
    """Return required tool gate for guide action if any."""
    return ACTION_TOOL_MAP.get(action)


PLANNER_ACTION_TOOL_MAP = {
    "navigate": TOOL_BROWSER,
    "click": TOOL_BROWSER,
    "type": TOOL_BROWSER,
    "wait_for_url": TOOL_BROWSER,
    "get_text": TOOL_BROWSER,
    "get_title": TOOL_BROWSER,
    "read_file": TOOL_FILESYSTEM,
    "write_file": TOOL_FILESYSTEM,
    "list_dir": TOOL_FILESYSTEM,
    "run_shell": TOOL_SHELL,
    "web_fetch": TOOL_WEB_FETCH,
    "web_search": TOOL_WEB_FETCH,
    "schedule": TOOL_SCHEDULER,
}

ESTIMATED_PRICING_USD_PER_1K = {
    "openai": {"input": 0.005, "output": 0.015},
    "anthropic": {"input": 0.008, "output": 0.024},
    "google": {"input": 0.0035, "output": 0.0105},
    "azure": {"input": 0.005, "output": 0.015},
}

PROVIDER_MAX_RETRIES = 1
PROVIDER_RETRY_BACKOFF_SECONDS = 0.35


def _estimate_tokens(text: str) -> int:
    """Estimate token count deterministically from raw text length."""
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))


def _is_retryable_provider_error(message: str) -> bool:
    text = str(message).strip().lower()
    if not text:
        return False
    retry_tokens = (
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    return any(token in text for token in retry_tokens)


def _extract_required_tools_from_plan(plan: dict) -> list[str]:
    """Derive required tool set from planner proposed actions."""
    tools: set[str] = set()
    actions = plan.get("proposed_actions", [])
    if not isinstance(actions, list):
        return []
    for action_obj in actions:
        if not isinstance(action_obj, dict):
            continue
        action_name = action_obj.get("action")
        if not isinstance(action_name, str):
            continue
        tool = PLANNER_ACTION_TOOL_MAP.get(action_name.strip())
        if tool:
            tools.add(tool)
    return sorted(tools)


def _build_planner_prompt(user_intent: str) -> str:
    """Build strict planner prompt that must return planner.v1 JSON only."""
    return (
        "You are a browser automation planning engine.\n"
        "Return strict JSON only with exact schema:\n"
        "{"
        "\"planner_schema_version\":\"planner.v1\","
        "\"intent\":\"...\","
        "\"proposed_actions\":["
        "{\"action\":\"navigate\",\"target\":\"<url>\",\"input\":\"\"},"
        "{\"action\":\"click\",\"target\":\"<css_selector>\",\"input\":\"\"},"
        "{\"action\":\"type\",\"target\":\"<css_selector>\",\"input\":\"<text>\"},"
        "{\"action\":\"wait_for_url\",\"target\":\"<url>\",\"input\":\"\"},"
        "{\"action\":\"get_text\",\"target\":\"<css_selector>\",\"input\":\"\"},"
        "{\"action\":\"get_title\",\"target\":\"\",\"input\":\"\"}"
        "]"
        "}\n"
        "Supported actions: navigate, click, type, wait_for_url, get_text, get_title.\n"
        "No markdown. No extra keys. No commentary.\n"
        f"User request: {user_intent.strip()}"
    )


def _estimate_preview_cost_usd(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        return 0.0
    rates = ESTIMATED_PRICING_USD_PER_1K.get(provider)
    if not rates:
        return 0.0
    input_cost = (input_tokens / 1000.0) * float(rates["input"])
    output_cost = (output_tokens / 1000.0) * float(rates["output"])
    return round(input_cost + output_cost, 6)


def _build_live_cost_estimate(
    provider_name: str,
    planner_prompt: str,
    assistant_prompt: str,
) -> dict:
    """Build deterministic planner/assistant cost estimates from prompt text."""
    planner_input = _estimate_tokens(planner_prompt)
    assistant_input = _estimate_tokens(assistant_prompt)
    planner_output = max(64, int(round(planner_input * 0.8))) if planner_input else 0
    assistant_output = max(64, int(round(assistant_input * 1.1))) if assistant_input else 0
    planner_cost = _estimate_preview_cost_usd(provider_name, planner_input, planner_output)
    assistant_cost = _estimate_preview_cost_usd(provider_name, assistant_input, assistant_output)
    return {
        "provider_name": provider_name,
        "planner": {
            "input_tokens": planner_input,
            "output_tokens": planner_output,
            "total_tokens": planner_input + planner_output,
            "cost_estimate_usd": planner_cost,
        },
        "assistant": {
            "input_tokens": assistant_input,
            "output_tokens": assistant_output,
            "total_tokens": assistant_input + assistant_output,
            "cost_estimate_usd": assistant_cost,
        },
    }


def _load_budget_snapshot(agent_id: str) -> dict:
    guard = CostGuard()
    return asyncio.run(guard.get_budget_status(agent_id))


def _resolve_provider_chain(requested_provider: str) -> list[str]:
    profile = load_profile()
    raw_chain = profile.get("provider_chain", ["ollama"])
    chain: list[str] = []
    if isinstance(raw_chain, list):
        for item in raw_chain:
            name = str(item).strip().lower()
            if name and name not in chain:
                chain.append(name)
    preferred = str(requested_provider).strip().lower()
    if preferred and preferred in chain:
        chain.remove(preferred)
        chain.insert(0, preferred)
    elif preferred:
        chain.insert(0, preferred)
    return chain[:5] or ["ollama"]


def _resolve_model_for_provider(provider_name: str, requested_model: str) -> str:
    """Return best model for provider using explicit request then profile settings."""
    model = str(requested_model).strip()
    if model:
        return model
    profile = load_profile()
    provider_settings = profile.get("provider_settings", {})
    provider = str(provider_name).strip().lower()
    if isinstance(provider_settings, dict):
        row = provider_settings.get(provider, {})
        if isinstance(row, dict):
            candidate = str(row.get("model_name", "")).strip()
            if candidate:
                return candidate
    fallback = str(profile.get("model_name", "llama3.2:3b")).strip()
    return fallback or "llama3.2:3b"


def _provider_is_usable(provider_name: str) -> tuple[bool, str]:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        if shutil.which("ollama") is None:
            return False, "ollama_not_installed"
        return True, ""
    if provider == "azure":
        status = get_secret_status().get(provider, {})
        if not bool(status.get("configured", False)):
            return False, "api_key_missing"
        if not os.getenv("BORISBOT_AZURE_OPENAI_ENDPOINT", "").strip():
            return False, "azure_endpoint_missing"
        return True, ""
    status = get_secret_status().get(provider, {})
    if not bool(status.get("configured", False)):
        return False, "api_key_missing"
    return True, ""


def _generate_plan_raw_with_ollama(user_intent: str, model_name: str) -> str:
    """Call Ollama generate endpoint and return raw planner output text."""
    if shutil.which("ollama") is None:
        raise ValueError("Ollama is not installed. Install it from Step 1 first.")
    prompt = _build_planner_prompt(user_intent)
    response = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=20.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Ollama generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Ollama response payload invalid")
    output = payload.get("response")
    if not isinstance(output, str):
        raise ValueError("Ollama response missing text output")
    return output


def _generate_plan_raw_with_openai(user_intent: str, model_name: str) -> str:
    """Call OpenAI chat completions endpoint and return text output."""
    api_key = get_provider_secret("openai")
    if not api_key:
        raise ValueError("OpenAI API key missing. Configure it in Provider Onboarding.")
    prompt = _build_planner_prompt(user_intent)
    endpoint = os.getenv("BORISBOT_OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"OpenAI generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response payload invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI response choice invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI response message invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI response content invalid")
    return content


def _generate_plan_raw_with_anthropic(user_intent: str, model_name: str) -> str:
    """Call Anthropic messages endpoint and return text output."""
    api_key = get_provider_secret("anthropic")
    if not api_key:
        raise ValueError("Anthropic API key missing. Configure it in Provider Onboarding.")
    prompt = _build_planner_prompt(user_intent)
    endpoint = os.getenv("BORISBOT_ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model_name,
            "max_tokens": 1200,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Anthropic generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Anthropic response payload invalid")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Anthropic response content invalid")
    first = content[0]
    if not isinstance(first, dict):
        raise ValueError("Anthropic response content item invalid")
    text = first.get("text")
    if not isinstance(text, str):
        raise ValueError("Anthropic response text missing")
    return text


def _generate_plan_raw_with_google(user_intent: str, model_name: str) -> str:
    """Call Google Gemini generateContent endpoint and return text output."""
    api_key = get_provider_secret("google")
    if not api_key:
        raise ValueError("Google API key missing. Configure it in Provider Onboarding.")
    prompt = _build_planner_prompt(user_intent)
    endpoint = os.getenv("BORISBOT_GOOGLE_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1beta/models/{model_name}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Google generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Google response payload invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Google response missing candidates")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("Google candidate invalid")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ValueError("Google content invalid")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Google content parts missing")
    part0 = parts[0]
    if not isinstance(part0, dict):
        raise ValueError("Google content part invalid")
    text = part0.get("text")
    if not isinstance(text, str):
        raise ValueError("Google response text missing")
    return text


def _generate_plan_raw_with_azure(user_intent: str, model_name: str) -> str:
    """Call Azure OpenAI chat completions endpoint and return text output."""
    api_key = get_provider_secret("azure")
    if not api_key:
        raise ValueError("Azure API key missing. Configure it in Provider Onboarding.")
    endpoint = os.getenv("BORISBOT_AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("Azure endpoint missing. Set BORISBOT_AZURE_OPENAI_ENDPOINT.")
    api_version = os.getenv("BORISBOT_AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
    deployment = model_name
    prompt = _build_planner_prompt(user_intent)
    response = httpx.post(
        f"{endpoint}/openai/deployments/{deployment}/chat/completions",
        params={"api-version": api_version},
        headers={"api-key": api_key, "content-type": "application/json"},
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Azure generate failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Azure response payload invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Azure response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Azure response choice invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Azure response message invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Azure response content invalid")
    return content


def _probe_provider_connection(provider_name: str, model_name: str) -> tuple[bool, str]:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False, f"ollama probe failed: HTTP {response.status_code}"
        return True, "ollama reachable"
    if provider == "openai":
        api_key = get_provider_secret("openai")
        if not api_key:
            return False, "OpenAI API key missing"
        endpoint = os.getenv("BORISBOT_OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
        response = httpx.get(
            f"{endpoint}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False, f"openai probe failed: HTTP {response.status_code}"
        return True, f"openai reachable ({model_name})"
    if provider == "anthropic":
        api_key = get_provider_secret("anthropic")
        if not api_key:
            return False, "Anthropic API key missing"
        endpoint = os.getenv("BORISBOT_ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
        response = httpx.get(
            f"{endpoint}/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False, f"anthropic probe failed: HTTP {response.status_code}"
        return True, f"anthropic reachable ({model_name})"
    if provider == "google":
        api_key = get_provider_secret("google")
        if not api_key:
            return False, "Google API key missing"
        endpoint = os.getenv("BORISBOT_GOOGLE_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
        response = httpx.get(
            f"{endpoint}/v1beta/models",
            params={"key": api_key},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False, f"google probe failed: HTTP {response.status_code}"
        return True, f"google reachable ({model_name})"
    if provider == "azure":
        api_key = get_provider_secret("azure")
        if not api_key:
            return False, "Azure API key missing"
        endpoint = os.getenv("BORISBOT_AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        if not endpoint:
            return False, "Azure endpoint missing (BORISBOT_AZURE_OPENAI_ENDPOINT)"
        api_version = os.getenv("BORISBOT_AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
        response = httpx.get(
            f"{endpoint}/openai/deployments",
            params={"api-version": api_version},
            headers={"api-key": api_key},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False, f"azure probe failed: HTTP {response.status_code}"
        return True, f"azure reachable ({model_name})"
    return False, f"provider '{provider}' probe not implemented"


def _generate_plan_raw_with_provider(provider_name: str, user_intent: str, model_name: str) -> str:
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        return _generate_plan_raw_with_ollama(user_intent, model_name=model_name)
    if provider == "openai":
        return _generate_plan_raw_with_openai(user_intent, model_name=model_name)
    if provider == "anthropic":
        return _generate_plan_raw_with_anthropic(user_intent, model_name=model_name)
    if provider == "google":
        return _generate_plan_raw_with_google(user_intent, model_name=model_name)
    if provider == "azure":
        return _generate_plan_raw_with_azure(user_intent, model_name=model_name)
    raise ValueError(f"Provider '{provider}' planner transport is not enabled in this build")


def _generate_chat_raw_with_ollama(prompt: str, model_name: str) -> str:
    """Call Ollama generate endpoint for freeform assistant chat."""
    if shutil.which("ollama") is None:
        raise ValueError("Ollama is not installed. Install it from Step 1 first.")
    response = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Ollama chat failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Ollama response payload invalid")
    output = payload.get("response")
    if not isinstance(output, str):
        raise ValueError("Ollama response missing text output")
    return output


def _generate_chat_raw_with_openai(prompt: str, model_name: str) -> str:
    """Call OpenAI chat completions for freeform assistant chat."""
    api_key = get_provider_secret("openai")
    if not api_key:
        raise ValueError("OpenAI API key missing. Configure it in Provider Onboarding.")
    endpoint = os.getenv("BORISBOT_OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "temperature": 0.2,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"OpenAI chat failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response payload invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI response choice invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI response message invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI response content invalid")
    return content


def _generate_chat_raw_with_anthropic(prompt: str, model_name: str) -> str:
    """Call Anthropic messages endpoint for freeform assistant chat."""
    api_key = get_provider_secret("anthropic")
    if not api_key:
        raise ValueError("Anthropic API key missing. Configure it in Provider Onboarding.")
    endpoint = os.getenv("BORISBOT_ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model_name,
            "max_tokens": 1200,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt.strip()}],
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Anthropic chat failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Anthropic response payload invalid")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Anthropic response content invalid")
    first = content[0]
    if not isinstance(first, dict):
        raise ValueError("Anthropic response content item invalid")
    text = first.get("text")
    if not isinstance(text, str):
        raise ValueError("Anthropic response text missing")
    return text


def _generate_chat_raw_with_google(prompt: str, model_name: str) -> str:
    """Call Google Gemini generateContent endpoint for freeform assistant chat."""
    api_key = get_provider_secret("google")
    if not api_key:
        raise ValueError("Google API key missing. Configure it in Provider Onboarding.")
    endpoint = os.getenv("BORISBOT_GOOGLE_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    response = httpx.post(
        f"{endpoint}/v1beta/models/{model_name}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Google chat failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Google response payload invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Google response missing candidates")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("Google candidate invalid")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ValueError("Google content invalid")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Google content parts missing")
    part0 = parts[0]
    if not isinstance(part0, dict):
        raise ValueError("Google content part invalid")
    text = part0.get("text")
    if not isinstance(text, str):
        raise ValueError("Google response text missing")
    return text


def _generate_chat_raw_with_azure(prompt: str, model_name: str) -> str:
    """Call Azure OpenAI chat completions for freeform assistant chat."""
    api_key = get_provider_secret("azure")
    if not api_key:
        raise ValueError("Azure API key missing. Configure it in Provider Onboarding.")
    endpoint = os.getenv("BORISBOT_AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("Azure endpoint missing. Set BORISBOT_AZURE_OPENAI_ENDPOINT.")
    api_version = os.getenv("BORISBOT_AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
    response = httpx.post(
        f"{endpoint}/openai/deployments/{model_name}/chat/completions",
        params={"api-version": api_version},
        headers={"api-key": api_key, "content-type": "application/json"},
        json={
            "messages": [{"role": "user", "content": prompt.strip()}],
            "temperature": 0.2,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise ValueError(f"Azure chat failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Azure response payload invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Azure response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Azure response choice invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Azure response message invalid")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Azure response content invalid")
    return content


def _generate_chat_raw_with_provider(provider_name: str, prompt: str, model_name: str) -> str:
    """Route assistant chat generation through configured provider transport."""
    provider = str(provider_name).strip().lower()
    if provider == "ollama":
        return _generate_chat_raw_with_ollama(prompt, model_name=model_name)
    if provider == "openai":
        return _generate_chat_raw_with_openai(prompt, model_name=model_name)
    if provider == "anthropic":
        return _generate_chat_raw_with_anthropic(prompt, model_name=model_name)
    if provider == "google":
        return _generate_chat_raw_with_google(prompt, model_name=model_name)
    if provider == "azure":
        return _generate_chat_raw_with_azure(prompt, model_name=model_name)
    raise ValueError(f"Provider '{provider}' chat transport is not enabled in this build")


def _build_dry_run_preview(intent: str, agent_id: str, model_name: str, provider_name: str) -> dict:
    """Build dry-run planner preview with strict schema + permission requirements."""
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Plan prompt cannot be empty.")
    budget_status = _load_budget_snapshot(agent_id)
    if bool(budget_status.get("blocked", False)):
        return {
            "status": "failed",
            "error_class": "cost_guard",
            "error_code": "BUDGET_BLOCKED",
            "message": "Planner calls are blocked by budget limits.",
            "budget": budget_status,
        }
    provider_chain = _resolve_provider_chain(provider_name)
    attempts: list[dict[str, str]] = []
    raw_output = ""
    selected_provider = ""
    for provider in provider_chain:
        usable, reason = _provider_is_usable(provider)
        if not usable:
            attempts.append({"provider": provider, "status": "skipped", "reason": reason})
            continue
        retries_left = PROVIDER_MAX_RETRIES
        while True:
            try:
                raw_output = _generate_plan_raw_with_provider(provider, intent, model_name=model_name)
                selected_provider = provider
                attempts.append({"provider": provider, "status": "ok"})
                break
            except ValueError as exc:
                reason_text = str(exc)
                if retries_left > 0 and _is_retryable_provider_error(reason_text):
                    attempts.append(
                        {
                            "provider": provider,
                            "status": "retrying",
                            "reason": reason_text,
                            "retries_left": str(retries_left),
                        }
                    )
                    retries_left -= 1
                    time.sleep(PROVIDER_RETRY_BACKOFF_SECONDS)
                    continue
                attempts.append({"provider": provider, "status": "failed", "reason": reason_text})
                break
        if raw_output:
            break
    if not raw_output:
        last_error = attempts[-1]["reason"] if attempts else "unknown"
        return {
            "status": "failed",
            "error_class": "llm_provider",
            "error_code": "LLM_PROVIDER_UNHEALTHY",
            "message": f"Planning failed: {last_error}",
            "attempts": attempts,
            "provider_chain": provider_chain,
            "budget": budget_status,
        }
    try:
        parsed = parse_planner_output(raw_output)
    except LLMInvalidOutputError as exc:
        return {
            "status": "failed",
            "error_class": exc.error_class,
            "error_code": exc.error_code,
            "message": str(exc),
            "raw_output": raw_output,
        }

    required_tools = _extract_required_tools_from_plan(parsed)
    try:
        preview_commands = validate_and_convert_plan(parsed)
    except LLMInvalidOutputError as exc:
        return {
            "status": "failed",
            "error_class": exc.error_class,
            "error_code": exc.error_code,
            "message": str(exc),
            "planner_output": parsed,
            "raw_output": raw_output,
        }
    required_permissions = [
        {
            "tool_name": tool,
            "decision": get_agent_tool_permission_sync(agent_id, tool),
        }
        for tool in required_tools
    ]

    prompt_tokens = _estimate_tokens(_build_planner_prompt(intent))
    completion_tokens = _estimate_tokens(raw_output)
    estimated_cost = _estimate_preview_cost_usd(
        selected_provider,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )
    return {
        "status": "ok",
        "planner_output": parsed,
        "validated_commands": preview_commands,
        "required_permissions": required_permissions,
        "token_estimate": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_estimate_usd": estimated_cost,
        "provider_name": selected_provider,
        "provider_attempts": attempts,
        "provider_chain": provider_chain,
        "raw_output": raw_output,
        "budget": budget_status,
    }


def _build_assistant_response(prompt: str, agent_id: str, model_name: str, provider_name: str) -> dict:
    """Build freeform assistant response with provider fallback and budget checks."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Assistant prompt cannot be empty.")
    budget_status = _load_budget_snapshot(agent_id)
    if bool(budget_status.get("blocked", False)):
        return {
            "status": "failed",
            "error_class": "cost_guard",
            "error_code": "BUDGET_BLOCKED",
            "message": "Assistant calls are blocked by budget limits.",
            "budget": budget_status,
        }

    provider_chain = _resolve_provider_chain(provider_name)
    attempts: list[dict[str, str]] = []
    selected_provider = ""
    output_text = ""
    for provider in provider_chain:
        usable, reason = _provider_is_usable(provider)
        if not usable:
            attempts.append({"provider": provider, "status": "skipped", "reason": reason})
            continue
        retries_left = PROVIDER_MAX_RETRIES
        while True:
            try:
                output_text = _generate_chat_raw_with_provider(provider, prompt, model_name=model_name).strip()
                selected_provider = provider
                attempts.append({"provider": provider, "status": "ok"})
                break
            except ValueError as exc:
                reason_text = str(exc)
                if retries_left > 0 and _is_retryable_provider_error(reason_text):
                    attempts.append(
                        {
                            "provider": provider,
                            "status": "retrying",
                            "reason": reason_text,
                            "retries_left": str(retries_left),
                        }
                    )
                    retries_left -= 1
                    time.sleep(PROVIDER_RETRY_BACKOFF_SECONDS)
                    continue
                attempts.append({"provider": provider, "status": "failed", "reason": reason_text})
                break
        if output_text:
            break
    if not output_text:
        last_error = attempts[-1]["reason"] if attempts else "unknown"
        return {
            "status": "failed",
            "error_class": "llm_provider",
            "error_code": "LLM_PROVIDER_UNHEALTHY",
            "message": f"Chat failed: {last_error}",
            "provider_attempts": attempts,
            "provider_chain": provider_chain,
            "budget": budget_status,
        }
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(output_text)
    estimated_cost = _estimate_preview_cost_usd(
        selected_provider,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )
    return {
        "status": "ok",
        "provider_name": selected_provider,
        "message": output_text,
        "provider_attempts": attempts,
        "provider_chain": provider_chain,
        "token_estimate": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_estimate_usd": estimated_cost,
        "budget": budget_status,
    }


def _extract_handoff_intent_from_assistant_trace(trace: dict) -> str:
    """Extract planner handoff intent from assistant chat trace payload."""
    if not isinstance(trace, dict) or str(trace.get("type", "")).strip() != "assistant_chat":
        raise ValueError("trace is not assistant_chat")
    stages = trace.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("assistant trace has no stages")
    created = stages[0]
    if not isinstance(created, dict):
        raise ValueError("assistant trace stage is invalid")
    data = created.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("assistant trace data is invalid")
    response = data.get("response", {})
    if not isinstance(response, dict):
        raise ValueError("assistant trace response is invalid")
    message = str(response.get("message", "")).strip()
    if not message:
        raise ValueError("assistant trace has no message")
    if len(message) > 4000:
        message = message[:4000].rstrip()
    return message


def extract_browser_ui_url(output: str) -> str:
    """Extract latest noVNC/browser URL printed by recorder output."""
    matches = re.findall(r"Open browser UI at:\s*(https?://\S+)", output or "")
    if not matches:
        return ""
    return _normalize_browser_ui_url(matches[-1].strip())


def _normalize_browser_ui_url(url: str) -> str:
    """Normalize browser UI URL to direct noVNC entrypoint with autoconnect defaults."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw

    path = parts.path or "/"
    lowered = path.lower()
    if lowered in {"", "/"}:
        path = "/vnc.html"
    elif lowered == "/novnc.html":
        path = "/vnc.html"

    query_map = parse_qs(parts.query, keep_blank_values=True)
    if path.lower() == "/vnc.html":
        if "autoconnect" not in query_map:
            query_map["autoconnect"] = ["1"]
        if "resize" not in query_map:
            query_map["resize"] = ["remote"]
        if "reconnect" not in query_map:
            query_map["reconnect"] = ["1"]
    query = urlencode(query_map, doseq=True)
    normalized = f"{parts.scheme}://{parts.netloc}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    return normalized


def _trace_already_executed(trace: dict) -> bool:
    stages = trace.get("stages", [])
    if not isinstance(stages, list):
        return False
    for stage in reversed(stages):
        if not isinstance(stage, dict):
            continue
        event = str(stage.get("event", "")).strip()
        if event == "approved_execute_submitted":
            return True
    return False


def _enforce_execute_permissions(
    agent_id: str,
    latest_preview: dict,
    *,
    approve_permission: bool,
) -> dict | None:
    """Validate all required tool permissions for execute-plan."""
    tools: list[str] = []
    required_permissions = latest_preview.get("required_permissions", [])
    if isinstance(required_permissions, list):
        for row in required_permissions:
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool_name", "")).strip()
            if tool_name:
                tools.append(tool_name)
    if not tools:
        tools = [TOOL_BROWSER]

    for tool_name in tools:
        decision = get_agent_tool_permission_sync(agent_id, tool_name)
        if decision == DECISION_DENY:
            return {
                "error": "permission_denied",
                "agent_id": agent_id,
                "tool_name": tool_name,
                "message": f"Tool '{tool_name}' is denied for agent '{agent_id}'.",
            }
        if decision == DECISION_PROMPT:
            if approve_permission:
                set_agent_tool_permission_sync(agent_id, tool_name, DECISION_ALLOW)
                continue
            return {
                "error": "permission_required",
                "agent_id": agent_id,
                "tool_name": tool_name,
                "message": f"Agent '{agent_id}' requires approval for tool '{tool_name}'.",
            }
    return None


def build_action_command(
    action: str,
    params: dict[str, str],
    workspace: Path,
    python_bin: str,
) -> list[str]:
    """Return a whitelisted CLI command for supported guide actions."""
    workflow_path = params.get("workflow_path", "workflows/real_login_test.json").strip()
    if not workflow_path:
        workflow_path = "workflows/real_login_test.json"

    if action == "docker_info":
        return ["docker", "info"]
    if action == "cleanup_sessions":
        return [python_bin, "-m", "borisbot.cli", "cleanup-browsers"]
    if action == "session_status":
        return [python_bin, "-m", "borisbot.cli", "session-status"]
    if action == "budget_status":
        agent = params.get("agent_id", "").strip() or "default"
        return [python_bin, "-m", "borisbot.cli", "budget-status", "--agent-id", agent]
    if action == "budget_set":
        cmd = [python_bin, "-m", "borisbot.cli", "budget-set"]
        system_limit = params.get("budget_system_daily", "").strip()
        agent_limit = params.get("budget_agent_daily", "").strip()
        monthly_limit = params.get("budget_monthly", "").strip()
        if system_limit:
            float(system_limit)
            cmd.extend(["--system-daily-limit-usd", system_limit])
        if agent_limit:
            float(agent_limit)
            cmd.extend(["--agent-daily-limit-usd", agent_limit])
        if monthly_limit:
            float(monthly_limit)
            cmd.extend(["--monthly-limit-usd", monthly_limit])
        if len(cmd) == 4:
            raise ValueError("Set at least one budget value before running budget_set")
        return cmd
    if action == "doctor":
        model = params.get("model_name", "").strip() or "llama3.2:3b"
        return [python_bin, "-m", "borisbot.cli", "doctor", "--model", model]
    if action == "ollama_check":
        return ["ollama", "--version"]
    if action == "ollama_install":
        return _resolve_ollama_install_command(sys.platform)
    if action == "ollama_start":
        return _resolve_ollama_start_command(sys.platform)
    if action == "ollama_pull":
        model = params.get("model_name", "").strip() or "llama3.2:3b"
        return ["ollama", "pull", model]
    if action == "llm_setup":
        model = params.get("model_name", "").strip() or "llama3.2:3b"
        return [python_bin, "-m", "borisbot.cli", "llm-setup", "--model", model, "--json"]
    if action == "bootstrap_setup":
        model = params.get("model_name", "").strip() or "llama3.2:3b"
        return [
            python_bin,
            "-m",
            "borisbot.cli",
            "setup",
            "--model",
            model,
            "--no-launch-guide",
            "--json",
        ]
    if action == "verify":
        return [python_bin, "-m", "borisbot.cli", "verify"]
    if action == "verify_agent_logic":
        return [python_bin, "-m", "borisbot.cli", "verify"]
    if action == "analyze":
        return [python_bin, "-m", "borisbot.cli", "analyze-workflow", workflow_path]
    if action == "lint":
        return [python_bin, "-m", "borisbot.cli", "lint-workflow", workflow_path]
    if action == "replay":
        return [python_bin, "-m", "borisbot.cli", "replay", workflow_path]
    if action == "release_check":
        return [python_bin, "-m", "borisbot.cli", "release-check", workflow_path]
    if action == "release_check_json":
        return [python_bin, "-m", "borisbot.cli", "release-check", workflow_path, "--json"]
    if action == "policy_safe_local":
        agent = params.get("agent_id", "").strip() or "default"
        return [python_bin, "-m", "borisbot.cli", "policy-apply", "--policy", "safe-local", "--agent-id", agent]
    if action == "policy_web_readonly":
        agent = params.get("agent_id", "").strip() or "default"
        return [python_bin, "-m", "borisbot.cli", "policy-apply", "--policy", "web-readonly", "--agent-id", agent]
    if action == "policy_automation":
        agent = params.get("agent_id", "").strip() or "default"
        return [python_bin, "-m", "borisbot.cli", "policy-apply", "--policy", "automation", "--agent-id", agent]
    if action == "policy_apply":
        agent = params.get("agent_id", "").strip() or "default"
        policy = params.get("policy_name", "").strip().lower() or "safe-local"
        if policy not in {"safe-local", "web-readonly", "automation"}:
            raise ValueError("policy_name must be one of: safe-local, web-readonly, automation")
        return [python_bin, "-m", "borisbot.cli", "policy-apply", "--policy", policy, "--agent-id", agent]
    if action == "record":
        task_id = params.get("task_id", "").strip() or "wf_new"
        start_url = params.get("start_url", "").strip() or "https://example.com"
        parsed = urlparse(start_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("start_url must include protocol and host, e.g. https://example.com")
        return [python_bin, "-m", "borisbot.cli", "record", task_id, "--start-url", start_url]

    raise ValueError(f"Unsupported action: {action}")


@dataclass
class GuideJob:
    """Background process metadata and output buffer."""

    job_id: str
    action: str
    command: list[str]
    params: dict[str, str] = field(default_factory=dict)
    status: str = "running"
    output: list[str] = field(default_factory=list)
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    process: subprocess.Popen[str] | None = None
    trace_id: str | None = None

    def append(self, line: str) -> None:
        self.output.append(line)
        if len(self.output) > 800:
            self.output = self.output[-800:]

    def to_dict(self) -> dict:
        output_text = "".join(self.output)
        return {
            "job_id": self.job_id,
            "action": self.action,
            "params": self.params,
            "command": " ".join(shlex.quote(part) for part in self.command),
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output": output_text,
            "browser_ui_url": extract_browser_ui_url(output_text),
        }


class GuideState:
    """Shared mutable state for the guide web app."""

    def __init__(self, workspace: Path, python_bin: str):
        self.workspace = workspace
        self.python_bin = python_bin
        self._jobs: dict[str, GuideJob] = {}
        self._traces: list[dict] = []
        self._inbox_items: list[dict] = []
        self._schedules: list[dict] = []
        self._lock = threading.Lock()
        self._counter = 0
        self._trace_counter = 0
        self._inbox_counter = 0
        self._schedule_counter = 0
        self._wizard_state: dict = self._load_wizard_state()

    def _load_wizard_state(self) -> dict:
        GUIDE_STATE_DIR.mkdir(parents=True, exist_ok=True)
        default_steps = [
            {"step_id": row["step_id"], "label": row["label"], "completed": False}
            for row in WIZARD_STEP_DEFS
        ]
        if not GUIDE_WIZARD_STATE_FILE.exists():
            return {"updated_at": datetime.utcnow().isoformat(), "steps": default_steps}
        try:
            payload = json.loads(GUIDE_WIZARD_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"updated_at": datetime.utcnow().isoformat(), "steps": default_steps}
        if not isinstance(payload, dict):
            return {"updated_at": datetime.utcnow().isoformat(), "steps": default_steps}
        existing = payload.get("steps", [])
        if not isinstance(existing, list):
            existing = []
        existing_map: dict[str, bool] = {}
        for row in existing:
            if not isinstance(row, dict):
                continue
            step_id = str(row.get("step_id", "")).strip()
            if not step_id:
                continue
            existing_map[step_id] = bool(row.get("completed", False))
        merged_steps = []
        for row in WIZARD_STEP_DEFS:
            step_id = row["step_id"]
            merged_steps.append(
                {
                    "step_id": step_id,
                    "label": row["label"],
                    "completed": bool(existing_map.get(step_id, False)),
                }
            )
        return {
            "updated_at": str(payload.get("updated_at", datetime.utcnow().isoformat())),
            "steps": merged_steps,
        }

    def _save_wizard_state_locked(self) -> None:
        self._wizard_state["updated_at"] = datetime.utcnow().isoformat()
        GUIDE_WIZARD_STATE_FILE.write_text(
            json.dumps(self._wizard_state, indent=2),
            encoding="utf-8",
        )

    def get_wizard_state(self) -> dict:
        with self._lock:
            steps = self._wizard_state.get("steps", [])
            if not isinstance(steps, list):
                steps = []
            total = len(steps)
            completed = sum(1 for row in steps if isinstance(row, dict) and bool(row.get("completed", False)))
            return {
                "updated_at": self._wizard_state.get("updated_at", ""),
                "steps": steps,
                "summary": {
                    "completed": completed,
                    "total": total,
                    "progress_percent": int((completed / total) * 100) if total else 0,
                },
            }

    def set_wizard_step(self, step_id: str, completed: bool) -> dict:
        key = str(step_id).strip()
        if not key:
            raise ValueError("step_id is required")
        with self._lock:
            steps = self._wizard_state.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("wizard state corrupted")
            found = False
            for row in steps:
                if not isinstance(row, dict):
                    continue
                if str(row.get("step_id", "")) == key:
                    row["completed"] = bool(completed)
                    found = True
                    break
            if not found:
                raise ValueError(f"unknown step_id '{key}'")
            self._save_wizard_state_locked()
            steps_out = list(steps)
        total = len(steps_out)
        done = sum(1 for row in steps_out if isinstance(row, dict) and bool(row.get("completed", False)))
        return {
            "updated_at": self._wizard_state.get("updated_at", ""),
            "steps": steps_out,
            "summary": {
                "completed": done,
                "total": total,
                "progress_percent": int((done / total) * 100) if total else 0,
            },
        }

    def list_inbox_items(self) -> list[dict]:
        with self._lock:
            items = list(reversed(self._inbox_items[-100:]))
        return items

    def add_inbox_item(self, intent: str, *, source: str = "manual", priority: str = "normal") -> dict:
        text = str(intent).strip()
        if not text:
            raise ValueError("intent is required")
        value = str(priority).strip().lower() or "normal"
        if value not in {"low", "normal", "high"}:
            raise ValueError("priority must be one of: low, normal, high")
        with self._lock:
            self._inbox_counter += 1
            item = {
                "item_id": f"inbox_{self._inbox_counter:05d}",
                "intent": text,
                "source": str(source).strip() or "manual",
                "priority": value,
                "status": "open",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self._inbox_items.append(item)
            return item

    def update_inbox_item(self, item_id: str, status: str) -> dict:
        key = str(item_id).strip()
        value = str(status).strip().lower()
        if value not in {"open", "in_progress", "done", "archived", "failed"}:
            raise ValueError("status must be one of: open, in_progress, done, archived, failed")
        with self._lock:
            for item in self._inbox_items:
                if str(item.get("item_id", "")) == key:
                    item["status"] = value
                    item["updated_at"] = datetime.utcnow().isoformat()
                    return item
        raise ValueError("inbox item not found")

    def delete_inbox_item(self, item_id: str) -> bool:
        key = str(item_id).strip()
        with self._lock:
            before = len(self._inbox_items)
            self._inbox_items = [item for item in self._inbox_items if str(item.get("item_id", "")) != key]
            return len(self._inbox_items) < before

    def create_schedule(self, intent: str, interval_minutes: int, *, agent_id: str = "default") -> dict:
        text = str(intent).strip()
        if not text:
            raise ValueError("intent is required")
        if int(interval_minutes) <= 0:
            raise ValueError("interval_minutes must be > 0")
        now = datetime.utcnow()
        with self._lock:
            self._schedule_counter += 1
            schedule = {
                "schedule_id": f"sched_{self._schedule_counter:05d}",
                "intent": text,
                "agent_id": str(agent_id).strip() or "default",
                "interval_minutes": int(interval_minutes),
                "enabled": True,
                "next_run_at": (now + timedelta(minutes=int(interval_minutes))).isoformat(),
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "last_enqueued_at": "",
            }
            self._schedules.append(schedule)
            return schedule

    def set_schedule_enabled(self, schedule_id: str, enabled: bool) -> dict:
        key = str(schedule_id).strip()
        with self._lock:
            for row in self._schedules:
                if str(row.get("schedule_id", "")) != key:
                    continue
                row["enabled"] = bool(enabled)
                row["updated_at"] = datetime.utcnow().isoformat()
                return row
        raise ValueError("schedule not found")

    def delete_schedule(self, schedule_id: str) -> bool:
        key = str(schedule_id).strip()
        with self._lock:
            before = len(self._schedules)
            self._schedules = [row for row in self._schedules if str(row.get("schedule_id", "")) != key]
            return len(self._schedules) < before

    def get_next_pending_item(self) -> dict | None:
        """Return the next 'open' inbox item (manual source only for now)."""
        with self._lock:
            for item in self._inbox_items:
                if item.get("status") == "open":
                    return item
        return None

    def tick_schedules(self) -> int:
        now = datetime.utcnow()
        enqueued = 0
        with self._lock:
            for row in self._schedules:
                if not bool(row.get("enabled", False)):
                    continue
                next_run_raw = str(row.get("next_run_at", "")).strip()
                try:
                    next_run = datetime.fromisoformat(next_run_raw)
                except Exception:
                    next_run = now + timedelta(minutes=int(row.get("interval_minutes", 1)))
                if next_run > now:
                    continue
                self._inbox_counter += 1
                item = {
                    "item_id": f"inbox_{self._inbox_counter:05d}",
                    "intent": str(row.get("intent", "")).strip(),
                    "source": f"schedule:{row.get('schedule_id', '')}",
                    "priority": "normal",
                    "status": "open",
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                }
                self._inbox_items.append(item)
                row["last_enqueued_at"] = now.isoformat()
                interval = int(row.get("interval_minutes", 1))
                row["next_run_at"] = (now + timedelta(minutes=interval)).isoformat()
                row["updated_at"] = now.isoformat()
                enqueued += 1
        return enqueued

    def list_schedules(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._schedules[-100:]))

    def workflows(self) -> list[str]:
        workflow_dir = self.workspace / "workflows"
        if not workflow_dir.exists():
            return []
        return sorted(str(p.relative_to(self.workspace)) for p in workflow_dir.glob("*.json"))

    def create_job(self, action: str, params: dict[str, str]) -> GuideJob:
        request_fingerprint = self._request_fingerprint(action, params)
        with self._lock:
            for existing in self._jobs.values():
                if existing.status != "running":
                    continue
                if existing.action != action:
                    continue
                if self._request_fingerprint(existing.action, existing.params) == request_fingerprint:
                    raise ValueError(
                        "An identical job is already running. Wait for it to finish before retrying."
                    )
        if action in {"record", "replay"}:
            running_browser_job = self._find_running_browser_job()
            if running_browser_job is not None:
                raise ValueError(
                    "A browser job is already running. Stop it before starting another record/replay."
                )
        command = build_action_command(action, params, self.workspace, self.python_bin)
        with self._lock:
            self._counter += 1
            job_id = f"job_{self._counter:04d}"
            job = GuideJob(job_id=job_id, action=action, command=command, params=params)
            self._jobs[job_id] = job
            trace = self._create_trace_locked(
                trace_type="action_run",
                data={
                    "action": action,
                    "params": params,
                    "command": " ".join(shlex.quote(part) for part in command),
                },
            )
            job.trace_id = trace["trace_id"]
        self._start_job(job)
        return job

    def _request_fingerprint(self, action: str, params: dict[str, str]) -> str:
        raw = json.dumps({"action": action, "params": params}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = [job.to_dict() for job in self._jobs.values()]
        jobs.sort(key=lambda row: row["started_at"], reverse=True)
        return jobs

    def runtime_status(self) -> dict:
        """Return runtime status snapshot for GUI display."""
        return _collect_runtime_status(self.python_bin)

    def list_traces(self) -> list[dict]:
        """Return recent traces sorted newest first."""
        with self._lock:
            return list(reversed(self._traces[-30:]))

    def list_trace_summaries(self) -> list[dict]:
        """Return compact summaries for recent traces."""
        traces = self.list_traces()
        summaries: list[dict] = []
        for trace in traces:
            stages = trace.get("stages", [])
            last_event = "unknown"
            if isinstance(stages, list) and stages:
                last = stages[-1]
                if isinstance(last, dict):
                    last_event = str(last.get("event", "unknown"))
            summaries.append(
                {
                    "trace_id": str(trace.get("trace_id", "")),
                    "type": str(trace.get("type", "")),
                    "created_at": str(trace.get("created_at", "")),
                    "stage_count": len(stages) if isinstance(stages, list) else 0,
                    "last_event": last_event,
                }
            )
        return summaries

    def add_plan_trace(self, *, agent_id: str, model_name: str, intent: str, preview: dict) -> dict:
        """Append dry-run planner trace entry."""
        with self._lock:
            return self._create_trace_locked(
                trace_type="plan_preview",
                data={
                    "agent_id": agent_id,
                    "model_name": model_name,
                    "intent": intent,
                    "preview": preview,
                },
            )

    def add_assistant_trace(self, *, agent_id: str, model_name: str, prompt: str, response: dict) -> dict:
        """Append assistant chat trace entry."""
        with self._lock:
            return self._create_trace_locked(
                trace_type="assistant_chat",
                data={
                    "agent_id": agent_id,
                    "model_name": model_name,
                    "prompt": prompt,
                    "response": response,
                },
            )

    def get_trace(self, trace_id: str) -> dict | None:
        """Fetch trace by id."""
        with self._lock:
            for trace in self._traces:
                if trace.get("trace_id") == trace_id:
                    return trace
        return None

    def append_trace_stage(self, trace_id: str, stage_data: dict) -> None:
        """Append stage to an existing trace."""
        self._append_trace_stage(trace_id, stage_data)

    def get_job(self, job_id: str) -> GuideJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None or job.process is None or job.status != "running":
            return False
        try:
            if os.name == "nt":
                job.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(job.process.pid, signal.SIGINT)
            return True
        except Exception:
            return False

    def _find_running_browser_job(self) -> GuideJob | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running" and job.action in {"record", "replay"}:
                    return job
        return None

    def _start_job(self, job: GuideJob) -> None:
        env = os.environ.copy()
        # Ensure child Python CLI output streams immediately into the guide UI.
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict = {
            "cwd": str(self.workspace),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["preexec_fn"] = os.setsid
        process = subprocess.Popen(job.command, **kwargs)
        job.process = process

        def _reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                job.append(line)
            process.wait()
            job.returncode = process.returncode
            job.finished_at = time.time()
            job.status = "completed" if process.returncode == 0 else "failed"
            if job.trace_id:
                self._append_trace_stage(
                    job.trace_id,
                    {
                        "event": "job_finished",
                        "status": job.status,
                        "returncode": job.returncode,
                    },
                )

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

    def _create_trace_locked(self, *, trace_type: str, data: dict) -> dict:
        self._trace_counter += 1
        trace = {
            "trace_id": f"trace_{self._trace_counter:05d}",
            "type": trace_type,
            "created_at": datetime.utcnow().isoformat(),
            "stages": [{"event": "created", "at": datetime.utcnow().isoformat(), "data": data}],
        }
        self._traces.append(trace)
        if len(self._traces) > 200:
            self._traces = self._traces[-200:]
        return trace

    def _append_trace_stage(self, trace_id: str, stage_data: dict) -> None:
        with self._lock:
            for trace in reversed(self._traces):
                if trace.get("trace_id") == trace_id:
                    trace.setdefault("stages", []).append(
                        {
                            "event": str(stage_data.get("event", "stage")),
                            "at": datetime.utcnow().isoformat(),
                            "data": stage_data,
                        }
                    )
                    return


def _build_support_bundle(state: GuideState, agent_id: str) -> dict:
    """Build compact diagnostic bundle for runtime + trace debugging."""
    agent = str(agent_id).strip() or "default"
    traces = state.list_traces()
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "agent_id": agent,
        "runtime_status": state.runtime_status(),
        "profile": load_profile(),
        "permissions": get_agent_permission_matrix_sync(agent),
        "wizard_state": state.get_wizard_state(),
        "task_inbox": state.list_inbox_items()[:20],
        "schedules": state.list_schedules()[:20],
        "trace_summaries": state.list_trace_summaries(),
        "recent_traces": traces[:5],
    }


def _collect_runtime_status(python_bin: str) -> dict:
    """Collect runtime diagnostic information."""
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": python_bin,
        },
        "docker": {
            "installed": shutil.which("docker") is not None,
            "running": False,  # We don't check for speed unless needed
        },
        "ollama": {
            "installed": shutil.which("ollama") is not None,
            "running": False,  # We don't check for speed unless needed
        },
        "log_dir": str(Path.home() / ".borisbot" / "logs"),
        "provider_matrix": {
            "ollama": {
                "installed": shutil.which("ollama") is not None,
                "running": False,
            }
        }
    }


def _make_handler(state: GuideState) -> Callable[..., BaseHTTPRequestHandler]:
    class GuideHandler(BaseHTTPRequestHandler):
        def _json_response(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

        def _serve_static(self, path: str, content_type: str) -> None:
            """Serve a static file from the static directory."""
            try:
                # Security check: ensure path stays within static dir
                base_path = Path(__file__).parent / "static"
                file_path = (base_path / path).resolve()
                if not str(file_path).startswith(str(base_path)):
                    self._json_response({"error": "forbidden"}, status=HTTPStatus.FORBIDDEN)
                    return
                
                if not file_path.exists():
                    self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return

                content = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._json_response({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._serve_static("index.html", "text/html; charset=utf-8")
                return
            if self.path.startswith("/static/"):
                # primitive static file server
                rel_path = self.path.replace("/static/", "", 1)
                if rel_path.endswith(".css"):
                    self._serve_static(rel_path, "text/css")
                    return
                if rel_path.endswith(".js"):
                    self._serve_static(rel_path, "application/javascript")
                    return
                # fallback
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            # API: System Check
            if self.path == "/api/system-check":
                checks = []
                # Check 1: Python version
                checks.append({
                    "label": "Python Environment",
                    "passed": sys.version_info >= (3, 10),
                    "message": f"Python {sys.version.split()[0]} detected"
                })
                # Check 2: Docker (optional but good)
                docker_path = shutil.which("docker")
                checks.append({
                    "label": "Docker Engine",
                    "passed": bool(docker_path),
                    "message": "Docker available" if docker_path else "Docker not found (agents will run locally only)"
                })
                # Check 3: Directories
                log_dir = Path.home() / ".borisbot" / "logs"
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    checks.append({ "label": "Write Permissions", "passed": True, "message": "Log directory writable" })
                except Exception as e:
                    checks.append({ "label": "Write Permissions", "passed": False, "message": f"Failed to write logs: {e}" })

                self._json_response({"items": checks})
                return

            # API: Provider Status (Granular)
            if self.path.startswith("/api/provider-status/"):
                provider = self.path.split("/")[-1]
                if provider == "ollama":
                    running = False
                    model = ""
                    installed = bool(shutil.which("ollama"))
                    if installed:
                        try:
                            # Try to hit localhost:11434
                            res = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
                            if res.status_code == 200:
                                running = True
                                models = res.json().get('models', [])
                                if models:
                                    model = models[0].get('name', 'unknown')
                        except Exception:
                            pass
                    
                    self._json_response({
                        "installed": installed,
                        "running": running,
                        "model": model
                    })
                    return

            # API: Recommended Models
            if self.path == "/api/recommended-models":
                self._json_response({
                    "items": [
                        {
                            "id": "llama3.2",
                            "name": "Llama 3.2",
                            "size": "2.0GB",
                            "description": "Meta's latest lightweight model. Balanced for speed and quality.",
                            "recommended": True
                        },
                        {
                            "id": "mistral",
                            "name": "Mistral 7B",
                            "size": "4.1GB",
                            "description": "High performant 7B model. Good for complex reasoning.",
                            "recommended": False
                        },
                        {
                            "id": "qwen2.5:3b",
                            "name": "Qwen 2.5 3B",
                            "size": "1.9GB",
                            "description": "Fast and capable model from Alibaba. Great for simple tasks.",
                            "recommended": False
                        },
                         {
                            "id": "deepseek-r1:1.5b",
                            "name": "DeepSeek R1 1.5B",
                            "size": "1.1GB",
                            "description": "Extremely fast distilled model. Good for quick checks.",
                            "recommended": False
                        }
                    ]
                })
                return



            if self.path == "/api/workflows":
                self._json_response({"items": state.workflows()})
                return
            if self.path == "/api/jobs":
                self._json_response({"items": state.list_jobs()})
                return
            if self.path == "/api/runtime-status":
                self._json_response(state.runtime_status())
                return
            if self.path == "/api/wizard-state":
                self._json_response(state.get_wizard_state())
                return
            if self.path == "/api/task-inbox":
                state.tick_schedules()
                self._json_response({"items": state.list_inbox_items()})
                return
            if self.path == "/api/schedules":
                state.tick_schedules()
                self._json_response({"items": state.list_schedules()})
                return
            if self.path.startswith("/api/ollama-setup-plan"):
                query = parse_qs(urlsplit(self.path).query)
                model_name = str(query.get("model_name", ["llama3.2:3b"])[0]).strip() or "llama3.2:3b"
                self._json_response(_build_ollama_setup_plan(model_name))
                return
            if self.path == "/api/profile":
                self._json_response(load_profile())
                return
            if self.path.startswith("/api/chat-history"):
                query = parse_qs(urlsplit(self.path).query)
                agent_id = str(query.get("agent_id", ["default"])[0]).strip() or "default"
                self._json_response({"agent_id": agent_id, "items": load_chat_history(agent_id)})
                return
            if self.path == "/api/provider-secrets":
                self._json_response({"providers": get_secret_status()})
                return
            if self.path.startswith("/api/permissions"):
                query = parse_qs(urlsplit(self.path).query)
                agent_id = str(query.get("agent_id", ["default"])[0]).strip() or "default"
                matrix = get_agent_permission_matrix_sync(agent_id)
                self._json_response({"agent_id": agent_id, "permissions": matrix})
                return
            if self.path.startswith("/api/support-bundle"):
                query = parse_qs(urlsplit(self.path).query)
                agent_id = str(query.get("agent_id", ["default"])[0]).strip() or "default"
                self._json_response(_build_support_bundle(state, agent_id))
                return
            if self.path == "/api/traces":
                self._json_response({"items": state.list_trace_summaries()})
                return
            if self.path.startswith("/api/traces/"):
                trace_id = self.path.split("/")[-1]
                trace = state.get_trace(trace_id)
                if not isinstance(trace, dict):
                    self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response(trace)
                return
            if self.path.startswith("/api/jobs/"):
                job_id = self.path.split("/")[-1]
                job = state.get_job(job_id)
                if job is None:
                    self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response(job.to_dict())
                return
            self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


        def do_POST(self) -> None:  # noqa: N802
            # API: Ollama Pull
            if self.path == "/api/ollama/pull":
                try:
                    payload = self._read_json()
                    model = payload.get("model_name") or payload.get("model")
                    if not model:
                        self._json_response({"error": "model_required"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    
                    # Create a job to pull the model
                    job = state.create_job("ollama_pull", {"model_name": str(model)})
                    self._json_response({"status": "pulling", "job_id": job.job_id}, status=HTTPStatus.ACCEPTED)
                    return
                except Exception as e:
                    self._json_response({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                    
            if self.path == "/api/run":
                payload = self._read_json()
                action = str(payload.get("action", "")).strip()
                params = payload.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                agent_id = str(params.get("agent_id", "default")).strip() or "default"
                required_tool = required_tool_for_action(action)
                if required_tool:
                    decision = get_agent_tool_permission_sync(agent_id, required_tool)
                    approve_permission = bool(payload.get("approve_permission", False))
                    if decision == DECISION_DENY:
                        self._json_response(
                            {
                                "error": "permission_denied",
                                "agent_id": agent_id,
                                "tool_name": required_tool,
                                "message": f"Tool '{required_tool}' is denied for agent '{agent_id}'.",
                            },
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return
                    if decision == DECISION_PROMPT and not approve_permission:
                        self._json_response(
                            {
                                "error": "permission_required",
                                "agent_id": agent_id,
                                "tool_name": required_tool,
                                "message": (
                                    f"Agent '{agent_id}' requires approval for tool '{required_tool}'."
                                ),
                            },
                            status=HTTPStatus.CONFLICT,
                        )
                        return
                    if decision == DECISION_PROMPT and approve_permission:
                        set_agent_tool_permission_sync(agent_id, required_tool, DECISION_ALLOW)
                try:
                    job = state.create_job(action, {k: str(v) for k, v in params.items()})
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(job.to_dict(), status=HTTPStatus.CREATED)
                return
            if self.path == "/api/profile":
                payload = self._read_json()
                try:
                    profile = save_profile(payload)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(profile, status=HTTPStatus.OK)
                return
            if self.path == "/api/wizard-state":
                payload = self._read_json()
                step_id = str(payload.get("step_id", "")).strip()
                completed = bool(payload.get("completed", False))
                try:
                    wizard = state.set_wizard_step(step_id, completed)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(wizard, status=HTTPStatus.OK)
                return
            if self.path == "/api/task-inbox":
                payload = self._read_json()
                action = str(payload.get("action", "add")).strip().lower() or "add"
                if action == "add":
                    intent = str(payload.get("intent", "")).strip()
                    priority = str(payload.get("priority", "normal")).strip().lower() or "normal"
                    try:
                        item = state.add_inbox_item(intent, source="manual", priority=priority)
                    except ValueError as exc:
                        self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._json_response({"status": "ok", "item": item, "items": state.list_inbox_items()}, status=HTTPStatus.OK)
                    return
                if action == "update":
                    item_id = str(payload.get("item_id", "")).strip()
                    status_value = str(payload.get("status", "open")).strip().lower() or "open"
                    try:
                        item = state.update_inbox_item(item_id, status_value)
                    except ValueError as exc:
                        self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._json_response({"status": "ok", "item": item, "items": state.list_inbox_items()}, status=HTTPStatus.OK)
                    return
                if action == "delete":
                    item_id = str(payload.get("item_id", "")).strip()
                    deleted = state.delete_inbox_item(item_id)
                    if not deleted:
                        self._json_response({"error": "item_not_found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._json_response({"status": "ok", "items": state.list_inbox_items()}, status=HTTPStatus.OK)
                    return
                self._json_response({"error": "unsupported action"}, status=HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/schedules":
                payload = self._read_json()
                action = str(payload.get("action", "create")).strip().lower() or "create"
                if action == "create":
                    intent = str(payload.get("intent", "")).strip()
                    interval_minutes = int(payload.get("interval_minutes", 0) or 0)
                    agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                    try:
                        item = state.create_schedule(intent, interval_minutes, agent_id=agent_id)
                    except ValueError as exc:
                        self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._json_response({"status": "ok", "item": item, "items": state.list_schedules()}, status=HTTPStatus.OK)
                    return
                if action == "toggle":
                    schedule_id = str(payload.get("schedule_id", "")).strip()
                    enabled = bool(payload.get("enabled", False))
                    try:
                        item = state.set_schedule_enabled(schedule_id, enabled)
                    except ValueError as exc:
                        self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._json_response({"status": "ok", "item": item, "items": state.list_schedules()}, status=HTTPStatus.OK)
                    return
                if action == "delete":
                    schedule_id = str(payload.get("schedule_id", "")).strip()
                    deleted = state.delete_schedule(schedule_id)
                    if not deleted:
                        self._json_response({"error": "schedule_not_found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._json_response({"status": "ok", "items": state.list_schedules()}, status=HTTPStatus.OK)
                    return
                self._json_response({"error": "unsupported action"}, status=HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/provider-secrets":
                payload = self._read_json()
                provider = str(payload.get("provider", "")).strip().lower()
                api_key = str(payload.get("api_key", "")).strip()
                try:
                    providers = set_provider_secret(provider, api_key)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"providers": providers}, status=HTTPStatus.OK)
                return
            if self.path == "/api/provider-test":
                payload = self._read_json()
                provider = str(payload.get("provider_name", "ollama")).strip() or "ollama"
                model_name = _resolve_model_for_provider(
                    provider,
                    str(payload.get("model_name", "")).strip(),
                )
                try:
                    ok, message = _probe_provider_connection(provider, model_name)
                except Exception as exc:
                    self._json_response({"status": "failed", "message": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(
                    {"status": "ok" if ok else "failed", "provider_name": provider, "message": message},
                    status=HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST,
                )
                return
            if self.path == "/api/cost-estimate":
                payload = self._read_json()
                provider_name = str(payload.get("provider_name", "ollama")).strip() or "ollama"
                planner_prompt = str(payload.get("planner_prompt", "")).strip()
                assistant_prompt = str(payload.get("assistant_prompt", "")).strip()
                self._json_response(
                    _build_live_cost_estimate(provider_name, planner_prompt, assistant_prompt),
                    status=HTTPStatus.OK,
                )
                return
            if self.path == "/api/chat-history":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                role = str(payload.get("role", "")).strip()
                text = str(payload.get("text", "")).strip()
                try:
                    items = append_chat_message(agent_id, role, text)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"agent_id": agent_id, "items": items}, status=HTTPStatus.OK)
                return
            if self.path == "/api/assistant-chat":
                payload = self._read_json()
                prompt = str(payload.get("prompt", "")).strip()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                decision = get_agent_tool_permission_sync(agent_id, TOOL_ASSISTANT)
                approve_permission = bool(payload.get("approve_permission", False))
                if decision == DECISION_DENY:
                    self._json_response(
                        {
                            "error": "permission_denied",
                            "agent_id": agent_id,
                            "tool_name": TOOL_ASSISTANT,
                            "message": f"Tool '{TOOL_ASSISTANT}' is denied for agent '{agent_id}'.",
                        },
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                if decision == DECISION_PROMPT and not approve_permission:
                    self._json_response(
                        {
                            "error": "permission_required",
                            "agent_id": agent_id,
                            "tool_name": TOOL_ASSISTANT,
                            "message": (
                                f"Agent '{agent_id}' requires approval for tool '{TOOL_ASSISTANT}'."
                            ),
                        },
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                if decision == DECISION_PROMPT and approve_permission:
                    set_agent_tool_permission_sync(agent_id, TOOL_ASSISTANT, DECISION_ALLOW)
                provider_name = str(payload.get("provider_name", "ollama")).strip() or "ollama"
                model_name = _resolve_model_for_provider(
                    provider_name,
                    str(payload.get("model_name", "")).strip(),
                )
                try:
                    response_payload = _build_assistant_response(
                        prompt,
                        agent_id=agent_id,
                        model_name=model_name,
                        provider_name=provider_name,
                    )
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if response_payload.get("status") != "ok":
                    self._json_response(response_payload, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.add_assistant_trace(
                    agent_id=agent_id,
                    model_name=model_name,
                    prompt=prompt,
                    response=response_payload,
                )
                response_payload["trace_id"] = trace["trace_id"]
                self._json_response(response_payload, status=HTTPStatus.OK)
                return
            if self.path == "/api/assistant-handoff":
                payload = self._read_json()
                trace_id = str(payload.get("trace_id", "")).strip()
                if not trace_id:
                    self._json_response({"error": "trace_id required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.get_trace(trace_id)
                if not isinstance(trace, dict):
                    self._json_response({"error": "trace_not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                try:
                    intent = _extract_handoff_intent_from_assistant_trace(trace)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response(
                    {"status": "ok", "source_trace_id": trace_id, "intent": intent},
                    status=HTTPStatus.OK,
                )
                return
            if self.path == "/api/chat-clear":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                clear_chat_history(agent_id)
                self._json_response({"agent_id": agent_id, "items": []}, status=HTTPStatus.OK)
                return
            if self.path == "/api/chat-clear-assistant":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                items = clear_chat_roles(agent_id, {"assistant_user", "assistant"})
                self._json_response({"agent_id": agent_id, "items": items}, status=HTTPStatus.OK)
                return
            if self.path == "/api/permissions":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                tool_name = str(payload.get("tool_name", "")).strip()
                decision = str(payload.get("decision", "")).strip()
                if tool_name not in ALLOWED_TOOLS:
                    self._json_response(
                        {"error": f"unsupported tool_name '{tool_name}'"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    set_agent_tool_permission_sync(agent_id, tool_name, decision)
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                matrix = get_agent_permission_matrix_sync(agent_id)
                self._json_response(
                    {"agent_id": agent_id, "tool_name": tool_name, "decision": decision, "permissions": matrix},
                    status=HTTPStatus.OK,
                )
                return
            if self.path == "/api/plan-preview":
                payload = self._read_json()
                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                intent = str(payload.get("intent", "")).strip()
                provider_name = str(payload.get("provider_name", "ollama")).strip() or "ollama"
                model_name = _resolve_model_for_provider(
                    provider_name,
                    str(payload.get("model_name", "")).strip(),
                )
                try:
                    preview = _build_dry_run_preview(
                        intent,
                        agent_id=agent_id,
                        model_name=model_name,
                        provider_name=provider_name,
                    )
                except ValueError as exc:
                    self._json_response({"status": "failed", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.add_plan_trace(
                    agent_id=agent_id,
                    model_name=model_name,
                    intent=intent,
                    preview=preview,
                )
                preview["trace_id"] = trace["trace_id"]
                self._json_response(preview, status=HTTPStatus.OK)
                return
            if self.path == "/api/execute-plan":
                payload = self._read_json()
                trace_id = str(payload.get("trace_id", "")).strip()
                force_execute = bool(payload.get("force", False))
                if not trace_id:
                    self._json_response({"error": "trace_id required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                trace = state.get_trace(trace_id)
                if not isinstance(trace, dict):
                    self._json_response({"error": "trace_not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                if _trace_already_executed(trace) and not force_execute:
                    self._json_response(
                        {
                            "error": "trace_already_executed",
                            "message": "This trace has already been executed. Re-run with force to execute again.",
                        },
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                stages = trace.get("stages", [])
                if not isinstance(stages, list) or not stages:
                    self._json_response({"error": "trace_has_no_stages"}, status=HTTPStatus.BAD_REQUEST)
                    return
                latest_preview = {}
                for stage in reversed(stages):
                    if isinstance(stage, dict):
                        data = stage.get("data", {})
                        if isinstance(data, dict) and isinstance(data.get("preview"), dict):
                            latest_preview = data["preview"]
                            break
                if not isinstance(latest_preview, dict) or latest_preview.get("status") != "ok":
                    self._json_response(
                        {"error": "trace_preview_not_executable"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                validated_commands = latest_preview.get("validated_commands", [])
                if not isinstance(validated_commands, list) or not validated_commands:
                    self._json_response(
                        {"error": "trace_has_no_validated_commands"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                agent_id = str(payload.get("agent_id", "default")).strip() or "default"
                approve_permission = bool(payload.get("approve_permission", False))
                permission_error = _enforce_execute_permissions(
                    agent_id,
                    latest_preview,
                    approve_permission=approve_permission,
                )
                if isinstance(permission_error, dict):
                    status = (
                        HTTPStatus.FORBIDDEN
                        if permission_error.get("error") == "permission_denied"
                        else HTTPStatus.CONFLICT
                    )
                    self._json_response(permission_error, status=status)
                    return

                generated_dir = state.workspace / "workflows" / "generated"
                generated_dir.mkdir(parents=True, exist_ok=True)
                task_id = f"plan_{trace_id}_{int(time.time())}"
                workflow_path = generated_dir / f"{task_id}.json"
                workflow_payload = {
                    "schema_version": "task_command.v1",
                    "task_id": task_id,
                    "commands": validated_commands,
                }
                workflow_path.write_text(json.dumps(workflow_payload, indent=2), encoding="utf-8")
                state.append_trace_stage(
                    trace_id,
                    {
                        "event": "approved_execute_requested",
                        "task_id": task_id,
                        "workflow_path": str(workflow_path),
                    },
                )
                try:
                    job = state.create_job("replay", {"workflow_path": str(workflow_path)})
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                state.append_trace_stage(
                    trace_id,
                    {
                        "event": "approved_execute_submitted",
                        "job_id": job.job_id,
                        "task_id": task_id,
                    },
                )
                self._json_response(
                    {
                        "status": "submitted",
                        "job_id": job.job_id,
                        "task_id": task_id,
                        "workflow_path": str(workflow_path),
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
                parts = self.path.split("/")
                if len(parts) < 4:
                    self._json_response({"error": "invalid_path"}, status=HTTPStatus.BAD_REQUEST)
                    return
                job_id = parts[3]
                stopped = state.stop_job(job_id)
                if not stopped:
                    self._json_response({"error": "job_not_running"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._json_response({"status": "stopping", "job_id": job_id})
                return
            self._json_response({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return GuideHandler


def _resolve_model_for_provider(provider: str, model: str) -> str:
    if model: return model
    
    # Try to load from profile
    try:
        profile = load_profile()
        settings = profile.get("provider_settings", {})
        provider_settings = settings.get(provider, {})
        if provider_settings.get("model_name"):
            return str(provider_settings["model_name"])
    except Exception:
        pass

    if provider == "ollama": return "llama3.2:3b"
    if provider == "openai": return "gpt-4o"
    if provider == "anthropic": return "claude-3-5-sonnet-latest"
    return "default"


def _extract_handoff_intent_from_assistant_trace(trace: dict) -> str:
    if trace.get("type") != "assistant_chat":
        raise ValueError("Trace is not an assistant_chat")
    
    stages = trace.get("stages", [])
    for stage in stages:
         data = stage.get("data", {})
         if isinstance(data, dict):
             # check helper response
             response = data.get("response", {})
             if isinstance(response, dict) and response.get("message"):
                 return str(response["message"])
    return "unknown"


def _trace_already_executed(trace: dict) -> bool:
    stages = trace.get("stages", [])
    for stage in stages:
        if stage.get("event") == "approved_execute_submitted":
            return True
    return False


def _enforce_execute_permissions(agent_id: str, preview: dict, approve_permission: bool = False) -> dict | None:
    # 1. Check if the preview already specifies required permissions (planner analysis)
    required_permissions = preview.get("required_permissions", [])
    if required_permissions:
        for req in required_permissions:
            tool = req.get("tool_name")
            if not tool:
                continue
            
            # We re-check the live permission status, as the plan might be stale
            decision = get_agent_tool_permission_sync(agent_id, tool)
            
            if decision == DECISION_DENY:
                return {
                    "error": "permission_denied",
                    "agent_id": agent_id,
                    "tool_name": tool,
                    "message": f"Tool '{tool}' is denied for agent '{agent_id}'.",
                }
            
            if decision == DECISION_PROMPT:
                if approve_permission:
                    set_agent_tool_permission_sync(agent_id, tool, DECISION_ALLOW)
                else:
                    return {
                        "error": "permission_required",
                        "agent_id": agent_id,
                        "tool_name": tool,
                        "message": f"Agent '{agent_id}' requires approval for tool '{tool}'.",
                    }
        return None

    # 2. If no explicit permissions in preview, fallback to scanning commands
    validated_commands = preview.get("validated_commands", [])
    required_tools = set()
    for cmd in validated_commands:
        action = cmd.get("action")
        tool = required_tool_for_action(action)
        if tool:
            required_tools.add(tool)

    # 3. If still no tools found, default to browser (legacy behavior / safety net)
    if not required_tools:
         required_tools.add(TOOL_BROWSER)

    for tool in required_tools:
        decision = get_agent_tool_permission_sync(agent_id, tool)
        if decision == DECISION_DENY:
            return {
                "error": "permission_denied",
                "agent_id": agent_id,
                "tool_name": tool,
                "message": f"Tool '{tool}' is denied for agent '{agent_id}'.",
            }
        
        if decision == DECISION_PROMPT:
            if approve_permission:
                set_agent_tool_permission_sync(agent_id, tool, DECISION_ALLOW)
            else:
                return {
                    "error": "permission_required",
                    "agent_id": agent_id,
                    "tool_name": tool,
                    "message": f"Agent '{agent_id}' requires approval for tool '{tool}'.",
                }

    return None


def run_guide_server(workspace: Path, host: str = "127.0.0.1", port: int = 7788, open_browser: bool = True) -> None:
    os.environ["BORISBOT_WORKSPACE"] = str(workspace.resolve())
    state = GuideState(workspace, python_bin=sys.executable)
    
    # Start background scheduler
    def _scheduler_loop() -> None:
        while True:
            try:
                state.tick_schedules()
            except Exception:
                pass
            time.sleep(10)
    
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    print(f"BorisBot guide available at: http://{host}:{port}")
    print("Press Ctrl+C to stop the guide server.")

    # Start ActionRunner
    runner = ActionRunner(state)
    runner.start()

    try:
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        httpd.server_close()


class ActionRunner(threading.Thread):
    def __init__(self, state: GuideState):
        super().__init__(daemon=True)
        self.state = state
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[ActionRunner] Error: {e}")
            time.sleep(2)

    def _tick(self):
        item = self.state.get_next_pending_item()
        if not item:
            return
        
        item_id = item["item_id"]
        intent = item["intent"]
        self.state.update_inbox_item(item_id, "in_progress")

        try:
            # 1. Defaults
            provider = "ollama"
            model = "llama3.2:3b"
            try:
                profile = load_profile()
                settings = profile.get("provider_settings", {})
                # Use configured primary provider
                configured_chain = profile.get("provider_chain", [])
                if configured_chain and isinstance(configured_chain, list):
                     provider = configured_chain[0]
                
                # Use configured model for that provider
                provider_config = settings.get(provider, {})
                if provider_config.get("model_name"):
                    model = provider_config.get("model_name")
            except Exception:
                pass

            # 2. Plan
            preview = _build_dry_run_preview(
                intent,
                agent_id="default",
                model_name=model,
                provider_name=provider,
            )
            
            if preview["status"] != "ok":
                raise ValueError(preview.get('message', 'unknown error'))

            # 3. Trace
            trace = self.state.add_plan_trace(
                 agent_id="default",
                 model_name=model,
                 intent=intent,
                 preview=preview,
            )

            # 4. Workflow
            validated_commands = preview.get("validated_commands", [])
            generated_dir = self.state.workspace / "workflows" / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            task_id = f"inbox_{item_id}_{int(time.time())}"
            workflow_path = generated_dir / f"{task_id}.json"
            
            workflow_payload = {
                "schema_version": "task_command.v1",
                "task_id": task_id,
                "commands": validated_commands,
            }
            workflow_path.write_text(json.dumps(workflow_payload, indent=2), encoding="utf-8")
            
            self.state.append_trace_stage(
                trace["trace_id"],
                {
                    "event": "approved_execute_requested",
                    "task_id": task_id,
                    "workflow_path": str(workflow_path),
                },
            )

            # 5. Execute (Replay)
            job = self.state.create_job("replay", {"workflow_path": str(workflow_path)})
            
            self.state.append_trace_stage(
                trace["trace_id"],
                {
                    "event": "approved_execute_submitted",
                    "job_id": job.job_id,
                    "task_id": task_id,
                },
            )
            
            # Optimistic completion of "scheduling"
            self.state.update_inbox_item(item_id, "done")

        except Exception as e:
            print(f"[ActionRunner] Task failed: {e}")
            self.state.update_inbox_item(item_id, "failed")


if __name__ == "__main__":
    run_guide_server(Path.cwd())
