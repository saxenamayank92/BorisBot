"""Strict planner output contract parsing and single-pass repair."""

from __future__ import annotations

import json
from typing import Any

from borisbot.llm.errors import LLMInvalidOutputError

try:
    from json_repair import repair_json as _repair_json_fn
except Exception:  # pragma: no cover - dependency optional at runtime
    _repair_json_fn = None

PLANNER_SCHEMA_VERSION = "planner.v1"
ALLOWED_TOP_LEVEL_FIELDS = {"planner_schema_version", "intent", "proposed_actions"}
ALLOWED_ACTION_FIELDS = {"action", "target", "input"}


def _extract_json_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMInvalidOutputError("planner output has no JSON object")
    return raw[start : end + 1]


def _single_repair_pass(raw: str) -> str:
    candidate = _extract_json_object(raw).strip()
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if open_braces > close_braces:
        candidate = candidate + ("}" * (open_braces - close_braces))
    return candidate


def _validate_action(action: Any, index: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise LLMInvalidOutputError(f"proposed_actions[{index}] must be object")
    keys = set(action.keys())
    if keys != ALLOWED_ACTION_FIELDS:
        raise LLMInvalidOutputError(
            f"proposed_actions[{index}] must contain exactly {sorted(ALLOWED_ACTION_FIELDS)}"
        )
    normalized = {
        "action": action.get("action"),
        "target": action.get("target"),
        "input": action.get("input"),
    }
    if not isinstance(normalized["action"], str) or not normalized["action"].strip():
        raise LLMInvalidOutputError(f"proposed_actions[{index}].action must be non-empty string")
    if not isinstance(normalized["target"], str):
        raise LLMInvalidOutputError(f"proposed_actions[{index}].target must be string")
    if not isinstance(normalized["input"], str):
        raise LLMInvalidOutputError(f"proposed_actions[{index}].input must be string")
    return normalized


def validate_planner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate strict planner.v1 schema with no extra fields."""
    keys = set(payload.keys())
    if keys != ALLOWED_TOP_LEVEL_FIELDS:
        raise LLMInvalidOutputError(
            f"planner payload must contain exactly {sorted(ALLOWED_TOP_LEVEL_FIELDS)}"
        )
    schema_version = payload.get("planner_schema_version")
    if schema_version != PLANNER_SCHEMA_VERSION:
        raise LLMInvalidOutputError(f"planner_schema_version must be {PLANNER_SCHEMA_VERSION}")

    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise LLMInvalidOutputError("intent must be non-empty string")

    actions = payload.get("proposed_actions")
    if not isinstance(actions, list):
        raise LLMInvalidOutputError("proposed_actions must be list")

    normalized_actions = [_validate_action(action, idx) for idx, action in enumerate(actions)]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "intent": intent.strip(),
        "proposed_actions": normalized_actions,
    }


def parse_planner_output(raw_text: str) -> dict[str, Any]:
    """Parse planner output with bounded repair attempts."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LLMInvalidOutputError("planner output empty")

    try:
        payload = json.loads(raw_text)
    except Exception:
        repaired = ""
        try:
            repaired = _single_repair_pass(raw_text)
            payload = json.loads(repaired)
        except Exception:
            if _repair_json_fn is None:
                raise LLMInvalidOutputError("planner output invalid after one repair pass")
            try:
                repaired_payload = _repair_json_fn(raw_text)
                if isinstance(repaired_payload, (dict, list)):
                    payload = repaired_payload
                elif isinstance(repaired_payload, str):
                    payload = json.loads(repaired_payload)
                else:
                    raise LLMInvalidOutputError("planner output invalid after json_repair")
            except Exception as repair_exc:
                raise LLMInvalidOutputError("planner output invalid after one repair pass") from repair_exc

    if not isinstance(payload, dict):
        raise LLMInvalidOutputError("planner output must be object")
    return validate_planner_payload(payload)
