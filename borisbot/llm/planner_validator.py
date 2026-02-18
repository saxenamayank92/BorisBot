"""Planner action validator and deterministic command conversion."""

from __future__ import annotations

from typing import Any

from borisbot.llm.errors import LLMInvalidOutputError

SUPPORTED_EXECUTOR_ACTIONS = {
    "navigate",
    "click",
    "type",
    "wait_for_url",
    "get_text",
    "get_title",
}
BLOCKED_ACTIONS = {"run_shell", "eval", "execute_js", "raw_js", "schedule"}


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise LLMInvalidOutputError(f"{field_name} must be string")
    return value.strip()


def validate_and_convert_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate planned actions and convert to deterministic executor commands."""
    actions = plan.get("proposed_actions")
    if not isinstance(actions, list):
        raise LLMInvalidOutputError("proposed_actions must be list")

    commands: list[dict[str, Any]] = []
    for idx, action_obj in enumerate(actions, start=1):
        if not isinstance(action_obj, dict):
            raise LLMInvalidOutputError(f"proposed_actions[{idx-1}] must be object")

        action = _require_str(action_obj.get("action"), f"proposed_actions[{idx-1}].action")
        target = _require_str(action_obj.get("target"), f"proposed_actions[{idx-1}].target")
        input_value = _require_str(action_obj.get("input"), f"proposed_actions[{idx-1}].input")

        if action in BLOCKED_ACTIONS:
            raise LLMInvalidOutputError(f"blocked action not allowed: {action}")
        if action not in SUPPORTED_EXECUTOR_ACTIONS:
            raise LLMInvalidOutputError(f"unsupported action: {action}")

        if action == "navigate":
            if not target:
                raise LLMInvalidOutputError("navigate.target must be non-empty URL")
            commands.append(
                {"id": str(idx), "action": "navigate", "params": {"url": target}}
            )
            continue

        if action in {"click", "wait_for_url", "get_text"}:
            if not target:
                raise LLMInvalidOutputError(f"{action}.target must be non-empty")
            params: dict[str, Any] = {"selector": target}
            commands.append({"id": str(idx), "action": action, "params": params})
            continue

        if action == "type":
            if not target:
                raise LLMInvalidOutputError("type.target must be non-empty selector")
            commands.append(
                {"id": str(idx), "action": "type", "params": {"selector": target, "text": input_value}}
            )
            continue

        if action == "get_title":
            commands.append({"id": str(idx), "action": "get_title", "params": {}})
            continue

        raise LLMInvalidOutputError(f"unsupported action: {action}")

    return commands

