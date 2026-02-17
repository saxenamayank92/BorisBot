"""Static workflow selector analyzer for deterministic replay risk reporting."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("borisbot.recorder.analyzer")

_THREE_PLUS_DIGITS = re.compile(r"\d{3,}")
_BROAD_TAGS = {"a", "button", "input", "div", "span", "p", "li"}


def _base_score(selector: str) -> tuple[int, str]:
    if selector.startswith("[data-testid="):
        return 95, "data-testid selector"
    if selector.startswith("[data-test="):
        return 93, "data-test selector"
    if selector.startswith("[data-qa="):
        return 93, "data-qa selector"
    if selector.startswith("#"):
        return 90, "id selector"
    if selector.startswith("[aria-label="):
        return 78, "aria-label selector"
    if selector.startswith("[name="):
        return 75, "name selector"
    if selector.startswith("[role="):
        return 65, "role selector"
    if "." in selector and not selector.startswith("["):
        return 55, "tag.class selector"
    if re.fullmatch(r"[a-z][a-z0-9-]*", selector):
        return 30, "tag-only selector"
    return 45, "generic selector"


def _selector_risks(selector: str) -> list[str]:
    risks: list[str] = []
    lowered = selector.lower()
    if _THREE_PLUS_DIGITS.search(selector):
        risks.append("contains long numeric token")
    if "css-" in lowered or "sc-" in lowered or ":r" in lowered:
        risks.append("contains dynamic style/react token")
    if "\\ " in selector:
        risks.append("contains escaped whitespace token")
    if len(selector) > 80:
        risks.append("very long selector")
    if re.fullmatch(r"[a-z][a-z0-9-]*", selector) and selector in _BROAD_TAGS:
        risks.append("overly broad tag selector")
    return risks


def _apply_penalties(base: int, risks: list[str]) -> int:
    score = base
    for risk in risks:
        if risk in {"contains long numeric token", "contains dynamic style/react token"}:
            score -= 15
        elif risk in {"contains escaped whitespace token", "very long selector"}:
            score -= 10
        elif risk == "overly broad tag selector":
            score -= 20
    return max(0, min(100, score))


def _score_band(score: int) -> str:
    if score >= 85:
        return "stable"
    if score >= 65:
        return "acceptable"
    if score >= 45:
        return "fragile"
    return "high_risk"


def analyze_workflow_payload(workflow: dict[str, Any]) -> dict[str, Any]:
    """Analyze selector quality from recorded workflow without executing commands."""
    commands = workflow.get("commands", [])
    selector_usage: dict[str, int] = {}
    for command in commands:
        if command.get("action") not in {"click", "type", "get_text"}:
            continue
        selector = str(command.get("params", {}).get("selector", "")).strip()
        if not selector:
            continue
        selector_usage[selector] = selector_usage.get(selector, 0) + 1

    command_reports: list[dict[str, Any]] = []
    scored_values: list[int] = []
    for command in commands:
        action = command.get("action")
        if action not in {"click", "type", "get_text"}:
            continue
        command_id = str(command.get("id", "unknown"))
        params = command.get("params", {})
        selector = str(params.get("selector", "")).strip()
        if not selector:
            command_reports.append(
                {
                    "command_id": command_id,
                    "action": action,
                    "score": 0,
                    "band": "high_risk",
                    "selector": "",
                    "reasons": ["missing selector"],
                }
            )
            scored_values.append(0)
            continue

        base, basis = _base_score(selector)
        reasons = [basis]
        risks = _selector_risks(selector)
        reasons.extend(risks)
        score = _apply_penalties(base, risks)

        duplicates = selector_usage.get(selector, 0)
        if duplicates > 1:
            dup_penalty = min(15, (duplicates - 1) * 5)
            score = max(0, score - dup_penalty)
            reasons.append(f"selector reused {duplicates} times")

        fallback_selectors = params.get("fallback_selectors")
        if isinstance(fallback_selectors, list) and fallback_selectors:
            score = min(100, score + 5)
            reasons.append("has fallback selectors")

        report = {
            "command_id": command_id,
            "action": action,
            "score": score,
            "band": _score_band(score),
            "selector": selector,
            "reasons": reasons,
        }
        command_reports.append(report)
        scored_values.append(score)

    average_score = round(sum(scored_values) / len(scored_values), 2) if scored_values else 0.0
    summary = {
        "selector_commands": len(scored_values),
        "average_score": average_score,
        "stable": sum(1 for item in command_reports if item["band"] == "stable"),
        "acceptable": sum(1 for item in command_reports if item["band"] == "acceptable"),
        "fragile": sum(1 for item in command_reports if item["band"] == "fragile"),
        "high_risk": sum(1 for item in command_reports if item["band"] == "high_risk"),
    }
    return {
        "task_id": workflow.get("task_id"),
        "schema_version": workflow.get("schema_version"),
        "summary": summary,
        "commands": command_reports,
    }


def analyze_workflow_file(path: Path) -> dict[str, Any]:
    """Load workflow JSON from disk and return static selector analysis report."""
    logger.info("Analyzing workflow selector quality from %s", path)
    workflow = json.loads(path.read_text(encoding="utf-8"))
    return analyze_workflow_payload(workflow)
