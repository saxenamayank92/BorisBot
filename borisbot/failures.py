"""Deterministic failure taxonomy and fingerprint utilities."""

from __future__ import annotations

import hashlib

from borisbot.contracts import ERROR_SCHEMA_V1


def build_failure(
    *,
    error_class: str,
    error_code: str,
    step_id: str,
    url: str,
    message: str,
    selector: str = "",
) -> dict[str, str]:
    """Build a stable failure payload for runtime outputs and persisted reports."""
    fingerprint_input = "|".join(
        [
            error_class,
            error_code,
            step_id or "",
            selector or "",
            url or "",
        ]
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    return {
        "error_schema_version": ERROR_SCHEMA_V1,
        "error_class": error_class,
        "error_code": error_code,
        "step_id": step_id,
        "url": url or "",
        "message": message,
        "fingerprint": fingerprint,
    }


def classify_failure(
    *,
    error: Exception,
    step_id: str,
    action: str = "",
    selector: str = "",
    url: str = "",
) -> dict[str, str]:
    """Classify runtime exception into versioned error taxonomy."""
    message = str(error)
    lower = message.lower()

    if "missing capability: browser" in lower:
        return build_failure(
            error_class="capability_rejected",
            error_code="CAP_BROWSER_MISSING",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )
    if "navigate required before interaction" in lower:
        return build_failure(
            error_class="capability_rejected",
            error_code="CAP_NAV_REQUIRED",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )
    if "domain not allowed:" in lower:
        return build_failure(
            error_class="capability_rejected",
            error_code="CAP_DOMAIN_NOT_ALLOWED",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )

    if "global daily cost limit reached" in lower:
        return build_failure(
            error_class="cost_rejected",
            error_code="COST_GLOBAL_DAILY_LIMIT",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )
    if "global monthly cost limit reached" in lower:
        return build_failure(
            error_class="cost_rejected",
            error_code="COST_GLOBAL_MONTHLY_LIMIT",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )
    if "agent daily cost limit reached" in lower:
        return build_failure(
            error_class="cost_rejected",
            error_code="COST_AGENT_DAILY_LIMIT",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )

    if "empty title" in lower or "invalid ready state" in lower:
        return build_failure(
            error_class="navigation_validation",
            error_code="NAV_INVALID_READY_STATE",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )

    if "strict mode" in lower:
        return build_failure(
            error_class="selector_ambiguous",
            error_code="SEL_AMBIGUOUS",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )
    if "not found" in lower or "waiting for selector" in lower:
        return build_failure(
            error_class="selector_not_found",
            error_code="SEL_NOT_FOUND",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )

    if "timeout" in lower:
        return build_failure(
            error_class="timeout",
            error_code="TIMEOUT_OPERATION",
            step_id=step_id,
            selector=selector,
            url=url,
            message=message,
        )

    default_code = "ACT_COMMAND_FAILED"
    if action == "click":
        default_code = "ACT_CLICK_FAILED"
    elif action == "type":
        default_code = "ACT_TYPE_FAILED"
    elif action == "navigate":
        default_code = "NAV_INVALID_READY_STATE"

    return build_failure(
        error_class="interaction_failed",
        error_code=default_code,
        step_id=step_id,
        selector=selector,
        url=url,
        message=message,
    )
