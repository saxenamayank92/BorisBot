"""Placeholder tests for BrowserManager scaffold surface area."""

import logging

from borisbot.supervisor.browser_manager import BrowserManager

logger = logging.getLogger(__name__)


def test_browser_manager_method_scaffold_exists() -> None:
    """Ensure BrowserManager exposes required lifecycle method stubs."""
    logger.info("Checking BrowserManager scaffold method availability.")
    expected_methods = (
        "request_session",
        "stop_session",
        "cleanup_orphan_containers",
        "health_check_sessions",
    )
    for method_name in expected_methods:
        assert hasattr(BrowserManager, method_name)
