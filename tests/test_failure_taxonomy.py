"""Tests for versioned failure taxonomy and deterministic fingerprinting."""

import unittest

from borisbot.failures import build_failure, classify_failure


class FailureTaxonomyTests(unittest.TestCase):
    """Validate error.v1 payload shape and classification behavior."""

    def test_build_failure_is_deterministic(self) -> None:
        first = build_failure(
            error_class="timeout",
            error_code="TIMEOUT_OPERATION",
            step_id="2",
            url="https://example.com",
            message="timed out",
            selector="#submit",
        )
        second = build_failure(
            error_class="timeout",
            error_code="TIMEOUT_OPERATION",
            step_id="2",
            url="https://example.com",
            message="timed out",
            selector="#submit",
        )
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["error_schema_version"], "error.v1")

    def test_classifies_capability_rejection(self) -> None:
        failure = classify_failure(
            error=RuntimeError("missing capability: BROWSER"),
            step_id="guard",
        )
        self.assertEqual(failure["error_class"], "capability_rejected")
        self.assertEqual(failure["error_code"], "CAP_BROWSER_MISSING")

    def test_classifies_selector_ambiguity(self) -> None:
        failure = classify_failure(
            error=RuntimeError("strict mode violation: locator matched 2 elements"),
            step_id="4",
            action="click",
            selector="button",
        )
        self.assertEqual(failure["error_class"], "selector_ambiguous")
        self.assertEqual(failure["error_code"], "SEL_AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
