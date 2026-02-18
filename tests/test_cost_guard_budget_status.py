"""Tests for deterministic budget status threshold computation."""

import unittest

from borisbot.llm.cost_guard import compute_budget_status


class CostGuardBudgetStatusTests(unittest.TestCase):
    """Validate fixed threshold status mapping semantics."""

    def test_budget_status_ok(self) -> None:
        self.assertEqual(compute_budget_status(spend=1.0, limit=20.0), "ok")

    def test_budget_status_warn_50(self) -> None:
        self.assertEqual(compute_budget_status(spend=10.0, limit=20.0), "warn_50")

    def test_budget_status_warn_80(self) -> None:
        self.assertEqual(compute_budget_status(spend=16.0, limit=20.0), "warn_80")

    def test_budget_status_blocked(self) -> None:
        self.assertEqual(compute_budget_status(spend=20.0, limit=20.0), "blocked")


if __name__ == "__main__":
    unittest.main()

