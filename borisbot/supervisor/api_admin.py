"""Supervisor admin endpoints for cloud cost guardrail management."""

import logging

from fastapi import APIRouter

from borisbot.llm.cost_guard import CostGuard

logger = logging.getLogger("borisbot.supervisor.api_admin")

router = APIRouter()
cost_guard = CostGuard()


@router.get("/admin/cost")
async def get_admin_cost():
    """Return current global spend and configured cost limits."""
    snapshot = await cost_guard.get_spend_snapshot()
    return {
        "daily_spend": snapshot["daily_spend"],
        "monthly_spend": snapshot["monthly_spend"],
        "daily_limit": snapshot["daily_limit"],
        "monthly_limit": snapshot["monthly_limit"],
    }


@router.post("/admin/cost/reset")
async def reset_admin_cost():
    """Reset current UTC day cloud usage accounting rows."""
    await cost_guard.reset_daily_usage()
    return {"status": "reset"}
