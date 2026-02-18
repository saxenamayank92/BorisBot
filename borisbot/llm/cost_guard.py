"""Deterministic cloud cost guardrails backed by supervisor database state."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import time
from uuid import uuid4

from borisbot.supervisor.database import get_db

logger = logging.getLogger("borisbot.llm.cost_guard")

SETTINGS_REFRESH_SECONDS = 30
DEFAULT_DAILY_BUDGET_USD = 20.0
DEFAULT_MONTHLY_BUDGET_USD = 300.0
WARNING_THRESHOLDS = (0.5, 0.8)
STATUS_SEVERITY = {"ok": 0, "warn_50": 1, "warn_80": 2, "blocked": 3}


def compute_budget_status(
    *,
    spend: float,
    limit: float,
    warning_thresholds: tuple[float, float] = WARNING_THRESHOLDS,
) -> str:
    """Return deterministic budget status: ok|warn_50|warn_80|blocked."""
    if limit <= 0:
        return "blocked"
    ratio = spend / limit
    if ratio >= 1.0:
        return "blocked"
    if ratio >= warning_thresholds[1]:
        return "warn_80"
    if ratio >= warning_thresholds[0]:
        return "warn_50"
    return "ok"


class CostLimitError(RuntimeError):
    """RuntimeError carrying a deterministic limit type for API translation."""

    def __init__(self, message: str, limit_type: str):
        super().__init__(message)
        self.limit_type = limit_type


class CostGuard:
    """Enforces global/agent cost budgets and records model token usage."""

    def __init__(self) -> None:
        self._settings_cache: dict[str, str] = {}
        self._pricing_cache: dict[str, dict[str, float | str]] = {}
        self._last_refresh_monotonic: float = 0.0

    async def _refresh_cache_if_needed(self, force: bool = False) -> None:
        """Refresh in-memory settings/pricing cache on a fixed interval."""
        now = time.monotonic()
        if not force and now - self._last_refresh_monotonic < SETTINGS_REFRESH_SECONDS:
            return

        settings: dict[str, str] = {}
        pricing: dict[str, dict[str, float | str]] = {}
        async for db in get_db():
            cursor = await db.execute("SELECT key, value FROM system_settings")
            rows = await cursor.fetchall()
            settings = {row["key"]: row["value"] for row in rows}

            cursor = await db.execute(
                """
                SELECT model_name, provider, input_cost_per_1k_tokens, output_cost_per_1k_tokens
                FROM model_pricing
                """
            )
            rows = await cursor.fetchall()
            pricing = {
                row["model_name"]: {
                    "provider": row["provider"],
                    "input_cost_per_1k_tokens": float(row["input_cost_per_1k_tokens"]),
                    "output_cost_per_1k_tokens": float(row["output_cost_per_1k_tokens"]),
                }
                for row in rows
            }

        self._settings_cache = settings
        self._pricing_cache = pricing
        self._last_refresh_monotonic = now

    @staticmethod
    def _utc_day_bounds() -> tuple[str, str]:
        """Return ISO UTC day start inclusive and next day exclusive bounds."""
        now = datetime.utcnow()
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
        return start.isoformat(), end.isoformat()

    @staticmethod
    def _utc_month_bounds() -> tuple[str, str]:
        """Return ISO UTC month start inclusive and next month exclusive bounds."""
        now = datetime.utcnow()
        start = datetime(now.year, now.month, 1)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1)
        else:
            end = datetime(now.year, now.month + 1, 1)
        return start.isoformat(), end.isoformat()

    async def _sum_usage(self, where_clause: str, params: tuple[object, ...]) -> float:
        """Return rounded spend aggregation from cloud_usage."""
        query = f"SELECT COALESCE(SUM(cost_usd), 0) AS spend FROM cloud_usage WHERE {where_clause}"
        async for db in get_db():
            cursor = await db.execute(query, params)
            row = await cursor.fetchone()
            spend = float(row["spend"] if row else 0.0)
        return round(spend, 6)

    async def get_daily_spend(self) -> float:
        """Return global UTC daily spend."""
        start, end = self._utc_day_bounds()
        return await self._sum_usage("created_at >= ? AND created_at < ?", (start, end))

    async def get_monthly_spend(self) -> float:
        """Return global UTC monthly spend."""
        start, end = self._utc_month_bounds()
        return await self._sum_usage("created_at >= ? AND created_at < ?", (start, end))

    async def get_agent_daily_spend(self, agent_id: str) -> float:
        """Return per-agent UTC daily spend."""
        start, end = self._utc_day_bounds()
        return await self._sum_usage(
            "agent_id = ? AND created_at >= ? AND created_at < ?",
            (agent_id, start, end),
        )

    async def _insert_limit_event(self, task_id: str, limit_type: str, agent_id: str) -> None:
        """Persist cost limit event for SSE consumers."""
        try:
            now = datetime.utcnow().isoformat()
            payload = json.dumps({"limit_type": limit_type, "agent_id": agent_id})
            async for db in get_db():
                await db.execute(
                    """
                    INSERT INTO task_events (id, task_id, event_type, payload, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid4()), task_id, "system_cost_limit_reached", payload, now),
                )
                await db.commit()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to insert cost limit event: %s", exc)

    async def _insert_budget_warning_event(
        self,
        task_id: str,
        limit_type: str,
        threshold: float,
        spend: float,
        limit: float,
        agent_id: str,
    ) -> None:
        """Persist one-time budget warning event."""
        payload = json.dumps(
            {
                "limit_type": limit_type,
                "threshold": threshold,
                "spend_usd": round(spend, 6),
                "limit_usd": round(limit, 6),
                "agent_id": agent_id,
            }
        )
        now = datetime.utcnow().isoformat()
        async for db in get_db():
            await db.execute(
                """
                INSERT INTO task_events (id, task_id, event_type, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid4()), task_id, "system_cost_threshold_warning", payload, now),
            )
            await db.commit()
            return

    async def _mark_warning_once(
        self,
        *,
        key: str,
        task_id: str,
        limit_type: str,
        threshold: float,
        spend: float,
        limit: float,
        agent_id: str,
    ) -> None:
        """Insert a warning marker if missing, then emit warning event exactly once."""
        async for db in get_db():
            cursor = await db.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
            row = await cursor.fetchone()
            if row is not None:
                return
            try:
                await db.execute(
                    "INSERT INTO system_settings (key, value) VALUES (?, ?)",
                    (key, datetime.utcnow().isoformat()),
                )
                await db.commit()
                break
            except Exception:
                return
        await self._insert_budget_warning_event(
            task_id=task_id,
            limit_type=limit_type,
            threshold=threshold,
            spend=spend,
            limit=limit,
            agent_id=agent_id,
        )

    async def _emit_budget_warnings(
        self,
        *,
        task_id: str,
        agent_id: str,
        daily_spend: float,
        daily_limit: float,
        agent_spend: float,
        agent_daily_limit: float,
    ) -> None:
        """Emit one-time 50%/80% warnings for global and agent daily budgets."""
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        for threshold in WARNING_THRESHOLDS:
            if daily_limit > 0 and daily_spend >= daily_limit * threshold:
                warning_key = f"budget_warning.global_daily.{int(threshold*100)}.{date_key}"
                await self._mark_warning_once(
                    key=warning_key,
                    task_id=task_id,
                    limit_type="global_daily",
                    threshold=threshold,
                    spend=daily_spend,
                    limit=daily_limit,
                    agent_id=agent_id,
                )
            if agent_daily_limit > 0 and agent_spend >= agent_daily_limit * threshold:
                warning_key = f"budget_warning.agent_daily.{agent_id}.{int(threshold*100)}.{date_key}"
                await self._mark_warning_once(
                    key=warning_key,
                    task_id=task_id,
                    limit_type="agent_daily",
                    threshold=threshold,
                    spend=agent_spend,
                    limit=agent_daily_limit,
                    agent_id=agent_id,
                )

    async def get_runtime_session_started_at(self) -> str:
        """Return persisted runtime session start timestamp or UTC day start fallback."""
        async for db in get_db():
            cursor = await db.execute(
                "SELECT value FROM system_settings WHERE key = 'runtime_session_started_at'"
            )
            row = await cursor.fetchone()
            if row and isinstance(row["value"], str) and row["value"].strip():
                return row["value"]
        start, _ = self._utc_day_bounds()
        return start

    async def get_usage_window(
        self,
        *,
        start_iso: str,
        end_iso: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, float]:
        """Return usage aggregation for a time window."""
        where = ["created_at >= ?"]
        params: list[object] = [start_iso]
        if end_iso:
            where.append("created_at < ?")
            params.append(end_iso)
        if agent_id:
            where.append("agent_id = ?")
            params.append(agent_id)
        clause = " AND ".join(where)
        async for db in get_db():
            cursor = await db.execute(
                f"""
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cost_usd), 0) AS spend
                FROM cloud_usage
                WHERE {clause}
                """,
                tuple(params),
            )
            row = await cursor.fetchone()
            return {
                "input_tokens": int(row["input_tokens"] if row else 0),
                "output_tokens": int(row["output_tokens"] if row else 0),
                "total_tokens": int((row["input_tokens"] if row else 0) + (row["output_tokens"] if row else 0)),
                "cost_usd": round(float(row["spend"] if row else 0.0), 6),
            }
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}

    async def check_can_execute(self, agent_id: str, task_id: str | None = None) -> None:
        """Raise RuntimeError when global/agent cloud budgets are exceeded."""
        await self._refresh_cache_if_needed()

        daily_limit = round(
            float(
                self._settings_cache.get(
                    "system_daily_limit_usd",
                    self._settings_cache.get("daily_budget_usd", DEFAULT_DAILY_BUDGET_USD),
                )
            ),
            6,
        )
        monthly_limit = round(float(self._settings_cache.get("monthly_budget_usd", DEFAULT_MONTHLY_BUDGET_USD)), 6)
        agent_daily_limit = round(
            float(
                self._settings_cache.get(
                    "agent_daily_limit_usd",
                    self._settings_cache.get("agent_daily_budget_usd", str(daily_limit)),
                )
            ),
            6,
        )

        daily_spend = await self.get_daily_spend()
        agent_spend = await self.get_agent_daily_spend(agent_id)
        await self._emit_budget_warnings(
            task_id=task_id or f"system_cost_guard:{agent_id}",
            agent_id=agent_id,
            daily_spend=daily_spend,
            daily_limit=daily_limit,
            agent_spend=agent_spend,
            agent_daily_limit=agent_daily_limit,
        )

        if daily_spend >= daily_limit:
            await self._insert_limit_event(task_id or f"system_cost_guard:{agent_id}", "global_daily", agent_id)
            raise CostLimitError("global daily cost limit reached", "global_daily")

        monthly_spend = await self.get_monthly_spend()
        if monthly_spend >= monthly_limit:
            await self._insert_limit_event(task_id or f"system_cost_guard:{agent_id}", "global_monthly", agent_id)
            raise CostLimitError("global monthly cost limit reached", "global_monthly")

        if agent_spend >= agent_daily_limit:
            await self._insert_limit_event(task_id or f"system_cost_guard:{agent_id}", "agent_daily", agent_id)
            raise CostLimitError("agent daily cost limit reached", "agent_daily")

    async def record_usage(
        self,
        agent_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record cloud usage row and return rounded computed USD cost."""
        await self._refresh_cache_if_needed()
        pricing = self._pricing_cache.get(model_name)
        if pricing is None:
            raise RuntimeError("unknown model pricing")

        input_cost = float(pricing["input_cost_per_1k_tokens"])
        output_cost = float(pricing["output_cost_per_1k_tokens"])
        cost_usd = round((input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost, 6)

        now = datetime.utcnow().isoformat()
        async for db in get_db():
            await db.execute(
                """
                INSERT INTO cloud_usage (
                    id, agent_id, model_name, input_tokens, output_tokens, cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), agent_id, model_name, input_tokens, output_tokens, cost_usd, now),
            )
            await db.commit()
        return cost_usd

    async def get_spend_snapshot(self) -> dict[str, float]:
        """Return current rounded spend and configured budget limits."""
        await self._refresh_cache_if_needed()
        daily_limit = round(
            float(
                self._settings_cache.get(
                    "system_daily_limit_usd",
                    self._settings_cache.get("daily_budget_usd", DEFAULT_DAILY_BUDGET_USD),
                )
            ),
            6,
        )
        monthly_limit = round(float(self._settings_cache.get("monthly_budget_usd", DEFAULT_MONTHLY_BUDGET_USD)), 6)
        daily_spend = await self.get_daily_spend()
        monthly_spend = await self.get_monthly_spend()
        return {
            "daily_spend": daily_spend,
            "monthly_spend": monthly_spend,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
        }

    async def get_budget_status(self, agent_id: str) -> dict[str, float | str]:
        """Return deterministic budget status snapshot for CLI/API."""
        await self._refresh_cache_if_needed()
        daily_limit = round(
            float(
                self._settings_cache.get(
                    "system_daily_limit_usd",
                    self._settings_cache.get("daily_budget_usd", DEFAULT_DAILY_BUDGET_USD),
                )
            ),
            6,
        )
        agent_daily_limit = round(
            float(
                self._settings_cache.get(
                    "agent_daily_limit_usd",
                    self._settings_cache.get("agent_daily_budget_usd", str(daily_limit)),
                )
            ),
            6,
        )
        daily_spend = await self.get_daily_spend()
        agent_spend = await self.get_agent_daily_spend(agent_id)
        global_status = compute_budget_status(spend=daily_spend, limit=daily_limit)
        agent_status = compute_budget_status(spend=agent_spend, limit=agent_daily_limit)
        combined_status = max((global_status, agent_status), key=lambda status: STATUS_SEVERITY[status])
        return {
            "daily_spend": daily_spend,
            "daily_limit": daily_limit,
            "daily_remaining": round(max(0.0, daily_limit - daily_spend), 6),
            "agent_daily_spend": agent_spend,
            "agent_daily_limit": agent_daily_limit,
            "global_status": global_status,
            "agent_status": agent_status,
            "status": combined_status,
        }

    async def get_agent_spend_today(self) -> dict[str, float]:
        """Return UTC daily spend grouped by agent."""
        start, end = self._utc_day_bounds()
        async for db in get_db():
            cursor = await db.execute(
                """
                SELECT agent_id, COALESCE(SUM(cost_usd), 0) AS spend
                FROM cloud_usage
                WHERE created_at >= ? AND created_at < ?
                GROUP BY agent_id
                """,
                (start, end),
            )
            rows = await cursor.fetchall()
        return {row["agent_id"]: round(float(row["spend"]), 6) for row in rows}

    async def reset_daily_usage(self) -> None:
        """Delete current UTC day cloud usage rows."""
        start, end = self._utc_day_bounds()
        async for db in get_db():
            await db.execute(
                "DELETE FROM cloud_usage WHERE created_at >= ? AND created_at < ?",
                (start, end),
            )
            await db.commit()
