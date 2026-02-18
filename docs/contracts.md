# BorisBot Contract Freeze

This document defines versioned contracts that are treated as stable runtime interfaces.

## Compatibility Policy

- Contract changes are additive by default.
- Breaking changes require a new schema version.
- CLI replay must reject unsupported schema versions.
- Existing persisted tasks/results/events must remain readable.

## Workflow Command Contract

- Version key: `schema_version`
- Current version: `task_command.v1`
- Shape:

```json
{
  "schema_version": "task_command.v1",
  "task_id": "string",
  "commands": [
    {
      "id": "string",
      "action": "navigate|click|type|wait_for_url|get_text|get_title",
      "params": {}
    }
  ]
}
```

Notes:
- `fallback_selectors` in `params` is optional and only used when replay is run with `--allow-fallback`.
- Optional `idempotency_key` on submitted task payload enables retry-safe enqueue semantics:
  - same key + same payload hash => deduplicated task response
  - same key + different payload hash => idempotency conflict

## Task Result Contract

- Version key: `schema_version`
- Current version: `task_result.v1`
- Shape:

```json
{
  "schema_version": "task_result.v1",
  "task_id": "string",
  "status": "completed|failed|rejected",
  "duration_ms": 0,
  "steps": []
}
```

## Task Event Payload Contract

- Payload version key: `schema_version`
- Current payload version: `task_event.v1`
- Events are emitted under `task_events.event_type` with JSON payload.

Example payload:

```json
{
  "schema_version": "task_event.v1",
  "command_id": "2",
  "status": "ok",
  "duration_ms": 125
}
```

## Failure Taxonomy Contract

- Version key: `error_schema_version`
- Current version: `error.v1`
- Required fields on failures:

```json
{
  "error_schema_version": "error.v1",
  "error_class": "navigation_validation|selector_not_found|selector_ambiguous|interaction_failed|timeout|capability_rejected|cost_rejected",
  "error_code": "string",
  "step_id": "string",
  "url": "string",
  "message": "string",
  "fingerprint": "sha256 hex"
}
```

Fingerprint input format:

`sha256(error_class + "|" + error_code + "|" + step_id + "|" + selector_or_empty + "|" + url_or_empty)`

URL normalization for fingerprint input:

- Lowercase hostname
- Host + path only (no protocol)
- Strip query string and fragment
- Strip trailing slash except root
- Example:
  - `https://www.linkedin.com/feed/?trk=xyz#top` -> `www.linkedin.com/feed`

## LLM Planner Contract

- Version key: `planner_schema_version`
- Current version: `planner.v1`
- Required top-level fields (exact set):
  - `planner_schema_version` (`planner.v1`)
  - `intent` (non-empty string)
  - `proposed_actions` (array)
- Required action fields (exact set):
  - `action` (string)
  - `target` (string)
  - `input` (string)

Planner output parsing rules:

- Accept strict JSON object only.
- Perform one repair pass only (extract JSON object and close missing braces).
- If still invalid after one pass, fail with `LLM_OUTPUT_INVALID`.
- Extra or missing fields are rejected.

## LLM Failure Codes

Frozen LLM failure taxonomy additions:

- `error_class=llm_provider`, `error_code=LLM_PROVIDER_UNHEALTHY`
- `error_class=llm_provider`, `error_code=LLM_TIMEOUT`
- `error_class=llm_invalid_output`, `error_code=LLM_OUTPUT_INVALID`

## Change Process

1. Add new contract constant in `borisbot/contracts.py`.
2. Update producer(s) to emit new version.
3. Keep old version readable by consumers.
4. Add migration/compatibility tests before switching defaults.
