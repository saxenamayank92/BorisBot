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

## Change Process

1. Add new contract constant in `borisbot/contracts.py`.
2. Update producer(s) to emit new version.
3. Keep old version readable by consumers.
4. Add migration/compatibility tests before switching defaults.
