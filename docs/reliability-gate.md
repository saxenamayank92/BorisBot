# Reliability Gate

This checklist is a hard release gate for browser runtime stability.

## Acceptance Criteria

- 5 consecutive cold replays must pass.
- Runs must be distributed across 2 different UTC dates.
- Each run must be executed after a fresh container restart.
- No silent fallback is allowed unless replay is run with `--allow-fallback`.

## Run Matrix

| Run | UTC Date | Workflow File | Restart Performed | Replay Command | Result | Task ID | Notes |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |

## Required Evidence

For each run, record:

- `borisbot replay ...` output JSON.
- `borisbot inspect <task_id>` output JSON.
- `docker ps` excerpt showing fresh container lifecycle for the replay agent.

## Suggested Procedure

1. Restart supervised browser container/session for the replay agent.
2. Run `borisbot replay workflows/<workflow>.json`.
3. Capture returned task id from result payload.
4. Run `borisbot inspect <task_id>`.
5. Fill one matrix row only if status is `completed`.

If any run fails, reset consecutive count to zero.
