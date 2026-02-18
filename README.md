# BorisBot

Local-first multi-agent runtime framework.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Start the supervisor
borisbot start

# Check status
borisbot status

# Spawn an agent
borisbot spawn my_agent

# Stop the supervisor
borisbot stop

# Launch guided web UI (beginner flow)
borisbot guide

# Show provider/token/budget runtime status
borisbot session-status

# Run golden planner regression suite
borisbot golden-check
```

Then open the shown local URL (default `http://127.0.0.1:7788`) and use the step cards to:
- check Docker and run `verify`
- start/stop recording with a URL and task id
- run analyze/lint/replay
- run release-check (human and `--json`)

## Guided UI Highlights

- One-touch local LLM setup: install Ollama (if missing), start runtime, pull selected model, then refresh session status.
- Persistent runtime profile: save `agent_name`, `primary_provider`, `provider_chain` (max 5), and `model_name`.
- Permission matrix: set per-agent `prompt|allow|deny` for `browser`, `filesystem`, `shell`, `web_fetch`, and `scheduler`.
- Planner dry-run and chat: send natural-language prompts, get validated `planner.v1` preview, token estimate, and required permissions.
- Trace auditability: inspect compact trace list, open full trace detail, and export trace JSON.

## Recorder Error Guidance

`borisbot record` now prints actionable messages for common failures:

- Docker daemon unavailable
- Maximum browser sessions reached
- Recorder port already in use
