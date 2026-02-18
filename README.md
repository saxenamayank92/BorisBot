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
```

Then open the shown local URL (default `http://127.0.0.1:7788`) and use the step cards to:
- check Docker and run `verify`
- start/stop recording with a URL and task id
- run analyze/lint/replay
- run release-check (human and `--json`)
