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

# One-command Ollama setup (install/start/pull)
borisbot llm-setup --model llama3.2:3b
# Machine-readable setup output
borisbot llm-setup --model llama3.2:3b --json

# Show provider/token/budget runtime status
borisbot session-status
# JSON status snapshot (provider-aware)
borisbot session-status --json

# Detailed spend + budget snapshot
borisbot budget-status --agent-id default
# Update budget limits
borisbot budget-set --system-daily-limit-usd 10 --agent-daily-limit-usd 5 --monthly-limit-usd 100

# Trace inspection from guide runtime
borisbot trace-list
borisbot trace-show trace_00001 --json

# Export support diagnostics bundle
borisbot support-bundle

# Run golden planner regression suite
borisbot golden-check

# Run CLI planner dry-run preview
borisbot plan-preview "Open example.com and read page title" --provider ollama

# Run CLI non-execution assistant chat
borisbot assistant-chat "Summarize deterministic browser automation tradeoffs" --provider ollama --approve-permission
# View persisted chat history
borisbot chat-history --agent-id default
# Clear assistant-only chat history
borisbot chat-clear --agent-id default --assistant-only

# Show provider readiness matrix in CLI
borisbot provider-status

# Probe connectivity for selected provider/model
borisbot provider-test --provider openai --model gpt-4o-mini
```

Then open the shown local URL (default `http://127.0.0.1:7788`) and use the step cards to:
- check Docker and run `verify`
- start/stop recording with a URL and task id
- run analyze/lint/replay
- run release-check (human and `--json`)

## Guided UI Highlights

- One-touch local LLM setup: install Ollama (if missing), start runtime, pull selected model, then refresh session status.
- Persistent runtime profile: save `agent_name`, `primary_provider`, `provider_chain` (max 5), `model_name`, and per-provider settings.
- API provider onboarding: configure `openai`, `anthropic`, `google`, and `azure` keys in GUI; keys are stored locally in `~/.borisbot/provider_secrets.json` and shown masked in UI.
- Planner provider transport support: `ollama`, `openai`, `anthropic`, `google`, and `azure` are implemented for dry-run generation (Azure additionally requires `BORISBOT_AZURE_OPENAI_ENDPOINT`).
- Permission matrix: set per-agent `prompt|allow|deny` for `assistant`, `browser`, `filesystem`, `shell`, `web_fetch`, and `scheduler`.
- Planner dry-run and chat: send natural-language prompts, get validated `planner.v1` preview, token estimate, provider-aware cost estimate, required permissions, and persistent per-agent chat history.
- Assistant-to-planner handoff: convert assistant trace output into a planner prompt draft; review and run dry-run before execution.
- Budget safety: dry-run planner is blocked when budget state is `blocked`.
- Budget controls: CLI + guide support budget status inspection and runtime limit updates.
- Runtime provider panel: GUI shows per-provider enabled/configured/usable state with quick diagnostics.
- Provider connectivity check: GUI button `Test Primary Provider` runs an immediate probe for current provider credentials/connectivity.
- Trace auditability: inspect compact trace list, open full trace detail, export trace JSON, and export support bundles.
- First-run wizard: guide tracks onboarding completion state (`Docker`, `LLM`, provider, permissions, first preview).
- Task inbox + scheduler: capture intents, prioritize work, and enqueue recurring intents on an interval.

## Recorder Error Guidance

`borisbot record` now prints actionable messages for common failures:

- Docker daemon unavailable
- Maximum browser sessions reached
- Recorder port already in use
