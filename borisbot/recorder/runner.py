"""Recorder runtime that captures browser events and immediately replays workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from uuid import uuid4

from borisbot.browser.actions import BrowserActions
from borisbot.browser.command_router import CommandRouter
from borisbot.browser.executor import BrowserExecutor
from borisbot.browser.task_runner import TaskRunner
from borisbot.recorder.server import RecordingServer
from borisbot.recorder.session import RecordingSession
from borisbot.supervisor.browser_manager import BrowserManager
from borisbot.supervisor.capability_manager import CapabilityManager

logger = logging.getLogger("borisbot.recorder.runner")

INJECTOR_JS = Path(__file__).parent / "injector.js"


async def run_record(task_id: str, output_dir: Path = Path("workflows")) -> dict:
    """Record a workflow and immediately replay it through TaskRunner."""
    session = RecordingSession(task_id=task_id)
    server = RecordingServer(session=session, port=7331)
    await server.start()

    manager = BrowserManager()
    agent_id = f"record_{task_id}"
    browser_session = None
    executor: BrowserExecutor | None = None

    try:
        capabilities = await CapabilityManager.get_capabilities(agent_id) or []
        has_browser_cap = any(row["capability_type"] == "BROWSER" for row in capabilities)
        if not has_browser_cap:
            await CapabilityManager.add_capability(agent_id, "BROWSER", "{}")

        browser_session = await manager.request_session(agent_id)
        executor = BrowserExecutor(browser_session["cdp_port"])
        await executor.connect()
        page = executor._require_page()
        context = page.context

        async def _record_event_binding(_source: object, event: dict) -> None:
            if not isinstance(event, dict):
                return
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            if isinstance(event_type, str):
                session.ingest(event_type, payload if isinstance(payload, dict) else {})

        await context.expose_binding("__BORIS_RECORD_EVENT__", _record_event_binding)

        injector_code = INJECTOR_JS.read_text(encoding="utf-8")
        injector_code = (
            "window.__BORIS_RECORD_HOST__ = 'host.docker.internal';\n"
            "window.__BORIS_RECORD_PORT__ = 7331;\n"
            f"{injector_code}"
        )
        await page.context.add_init_script(injector_code)
        await page.evaluate(injector_code)

        print(f"\nRecording task: {task_id}")
        print("Perform actions in browser. Press Ctrl+C when finished.\n")

        stop_event = asyncio.Event()

        def _stop() -> None:
            stop_event.set()

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, _stop)
        except NotImplementedError:
            logger.warning("Signal handlers not supported on this platform.")

        await stop_event.wait()

        workflow = session.finalize()
        current_url = page.url
        commands = workflow.get("commands", [])
        has_navigate = any(cmd.get("action") == "navigate" for cmd in commands)
        if commands and not has_navigate and isinstance(current_url, str) and current_url.strip():
            commands.insert(
                0,
                {
                    "id": "0",
                    "action": "navigate",
                    "params": {"url": current_url},
                },
            )
            for idx, cmd in enumerate(commands, start=1):
                cmd["id"] = str(idx)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.json"
        output_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

        print(f"\nRecorded {len(workflow['commands'])} commands")
        print(f"Saved -> {output_path}\n")

        print("Replaying recorded workflow...\n")
        replay_workflow = dict(workflow)
        replay_workflow["task_id"] = f"{task_id}_replay_{uuid4().hex[:8]}"
        actions = BrowserActions(executor)
        router = CommandRouter(actions)
        runner = TaskRunner(router, agent_id=agent_id, pre_persisted=False, worker_id="direct")
        result = await runner.run(replay_workflow)

        print("Replay result:")
        print(json.dumps(result, indent=2))
        if result.get("status") != "completed":
            print("\nReplay failed. Review selectors.\n")
        else:
            print("\nReplay successful.\n")
        return result
    finally:
        await server.stop()
        if executor is not None:
            await executor.close()
        try:
            await manager.stop_session(agent_id)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Failed to stop recorder session: %s", exc)
