# Global state for the supervisor
# Currently, most state is persisted in SQLite, but we might need
# in-memory tracking for active websocket connections or similar in the future.

class ProcessState:
    """
    In-memory state for tracking subprocess handles.
    This is NOT for persistent data, but for runtime handles (popen objects).
    """
    _processes = {}

    @classmethod
    def register(cls, agent_id: str, process):
        cls._processes[agent_id] = process

    @classmethod
    def get(cls, agent_id: str):
        return cls._processes.get(agent_id)

    @classmethod
    def remove(cls, agent_id: str):
        if agent_id in cls._processes:
            del cls._processes[agent_id]
