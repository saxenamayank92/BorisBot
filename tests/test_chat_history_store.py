"""Tests for persistent guide chat history store."""

import tempfile
import unittest
from pathlib import Path

from borisbot.guide.chat_history_store import (
    append_chat_message,
    clear_chat_roles,
    clear_chat_history,
    load_chat_history,
)


class ChatHistoryStoreTests(unittest.TestCase):
    def test_append_and_load_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chat_history" / "agent.json"
            from borisbot.guide import chat_history_store as mod

            original = mod.CHAT_HISTORY_DIR
            mod.CHAT_HISTORY_DIR = Path(tmpdir) / "chat_history"
            try:
                append_chat_message("agent-a", "user", "hello")
                append_chat_message("agent-a", "planner", "world")
                items = load_chat_history("agent-a")
                self.assertEqual(len(items), 2)
                self.assertEqual(items[0]["role"], "user")
                self.assertEqual(items[1]["role"], "planner")
            finally:
                mod.CHAT_HISTORY_DIR = original

    def test_clear_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from borisbot.guide import chat_history_store as mod

            original = mod.CHAT_HISTORY_DIR
            mod.CHAT_HISTORY_DIR = Path(tmpdir) / "chat_history"
            try:
                append_chat_message("agent-a", "user", "hello")
                clear_chat_history("agent-a")
                items = load_chat_history("agent-a")
                self.assertEqual(items, [])
            finally:
                mod.CHAT_HISTORY_DIR = original

    def test_clear_chat_roles_only_removes_selected_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from borisbot.guide import chat_history_store as mod

            original = mod.CHAT_HISTORY_DIR
            mod.CHAT_HISTORY_DIR = Path(tmpdir) / "chat_history"
            try:
                append_chat_message("agent-a", "user", "hello")
                append_chat_message("agent-a", "assistant_user", "ask")
                append_chat_message("agent-a", "assistant", "answer")
                remaining = clear_chat_roles("agent-a", {"assistant_user", "assistant"})
                self.assertEqual(len(remaining), 1)
                self.assertEqual(remaining[0]["role"], "user")
            finally:
                mod.CHAT_HISTORY_DIR = original


if __name__ == "__main__":
    unittest.main()
