"""Tests for persistent runtime profile configuration."""

import tempfile
import unittest
from pathlib import Path

from borisbot.supervisor.profile_config import load_profile, save_profile, validate_profile


class ProfileConfigTests(unittest.TestCase):
    """Validate profile schema and provider-chain constraints."""

    def test_validate_profile_rejects_excess_chain(self) -> None:
        with self.assertRaises(ValueError):
            validate_profile(
                {
                    "schema_version": "profile.v1",
                    "agent_name": "a",
                    "primary_provider": "ollama",
                    "provider_chain": ["ollama", "openai", "anthropic", "google", "azure", "extra"],
                    "model_name": "llama3.2:3b",
                }
            )

    def test_save_and_load_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            profile = save_profile(
                {
                    "schema_version": "profile.v1",
                    "agent_name": "boris",
                    "primary_provider": "ollama",
                    "provider_chain": ["ollama", "openai"],
                    "model_name": "llama3.2:3b",
                },
                path=path,
            )
            loaded = load_profile(path=path)
            self.assertEqual(profile["agent_name"], loaded["agent_name"])
            self.assertEqual(loaded["provider_chain"], ["ollama", "openai"])


if __name__ == "__main__":
    unittest.main()

