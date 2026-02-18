"""Tests for provider secret persistence and masking."""

import tempfile
import unittest
from pathlib import Path

from borisbot.supervisor.provider_secrets import get_secret_status, set_provider_secret


class ProviderSecretsTests(unittest.TestCase):
    def test_set_and_mask_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "provider_secrets.json"
            status = set_provider_secret("openai", "sk-test-key-12345678", path=path)
            self.assertTrue(status["openai"]["configured"])
            self.assertIn("...", str(status["openai"]["masked"]))

    def test_clear_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "provider_secrets.json"
            set_provider_secret("openai", "sk-test-key-12345678", path=path)
            status = set_provider_secret("openai", "", path=path)
            self.assertFalse(status["openai"]["configured"])
            status2 = get_secret_status(path=path)
            self.assertFalse(status2["openai"]["configured"])


if __name__ == "__main__":
    unittest.main()
