"""Unit tests for BrowserManager user-facing URL helpers."""

import unittest

from borisbot.supervisor.browser_manager import build_novnc_url


class BrowserManagerUrlTests(unittest.TestCase):
    """Validate noVNC URL format used by recorder/guide output."""

    def test_build_novnc_url_uses_direct_autoconnect_client(self) -> None:
        url = build_novnc_url(6080)
        self.assertEqual(
            url,
            "http://localhost:6080/vnc.html?autoconnect=1&resize=remote&reconnect=1",
        )


if __name__ == "__main__":
    unittest.main()

