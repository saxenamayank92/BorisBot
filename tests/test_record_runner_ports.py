"""Tests for recorder port selection behavior."""

import socket
import unittest

from borisbot.recorder.runner import _choose_recording_port


class RecorderPortSelectionTests(unittest.TestCase):
    """Ensure recorder can recover when default port is occupied."""

    def test_choose_port_falls_back_when_preferred_in_use(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        occupied_port = int(probe.getsockname()[1])
        probe.close()

        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", occupied_port))
        holder.listen(1)
        try:
            port = _choose_recording_port(preferred_port=occupied_port)
            self.assertNotEqual(port, occupied_port)
            self.assertGreater(port, 0)
        finally:
            holder.close()


if __name__ == "__main__":
    unittest.main()
