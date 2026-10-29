"""Serial frame formatting / rate limiting (no hardware needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serial_link import SerialLink  # noqa: E402


class FakeLink(SerialLink):
    def __init__(self, **kw):
        self.port = "fake"
        self.baud = 115200
        self.min_interval = kw.get("min_interval", 0.0)
        self.heartbeat = kw.get("heartbeat", 3600.0)
        self.reset_wait = 0.0
        self.reconnect_interval = 3600.0
        self.verbose = False
        self._ser = object()          # pretend the port is open
        self._last_value = None
        self._last_sent = 0.0
        self._last_attempt = 0.0
        self._last_error = None
        self.frames_sent = 0
        self.frames = []

    def _write(self, text):
        self.frames.append(text)


class TestSerialFormat(unittest.TestCase):
    def test_format(self):
        self.assertEqual(SerialLink._format(0, []), "FC 0 0\n")
        self.assertEqual(SerialLink._format(3, [3]), "FC 3 1 3\n")
        self.assertEqual(SerialLink._format(5, [2, 3]), "FC 5 2 2 3\n")

    def test_format_with_sides(self):
        self.assertEqual(SerialLink._format(3, [3], ["R"]), "FC 3 1 3:R\n")
        self.assertEqual(
            SerialLink._format(5, [2, 3], ["L", "R"]), "FC 5 2 2:L 3:R\n"
        )
        self.assertEqual(SerialLink._format(2, [2], [None]), "FC 2 1 2\n")

    def test_only_send_on_change(self):
        link = FakeLink()
        self.assertTrue(link.send(3, [3]))
        self.assertFalse(link.send(3, [3]))
        self.assertTrue(link.send(4, [4]))
        self.assertEqual(link.frames, ["FC 3 1 3\n", "FC 4 1 4\n"])

    def test_heartbeat_resends_same_value(self):
        link = FakeLink(heartbeat=0.0)
        link.send(2, [2])
        link.send(2, [2])
        self.assertEqual(len(link.frames), 2)

    def test_two_hands(self):
        link = FakeLink()
        link.send(5, [2, 3], ["L", "R"])
        self.assertEqual(link.frames[0], "FC 5 2 2:L 3:R\n")

    def test_side_change_is_a_new_value(self):
        link = FakeLink()
        link.send(3, [3], [None])
        self.assertTrue(link.send(3, [3], ["R"]))
        self.assertFalse(link.send(3, [3], ["R"]))


if __name__ == "__main__":
    unittest.main()
