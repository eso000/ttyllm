"""Tests for raw terminal mode and the alternate screen."""

import contextlib
import io
import os
import pty
import sys
import termios
import unittest
from unittest import mock

from ai_chat.terminal import raw_terminal

class TestRawTerminal(unittest.TestCase):
    """Tests for raw terminal."""
    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        self.fd = slave
        self.stdin = mock.Mock()
        self.stdin.fileno.return_value = slave
        patcher = mock.patch.object(sys, "stdin", self.stdin)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _lflag(self) -> int:
        """Test helper: lflag."""
        return int(termios.tcgetattr(self.fd)[3])

    def test_input_is_raw_inside_and_restored_after(self) -> None:
        """Input is raw inside and restored after."""
        before = termios.tcgetattr(self.fd)
        wanted = termios.ICANON | termios.ECHO | termios.ISIG
        self.assertEqual(self._lflag() & wanted, wanted)
        with contextlib.redirect_stdout(io.StringIO()):
            with raw_terminal():
                self.assertEqual(self._lflag() & wanted, 0)
        self.assertEqual(termios.tcgetattr(self.fd), before)

    def test_alternate_screen_is_entered_and_left(self) -> None:
        """Alternate screen is entered and left."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with raw_terminal():
                self.assertIn("\033[?1049h", out.getvalue())
                self.assertNotIn("\033[?1049l", out.getvalue())
        self.assertIn("\033[?1049l", out.getvalue())
        self.assertIn("\033[?25h", out.getvalue())

    def test_terminal_is_restored_after_an_error(self) -> None:
        """Terminal is restored after an error."""
        before = termios.tcgetattr(self.fd)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                with raw_terminal():
                    raise RuntimeError("boom")
        self.assertEqual(termios.tcgetattr(self.fd), before)

    def test_non_tty_leaves_the_screen_untouched(self) -> None:
        """Non tty leaves the screen untouched."""
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        self.stdin.fileno.return_value = read_fd
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(termios.error):
                with raw_terminal():
                    pass
        self.assertEqual(out.getvalue(), "")

if __name__ == "__main__":
    unittest.main()
