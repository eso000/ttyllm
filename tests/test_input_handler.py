"""Tests for the input handler: key bindings, line editing, scrolling and escape sequences."""

import os
import sys
import threading
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

from ai_chat import view as view_module
from ai_chat.input_handler import KEY_LABELS, InputHandler, Key
from ai_chat.models import Model, ModelList
from test_input_keys import content, press
from ai_chat.state import ViewMode
from ai_chat.supervisor import Supervisor


@contextmanager
def stdin_with(data: bytes, close: bool = True) -> Iterator[None]:
    """make sys.stdin a pipe holding data, optionally leaving the write end open"""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, data)
    if close:
        os.close(write_fd)
    stdin = mock.Mock()
    stdin.fileno.return_value = read_fd
    try:
        with mock.patch.object(sys, "stdin", stdin):
            yield
    finally:
        os.close(read_fd)
        if not close:
            os.close(write_fd)


class HandlerTestCase(unittest.TestCase):
    """Shared test case for handler."""

    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        self.sv: Supervisor = Supervisor()
        self.handler: InputHandler = self.sv.keyboard
        patcher = mock.patch.object(view_module, "fit_text_to_box")
        patcher.start()
        self.addCleanup(patcher.stop)

    def type(self, text: str) -> None:
        """Test helper: type."""
        for ch in text:
            self.handler.handle_key(ch)


class TestKeyTable(HandlerTestCase):
    """Tests for key table."""

    def test_every_key_has_an_action(self) -> None:
        """Every key has an action."""
        self.assertEqual({int(key) for key in Key}, set(self.handler.keys))

    def test_keys_are_the_documented_control_codes(self) -> None:
        """Keys are the documented control codes."""
        for key, letter in (
            (Key.CTRL_S, "S"),
            (Key.CTRL_B, "B"),
            (Key.CTRL_D, "D"),
            (Key.CTRL_N, "N"),
            (Key.CTRL_P, "P"),
            (Key.CTRL_R, "R"),
            (Key.CTRL_E, "E"),
            (Key.CTRL_Q, "Q"),
            (Key.CTRL_U, "U"),
            (Key.CTRL_C, "C"),
        ):
            with self.subTest(key=key.name):
                self.assertEqual(key, ord(letter) - 64)

    def test_actions_are_the_supervisor_methods(self) -> None:
        """Actions are the supervisor methods."""
        self.assertEqual(self.handler.keys[Key.CTRL_S], self.sv.split)
        self.assertEqual(self.handler.keys[Key.CTRL_B], self.sv.create_buffer)
        self.assertEqual(self.handler.keys[Key.CTRL_D], self.sv.buffer_delete)
        self.assertEqual(self.handler.keys[Key.CTRL_R], self.sv.redo)
        self.assertEqual(self.handler.keys[Key.CTRL_E], self.sv.undo)
        self.assertEqual(self.handler.keys[Key.CTRL_C], self.sv.handle_sigint)


class TestHandleKey(HandlerTestCase):
    """Tests for handle key."""

    def test_a_key_runs_its_action(self) -> None:
        """A key runs its action."""
        self.handler.handle_key("\x02")
        self.assertEqual(len(self.sv.buffers), 2)

    def test_redo_and_undo_run_without_the_ui_lock(self) -> None:
        """Redo and undo run without the UI lock, they can wait for a request."""
        for key in (Key.CTRL_R, Key.CTRL_E):
            with self.subTest(key=key.name):
                free = []

                def action() -> None:
                    """Test helper: action."""
                    taken = []

                    def take() -> None:
                        """Test helper: take the lock from another thread."""
                        if self.sv.ui.cond.acquire(timeout=1):
                            taken.append(True)
                            self.sv.ui.cond.release()

                    thread = threading.Thread(target=take)
                    thread.start()
                    thread.join(2)
                    free.append(bool(taken))

                self.handler.keys[key] = action
                self.handler.handle_key(chr(key))
                self.assertEqual(free, [True])

    def test_a_replaced_action_is_used(self) -> None:
        """A replaced action is used."""
        calls = []
        self.handler.keys[Key.CTRL_B] = lambda: calls.append("b")
        self.handler.handle_key("\x02")
        self.assertEqual(calls, ["b"])
        self.assertEqual(len(self.sv.buffers), 1)

    def test_ctrl_h_shows_the_help_view(self) -> None:
        """Ctrl h shows the help view."""
        self.handler.handle_key("\x08")
        self.assertIs(self.sv.ui.mode, ViewMode.HELP)
        self.assertEqual(self.sv.ui.scroll, 0)

    def test_printable_characters_are_typed(self) -> None:
        """Printable characters are typed."""
        self.type("hi")
        self.assertEqual(content(self.sv), "hi")

    def test_unknown_control_characters_are_ignored(self) -> None:
        """Unknown control characters are ignored."""
        self.type("\x01\x00\x0c")
        self.assertEqual(content(self.sv), "")
        self.assertEqual(self.sv.ui.command, "")

    def test_the_lock_is_released_when_an_action_fails(self) -> None:
        """The lock is released when an action fails."""

        def boom() -> None:
            """Test helper: boom."""
            raise RuntimeError("boom")

        self.handler.keys[Key.CTRL_B] = boom
        with self.assertRaises(RuntimeError):
            self.handler.handle_key("\x02")
        acquired: list[bool] = []
        thread = threading.Thread(
            target=lambda: acquired.append(self.sv.ui.cond.acquire(timeout=1))
        )
        thread.start()
        thread.join(2)
        self.assertEqual(acquired, [True])

    def _lock_is_held_by_another_thread_view(self) -> list[bool]:
        """probe from a second thread whether the input lock is currently held"""
        results: list[bool] = []

        def probe() -> None:
            """Test helper: probe."""
            got = self.sv.ui.cond.acquire(timeout=0.1)
            if got:
                self.sv.ui.cond.release()
            results.append(not got)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(2)
        return results

    def test_control_key_actions_run_with_the_input_lock_held(self) -> None:
        """Control key actions run with the input lock held."""
        seen: list[list[bool]] = []
        self.handler.keys[Key.CTRL_B] = lambda: seen.append(
            self._lock_is_held_by_another_thread_view()
        )
        self.handler.handle_key("\x02")
        self.assertEqual(seen, [[True]])

    def test_typing_runs_with_the_input_lock_held(self) -> None:
        """Typing runs with the input lock held."""
        seen: list[list[bool]] = []
        self.handler.type_char = lambda ch: seen.append(  # type: ignore[method-assign]
            self._lock_is_held_by_another_thread_view()
        )
        self.handler.handle_key("a")
        self.assertEqual(seen, [[True]])

    def test_enter_runs_with_the_input_lock_held(self) -> None:
        """Enter runs with the input lock held."""
        seen: list[list[bool]] = []
        self.handler.enter = lambda: seen.append(  # type: ignore[method-assign]
            self._lock_is_held_by_another_thread_view()
        )
        self.handler.handle_key("\r")
        self.assertEqual(seen, [[True]])

    def test_the_lock_is_free_between_keys(self) -> None:
        """The lock is free between keys."""
        self.handler.handle_key("a")
        self.assertEqual(self._lock_is_held_by_another_thread_view(), [False])

    def test_a_notification_swallows_exactly_one_key(self) -> None:
        """A notification swallows exactly one key."""
        self.sv.ui.notification = "boom"
        self.type("ab")
        self.assertEqual(content(self.sv), "b")


class TestLineEditing(HandlerTestCase):
    """Tests for line editing."""

    def test_toggle_command_line(self) -> None:
        """Toggle command line."""
        self.handler.handle_key("\x11")
        self.assertEqual(self.sv.ui.command, "/")
        self.handler.handle_key("\x11")
        self.assertEqual(self.sv.ui.command, "")

    def test_the_command_line_is_redrawn_after_an_edit(self) -> None:
        """The command line is redrawn after an edit."""
        with mock.patch.object(self.sv.view, "draw_command_line") as redraw:
            self.type(":a")
            self.handler.handle_key("\x7f")
        self.assertEqual(redraw.call_count, 3)

    def test_clearing_the_last_command_character_redraws_the_chat(self) -> None:
        """Clearing the last command character redraws the chat."""
        self.type(":")
        self.sv.current_buffer.updated = False
        self.handler.handle_key("\x7f")
        self.assertTrue(self.sv.current_buffer.updated)

    def test_backspace_on_a_message_returns_to_the_chat(self) -> None:
        """Backspace on a message returns to the chat."""
        self.sv.ui.set_mode(ViewMode.DATA, 4)
        self.handler.handle_key("\x7f")
        self.assertIs(self.sv.ui.mode, ViewMode.NORMAL)

    def test_tab_completes_with_the_known_models(self) -> None:
        """Tab completes with the known models."""
        self.sv.models = ModelList([Model("llama3.2")])
        self.type(":sm ll\t")
        self.assertEqual(self.sv.ui.command, ":sm llama3.2")

    def test_tab_does_not_redraw_when_nothing_completes(self) -> None:
        """Tab does not redraw when nothing completes."""
        self.type(":sm zz")
        with mock.patch.object(self.sv.view, "draw_command_line") as redraw:
            self.handler.handle_key("\t")
        redraw.assert_not_called()


class TestSubmit(HandlerTestCase):
    """Tests for submit."""

    def test_enter_signals_the_watcher(self) -> None:
        """Enter signals the watcher."""
        self.type("hi\r")
        self.assertTrue(self.sv.ui.submitted)

    def test_submit_wakes_a_waiting_thread(self) -> None:
        """Submit wakes a waiting thread."""
        woke = threading.Event()

        def wait() -> None:
            """Test helper: wait."""
            with self.sv.ui.cond:
                while not self.sv.ui.submitted:
                    self.sv.ui.cond.wait()
            woke.set()

        thread = threading.Thread(target=wait)
        thread.start()
        self.type("hi\r")
        self.assertTrue(woke.wait(2))
        thread.join(2)

    def test_quit_stops_before_submitting(self) -> None:
        """Quit stops before submitting."""
        self.sv.run = True
        self.type(":q\r")
        self.assertFalse(self.sv.run)
        self.assertTrue(self.sv.ui.submitted)

    def test_enter_in_a_view_does_not_submit(self) -> None:
        """Enter in a view does not submit."""
        self.sv.ui.set_mode(ViewMode.HELP, 0)
        self.handler.handle_key("\r")
        self.assertFalse(self.sv.ui.submitted)
        self.assertIs(self.sv.ui.mode, ViewMode.NORMAL)


class TestReadEscape(unittest.TestCase):
    """Tests for read escape."""

    def _read(self, data: bytes, close: bool = True) -> str | None:
        """run read_escape on a thread so that a hang fails the test, not the suite"""
        with stdin_with(data, close):
            result: list[str | None] = []
            thread = threading.Thread(
                target=lambda: result.append(InputHandler.read_escape()), daemon=True
            )
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive(), "read_escape did not return")
            return result[0]

    def test_arrow_sequences(self) -> None:
        """Arrow sequences."""
        for rest in ("[A", "[B", "[C", "[D", "OA", "OB"):
            with self.subTest(rest=rest):
                self.assertEqual(self._read(rest.encode()), rest)

    def test_longer_sequences_end_at_their_final_character(self) -> None:
        """Longer sequences end at their final character."""
        self.assertEqual(self._read(b"[1;5A"), "[1;5A")
        self.assertEqual(self._read(b"[3~"), "[3~")

    def test_it_stops_at_the_end_of_the_sequence(self) -> None:
        """It stops at the end of the sequence."""
        with stdin_with(b"[Axyz"):
            self.assertEqual(InputHandler.read_escape(), "[A")
            self.assertEqual(os.read(sys.stdin.fileno(), 10), b"xyz")

    def test_nothing_to_read_gives_none_without_blocking(self) -> None:
        """Nothing to read gives none without blocking."""
        self.assertIsNone(self._read(b"", close=False))

    def test_another_escape_cancels_the_sequence(self) -> None:
        """Another escape cancels the sequence."""
        self.assertIsNone(self._read(b"[\x1bA"))

    def test_stdin_closing_mid_sequence_gives_none(self) -> None:
        """Stdin closing mid sequence gives none."""
        self.assertIsNone(self._read(b"["))
        self.assertIsNone(self._read(b""))

    def test_bytes_that_are_not_utf8_do_not_crash(self) -> None:
        """Bytes that are not utf8 do not crash."""
        self.assertEqual(self._read(b"\xe4A"), "\ufffdA")


class TestApplyEscape(HandlerTestCase):
    """Tests for apply escape."""

    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        super().setUp()
        self.sv.ui.scroll = 5
        self.sv.ui.scroll_max = 30

    def test_up_down_left_right(self) -> None:
        """Up down left right."""
        for sequence, expected in (
            ("[A", 4),
            ("OA", 4),
            ("[B", 6),
            ("OB", 6),
            ("[D", -1),
            ("[C", 1),
        ):
            with self.subTest(sequence=sequence):
                self.sv.ui.scroll = 5
                self.handler.apply_escape(sequence)
                self.assertEqual(self.sv.ui.scroll, expected)

    def test_other_sequences_change_nothing(self) -> None:
        """Other sequences change nothing."""
        for sequence in ("[3~", "[1;5A", "[1;5D", "[E", ""):
            with self.subTest(sequence=sequence):
                self.handler.apply_escape(sequence)
                self.assertEqual(self.sv.ui.scroll, 5)


class TestEscapeKey(unittest.TestCase):
    """Tests for escape key."""

    def _run(self, data: bytes) -> tuple[Supervisor, list[BaseException | None]]:
        """Test helper: run."""
        errors: list[BaseException | None] = []
        old_hook = threading.excepthook
        threading.excepthook = lambda args: errors.append(args.exc_value)
        try:
            sv = press(data)
        finally:
            threading.excepthook = old_hook
        return sv, errors

    def test_a_lone_escape_at_the_end_of_input_is_harmless(self) -> None:
        """A lone escape at the end of input is harmless."""
        sv, errors = self._run(b"\x1b")
        self.assertEqual(errors, [])
        self.assertFalse(sv.run)

    def test_input_ending_inside_a_sequence_is_harmless(self) -> None:
        """Input ending inside a sequence is harmless."""
        sv, errors = self._run(b"ab\x1b[")
        self.assertEqual(errors, [])
        self.assertEqual(content(sv), "ab")

    def test_a_non_utf8_byte_inside_a_sequence_is_harmless(self) -> None:
        """A non utf8 byte inside a sequence is harmless."""
        sv, errors = self._run(b"a\x1b\xe4Ab")
        self.assertEqual(errors, [])
        self.assertEqual(content(sv), "ab")

    def test_escape_notifies_the_renderer_after_a_sequence(self) -> None:
        """Escape notifies the renderer after a sequence."""
        sv = Supervisor()
        sv.current_buffer.updated = False
        press(b"\x1b[B", sv)
        self.assertTrue(sv.current_buffer.updated)

    def test_escape_without_a_sequence_does_not_notify(self) -> None:
        """Escape without a sequence does not notify."""
        sv = Supervisor()
        sv.current_buffer.updated = False
        press(b"\x1b\x1b", sv)
        self.assertFalse(sv.current_buffer.updated)


class TestHelpText(HandlerTestCase):
    """Tests for help text."""

    def test_shortcuts_use_the_labels_and_action_descriptions(self) -> None:
        """Shortcuts use the labels and action descriptions."""
        text = self.handler.help_text()
        self.assertIn("shortcuts", text)
        self.assertIn("Ctrl+S:\tcreates new buffer from current", text)
        self.assertIn("Ctrl+Q:\topen the command line, or close it again", text)
        self.assertIn("Esc:\tscroll the view with the arrow keys", text)
        self.assertIn("Enter:\tsubmit message or command, or leave a view", text)
        self.assertIn("y / n:\tapprove / deny a tool call", text)

    def test_every_binding_is_listed_with_a_description(self) -> None:
        """Every binding is listed with a description."""
        text = self.handler.help_text()
        for code in self.handler.keys:
            with self.subTest(key=KEY_LABELS[code]):
                self.assertIn(f"{KEY_LABELS[code]}:\t", text)

    def test_no_shortcut_is_left_undescribed(self) -> None:
        """No shortcut is left undescribed."""
        for line in self.handler.help_text().splitlines()[2:]:
            with self.subTest(line=line):
                self.assertTrue(line.split(":\t", 1)[-1].strip())


if __name__ == "__main__":
    unittest.main()
