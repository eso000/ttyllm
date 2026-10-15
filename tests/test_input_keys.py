"""Tests for keystroke handling end to end through the supervisor."""

import os
import sys
import tempfile
import threading
import unittest
from typing import Any
from unittest import mock

from ai_chat import view as view_module
from ai_chat.models import Model, ModelList
from ai_chat.state import ViewMode
from ai_chat.supervisor import Supervisor

CTRL_B, CTRL_C, CTRL_D, CTRL_E = b"\x02", b"\x03", b"\x04", b"\x05"
CTRL_N, CTRL_P, CTRL_Q, CTRL_R = b"\x0e", b"\x10", b"\x11", b"\x12"
CTRL_S, CTRL_U, TAB, BACKSPACE = b"\x13", b"\x15", b"\t", b"\x7f"
UP, DOWN, RIGHT, LEFT = b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D"


def press(data: bytes, sv: Supervisor | None = None, **state: Any) -> Supervisor:
    """feed raw bytes through the input thread until stdin closes"""
    sv = sv or Supervisor()
    sv.run = True
    for name, value in state.items():
        setattr(sv.ui, name, value)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, data)
    os.close(write_fd)
    stdin = mock.Mock()
    stdin.fileno.return_value = read_fd
    with (
        mock.patch.object(sys, "stdin", stdin),
        mock.patch.object(view_module, "fit_text_to_box"),
    ):
        thread = threading.Thread(target=sv.keyboard.run)
        thread.start()
        thread.join(2)
    os.close(read_fd)
    assert not thread.is_alive(), "input thread did not finish"
    return sv


def content(sv: Supervisor) -> str:
    """Test helper: content."""
    return str(sv.current_data["messages"][-1]["content"])


def pending_call() -> dict[str, Any]:
    """Test helper: pending call."""
    return {"id": "1", "name": "tool", "args": {}, "approved": None, "result": None}


class TestBufferKeys(unittest.TestCase):
    """Tests for buffer keys."""

    def test_ctrl_b_creates_a_buffer(self) -> None:
        """Ctrl b creates a buffer."""
        sv = press(CTRL_B)
        self.assertEqual(len(sv.buffers), 2)
        self.assertEqual(sv.current_buffer_id, 1)

    def test_ctrl_p_and_ctrl_n_switch_buffers(self) -> None:
        """Ctrl p and ctrl n switch buffers."""
        self.assertEqual(press(CTRL_B + CTRL_P).current_buffer_id, 0)
        self.assertEqual(press(CTRL_B + CTRL_P + CTRL_N).current_buffer_id, 1)

    def test_ctrl_s_splits_the_buffer(self) -> None:
        """Ctrl s splits the buffer."""
        sv = Supervisor()
        sv.current_data["model"] = "llama"
        press(CTRL_S, sv)
        self.assertEqual(len(sv.buffers), 2)
        self.assertEqual(sv.current_buffer_id, 1)
        self.assertEqual(sv.current_data["model"], "llama")

    def test_ctrl_d_deletes_the_buffer(self) -> None:
        """Ctrl d deletes the buffer."""
        sv = press(CTRL_B + CTRL_D)
        self.assertEqual(len(sv.buffers), 1)
        self.assertEqual(sv.current_buffer_id, 0)

    def test_ctrl_r_regenerates_by_queueing_the_buffer(self) -> None:
        """Ctrl r regenerates by queueing the buffer."""
        sv = press(CTRL_R)
        self.assertTrue(sv.queue.has_scheduled)

    def test_ctrl_e_reverts_to_an_empty_user_message(self) -> None:
        """Ctrl e reverts to an empty user message."""
        sv = press(b"hello" + CTRL_E)
        last = sv.current_data["messages"][-1]
        self.assertEqual((last["role"], last["content"]), ("user", ""))

    def test_ctrl_c_removes_the_buffer_from_the_queue(self) -> None:
        """Ctrl c removes the buffer from the queue."""
        sv = Supervisor()
        sv.queue.schedule(sv.current_buffer)
        self.assertTrue(sv.current_buffer.queued)
        press(CTRL_C, sv)
        self.assertFalse(sv.queue.has_scheduled)
        self.assertFalse(sv.current_buffer.queued)

    def test_unbound_control_keys_are_ignored(self) -> None:
        """Unbound control keys are ignored."""
        sv = press(b"\x01\x07\x0b")
        self.assertEqual(content(sv), "")
        self.assertEqual(sv.ui.command, "")
        self.assertEqual(len(sv.buffers), 1)


class TestCommandLine(unittest.TestCase):
    """Tests for command line."""

    def test_ctrl_q_toggles_command_mode(self) -> None:
        """Ctrl q toggles command mode."""
        self.assertEqual(press(CTRL_Q).ui.command, "/")
        self.assertEqual(press(CTRL_Q + CTRL_Q).ui.command, "")

    def test_colon_or_slash_at_an_empty_message_starts_a_command(self) -> None:
        """Colon or slash at an empty message starts a command."""
        self.assertEqual(press(b":sm x").ui.command, ":sm x")
        self.assertEqual(press(b"/sm x").ui.command, "/sm x")
        self.assertEqual(content(press(b":sm x")), "")

    def test_colon_inside_a_message_is_text(self) -> None:
        """Colon inside a message is text."""
        sv = press(b"a:b")
        self.assertEqual(content(sv), "a:b")
        self.assertEqual(sv.ui.command, "")

    def test_colon_while_a_request_runs_starts_a_command(self) -> None:
        """Colon while a request runs starts a command."""
        sv = Supervisor()
        sv.current_buffer.in_request = True
        sv.current_buffer.chat_data["messages"][-1]["content"] = "busy"
        press(b":", sv)
        self.assertEqual(sv.ui.command, ":")

    def test_ctrl_u_clears_the_command(self) -> None:
        """Ctrl u clears the command."""
        self.assertEqual(press(b":ab" + CTRL_U).ui.command, "")

    def test_ctrl_u_without_a_command_clears_the_message(self) -> None:
        """Ctrl u without a command clears the message."""
        self.assertEqual(content(press(b"hello" + CTRL_U)), "")

    def test_backspace_edits_the_command(self) -> None:
        """Backspace edits the command."""
        self.assertEqual(press(b":ab" + BACKSPACE).ui.command, ":a")
        self.assertEqual(press(b":" + BACKSPACE).ui.command, "")

    def test_backspace_without_a_command_edits_the_message(self) -> None:
        """Backspace without a command edits the message."""
        self.assertEqual(content(press(b"abc" + BACKSPACE)), "ab")

    def test_backspace_without_a_command_returns_to_the_chat(self) -> None:
        """Backspace without a command returns to the chat."""
        sv = press(BACKSPACE, mode=ViewMode.HELP, scroll=4)
        self.assertIs(sv.ui.mode, ViewMode.NORMAL)
        self.assertEqual(sv.ui.scroll, -1)


class TestTyping(unittest.TestCase):
    """Tests for typing."""

    def test_text_goes_into_the_message(self) -> None:
        """Text goes into the message."""
        sv = press(b"hi there")
        self.assertEqual(content(sv), "hi there")
        self.assertEqual(sv.ui.command, "")

    def test_text_is_not_added_while_a_request_runs(self) -> None:
        """Text is not added while a request runs."""
        sv = Supervisor()
        sv.current_buffer.in_request = True
        press(b"x", sv)
        self.assertEqual(content(sv), "")
        self.assertEqual(sv.ui.command, "")

    def test_typing_returns_to_the_chat(self) -> None:
        """Typing returns to the chat."""
        sv = press(b"a", mode=ViewMode.DATA, scroll=3)
        self.assertIs(sv.ui.mode, ViewMode.NORMAL)
        self.assertEqual(sv.ui.scroll, -1)


class TestEnter(unittest.TestCase):
    """Tests for enter."""

    def test_enter_submits_a_message(self) -> None:
        """Enter submits a message."""
        sv = press(b"hi\r")
        self.assertTrue(sv.ui.submitted)
        self.assertEqual(content(sv), "hi")

    def test_newline_submits_like_enter(self) -> None:
        """Newline submits like enter."""
        self.assertTrue(press(b"hi\n").ui.submitted)

    def test_enter_submits_a_command_and_keeps_it_for_the_watcher(self) -> None:
        """Enter submits a command and keeps it for the watcher."""
        sv = press(b":bc\r")
        self.assertTrue(sv.ui.submitted)
        self.assertEqual(sv.ui.command, ":bc")

    def test_quit_command_stops_the_app(self) -> None:
        """Quit command stops the app."""
        for command in (b":q\r", b"/q\r"):
            with self.subTest(command=command):
                sv = press(command)
                self.assertFalse(sv.run)
                self.assertTrue(sv.ui.submitted)

    def test_enter_in_another_view_goes_back_to_the_chat(self) -> None:
        """Enter in another view goes back to the chat."""
        sv = press(b"\r", mode=ViewMode.HELP, scroll=4)
        self.assertIs(sv.ui.mode, ViewMode.NORMAL)
        self.assertEqual(sv.ui.scroll, -1)
        self.assertFalse(sv.ui.submitted)


class TestToolApproval(unittest.TestCase):
    """Tests for tool approval."""

    def _press(self, data: bytes, **state: Any) -> tuple[Supervisor, dict[str, Any]]:
        """Test helper: press."""
        call = pending_call()
        sv = Supervisor()
        sv.current_buffer.set_pending([call])
        return press(data, sv, **state), call

    def test_y_approves_and_resumes_the_chat(self) -> None:
        """Y approves and resumes the chat."""
        sv, call = self._press(b"y")
        self.assertIs(call["approved"], True)
        self.assertTrue(sv.queue.has_scheduled)

    def test_n_denies(self) -> None:
        """N denies."""
        _, call = self._press(b"n")
        self.assertIs(call["approved"], False)

    def test_answers_are_ignored_while_typing_a_command(self) -> None:
        """Answers are ignored while typing a command."""
        sv, call = self._press(b":y")
        self.assertIsNone(call["approved"])
        self.assertEqual(sv.ui.command, ":y")

    def test_answers_are_ignored_outside_the_chat_view(self) -> None:
        """Answers are ignored outside the chat view."""
        _, call = self._press(b"y", mode=ViewMode.HELP)
        self.assertIsNone(call["approved"])

    def test_y_without_a_pending_call_is_text(self) -> None:
        """Y without a pending call is text."""
        self.assertEqual(content(press(b"yes")), "yes")


class TestNotification(unittest.TestCase):
    """Tests for notification."""

    def test_the_first_key_only_dismisses_it(self) -> None:
        """The first key only dismisses it."""
        sv = press(b"ab", notification="cannot reach server")
        self.assertEqual(sv.ui.notification, "")
        self.assertEqual(content(sv), "b")

    def test_a_dismissing_key_is_not_executed(self) -> None:
        """A dismissing key is not executed."""
        sv = press(CTRL_B, notification="boom")
        self.assertEqual(len(sv.buffers), 1)


class TestEscapeSequences(unittest.TestCase):
    """Tests for escape sequences."""

    def test_arrows_scroll(self) -> None:
        """Arrows scroll."""
        self.assertEqual(press(UP, scroll_max=30).ui.scroll, 29)
        self.assertEqual(press(DOWN, scroll=5, scroll_max=30).ui.scroll, 6)

    def test_application_mode_arrows_scroll_too(self) -> None:
        """Application mode arrows scroll too."""
        self.assertEqual(press(b"\x1bOA", scroll_max=30).ui.scroll, 29)
        self.assertEqual(press(b"\x1bOB", scroll=5, scroll_max=30).ui.scroll, 6)

    def test_left_and_right_jump_to_the_end_and_the_top(self) -> None:
        """Left and right jump to the end and the top."""
        self.assertEqual(press(RIGHT, scroll=7).ui.scroll, 1)
        self.assertEqual(press(LEFT, scroll=7).ui.scroll, -1)

    def test_arrows_are_ignored_while_typing_a_command(self) -> None:
        """Arrows are ignored while typing a command."""
        sv = press(b":" + UP, scroll=5, scroll_max=30)
        self.assertEqual(sv.ui.scroll, 5)

    def test_unknown_sequences_change_nothing(self) -> None:
        """Unknown sequences change nothing."""
        sv = press(b"\x1b[3~", scroll=5, scroll_max=30)
        self.assertEqual(sv.ui.scroll, 5)
        self.assertEqual(content(sv), "")

    def test_a_sequence_does_not_leak_into_the_message(self) -> None:
        """A sequence does not leak into the message."""
        self.assertEqual(content(press(b"a" + UP + b"b")), "ab")

    def test_escape_followed_by_escape_is_dropped(self) -> None:
        """Escape followed by escape is dropped."""
        self.assertEqual(content(press(b"\x1b\x1bx")), "x")


class TestTabCompletion(unittest.TestCase):
    """Tests for tab completion."""

    MODELS = ["llama3.2", "llama3.1", "mistral"]

    def _complete(self, command: str) -> str:
        """Test helper: complete."""
        sv = Supervisor()
        sv.models = ModelList([Model(name) for name in self.MODELS])
        return press(command.encode() + TAB, sv).ui.command

    def test_model_names_complete_to_the_common_prefix(self) -> None:
        """Model names complete to the common prefix."""
        self.assertEqual(self._complete(":sm ll"), ":sm llama3.")

    def test_a_single_model_completes_fully(self) -> None:
        """A single model completes fully."""
        self.assertEqual(self._complete(":sm mi"), ":sm mistral")

    def test_slash_commands_complete_too(self) -> None:
        """Slash commands complete too."""
        self.assertEqual(self._complete("/sm mi"), "/sm mistral")

    def test_no_match_changes_nothing(self) -> None:
        """No match changes nothing."""
        self.assertEqual(self._complete(":sm zz"), ":sm zz")

    def test_a_command_without_an_argument_is_not_completed(self) -> None:
        """A command without an argument is not completed."""
        self.assertEqual(self._complete(":sm"), ":sm")

    def test_other_commands_are_not_completed(self) -> None:
        """Other commands are not completed."""
        self.assertEqual(self._complete(":o st"), ":o st")

    def test_tab_without_a_command_does_nothing(self) -> None:
        """Tab without a command does nothing."""
        self.assertEqual(content(press(TAB)), "")
        self.assertEqual(press(TAB).ui.command, "")

    def test_file_paths_complete(self) -> None:
        """File paths complete."""
        with tempfile.TemporaryDirectory() as folder:
            for name in ("alpha.json", "alpine.json", "beta.json"):
                open(os.path.join(folder, name), "w", encoding="utf-8").close()
            os.mkdir(os.path.join(folder, "sub"))
            for command, expected in (
                (":e", "alp"),
                (":i", "alp"),
                (":image", "alp"),
            ):
                with self.subTest(command=command):
                    self.assertEqual(
                        self._complete(f"{command} {folder}/al"),
                        f"{command} {folder}/{expected}",
                    )
            self.assertEqual(
                self._complete(f":i {folder}/be"), f":i {folder}/beta.json"
            )
            self.assertEqual(self._complete(f":i {folder}/su"), f":i {folder}/sub/")


if __name__ == "__main__":
    unittest.main()
