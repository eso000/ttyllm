"""Tests for the request queue, worker threads and cancellation."""

import os
import sys
import threading
import time
import unittest
from typing import Any
from unittest import mock

from ai_chat import supervisor as supervisor_module
from ai_chat import view as view_module
from ai_chat.buffer import Buffer
from ai_chat.requester import Requester
from ai_chat.request_queue import RequestQueue
from ai_chat.state import ViewMode
from ai_chat.supervisor import Supervisor
from test_requester import _http_response, _serve_once

class TestRenderThread(unittest.TestCase):
    """Tests for render thread."""
    def test_survives_rapid_buffer_switching(self) -> None:
        """Survives rapid buffer switching."""
        sv = Supervisor()
        sv.buffers.append(Buffer())
        sv.run = True
        errors = []
        old_hook = threading.excepthook
        old_interval = sys.getswitchinterval()
        threading.excepthook = lambda args: errors.append(args.exc_value)
        sys.setswitchinterval(1e-6)
        try:
            with mock.patch.object(
                view_module, "fit_to_box", lambda *a, **k: (10, -1)
            ), mock.patch.object(view_module, "fit_text_to_box"):
                render = threading.Thread(target=sv.update_buffer)
                render.start()
                deadline = time.time() + 1
                switches = 0
                while time.time() < deadline and render.is_alive():
                    sv.change_buffer(switches % 2)
                    switches += 1
                alive = render.is_alive()
                sv.stop()
                render.join(2)
        finally:
            sys.setswitchinterval(old_interval)
            threading.excepthook = old_hook
        self.assertEqual(errors, [])
        self.assertTrue(alive)
        self.assertFalse(render.is_alive())

    def test_scroll_during_render_is_not_overwritten(self) -> None:
        """Scroll during render is not overwritten."""
        sv = Supervisor()
        sv.run = True
        sv.ui.scroll = 0
        sv.current_buffer.updated = False  # a new buffer starts out as updated

        def scrolling_draw(*args: Any, **kwargs: Any) -> tuple[int, int]:
            """Test helper: scrolling draw."""
            sv.ui.scroll = 7  # user scrolls while the screen is drawn
            return (10, 3)

        with mock.patch.object(
            view_module, "fit_to_box", scrolling_draw
        ), mock.patch.object(view_module, "fit_text_to_box"):
            render = threading.Thread(target=sv.update_buffer)
            render.start()
            sv.current_buffer.notify()
            deadline = time.time() + 2
            while sv.ui.scroll_max == -1 and time.time() < deadline:
                time.sleep(0.01)
            sv.stop()
            render.join(2)
        self.assertEqual(sv.ui.scroll_max, 10)
        self.assertEqual(sv.ui.scroll, 7)

class TestCancel(unittest.TestCase):
    """Tests for cancel."""
    def test_idle_cancel_leaves_nothing_behind(self) -> None:
        """Idle cancel leaves nothing behind."""
        chat = Buffer()
        for _ in range(5):
            chat.cancel_request()
        self.assertFalse(chat.cancel)
        with self.assertRaises(BlockingIOError):
            os.read(chat.pipe[0], 1)

    def test_cancel_during_request_is_recorded_and_cleared(self) -> None:
        """Cancel during request is recorded and cleared."""
        chat = Buffer()
        chat.in_request = True
        for _ in range(3):
            chat.cancel_request()
        self.assertTrue(chat.cancel)
        chat.clear_cancel()
        self.assertFalse(chat.cancel)
        with self.assertRaises(BlockingIOError):
            os.read(chat.pipe[0], 1)

    def test_next_request_is_not_aborted_by_old_cancels(self) -> None:
        """Next request is not aborted by old cancels."""
        body = {
            "message": {"content": "hi"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        port, _ = _serve_once(_http_response(body))
        chat = Buffer()
        chat.in_request = True
        for _ in range(3):
            chat.cancel_request()
        chat.in_request = False
        self.assertIsNone(Requester(port=port).get_reply(chat))
        self.assertEqual(chat.chat_data["messages"][-2]["content"], "hi")

    def test_request_has_no_startup_delay(self) -> None:
        """Request has no startup delay."""
        body = {
            "message": {"content": "hi"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        port, _ = _serve_once(_http_response(body))
        started = time.time()
        Requester(port=port).get_reply(Buffer())
        self.assertLess(time.time() - started, 0.4)

    def test_pipe_is_closed_with_the_buffer(self) -> None:
        """Pipe is closed with the buffer."""
        chat = Buffer()
        read_fd, write_fd = chat.pipe
        chat.__del__()
        for fd in (read_fd, write_fd):
            with self.assertRaises(OSError):
                os.fstat(fd)
        chat.pipe = ()  # type: ignore[assignment]

class TestRequestQueue(unittest.TestCase):
    """Tests for request queue."""
    def test_concurrent_schedule_enqueues_once(self) -> None:
        """Concurrent schedule enqueues once."""
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old_interval)
        for _ in range(100):
            queue = RequestQueue()
            chat = Buffer()
            barrier = threading.Barrier(8)

            def schedule() -> None:
                """Test helper: schedule."""
                barrier.wait()
                queue.schedule(chat)

            threads = [threading.Thread(target=schedule) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(queue.qu), 1)

class TestInput(unittest.TestCase):
    """Tests for input."""
    def _feed(self, sv: Supervisor, data: bytes) -> threading.Thread:
        """Test helper: feed."""
        read_fd, write_fd = os.pipe()
        os.write(write_fd, data)
        os.close(write_fd)
        stdin = mock.Mock()
        stdin.fileno.return_value = read_fd
        with mock.patch.object(sys, "stdin", stdin):
            thread = threading.Thread(target=sv.keyboard.run)
            thread.start()
            thread.join(2)
        os.close(read_fd)
        return thread

    def test_multibyte_character_is_typed_not_fatal(self) -> None:
        """Multibyte character is typed not fatal."""
        sv = Supervisor()
        sv.run = True
        thread = self._feed(sv, "ä".encode())
        self.assertFalse(thread.is_alive())
        self.assertEqual(sv.current_data["messages"][-1]["content"], "ä")

    def test_closed_stdin_stops_the_app(self) -> None:
        """Closed stdin stops the app."""
        sv = Supervisor()
        sv.run = True
        thread = self._feed(sv, b"")
        self.assertFalse(thread.is_alive())
        self.assertFalse(sv.run)

class TestInputUIState(unittest.TestCase):
    """Tests for input ui state."""
    def _run(self, data: bytes, **state: Any) -> Supervisor:
        """Test helper: run."""
        sv = Supervisor()
        sv.run = True
        for name, value in state.items():
            setattr(sv.ui, name, value)
        with mock.patch.object(view_module, "fit_text_to_box"):
            TestInput()._feed(sv, data)
        return sv

    def test_up_arrow_scrolls_from_the_end(self) -> None:
        """Up arrow scrolls from the end."""
        sv = self._run(b"\x1b[A", scroll=-1, scroll_max=30)
        self.assertEqual(sv.ui.scroll, 29)

    def test_down_arrow_follows_the_end_again(self) -> None:
        """Down arrow follows the end again."""
        sv = self._run(b"\x1b[B", scroll=30, scroll_max=30)
        self.assertEqual(sv.ui.scroll, -1)

    def test_left_and_right_arrows_jump(self) -> None:
        """Left and right arrows jump."""
        self.assertEqual(self._run(b"\x1b[C", scroll=7).ui.scroll, 1)
        self.assertEqual(self._run(b"\x1b[D", scroll=7).ui.scroll, -1)

    def test_typing_a_colon_starts_a_command(self) -> None:
        """Typing a colon starts a command."""
        sv = self._run(b":sm x")
        self.assertEqual(sv.ui.command, ":sm x")

    def test_a_key_dismisses_the_notification_and_is_not_typed(self) -> None:
        """A key dismisses the notification and is not typed."""
        sv = self._run(b"a", notification="cannot reach server")
        self.assertEqual(sv.ui.notification, "")
        self.assertEqual(sv.current_data["messages"][-1]["content"], "")

    def test_typing_returns_to_the_chat(self) -> None:
        """Typing returns to the chat."""
        sv = self._run(b"a", mode=ViewMode.HELP, scroll=4)
        self.assertIs(sv.ui.mode, ViewMode.NORMAL)
        self.assertEqual(sv.ui.scroll, -1)

    def test_enter_leaves_another_view(self) -> None:
        """Enter leaves another view."""
        sv = self._run(b"\r", mode=ViewMode.MODELS, scroll=4)
        self.assertIs(sv.ui.mode, ViewMode.NORMAL)
        self.assertEqual(sv.ui.scroll, -1)

class TestWatcher(unittest.TestCase):
    """Tests for watcher."""
    def test_command_is_taken_atomically_and_cleared(self) -> None:
        """Command is taken atomically and cleared."""
        sv = Supervisor()
        sv.run = True
        seen = []
        sv.commands.execute = lambda command: seen.append(command)  # type: ignore
        sv.view.draw_command_line = lambda: None  # type: ignore[method-assign]
        with sv.ui.cond:
            sv.ui.command = "/n"
            sv.ui.submitted = True
        watcher = threading.Thread(target=sv.watcher)
        watcher.start()
        deadline = time.time() + 2
        while not seen and time.time() < deadline:
            time.sleep(0.01)
        sv.stop()
        watcher.join(2)
        self.assertEqual(seen, ["/n"])
        self.assertEqual(sv.ui.command, "")

class TestGuardedThreads(unittest.TestCase):
    """Tests for guarded threads."""
    def test_failing_worker_is_reported_and_restarted(self) -> None:
        """Failing worker is reported and restarted."""
        sv = Supervisor()
        sv.run = True
        calls = []

        def flaky() -> None:
            """Test helper: flaky."""
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")

        with mock.patch.object(supervisor_module.time, "sleep"):
            sv._guarded("worker", flaky)()
        self.assertEqual(len(calls), 2)
        self.assertIn("boom", sv.ui.notification)

if __name__ == "__main__":
    unittest.main()
