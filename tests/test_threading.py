import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supervisor as supervisor_module
from buffer import Buffer
from requester import Requester
from supervisor import Queue, Supervisor
from test_requester import _http_response, _serve_once


def _make_supervisor():
    with mock.patch.object(Requester, "get_models", return_value=[]), mock.patch(
        "supervisor.signal.signal"
    ):
        sv = Supervisor()
    sv.notification = ""
    return sv


class TestRenderThread(unittest.TestCase):
    def test_survives_rapid_buffer_switching(self):
        sv = _make_supervisor()
        sv.buffers.append(Buffer())
        sv.run = True
        errors = []
        old_hook = threading.excepthook
        old_interval = sys.getswitchinterval()
        threading.excepthook = lambda args: errors.append(args.exc_value)
        sys.setswitchinterval(1e-6)
        try:
            with mock.patch.object(
                supervisor_module, "fit_to_box", lambda *a, **k: (10, -1)
            ), mock.patch.object(supervisor_module, "fit_text_to_box"):
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

    def test_scroll_during_render_is_not_overwritten(self):
        sv = _make_supervisor()
        sv.run = True
        sv.buffer_last = 0

        def scrolling_draw(*args, **kwargs):
            sv.buffer_last = 7  # user scrolls while the screen is drawn
            return (10, 3)

        with mock.patch.object(
            supervisor_module, "fit_to_box", scrolling_draw
        ), mock.patch.object(supervisor_module, "fit_text_to_box"):
            render = threading.Thread(target=sv.update_buffer)
            render.start()
            sv.current_buffer.notify()
            deadline = time.time() + 2
            while sv.buffer_lines == -1 and time.time() < deadline:
                time.sleep(0.01)
            sv.stop()
            render.join(2)
        self.assertEqual(sv.buffer_lines, 10)
        self.assertEqual(sv.buffer_last, 7)


class TestCancel(unittest.TestCase):
    def test_idle_cancel_leaves_nothing_behind(self):
        chat = Buffer()
        for _ in range(5):
            chat.cancel_request()
        self.assertFalse(chat.cancel)
        with self.assertRaises(BlockingIOError):
            os.read(chat.pipe[0], 1)

    def test_cancel_during_request_is_recorded_and_cleared(self):
        chat = Buffer()
        chat.in_request = True
        for _ in range(3):
            chat.cancel_request()
        self.assertTrue(chat.cancel)
        chat.clear_cancel()
        self.assertFalse(chat.cancel)
        with self.assertRaises(BlockingIOError):
            os.read(chat.pipe[0], 1)

    def test_next_request_is_not_aborted_by_old_cancels(self):
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

    def test_request_has_no_startup_delay(self):
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

    def test_pipe_is_closed_with_the_buffer(self):
        chat = Buffer()
        read_fd, write_fd = chat.pipe
        chat.__del__()
        for fd in (read_fd, write_fd):
            with self.assertRaises(OSError):
                os.fstat(fd)
        chat.pipe = ()


class TestQueue(unittest.TestCase):
    def test_concurrent_schedule_enqueues_once(self):
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old_interval)
        for _ in range(100):
            queue = Queue()
            chat = Buffer()
            barrier = threading.Barrier(8)

            def schedule():
                barrier.wait()
                queue.schedule(chat)

            threads = [threading.Thread(target=schedule) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(queue.qu), 1)


class TestInput(unittest.TestCase):
    def _feed(self, sv, data):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, data)
        os.close(write_fd)
        stdin = mock.Mock()
        stdin.fileno.return_value = read_fd
        with mock.patch.object(supervisor_module.sys, "stdin", stdin):
            thread = threading.Thread(target=sv.get_input)
            thread.start()
            thread.join(2)
        os.close(read_fd)
        return thread

    def test_multibyte_character_is_typed_not_fatal(self):
        sv = _make_supervisor()
        sv.run = True
        thread = self._feed(sv, "ä".encode())
        self.assertFalse(thread.is_alive())
        self.assertEqual(sv.current_data["messages"][-1]["content"], "ä")

    def test_closed_stdin_stops_the_app(self):
        sv = _make_supervisor()
        sv.run = True
        thread = self._feed(sv, b"")
        self.assertFalse(thread.is_alive())
        self.assertFalse(sv.run)


class TestWatcher(unittest.TestCase):
    def test_command_is_taken_atomically_and_cleared(self):
        sv = _make_supervisor()
        sv.run = True
        seen = []
        sv.process_command = lambda command: seen.append(command)
        sv.update_command = lambda: None
        with sv.input_cond:
            sv.command = "/n"
            sv.input_done = True
        watcher = threading.Thread(target=sv.watcher)
        watcher.start()
        deadline = time.time() + 2
        while not seen and time.time() < deadline:
            time.sleep(0.01)
        sv.stop()
        watcher.join(2)
        self.assertEqual(seen, ["/n"])
        self.assertEqual(sv.command, "")


class TestGuardedThreads(unittest.TestCase):
    def test_failing_worker_is_reported_and_restarted(self):
        sv = _make_supervisor()
        sv.run = True
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")

        with mock.patch.object(supervisor_module.time, "sleep"):
            sv._guarded(flaky)()
        self.assertEqual(len(calls), 2)
        self.assertIn("boom", sv.notification)


if __name__ == "__main__":
    unittest.main()
