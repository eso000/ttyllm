import contextlib
import io
import os
import signal
import sys
import threading
import unittest
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supervisor as supervisor_module
import view as view_module
from buffer import Buffer
from requester import Model, Requester
from supervisor import Supervisor
from test_input_handler import stdin_with


def message(role: str, content: str = "") -> dict[str, Any]:
    return {"role": role, "content": content}


class TestConstruction(unittest.TestCase):
    def test_creating_a_supervisor_has_no_side_effects(self) -> None:
        out = io.StringIO()
        with mock.patch.object(Requester, "get_models") as get_models, mock.patch.object(
            supervisor_module.signal, "signal"
        ) as register, contextlib.redirect_stdout(out):
            sv = Supervisor()
        get_models.assert_not_called()
        register.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(sv.ui.notification, "")
        self.assertEqual(sv.models_list, [])

    def test_a_new_supervisor_has_one_empty_buffer_and_nothing_queued(self) -> None:
        sv = Supervisor()
        self.assertEqual(len(sv.buffers), 1)
        self.assertEqual(sv.current_buffer_id, 0)
        self.assertFalse(sv.queue.has_scheduled)

    def test_the_parts_are_wired_to_the_supervisor(self) -> None:
        sv = Supervisor()
        self.assertIs(sv.commands.sv, sv)
        self.assertIs(sv.keyboard.sv, sv)
        self.assertIs(sv.view.sv, sv)


class TestBufferNavigation(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()

    def test_create_adds_and_selects_a_blank_buffer(self) -> None:
        self.sv.create_buffer()
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertEqual(self.sv.current_buffer_id, 1)

    def test_next_and_previous_stop_at_the_ends(self) -> None:
        self.sv.buffer_previous()
        self.sv.buffer_next()
        self.assertEqual(self.sv.current_buffer_id, 0)
        self.sv.create_buffer()
        self.sv.create_buffer()
        self.sv.buffer_next()
        self.assertEqual(self.sv.current_buffer_id, 2)
        self.sv.buffer_previous()
        self.sv.buffer_previous()
        self.sv.buffer_previous()
        self.assertEqual(self.sv.current_buffer_id, 0)

    def test_switching_wakes_the_renderer_of_the_new_buffer(self) -> None:
        self.sv.create_buffer()
        self.sv.buffers[0].updated = False
        self.sv.buffer_previous()
        self.assertTrue(self.sv.buffers[0].updated)


class TestSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.original = self.sv.current_buffer
        self.original.chat_data["messages"] += [message("user", "hi")]
        self.original.tokens = 17
        self.original.visual_data["un"] = "me"

    def test_split_copies_chat_looks_and_tokens(self) -> None:
        self.sv.split()
        copy = self.sv.current_buffer
        self.assertIsNot(copy, self.original)
        self.assertEqual(copy.chat_data, self.original.chat_data)
        self.assertEqual(copy.visual_data, self.original.visual_data)
        self.assertEqual(copy.tokens, 17)

    def test_the_copy_is_independent(self) -> None:
        self.sv.split()
        self.sv.current_buffer.chat_data["messages"][-1]["content"] = "changed"
        self.sv.current_buffer.visual_data["un"] = "other"
        self.assertEqual(self.original.chat_data["messages"][-1]["content"], "hi")
        self.assertEqual(self.original.visual_data["un"], "me")

    def test_split_selects_the_copy(self) -> None:
        self.sv.split()
        self.assertEqual(self.sv.current_buffer_id, 1)

    def test_split_is_refused_while_tool_calls_are_pending(self) -> None:
        self.original.set_pending(
            [{"id": "1", "name": "t", "args": {}, "approved": None, "result": None}]
        )
        self.sv.split()
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIn("pending tool calls", self.sv.ui.notification)


class TestDelete(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.sv.buffers = [Buffer(), Buffer(), Buffer()]

    def test_deleting_selects_the_buffer_that_moves_up(self) -> None:
        self.sv.current_buffer_id = 1
        following = self.sv.buffers[2]
        self.sv.buffer_delete()
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertIs(self.sv.current_buffer, following)

    def test_deleting_the_last_buffer_selects_the_one_before(self) -> None:
        self.sv.current_buffer_id = 2
        before = self.sv.buffers[1]
        self.sv.buffer_delete()
        self.assertIs(self.sv.current_buffer, before)
        self.assertEqual(self.sv.current_buffer_id, 1)

    def test_deleting_the_only_buffer_leaves_a_fresh_one(self) -> None:
        self.sv.buffers = [self.sv.buffers[0]]
        old = self.sv.current_buffer
        self.sv.buffer_delete()
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIsNot(self.sv.current_buffer, old)

    def test_deleting_drops_the_buffer_from_the_queue(self) -> None:
        doomed = self.sv.current_buffer
        self.sv.queue.schedule(doomed)
        self.sv.buffer_delete()
        self.assertFalse(self.sv.queue.has_scheduled)
        self.assertFalse(doomed.queued)

    def test_deleting_declines_pending_tool_calls(self) -> None:
        doomed = self.sv.current_buffer
        call: dict[str, Any] = {
            "id": "1", "name": "t", "args": {}, "approved": None, "result": None
        }
        doomed.set_pending([call])
        self.sv.buffer_delete()
        self.assertFalse(call["approved"])

    def test_deleting_wakes_the_renderer_of_the_deleted_buffer(self) -> None:
        doomed = self.sv.current_buffer
        waiting, woken = threading.Event(), threading.Event()

        def wait() -> None:
            with doomed.cond:
                waiting.set()
                notified = doomed.cond.wait(2)
            if notified:
                woken.set()

        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(waiting.wait(2))
        with doomed.cond:  # only available once the thread is really waiting
            pass
        self.sv.buffer_delete()
        self.assertTrue(woken.wait(2))
        thread.join(2)


class TestMessageOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.buffer = self.sv.current_buffer
        self.buffer.chat_data["messages"] = [
            message("system", "rules"),
            message("user", "question"),
            message("assistant", "answer"),
            message("user"),
        ]

    def roles(self) -> list[str]:
        return [m["role"] for m in self.buffer.chat_data["messages"]]

    def test_redo_drops_the_reply_and_queues_the_buffer(self) -> None:
        self.sv.redo()
        self.assertEqual(self.roles(), ["system", "user"])
        self.assertTrue(self.buffer.queued)

    def test_undo_reverts_to_an_empty_user_message(self) -> None:
        self.sv.undo()
        self.assertEqual(self.roles(), ["system", "user"])
        self.assertEqual(self.buffer.chat_data["messages"][-1]["content"], "")
        self.assertFalse(self.buffer.queued)

    def test_correct_removes_the_last_message(self) -> None:
        self.sv.correct()
        self.assertEqual(self.roles(), ["system", "user", "assistant"])

    def test_start_over_keeps_only_a_system_message(self) -> None:
        self.sv.start_over()
        self.assertEqual(self.roles(), ["system"])

    def test_operations_act_on_the_current_buffer_only(self) -> None:
        other = Buffer()
        self.sv.buffers.append(other)
        before = list(other.chat_data["messages"])
        self.sv.start_over()
        self.assertEqual(other.chat_data["messages"], before)


@contextlib.contextmanager
def fake_terminal_into(events: list[str]) -> Iterator[None]:
    events.append("terminal in")
    try:
        yield
    finally:
        events.append("terminal out")


class TestRunChat(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.events: list[str] = []
        self.previous_handler: Callable[..., None] = lambda *args: None
        for patcher in (
            mock.patch.object(supervisor_module, "raw_terminal", self._terminal),
            mock.patch.object(view_module, "fit_text_to_box"),
            mock.patch.object(view_module, "fit_to_box", return_value=(1, -1)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.register = mock.patch.object(
            supervisor_module.signal, "signal", side_effect=self._register
        ).start()
        self.addCleanup(mock.patch.stopall)

    def _terminal(self) -> contextlib.AbstractContextManager[None]:
        return fake_terminal_into(self.events)

    def _register(self, number: int, handler: Any) -> Any:
        restoring = handler is self.previous_handler
        self.events.append("signal restored" if restoring else "signal set")
        return self.previous_handler

    def _get_models(self, models: list[Model]) -> mock._patch[mock.MagicMock]:
        def get_models() -> list[Model]:
            self.events.append("models")
            return models

        return mock.patch.object(Requester, "get_models", side_effect=get_models)

    def _run(self, models: list[Model]) -> None:
        with self._get_models(models), stdin_with(b""):
            self.sv.run_chat()

    def test_models_are_loaded_before_the_terminal_is_taken_over(self) -> None:
        self._run([Model("llama3.2")])
        self.assertLess(self.events.index("models"), self.events.index("terminal in"))
        self.assertEqual(self.sv.models_list, ["llama3.2"])
        self.assertIn("llama3.2", self.sv.models_short)

    def test_a_server_problem_is_reported_on_start(self) -> None:
        self._run([])
        self.assertIn("no models found", self.sv.ui.notification)

    def test_working_models_start_without_a_notification(self) -> None:
        self._run([Model("llama3.2")])
        self.assertEqual(self.sv.ui.notification, "")

    def test_the_resize_handler_lives_inside_the_terminal_session(self) -> None:
        self._run([Model("m")])
        self.assertEqual(
            [e for e in self.events if e != "models"],
            ["terminal in", "signal set", "signal restored", "terminal out"],
        )

    def test_the_resize_handler_is_the_supervisors_and_is_restored(self) -> None:
        self._run([Model("m")])
        first, last = self.register.call_args_list[0], self.register.call_args_list[-1]
        self.assertEqual(first.args, (signal.SIGWINCH, self.sv.handle_sigwinch))
        self.assertEqual(last.args, (signal.SIGWINCH, self.previous_handler))

    def test_a_missing_previous_handler_is_restored_as_default(self) -> None:
        self.register.side_effect = lambda number, handler: None
        self._run([Model("m")])
        self.assertEqual(self.register.call_args_list[-1].args[1], signal.SIG_DFL)

    def test_everything_is_undone_when_starting_a_thread_fails(self) -> None:
        class BrokenThread:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("cannot start")

        with self._get_models([Model("m")]), mock.patch.object(
            supervisor_module.threading, "Thread", BrokenThread
        ), self.assertRaises(RuntimeError):
            self.sv.run_chat()
        self.assertEqual(self.events[-2:], ["signal restored", "terminal out"])

    def test_every_loop_runs_on_its_own_named_thread(self) -> None:
        names: dict[str, str] = {}
        others_started = threading.Event()

        def record(key: str, target: Callable[[], None]) -> Callable[[], None]:
            def inner() -> None:
                names[key] = threading.current_thread().name
                if len(names) == 4:
                    others_started.set()
                if key == "input":  # it stops the chat once stdin ends
                    others_started.wait(2)
                target()

            return inner

        self.sv.keyboard.run = record("input", self.sv.keyboard.run)  # type: ignore
        self.sv.update_buffer = record("render", self.sv.update_buffer)  # type: ignore
        self.sv.watcher = record("dispatch", self.sv.watcher)  # type: ignore
        self.sv.queue_manager = record("replies", self.sv.queue_manager)  # type: ignore
        self._run([Model("m")])
        self.assertEqual(
            names,
            {
                "input": "input",
                "render": "render",
                "dispatch": "dispatch",
                "replies": "replies",
            },
        )

    def test_the_threads_stop_when_input_ends(self) -> None:
        self._run([Model("m")])
        self.assertFalse(self.sv.run)
        self.assertEqual(
            [t.name for t in threading.enumerate() if t.name in (
                "input", "render", "dispatch", "replies")],
            [],
        )


class TestResize(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.calls: list[str] = []
        for target, name in (
            (supervisor_module, "clean_screen"),
            (self.sv.view, "draw_command_line"),
        ):
            patcher = mock.patch.object(
                target, name, side_effect=lambda name=name: self.calls.append(name)
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def _resize(self, columns: int, lines: int) -> None:
        with mock.patch("state.shutil.get_terminal_size") as size:
            size.return_value = os.terminal_size((columns, lines))
            self.sv.handle_sigwinch(signal.SIGWINCH, None)

    def test_the_new_size_is_taken_over(self) -> None:
        self._resize(132, 43)
        self.assertEqual((self.sv.ui.columns, self.sv.ui.lines), (132, 43))

    def test_the_screen_is_cleared_and_the_status_line_redrawn_in_that_order(self) -> None:
        self._resize(100, 30)
        self.assertEqual(self.calls, ["clean_screen", "draw_command_line"])

    def test_the_renderer_is_woken_to_redraw_the_chat(self) -> None:
        self.sv.current_buffer.updated = False
        self._resize(100, 30)
        self.assertTrue(self.sv.current_buffer.updated)


class TestGuarded(unittest.TestCase):
    def test_a_failure_names_the_thread(self) -> None:
        sv = Supervisor()
        sv.run = True
        calls: list[int] = []

        def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("bad value")

        with mock.patch.object(supervisor_module.time, "sleep"):
            sv._guarded("render", flaky)()
        self.assertEqual(len(calls), 2)
        self.assertIn("render thread failed", sv.ui.notification)
        self.assertIn("bad value", sv.ui.notification)

    def test_it_does_not_restart_a_stopped_chat(self) -> None:
        sv = Supervisor()
        sv.run = False
        calls: list[int] = []
        sv._guarded("render", lambda: calls.append(1))()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
