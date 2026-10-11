import dataclasses
import unittest
from typing import Any
from unittest import mock

from ai_chat import view as view_module
from ai_chat.buffer import Buffer
from ai_chat.models import Model, ModelList
from ai_chat.state import ViewMode
from ai_chat.supervisor import Supervisor

def _text(box: Any) -> str:
    return "\n".join(box.lines)

class ViewTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = Supervisor()
        self.view = self.sv.view
        self.ui = self.sv.ui
        self.ui.columns, self.ui.lines = 80, 24
        self.buffer = self.sv.current_buffer

class TestCompose(ViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.buffer.chat_data["messages"][0]["content"] = "the system prompt"
        self.buffer.chat_data["messages"].append(
            {"role": "user", "content": "hello there"}
        )
        self.sv.models = ModelList([Model("MODEL LIST")])

    def test_normal_mode_shows_the_chat_with_a_cursor(self) -> None:
        box, hide = self.view.compose(self.buffer, -1)
        self.assertIn("hello there", _text(box))
        self.assertFalse(hide)

    def test_system_mode_shows_the_system_prompt(self) -> None:
        self.ui.mode = ViewMode.SYSTEM
        box, hide = self.view.compose(self.buffer, -1)
        self.assertIn("the system prompt", _text(box))
        self.assertTrue(hide)

    def test_data_mode_shows_raw_data(self) -> None:
        self.ui.mode = ViewMode.DATA
        box, hide = self.view.compose(self.buffer, -1)
        self.assertIn('"role"', _text(box))
        self.assertTrue(hide)

    def test_help_mode_shows_the_command_help(self) -> None:
        self.ui.mode = ViewMode.HELP
        box, hide = self.view.compose(self.buffer, -1)
        self.assertEqual(_text(box).split("\n")[0][:8], "commands")
        self.assertIn("show chat", _text(box))
        self.assertTrue(hide)

    def test_models_mode_shows_the_model_list(self) -> None:
        self.ui.mode = ViewMode.MODELS
        box, hide = self.view.compose(self.buffer, -1)
        self.assertIn("MODEL LIST", _text(box))
        self.assertTrue(hide)

    def test_help_and_models_are_only_read_for_their_own_view(self) -> None:
        with mock.patch.object(self.sv.commands, "help_text") as help_text:
            self.view.compose(self.buffer, -1)
            self.ui.mode = ViewMode.MODELS
            self.view.compose(self.buffer, -1)
        help_text.assert_not_called()

    def test_notification_wins_over_every_mode(self) -> None:
        self.ui.notification = "cannot reach server"
        for mode in ViewMode:
            with self.subTest(mode=mode):
                self.ui.mode = mode
                box, hide = self.view.compose(self.buffer, -1)
                self.assertIn("cannot reach server", _text(box))
                self.assertNotIn("hello there", _text(box))
                self.assertTrue(hide)

    def test_scroll_is_passed_to_the_box(self) -> None:
        self.ui.mode = ViewMode.HELP
        self.assertEqual(self.view.compose(self.buffer, 7)[0].last, 7)

    def test_text_is_wrapped_to_the_width_without_the_border(self) -> None:
        self.assertEqual(self.view.box_width, 76)
        self.ui.columns = 40
        self.assertEqual(self.view.box_width, 36)

    def test_compose_does_not_change_the_ui_state(self) -> None:
        for mode in ViewMode:
            self.ui.mode = mode
            self.ui.scroll, self.ui.scroll_max = 3, 9
            self.ui.notification = "n" if mode is ViewMode.NORMAL else ""
            before = dataclasses.replace(self.ui)
            self.view.compose(self.buffer, self.ui.scroll)
            self.assertEqual(self.ui, before, mode)

    def test_the_view_follows_the_ui_state_of_the_supervisor(self) -> None:
        other = type(self.ui)(columns=50, lines=10)
        self.sv.ui = other
        self.assertIs(self.view.ui, other)
        self.assertEqual(self.view.box_width, 46)

class TestDraw(ViewTestCase):
    def test_draw_uses_the_terminal_size(self) -> None:
        box, _ = self.view.compose(self.buffer, -1)
        with mock.patch.object(view_module, "fit_to_box", return_value=(5, -1)) as fit:
            result = self.view.draw(box, True)
        self.assertEqual(result, (5, -1))
        fit.assert_called_once_with(box, 76, 19, 2, 3, hide_cursor=True)

class TestStatusLine(ViewTestCase):
    def _buffers(self, count: int) -> list[Buffer]:
        self.sv.buffers = [Buffer() for _ in range(count)]
        for number, buffer in enumerate(self.sv.buffers):
            buffer.chat_data["model"] = f"model{number}"
        return self.sv.buffers

    def test_shows_model_tokens_and_position_of_the_current_buffer(self) -> None:
        buffers = self._buffers(2)
        buffers[1].tokens = 42
        self.sv.current_buffer_id = 1
        text = self.view.status_text()
        self.assertIn("model: model1", text)
        self.assertIn("tokens: 42", text)
        self.assertIn("buffer: 2 of 2", text)

    def test_the_status_follows_buffer_switches(self) -> None:
        self._buffers(2)
        self.sv.change_buffer(0)
        self.assertIn("model: model0", self.view.status_text())
        self.sv.change_buffer(1)
        self.assertIn("model: model1", self.view.status_text())

    def test_idle_buffer_has_no_request_marker(self) -> None:
        self._buffers(1)
        text = self.view.status_text()
        self.assertNotIn("in_request", text)
        self.assertNotIn("queued", text)
        self.assertNotIn("approval", text)

    def test_request_markers(self) -> None:
        buffers = self._buffers(1)
        buffers[0].queued = True
        self.assertIn("queued", self.view.status_text())
        buffers[0].in_request = True
        text = self.view.status_text()
        self.assertIn("in_request", text)
        self.assertNotIn("queued", text)

    def test_lists_every_buffer_waiting_for_approval(self) -> None:
        buffers = self._buffers(3)
        for number in (0, 2):
            buffers[number].set_pending(
                [{"id": "1", "name": "t", "args": {}, "approved": None, "result": None}]
            )
        self.sv.current_buffer_id = 1
        self.assertIn("approval needed in: 1, 3", self.view.status_text())

    def test_command_line_replaces_the_status_while_typing(self) -> None:
        self.ui.command = "/sm lla"
        with mock.patch.object(view_module, "fit_text_to_box") as fit:
            self.view.draw_command_line()
        self.assertEqual(fit.call_args.args[0], "/sm lla")
        self.assertEqual(fit.call_args.args[6], view_module.COMMAND_BACKGROUND)

    def test_status_is_drawn_when_no_command_is_typed(self) -> None:
        with mock.patch.object(view_module, "fit_text_to_box") as fit:
            self.view.draw_command_line()
        self.assertIn("model:", fit.call_args.args[0])
        self.assertTrue(fit.call_args.kwargs["hide_cursor"])

if __name__ == "__main__":
    unittest.main()
