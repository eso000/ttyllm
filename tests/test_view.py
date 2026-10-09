import dataclasses
import os
import sys
import unittest
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import view as view_module
from buffer import Buffer
from state import UIState, ViewMode
from view import View


def _make_view(**overrides: Any) -> tuple[View, UIState]:
    ui = UIState(columns=80, lines=24, **overrides)
    return View(ui, lambda: "HELP TEXT", lambda: "MODEL LIST"), ui


def _text(box: Any) -> str:
    return "\n".join(box.lines)


class TestCompose(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = Buffer()
        self.buffer.chat_data["messages"][0]["content"] = "the system prompt"
        self.buffer.chat_data["messages"].append(
            {"role": "user", "content": "hello there"}
        )

    def test_normal_mode_shows_the_chat_with_a_cursor(self) -> None:
        view, _ = _make_view()
        box, hide = view.compose(self.buffer, -1)
        self.assertIn("hello there", _text(box))
        self.assertFalse(hide)

    def test_system_mode_shows_the_system_prompt(self) -> None:
        view, _ = _make_view(mode=ViewMode.SYSTEM)
        box, hide = view.compose(self.buffer, -1)
        self.assertIn("the system prompt", _text(box))
        self.assertTrue(hide)

    def test_data_mode_shows_raw_data(self) -> None:
        view, _ = _make_view(mode=ViewMode.DATA)
        box, hide = view.compose(self.buffer, -1)
        self.assertIn('"role"', _text(box))
        self.assertTrue(hide)

    def test_help_and_models_come_from_the_providers(self) -> None:
        view, ui = _make_view(mode=ViewMode.HELP)
        self.assertIn("HELP TEXT", _text(view.compose(self.buffer, -1)[0]))
        ui.mode = ViewMode.MODELS
        box, hide = view.compose(self.buffer, -1)
        self.assertIn("MODEL LIST", _text(box))
        self.assertTrue(hide)

    def test_providers_are_only_called_for_their_own_view(self) -> None:
        help_text, models_text = mock.Mock(return_value=""), mock.Mock(return_value="")
        ui = UIState()
        view = View(ui, help_text, models_text)
        view.compose(self.buffer, -1)
        help_text.assert_not_called()
        models_text.assert_not_called()

    def test_notification_wins_over_every_mode(self) -> None:
        for mode in ViewMode:
            with self.subTest(mode=mode):
                view, ui = _make_view(mode=mode)
                ui.notification = "cannot reach server"
                box, hide = view.compose(self.buffer, -1)
                self.assertIn("cannot reach server", _text(box))
                self.assertNotIn("hello there", _text(box))
                self.assertTrue(hide)

    def test_scroll_is_passed_to_the_box(self) -> None:
        view, _ = _make_view(mode=ViewMode.HELP)
        self.assertEqual(view.compose(self.buffer, 7)[0].last, 7)

    def test_text_is_wrapped_to_the_width_without_the_border(self) -> None:
        view, ui = _make_view(mode=ViewMode.HELP)
        self.assertEqual(view.box_width, 76)
        ui.columns = 40
        self.assertEqual(view.box_width, 36)

    def test_compose_does_not_change_the_ui_state(self) -> None:
        for mode in ViewMode:
            view, ui = _make_view(mode=mode, scroll=3, scroll_max=9)
            ui.notification = "n" if mode is ViewMode.NORMAL else ""
            before = dataclasses.asdict(ui)
            view.compose(self.buffer, ui.scroll)
            self.assertEqual(dataclasses.asdict(ui), before, mode)


class TestDraw(unittest.TestCase):
    def test_draw_uses_the_terminal_size(self) -> None:
        view, _ = _make_view()
        box, _ = view.compose(Buffer(), -1)
        with mock.patch.object(view_module, "fit_to_box", return_value=(5, -1)) as fit:
            result = view.draw(box, True)
        self.assertEqual(result, (5, -1))
        fit.assert_called_once_with(box, 76, 19, 2, 3, hide_cursor=True)


class TestStatusLine(unittest.TestCase):
    def _buffers(self, count: int = 1) -> list[Buffer]:
        buffers = [Buffer() for _ in range(count)]
        for number, buffer in enumerate(buffers):
            buffer.chat_data["model"] = f"model{number}"
        return buffers

    def test_shows_model_tokens_and_position(self) -> None:
        view, _ = _make_view()
        buffers = self._buffers(2)
        buffers[1].tokens = 42
        text = view.status_text(buffers, 1)
        self.assertIn("model: model1", text)
        self.assertIn("tokens: 42", text)
        self.assertIn("buffer: 2 of 2", text)

    def test_idle_buffer_has_no_request_marker(self) -> None:
        view, _ = _make_view()
        text = view.status_text(self._buffers(), 0)
        self.assertNotIn("in_request", text)
        self.assertNotIn("queued", text)
        self.assertNotIn("approval", text)

    def test_request_markers(self) -> None:
        view, _ = _make_view()
        buffers = self._buffers()
        buffers[0].queued = True
        self.assertIn("queued", view.status_text(buffers, 0))
        buffers[0].in_request = True
        text = view.status_text(buffers, 0)
        self.assertIn("in_request", text)
        self.assertNotIn("queued", text)

    def test_lists_every_buffer_waiting_for_approval(self) -> None:
        view, _ = _make_view()
        buffers = self._buffers(3)
        for number in (0, 2):
            buffers[number].set_pending(
                [{"id": "1", "name": "t", "args": {}, "approved": None, "result": None}]
            )
        self.assertIn("approval needed in: 1, 3", view.status_text(buffers, 1))

    def test_command_line_replaces_the_status_while_typing(self) -> None:
        view, ui = _make_view(command="/sm lla")
        with mock.patch.object(view_module, "fit_text_to_box") as fit:
            view.draw_command_line(self._buffers(), 0)
        self.assertEqual(fit.call_args.args[0], "/sm lla")
        self.assertEqual(fit.call_args.args[6], view_module.COMMAND_BACKGROUND)

    def test_status_is_drawn_when_no_command_is_typed(self) -> None:
        view, _ = _make_view()
        with mock.patch.object(view_module, "fit_text_to_box") as fit:
            view.draw_command_line(self._buffers(), 0)
        self.assertIn("model: model0", fit.call_args.args[0])
        self.assertTrue(fit.call_args.kwargs["hide_cursor"])


if __name__ == "__main__":
    unittest.main()
