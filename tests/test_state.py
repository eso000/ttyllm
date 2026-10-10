import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import UIState, ViewMode


class TestDefaults(unittest.TestCase):
    def test_starts_in_the_chat_following_its_end(self) -> None:
        ui = UIState()
        self.assertIs(ui.mode, ViewMode.NORMAL)
        self.assertEqual(ui.scroll, -1)
        self.assertEqual(ui.command, "")
        self.assertEqual(ui.notification, "")

    def test_instances_do_not_share_state(self) -> None:
        first, second = UIState(), UIState()
        first.command = "/x"
        self.assertEqual(second.command, "")


class TestTerminalSize(unittest.TestCase):
    def test_from_terminal_reads_the_size(self) -> None:
        with mock.patch("state.shutil.get_terminal_size") as size:
            size.return_value = os.terminal_size((120, 40))
            ui = UIState.from_terminal()
        self.assertEqual((ui.columns, ui.lines), (120, 40))

    def test_resize_picks_up_a_new_size(self) -> None:
        ui = UIState()
        with mock.patch("state.shutil.get_terminal_size") as size:
            size.return_value = os.terminal_size((60, 20))
            ui.resize()
        self.assertEqual((ui.columns, ui.lines), (60, 20))


class TestModes(unittest.TestCase):
    def test_set_mode_with_a_scroll_position(self) -> None:
        ui = UIState(scroll=9)
        ui.set_mode(ViewMode.HELP, 0)
        self.assertIs(ui.mode, ViewMode.HELP)
        self.assertEqual(ui.scroll, 0)

    def test_set_mode_without_a_scroll_keeps_the_position(self) -> None:
        ui = UIState(scroll=9)
        ui.set_mode(ViewMode.SYSTEM)
        self.assertIs(ui.mode, ViewMode.SYSTEM)
        self.assertEqual(ui.scroll, 9)

    def test_scroll_zero_is_a_position_not_a_keep_marker(self) -> None:
        ui = UIState(scroll=9)
        ui.set_mode(ViewMode.DATA, 0)
        self.assertEqual(ui.scroll, 0)

    def test_show_chat_follows_the_end(self) -> None:
        ui = UIState(mode=ViewMode.MODELS, scroll=4)
        ui.show_chat()
        self.assertIs(ui.mode, ViewMode.NORMAL)
        self.assertEqual(ui.scroll, -1)


class TestScrolling(unittest.TestCase):
    def test_up_from_the_end_goes_to_the_last_full_page(self) -> None:
        ui = UIState(scroll=-1, scroll_max=30)
        ui.scroll_up()
        self.assertEqual(ui.scroll, 29)

    def test_up_moves_one_line_and_stops_at_the_top(self) -> None:
        ui = UIState(scroll=2, scroll_max=30)
        ui.scroll_up()
        ui.scroll_up()
        self.assertEqual(ui.scroll, 0)
        ui.scroll_up()
        self.assertEqual(ui.scroll, 0)

    def test_down_moves_one_line(self) -> None:
        ui = UIState(scroll=5, scroll_max=30)
        ui.scroll_down()
        self.assertEqual(ui.scroll, 6)

    def test_down_past_the_last_line_follows_the_end(self) -> None:
        ui = UIState(scroll=30, scroll_max=30)
        ui.scroll_down()
        self.assertEqual(ui.scroll, -1)

    def test_down_while_following_stays_following(self) -> None:
        ui = UIState(scroll=-1, scroll_max=30)
        ui.scroll_down()
        self.assertEqual(ui.scroll, -1)

    def test_arrows_are_ignored_while_a_command_is_typed(self) -> None:
        ui = UIState(scroll=5, scroll_max=30, command="/m")
        ui.scroll_up()
        self.assertEqual(ui.scroll, 5)
        ui.scroll_down()
        self.assertEqual(ui.scroll, 5)

    def test_jump_to_end_and_top(self) -> None:
        ui = UIState(scroll=5, scroll_max=30)
        ui.scroll_to_top()
        self.assertEqual(ui.scroll, 1)
        ui.scroll_to_end()
        self.assertEqual(ui.scroll, -1)


class TestNotification(unittest.TestCase):
    def test_set_scrolls_to_the_start(self) -> None:
        ui = UIState(scroll=12)
        ui.set_notification("boom")
        self.assertEqual(ui.notification, "boom")
        self.assertEqual(ui.scroll, 0)

    def test_dismiss_reports_whether_there_was_a_message(self) -> None:
        ui = UIState()
        self.assertFalse(ui.dismiss_notification())
        ui.set_notification("boom")
        self.assertTrue(ui.dismiss_notification())
        self.assertEqual(ui.notification, "")
        self.assertFalse(ui.dismiss_notification())


class TestTakeCommand(unittest.TestCase):
    def test_returns_and_clears_the_command(self) -> None:
        ui = UIState(command="/sm llama")
        self.assertEqual(ui.take_command(), "/sm llama")
        self.assertEqual(ui.command, "")
        self.assertEqual(ui.take_command(), "")

    def test_submit_marks_the_command_and_taking_it_consumes_the_mark(self) -> None:
        ui = UIState(command="/sm llama")
        self.assertFalse(ui.submitted)
        ui.submit()
        self.assertTrue(ui.submitted)
        self.assertEqual(ui.take_command(), "/sm llama")
        self.assertFalse(ui.submitted)

    def test_submit_wakes_a_thread_waiting_for_a_command(self) -> None:
        ui = UIState(command="/n")
        woke = threading.Event()

        def wait() -> None:
            with ui.cond:
                while not ui.submitted:
                    ui.cond.wait()
            woke.set()

        thread = threading.Thread(target=wait)
        thread.start()
        ui.submit()
        self.assertTrue(woke.wait(2))
        thread.join(2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
