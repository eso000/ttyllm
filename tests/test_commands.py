import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer import Buffer
from commands import Commands
from requester import Model, Requester
from state import ViewMode
from supervisor import Supervisor


def _make_supervisor() -> Supervisor:
    with mock.patch.object(Requester, "get_models", return_value=[]), mock.patch(
        "supervisor.signal.signal"
    ):
        sv = Supervisor()
    sv.ui.notification = ""
    return sv


class CommandsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = _make_supervisor()
        self.commands = self.sv.commands
        self.sv.buffers[0].add_msg("system")


class TestExecute(CommandsTestCase):
    def test_quit_returns_false(self) -> None:
        self.assertFalse(self.commands.execute(":q"))
        self.assertFalse(self.commands.execute("/q"))

    def test_other_commands_return_true(self) -> None:
        self.assertTrue(self.commands.execute(":n"))
        self.assertTrue(self.commands.execute(":does-not-exist"))

    def test_text_without_prefix_is_ignored(self) -> None:
        self.assertTrue(self.commands.execute("hello"))
        self.assertEqual(self.sv.ui.mode, ViewMode.NORMAL)

    def test_supervisor_delegates_process_command(self) -> None:
        self.assertFalse(self.sv.process_command(":q"))

    def test_extra_arguments_do_not_break_argless_commands(self) -> None:
        for name in ("n", "d", "S", "h", "m", "ai", "us", "sys", "mr"):
            with self.subTest(command=name):
                self.assertTrue(self.commands.execute(f":{name} unexpected extra"))


class TestViews(CommandsTestCase):
    def test_each_view_command_selects_its_state(self) -> None:
        for name, state in (
            ("m", ViewMode.MODELS),
            ("h", ViewMode.HELP),
            ("d", ViewMode.DATA),
            ("S", ViewMode.SYSTEM),
            ("n", ViewMode.NORMAL),
        ):
            with self.subTest(command=name):
                self.commands.execute(f":{name}")
                self.assertEqual(self.sv.ui.mode, state)

    def test_scroll_is_reset_per_view(self) -> None:
        self.sv.ui.scroll = 7
        self.commands.execute(":d")
        self.assertEqual(self.sv.ui.scroll, 0)
        self.commands.execute(":n")
        self.assertEqual(self.sv.ui.scroll, -1)

    def test_system_view_keeps_the_scroll_position(self) -> None:
        self.sv.ui.scroll = 7
        self.commands.execute(":S")
        self.assertEqual(self.sv.ui.scroll, 7)

    def test_view_marks_buffer_updated(self) -> None:
        self.sv.current_buffer.updated = False
        self.commands.execute(":h")
        self.assertTrue(self.sv.current_buffer.updated)


class TestRoles(CommandsTestCase):
    def test_role_commands_change_the_current_message(self) -> None:
        for name, role in (("ai", "assistant"), ("us", "user"), ("sys", "system")):
            with self.subTest(command=name):
                self.commands.execute(f":{name}")
                self.assertEqual(self.sv.current_data["messages"][-1]["role"], role)


class TestBuffers(CommandsTestCase):
    def test_create_and_navigate(self) -> None:
        self.commands.execute(":bc")
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertEqual(self.sv.current_buffer_id, 1)
        self.commands.execute(":bp")
        self.assertEqual(self.sv.current_buffer_id, 0)
        self.commands.execute(":bn")
        self.assertEqual(self.sv.current_buffer_id, 1)

    def test_navigation_stops_at_the_ends(self) -> None:
        self.commands.execute(":bp")
        self.assertEqual(self.sv.current_buffer_id, 0)
        self.commands.execute(":bn")
        self.assertEqual(self.sv.current_buffer_id, 0)

    def test_split_copies_the_chat(self) -> None:
        self.sv.current_data["model"] = "llama"
        self.commands.execute(":s")
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertEqual(self.sv.current_buffer_id, 1)
        self.assertEqual(self.sv.current_data["model"], "llama")
        self.assertIsNot(self.sv.buffers[0].chat_data, self.sv.buffers[1].chat_data)

    def test_split_is_refused_with_pending_tool_calls(self) -> None:
        self.sv.current_buffer.pending_calls = [{"id": "1"}]  # type: ignore[list-item]
        self.commands.execute(":s")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIn("pending tool calls", self.sv.ui.notification)

    def test_delete_last_buffer_leaves_a_fresh_one(self) -> None:
        old = self.sv.current_buffer
        self.commands.execute(":bd")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIsNot(self.sv.current_buffer, old)

    def test_delete_last_in_list_moves_selection_back(self) -> None:
        self.commands.execute(":bc")
        self.commands.execute(":bd")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertEqual(self.sv.current_buffer_id, 0)

    def test_delete_removes_buffer_from_queue(self) -> None:
        buffer = self.sv.current_buffer
        self.sv.queue.schedule(buffer)
        self.commands.execute(":bd")
        self.assertFalse(self.sv.queue.has_scheduled)
        self.assertFalse(buffer.queued)

    def test_redo_schedules_the_buffer(self) -> None:
        self.sv.current_buffer.add_msg("user")
        self.commands.execute(":r")
        self.assertTrue(self.sv.queue.has_scheduled)


class TestModelCommands(CommandsTestCase):
    def test_switch_model(self) -> None:
        self.commands.execute(":sm llama3.2")
        self.assertEqual(self.sv.current_data["model"], "llama3.2")

    def test_switch_model_without_argument_is_a_noop(self) -> None:
        before = self.sv.current_data["model"]
        self.commands.execute(":sm")
        self.assertEqual(self.sv.current_data["model"], before)

    def test_refresh_reports_a_problem(self) -> None:
        with mock.patch.object(Requester, "get_models", return_value=[]):
            self.commands.execute(":mr")
        self.assertIn("no models found", self.sv.ui.notification)

    def test_refresh_reports_the_count(self) -> None:
        with mock.patch.object(
            Requester, "get_models", return_value=[Model("a"), Model("b")]
        ):
            self.commands.execute(":mr")
        self.assertEqual(self.sv.ui.notification, "loaded 2 models")


class TestOptions(CommandsTestCase):
    def test_set_stream(self) -> None:
        self.commands.execute(":o stream 0")
        self.assertFalse(self.sv.current_data["stream"])
        self.commands.execute(":o stream 1")
        self.assertTrue(self.sv.current_data["stream"])

    def test_unknown_option_is_reported(self) -> None:
        self.commands.execute(":o bogus 1")
        self.assertIn("bogus", self.sv.ui.notification)

    def test_missing_value_is_reported(self) -> None:
        self.commands.execute(":o k")
        self.assertEqual(self.sv.ui.notification, "no data provided")

    def test_without_arguments_shows_options(self) -> None:
        self.commands.execute(":o")
        self.assertIn("top_k", self.sv.ui.notification)


class TestVisual(CommandsTestCase):
    def test_set_names(self) -> None:
        self.commands.execute(":v un me")
        self.assertEqual(self.sv.current_buffer.user_name, "me")

    def test_invalid_color_is_reported(self) -> None:
        self.commands.execute(":v uc nonsense")
        self.assertIn("invalid color code", self.sv.ui.notification)

    def test_unknown_key_is_reported(self) -> None:
        self.commands.execute(":v zz 1")
        self.assertIn("zz", self.sv.ui.notification)


class TestFiles(CommandsTestCase):
    def test_export_then_import_round_trips(self) -> None:
        self.sv.current_data["model"] = "roundtrip"
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "chat.json")
            self.commands.execute(f":e {path}")
            with open(path, encoding="utf-8") as file:
                self.assertEqual(json.load(file)["chat"]["model"], "roundtrip")
            self.commands.execute(":bc")
            self.commands.execute(f":i {path}")
        self.assertEqual(self.sv.current_data["model"], "roundtrip")

    def test_import_missing_file_is_reported(self) -> None:
        self.commands.execute(":i /does/not/exist.json")
        self.assertIn("not found", self.sv.ui.notification)

    def test_import_corrupted_file_is_reported(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
            file.write("{not json")
            file.flush()
            self.commands.execute(f":i {file.name}")
        self.assertIn("corrupted", self.sv.ui.notification)

    def test_export_to_missing_directory_is_reported(self) -> None:
        self.commands.execute(":e /does/not/exist/chat.json")
        self.assertIn("not found", self.sv.ui.notification)


class TestImages(CommandsTestCase):
    def _user_message(self) -> None:
        self.sv.current_buffer.change_msg("user")

    def test_add_and_remove_image(self) -> None:
        self._user_message()
        with tempfile.NamedTemporaryFile() as file:
            file.write(b"\x89PNG")
            file.flush()
            self.commands.execute(f":image {file.name}")
        message = self.sv.current_data["messages"][-1]
        self.assertEqual(len(message["images"]), 1)
        self.commands.execute(":rmi")
        self.assertNotIn("images", self.sv.current_data["messages"][-1])

    def test_image_is_only_added_to_user_messages(self) -> None:
        self.sv.current_buffer.change_msg("assistant")
        with tempfile.NamedTemporaryFile() as file:
            self.commands.execute(f":image {file.name}")
        self.assertNotIn("images", self.sv.current_data["messages"][-1])

    def test_missing_image_is_reported(self) -> None:
        self._user_message()
        self.commands.execute(":image /does/not/exist.png")
        self.assertIn("not found", self.sv.ui.notification)


class TestHelpText(CommandsTestCase):
    def test_merged_commands_keep_distinct_descriptions(self) -> None:
        text = self.commands.help_text()
        for fragment in (
            "n:\tshow chat",
            "d:\tshow data",
            "S:\tshow system prompt",
            "ai:\tchange current message to assistant",
            "us:\tchange current message to user",
            "sys:\tchange current message to system",
        ):
            self.assertIn(fragment, text)

    def test_every_command_has_a_description(self) -> None:
        for name, command in Commands(self.sv).table.items():
            self.assertTrue(command.description, name)


if __name__ == "__main__":
    unittest.main()
