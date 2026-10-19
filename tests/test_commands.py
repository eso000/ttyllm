"""Tests for the slash and colon commands, their table and help text."""

import dataclasses
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from ttyllm.commands import Commands, _first_line, _rest_lines
from ttyllm.models import Model
from ttyllm.requester import Requester
from ttyllm.state import ViewMode
from ttyllm.supervisor import Supervisor


class CommandsTestCase(unittest.TestCase):
    """Shared test case for commands."""

    def setUp(self) -> None:
        """Set up the fixtures for each test."""
        self.sv = Supervisor()
        self.commands = self.sv.commands
        self.sv.buffers[0].add_msg("system")


class TestExecute(CommandsTestCase):
    """Tests for execute."""

    def test_quit_returns_false(self) -> None:
        """Quit returns false."""
        self.assertFalse(self.commands.execute(":q"))
        self.assertFalse(self.commands.execute("/q"))

    def test_other_commands_return_true(self) -> None:
        """Other commands return true."""
        self.assertTrue(self.commands.execute(":n"))
        self.assertTrue(self.commands.execute(":does-not-exist"))

    def test_text_without_prefix_is_ignored(self) -> None:
        """Text without prefix is ignored."""
        self.assertTrue(self.commands.execute("hello"))
        self.assertEqual(self.sv.ui.mode, ViewMode.NORMAL)

    def test_extra_arguments_do_not_break_argless_commands(self) -> None:
        """Extra arguments do not break argless commands."""
        argless = [n for n, c in self.commands.table.items() if not c.takes_args]
        self.assertGreater(len(argless), 15)
        with mock.patch.object(Requester, "get_models", return_value=[]):
            for name in argless:
                with self.subTest(command=name):
                    sv = Supervisor()
                    self.assertTrue(sv.commands.execute(f":{name} unexpected extra"))


class TestCommandTable(CommandsTestCase):
    """Tests for command table."""

    def test_buffer_commands_are_the_supervisors_own_methods(self) -> None:
        """Buffer commands are the supervisors own methods."""
        for name, method in (
            ("s", self.sv.split),
            ("r", self.sv.redo),
            ("R", self.sv.start_over),
            ("rr", self.sv.undo),
            ("bc", self.sv.create_buffer),
            ("bn", self.sv.buffer_next),
            ("bp", self.sv.buffer_previous),
            ("bd", self.sv.buffer_delete),
            ("-", self.sv.correct),
        ):
            with self.subTest(command=name):
                self.assertEqual(self.commands.table[name].handler, method)

    def test_exactly_the_commands_with_arguments_take_them(self) -> None:
        """Exactly the commands with arguments take them."""
        takes = {n for n, c in self.commands.table.items() if c.takes_args}
        self.assertEqual(takes, {"sm", "i", "e", "endp", "v", "o", "image"})

    def test_arguments_reach_the_commands_that_take_them(self) -> None:
        """Arguments reach the commands that take them."""
        for name in ("sm", "i", "e", "endp", "v", "o", "image"):
            with self.subTest(command=name):
                handler = mock.Mock()
                entry = dataclasses.replace(self.commands.table[name], handler=handler)
                self.commands.table[name] = entry
                self.commands.execute(f":{name} first second")
                handler.assert_called_once_with("first", "second")

    def test_a_command_without_arguments_is_called_bare(self) -> None:
        """A command without arguments is called bare."""
        handler = mock.Mock()
        self.commands.table["bc"] = dataclasses.replace(
            self.commands.table["bc"], handler=handler
        )
        self.commands.execute(":bc ignored words")
        handler.assert_called_once_with()

    def test_unknown_commands_are_ignored(self) -> None:
        """Unknown commands are ignored."""
        before = len(self.sv.buffers)
        self.assertTrue(self.commands.execute(":nothing here"))
        self.assertEqual(len(self.sv.buffers), before)

    def test_every_description_comes_from_a_docstring_or_is_explicit(self) -> None:
        """Every description comes from a docstring or is explicit."""
        for name, entry in self.commands.table.items():
            with self.subTest(command=name):
                self.assertTrue(entry.description.strip())


class TestViews(CommandsTestCase):
    """Tests for views."""

    def test_each_view_command_selects_its_state(self) -> None:
        """Each view command selects its state."""
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
        """Scroll is reset per view."""
        self.sv.ui.scroll = 7
        self.commands.execute(":d")
        self.assertEqual(self.sv.ui.scroll, 0)
        self.commands.execute(":n")
        self.assertEqual(self.sv.ui.scroll, -1)

    def test_system_view_keeps_the_scroll_position(self) -> None:
        """System view keeps the scroll position."""
        self.sv.ui.scroll = 7
        self.commands.execute(":S")
        self.assertEqual(self.sv.ui.scroll, 7)

    def test_view_marks_buffer_updated(self) -> None:
        """View marks buffer updated."""
        self.sv.current_buffer.updated = False
        self.commands.execute(":h")
        self.assertTrue(self.sv.current_buffer.updated)

    def test_view_takes_the_ui_lock_before_the_buffer_lock(self) -> None:
        """a set_view that held the buffer lock while taking the ui lock would
        invert the input thread's ui -> buffer order and deadlock"""
        ui, buffer = self.sv.ui, self.sv.current_buffer
        ui.cond.acquire()
        thread = threading.Thread(target=self.commands.execute, args=(":h",))
        thread.start()
        # while the ui lock is held, the buffer lock must still be free
        free = buffer.cond.acquire(timeout=1)
        if free:
            buffer.cond.release()
        ui.cond.release()
        thread.join(2)
        self.assertTrue(free, "set_view took the buffer lock before the ui lock")
        self.assertFalse(thread.is_alive())


class TestRoles(CommandsTestCase):
    """Tests for roles."""

    def test_role_commands_change_the_current_message(self) -> None:
        """Role commands change the current message."""
        for name, role in (("ai", "assistant"), ("us", "user"), ("sys", "system")):
            with self.subTest(command=name):
                self.commands.execute(f":{name}")
                self.assertEqual(self.sv.current_data["messages"][-1]["role"], role)


class TestBuffers(CommandsTestCase):
    """Tests for buffers."""

    def test_create_and_navigate(self) -> None:
        """Create and navigate."""
        self.commands.execute(":bc")
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertEqual(self.sv.current_buffer_id, 1)
        self.commands.execute(":bp")
        self.assertEqual(self.sv.current_buffer_id, 0)
        self.commands.execute(":bn")
        self.assertEqual(self.sv.current_buffer_id, 1)

    def test_navigation_stops_at_the_ends(self) -> None:
        """Navigation stops at the ends."""
        self.commands.execute(":bp")
        self.assertEqual(self.sv.current_buffer_id, 0)
        self.commands.execute(":bn")
        self.assertEqual(self.sv.current_buffer_id, 0)

    def test_split_copies_the_chat(self) -> None:
        """Split copies the chat."""
        self.sv.current_data["model"] = "llama"
        self.commands.execute(":s")
        self.assertEqual(len(self.sv.buffers), 2)
        self.assertEqual(self.sv.current_buffer_id, 1)
        self.assertEqual(self.sv.current_data["model"], "llama")
        self.assertIsNot(self.sv.buffers[0].chat_data, self.sv.buffers[1].chat_data)

    def test_split_is_refused_with_pending_tool_calls(self) -> None:
        """Split is refused with pending tool calls."""
        self.sv.current_buffer.pending_calls = [{"id": "1"}]  # type: ignore[list-item]
        self.commands.execute(":s")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIn("pending tool calls", self.sv.ui.notification)

    def test_delete_last_buffer_leaves_a_fresh_one(self) -> None:
        """Delete last buffer leaves a fresh one."""
        old = self.sv.current_buffer
        self.commands.execute(":bd")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertIsNot(self.sv.current_buffer, old)

    def test_delete_last_in_list_moves_selection_back(self) -> None:
        """Delete last in list moves selection back."""
        self.commands.execute(":bc")
        self.commands.execute(":bd")
        self.assertEqual(len(self.sv.buffers), 1)
        self.assertEqual(self.sv.current_buffer_id, 0)

    def test_delete_removes_buffer_from_queue(self) -> None:
        """Delete removes buffer from queue."""
        buffer = self.sv.current_buffer
        self.sv.queue.schedule(buffer)
        self.commands.execute(":bd")
        self.assertFalse(self.sv.queue.has_scheduled)
        self.assertFalse(buffer.queued)

    def test_redo_schedules_the_buffer(self) -> None:
        """Redo schedules the buffer."""
        self.sv.current_buffer.add_msg("user")
        self.commands.execute(":r")
        self.assertTrue(self.sv.queue.has_scheduled)


class TestModelCommands(CommandsTestCase):
    """Tests for model commands."""

    def test_switch_model(self) -> None:
        """Switch model."""
        self.commands.execute(":sm llama3.2")
        self.assertEqual(self.sv.current_data["model"], "llama3.2")

    def test_switch_model_without_argument_is_a_noop(self) -> None:
        """Switch model without argument is a noop."""
        before = self.sv.current_data["model"]
        self.commands.execute(":sm")
        self.assertEqual(self.sv.current_data["model"], before)

    def test_refresh_reports_a_problem(self) -> None:
        """Refresh reports a problem."""
        with mock.patch.object(Requester, "get_models", return_value=[]):
            self.commands.execute(":mr")
        self.assertIn("no models found", self.sv.ui.notification)

    def test_refresh_reports_the_count(self) -> None:
        """Refresh reports the count."""
        with mock.patch.object(
            Requester, "get_models", return_value=[Model("a"), Model("b")]
        ):
            self.commands.execute(":mr")
        self.assertEqual(self.sv.ui.notification, "loaded 2 models")


class TestEndpoint(CommandsTestCase):
    """Tests for endpoint."""

    def _with_models(self):
        """Test helper: with models."""
        return mock.patch.object(Requester, "get_models", return_value=[Model("a")])

    def test_sets_host_and_port_from_host_colon_port(self) -> None:
        """Sets host and port from host colon port."""
        with self._with_models():
            self.commands.execute(":endp 192.168.1.5:8080")
        self.assertEqual(self.sv.requester.host, "192.168.1.5")
        self.assertEqual(self.sv.requester.port, 8080)
        self.assertIn("192.168.1.5:8080", self.sv.ui.notification)

    def test_two_arguments_set_host_and_port(self) -> None:
        """Two arguments set host and port."""
        with self._with_models():
            self.commands.execute(":endp 10.0.0.2 11434")
        self.assertEqual(self.sv.requester.host, "10.0.0.2")
        self.assertEqual(self.sv.requester.port, 11434)

    def test_https_url_enables_tls(self) -> None:
        """Https url enables tls."""
        with self._with_models():
            self.commands.execute(":endp https://api.example.com:443")
        self.assertTrue(self.sv.requester.tls)
        self.assertEqual(self.sv.requester.port, 443)

    def test_scheme_less_token_keeps_the_current_port(self) -> None:
        """Scheme less token keeps the current port."""
        before = self.sv.requester.port
        with self._with_models():
            self.commands.execute(":endp otherhost")
        self.assertEqual(self.sv.requester.host, "otherhost")
        self.assertEqual(self.sv.requester.port, before)

    def test_without_arguments_shows_the_endpoint(self) -> None:
        """Without arguments shows the endpoint."""
        self.commands.execute(":endp")
        self.assertIn(self.sv.requester.endpoint, self.sv.ui.notification)

    def test_slash_prefix_shows_the_endpoint(self) -> None:
        """Slash prefix shows the endpoint."""
        self.commands.execute("/endp")
        self.assertEqual(
            self.sv.ui.notification, f"endpoint: {self.sv.requester.endpoint}"
        )

    def test_empty_arguments_show_the_endpoint_too(self) -> None:
        """Empty arguments show the endpoint too."""
        before = self.sv.requester.endpoint
        self.commands.execute(":endp  ")
        self.assertIn(self.sv.requester.endpoint, self.sv.ui.notification)
        self.assertEqual(self.sv.requester.endpoint, before)

    def test_invalid_port_is_reported_and_nothing_changes(self) -> None:
        """Invalid port is reported and nothing changes."""
        before = self.sv.requester.endpoint
        self.commands.execute(":endp host:notaport")
        self.assertIn("invalid endpoint", self.sv.ui.notification)
        self.assertEqual(self.sv.requester.endpoint, before)

    def test_port_out_of_range_is_reported(self) -> None:
        """Port out of range is reported."""
        self.commands.execute(":endp host 99999")
        self.assertIn("out of range", self.sv.ui.notification)

    def test_too_many_arguments_are_reported(self) -> None:
        """Too many arguments are reported."""
        self.commands.execute(":endp a b c")
        self.assertIn("usage", self.sv.ui.notification)


class TestOptions(CommandsTestCase):
    """Tests for options."""

    def test_set_stream(self) -> None:
        """Set stream."""
        self.commands.execute(":o stream 0")
        self.assertFalse(self.sv.current_data["stream"])
        self.commands.execute(":o stream 1")
        self.assertTrue(self.sv.current_data["stream"])

    def test_set_fractional_temperature(self) -> None:
        """Set fractional temperature."""
        self.commands.execute(":o t 0.8")
        self.assertEqual(self.sv.current_buffer.temperature, 0.8)

    def test_invalid_value_is_reported_and_nothing_changes(self) -> None:
        """Invalid value is reported and nothing changes."""
        for command in (
            ":o k abc",
            ":o p abc",
            ":o t abc",
            ":o pen abc",
            ":o stream abc",
        ):
            with self.subTest(command=command):
                before = dict(self.sv.current_buffer.options)
                self.commands.execute(command)
                self.assertIn("invalid value", self.sv.ui.notification)
                self.assertEqual(self.sv.current_buffer.options, before)

    def test_unknown_option_is_reported(self) -> None:
        """Unknown option is reported."""
        self.commands.execute(":o bogus 1")
        self.assertIn("bogus", self.sv.ui.notification)

    def test_missing_value_is_reported(self) -> None:
        """Missing value is reported."""
        self.commands.execute(":o k")
        self.assertEqual(self.sv.ui.notification, "no data provided")

    def test_without_arguments_shows_options(self) -> None:
        """Without arguments shows options."""
        self.commands.execute(":o")
        self.assertIn("top_k", self.sv.ui.notification)


class TestVisual(CommandsTestCase):
    """Tests for visual."""

    def test_set_names(self) -> None:
        """Set names."""
        self.commands.execute(":v un me")
        self.assertEqual(self.sv.current_buffer.user_name, "me")

    def test_invalid_color_is_reported(self) -> None:
        """Invalid color is reported."""
        self.commands.execute(":v uc nonsense")
        self.assertIn("invalid color code", self.sv.ui.notification)

    def test_unknown_key_is_reported(self) -> None:
        """Unknown key is reported."""
        self.commands.execute(":v zz 1")
        self.assertIn("zz", self.sv.ui.notification)


class TestFiles(CommandsTestCase):
    """Tests for files."""

    def test_export_then_import_round_trips(self) -> None:
        """Export then import round trips."""
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
        """Import missing file is reported."""
        self.commands.execute(":i /does/not/exist.json")
        self.assertIn("not found", self.sv.ui.notification)

    def test_import_corrupted_file_is_reported(self) -> None:
        """Import corrupted file is reported."""
        with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
            file.write("{not json")
            file.flush()
            self.commands.execute(f":i {file.name}")
        self.assertIn("corrupted", self.sv.ui.notification)

    def test_export_to_missing_directory_is_reported(self) -> None:
        """Export to missing directory is reported."""
        self.commands.execute(":e /does/not/exist/chat.json")
        self.assertIn("not found", self.sv.ui.notification)


class TestImages(CommandsTestCase):
    """Tests for images."""

    def _user_message(self) -> None:
        """Test helper: user message."""
        self.sv.current_buffer.change_msg("user")

    def test_add_and_remove_image(self) -> None:
        """Add and remove image."""
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
        """Image is only added to user messages."""
        self.sv.current_buffer.change_msg("assistant")
        with tempfile.NamedTemporaryFile() as file:
            self.commands.execute(f":image {file.name}")
        self.assertNotIn("images", self.sv.current_data["messages"][-1])

    def test_missing_image_is_reported(self) -> None:
        """Missing image is reported."""
        self._user_message()
        self.commands.execute(":image /does/not/exist.png")
        self.assertIn("not found", self.sv.ui.notification)


class TestHelpText(CommandsTestCase):
    """Tests for help text."""

    def test_merged_commands_keep_distinct_descriptions(self) -> None:
        """Merged commands keep distinct descriptions."""
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
        """Every command has a description."""
        for name, command in Commands(self.sv).table.items():
            self.assertTrue(command.description, name)

    def test_multiline_docstrings_add_a_details_section(self) -> None:
        """Multiline docstrings add a details section."""
        text = self.commands.help_text()
        self.assertIn("o:\tshow or set sampling options", text)
        self.assertIn("details:", text)
        self.assertIn("o:\nk: top_k (int)", text)
        self.assertIn("v:\nun: user name", text)

    def test_only_multiline_commands_are_in_the_details(self) -> None:
        """Only multiline commands are in the details."""
        text = self.commands.help_text()
        details = text.split("details:\n\n", 1)[1]
        for name, command in self.commands.table.items():
            if command.detail is None:
                self.assertNotIn(f"{name}:\n", details)


class TestDocstringSplit(unittest.TestCase):
    """Tests for docstring split."""

    def test_the_first_line_becomes_the_description(self) -> None:
        """The first line becomes the description."""
        self.assertEqual(_first_line("summary\n\nmore\nlines"), "summary")

    def test_the_rest_becomes_the_detail(self) -> None:
        """The rest becomes the detail."""
        self.assertEqual(_rest_lines("summary\n\nmore\nlines"), "more\nlines")

    def test_a_single_line_docstring_has_no_detail(self) -> None:
        """A single line docstring has no detail."""
        self.assertEqual(_first_line("only a summary"), "only a summary")
        self.assertIsNone(_rest_lines("only a summary"))

    def test_a_missing_docstring_is_empty(self) -> None:
        """A missing docstring is empty."""
        self.assertEqual(_first_line(None), "")
        self.assertIsNone(_rest_lines(None))


if __name__ == "__main__":
    unittest.main()
