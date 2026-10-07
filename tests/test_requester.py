import os
import sys
import json
import socket
import threading
import unittest
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requester
from requester import Model, Requester, _parse_models
from supervisor import Supervisor
from buffer import Buffer

GIB = 1073741824


def _http_response(payload: Any, content_type: str = "application/json") -> bytes:
    body = json.dumps(payload).encode()
    head = (
        "HTTP/1.1 200 OK\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    ).encode()
    return head + body


def _serve_once(response: bytes) -> tuple[int, threading.Thread]:
    """start a one-shot http server, return its port"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def run() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                conn.recv(1024)
                conn.sendall(response)
        except OSError:
            pass
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


def _closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestParseModels(unittest.TestCase):
    def test_openai_provides_names_only(self) -> None:
        models = _parse_models({"data": [{"id": "gpt-4o"}, {"id": "o1"}]})
        self.assertEqual([m.name for m in models], ["gpt-4o", "o1"])
        self.assertTrue(all(m.size_gb is None for m in models))
        self.assertTrue(all(m.parameter_size is None for m in models))

    def test_ollama_provides_size_and_params(self) -> None:
        models = _parse_models(
            {
                "models": [
                    {
                        "name": "llama3.2",
                        "size": 2 * GIB,
                        "details": {"parameter_size": "3B"},
                    }
                ]
            }
        )
        self.assertEqual(models, [Model("llama3.2", 2, "3B")])

    def test_ollama_entry_without_size(self) -> None:
        models = _parse_models({"models": [{"name": "mystery"}]})
        self.assertEqual(models, [Model("mystery", None, None)])

    def test_ollama_entry_with_null_details(self) -> None:
        models = _parse_models({"models": [{"name": "m", "size": 1, "details": None}]})
        self.assertEqual(models, [Model("m", 0, None)])

    def test_missing_keys_returns_empty(self) -> None:
        self.assertEqual(_parse_models({}), [])
        self.assertEqual(_parse_models({"object": "list"}), [])

    def test_empty_lists_return_empty(self) -> None:
        self.assertEqual(_parse_models({"models": []}), [])
        self.assertEqual(_parse_models({"data": []}), [])

    def test_malformed_entries_are_skipped(self) -> None:
        models = _parse_models({"models": [{"size": 1}, {"name": "ok"}]})
        self.assertEqual([m.name for m in models], ["ok"])
        models = _parse_models({"data": [{"nope": 1}, {"id": "ok"}]})
        self.assertEqual([m.name for m in models], ["ok"])

    def test_non_dict_returns_empty(self) -> None:
        self.assertEqual(_parse_models(None), [])
        self.assertEqual(_parse_models("models"), [])


class TestFormatModel(unittest.TestCase):
    def test_openai_names_only(self) -> None:
        self.assertEqual(Supervisor._format_model(Model("gpt-4o")), "gpt-4o\n")

    def test_ollama_with_size(self) -> None:
        line = Supervisor._format_model(Model("llama3.2", 2, "3B"))
        self.assertEqual(line, f"{'llama3.2':<20}\t 2 GB\t3B\n")


class TestGetModels(unittest.TestCase):
    def test_ollama_response(self) -> None:
        port, _ = _serve_once(
            _http_response(
                {"models": [{"name": "llama3.2", "size": 2 * GIB,
                             "details": {"parameter_size": "3B"}}]}
            )
        )
        result = Requester(port=port).get_models()
        self.assertEqual(result, [Model("llama3.2", 2, "3B")])
        self.assertIsNone(Requester(port=port).models_error)

    def test_openai_response(self) -> None:
        port, _ = _serve_once(_http_response({"data": [{"id": "gpt-4o"}]}))
        request = Requester(port=port, open_api=True)
        self.assertEqual(request.get_models(), [Model("gpt-4o")])
        self.assertIsNone(request.models_error)

    def test_no_server_returns_empty_and_reports_error(self) -> None:
        request = Requester(port=_closed_port())
        self.assertEqual(request.get_models(), [])
        self.assertIsNotNone(request.models_error)

    def test_wrong_content_type_reports_error(self) -> None:
        port, _ = _serve_once(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
        request = Requester(port=port)
        self.assertEqual(request.get_models(), [])
        self.assertIn("unexpected response", request.models_error)  # type: ignore[arg-type]

    def test_error_is_cleared_on_recovery(self) -> None:
        request = Requester(port=_closed_port())
        request.get_models()
        self.assertIsNotNone(request.models_error)
        port, _ = _serve_once(_http_response({"data": [{"id": "gpt-4o"}]}))
        request.set_endpoint(("127.0.0.1", port))
        request.get_models()
        self.assertIsNone(request.models_error)


class TestSupervisorModelAttributes(unittest.TestCase):
    def test_load_models_on_unreachable_server_is_safe(self) -> None:
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.requester = Requester(port=_closed_port())
        supervisor.models = None  # type: ignore[assignment]
        supervisor.models_list = None  # type: ignore[assignment]
        supervisor.models_short = None  # type: ignore[assignment]
        supervisor.load_models()
        self.assertEqual(supervisor.models, [])
        self.assertEqual(supervisor.models_list, [])
        self.assertEqual(supervisor.models_short, "")

    def test_load_models_from_openai_names_only(self) -> None:
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.requester = Requester(port=_closed_port(), open_api=True)
        supervisor.requester.get_models = lambda: [Model("gpt-4o"), Model("o1")]  # type: ignore[method-assign]
        supervisor.load_models()
        self.assertEqual(supervisor.models_list, ["gpt-4o", "o1"])
        self.assertEqual(supervisor.models_short, "gpt-4o\no1\n")


class TestReplyErrors(unittest.TestCase):
    def test_connection_error_is_returned(self) -> None:
        chat = Buffer()
        error = Requester(port=_closed_port()).get_reply(chat)
        self.assertIn("ConnectionRefusedError", error)  # type: ignore[arg-type]
        self.assertFalse(chat.in_request)

    def test_http_error_is_returned(self) -> None:
        port, _ = _serve_once(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        error = Requester(port=port).get_reply(Buffer())
        self.assertIn("404", error)  # type: ignore[arg-type]

    def test_cancelled_request_reports_nothing(self) -> None:
        chat = Buffer()
        port, _ = _serve_once(b"")
        requester = Requester(port=port)

        def cancelled_header(*_: Any) -> None:
            chat.cancel = True
            raise OSError("interrupted")

        requester._get_header = cancelled_header  # type: ignore
        self.assertIsNone(requester.get_reply(chat))


class TestNotification(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = Supervisor.__new__(Supervisor)
        self.supervisor.requester = Requester(port=_closed_port())
        self.supervisor.notification = ""
        self.supervisor.buffers = [Buffer()]
        self.supervisor.current_buffer_id = 0
        self.supervisor.buffer_last = -1
        self.supervisor.models_list = []

    def test_unreachable_server_is_reported(self) -> None:
        self.supervisor.requester.models_error = "refused"
        self.assertTrue(self.supervisor._report_models())
        self.assertIn("cannot reach", self.supervisor.notification)
        self.assertIn("refused", self.supervisor.notification)

    def test_no_models_is_reported(self) -> None:
        self.assertTrue(self.supervisor._report_models())
        self.assertIn("no models found", self.supervisor.notification)

    def test_models_present_stays_silent(self) -> None:
        self.supervisor.models_list = ["llama3.2"]
        self.assertFalse(self.supervisor._report_models())
        self.assertEqual(self.supervisor.notification, "")

    def test_reply_error_becomes_notification(self) -> None:
        self.supervisor.queue = None  # type: ignore[assignment]
        self.supervisor.requester.get_reply = lambda buffer: "boom"  # type: ignore
        self.supervisor.get_reply(self.supervisor.buffers[0])
        self.assertEqual(self.supervisor.notification, "boom")

    def test_notification_marks_buffer_for_redraw(self) -> None:
        self.supervisor.show_notification("hello")
        self.assertTrue(self.supervisor.buffers[0].updated)
        self.assertEqual(self.supervisor.buffer_last, 0)


class TestHelp(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = Supervisor.__new__(Supervisor)
        self.supervisor.buffers = [Buffer()]
        self.supervisor.current_buffer_id = 0
        self.supervisor.notification = ""
        self.supervisor.buffer_last = -1
        self.supervisor.state = None  # type: ignore[assignment]
        self.supervisor.commands = {}
        for name in ("help", "switch_model", "remove_image"):
            self.supervisor.commands[name] = getattr(Supervisor, name).__get__(
                self.supervisor
            )

    def test_help_is_a_scrollable_view_not_a_notification(self) -> None:
        self.supervisor.help()
        self.assertEqual(self.supervisor.state.name, "HELP")
        self.assertEqual(self.supervisor.buffer_last, 0)
        self.assertEqual(self.supervisor.notification, "")

    def test_help_lists_every_command_with_a_description(self) -> None:
        text = self.supervisor._help_text()
        for name in self.supervisor.commands:
            self.assertIn(f"{name}:", text)
        self.assertNotIn("None", text)

    def test_every_real_command_is_documented(self) -> None:
        with mock.patch.object(Requester, "get_models", return_value=[]), \
                mock.patch("supervisor.signal.signal"):
            supervisor = Supervisor()
        undocumented = [n for n, f in supervisor.commands.items() if not f.__doc__]
        self.assertEqual(undocumented, [])


if __name__ == "__main__":
    unittest.main()
