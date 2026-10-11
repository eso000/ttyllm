import json
import os
import re
import select
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
from typing import Any
from unittest import mock

from ai_chat import requester
from ai_chat.models import Model, ModelList
from ai_chat.requester import Requester
from ai_chat.commands import Commands
from ai_chat.state import UIState
from ai_chat.supervisor import Supervisor
from ai_chat.buffer import Buffer

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

def _sse_response(*events: str) -> bytes:
    """build a chunked text/event-stream http response from raw sse events"""
    chunks = b"".join(
        f"{len(event.encode()):x}\r\n".encode() + event.encode() + b"\r\n"
        for event in events
    )
    head = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
    ).encode()
    return head + chunks + b"0\r\n\r\n"

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

def _serve_stall(
    connected: threading.Event | None = None,
) -> tuple[int, threading.Thread]:
    """start a server that accepts one connection and never sends a body"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def run() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                conn.recv(1024)
                if connected is not None:
                    connected.set()
                while conn.recv(1024):
                    pass
        except OSError:
            pass
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread

def _self_signed_cert(directory: str) -> tuple[str, str]:
    """generate a throwaway self-signed cert for localhost with the openssl cli"""
    certfile = os.path.join(directory, "cert.pem")
    keyfile = os.path.join(directory, "key.pem")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            keyfile,
            "-out",
            certfile,
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return certfile, keyfile

def _serve_tls_once(
    response: bytes, certfile: str, keyfile: str
) -> tuple[int, threading.Thread]:
    """start a one-shot https server that speaks TLS with the given certificate"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)

    def run() -> None:
        try:
            conn, _ = server.accept()
            with context.wrap_socket(conn, server_side=True) as tls:
                tls.recv(1024)
                tls.sendall(response)
        except OSError:
            pass
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread

def _unverified_context() -> ssl.SSLContext:
    """a client context that skips verification, for the self-signed test server"""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

class TestModelListFromResponse(unittest.TestCase):
    def test_openai_provides_names_only(self) -> None:
        models = ModelList.from_response({"data": [{"id": "gpt-4o"}, {"id": "o1"}]})
        self.assertEqual(models.names, ["gpt-4o", "o1"])
        self.assertTrue(all(m.size_gb is None for m in models))
        self.assertTrue(all(m.parameter_size is None for m in models))

    def test_ollama_provides_size_and_params(self) -> None:
        models = ModelList.from_response(
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
        self.assertEqual(list(models), [Model("llama3.2", 2, "3B")])

    def test_ollama_entry_without_size(self) -> None:
        models = ModelList.from_response({"models": [{"name": "mystery"}]})
        self.assertEqual(list(models), [Model("mystery", None, None)])

    def test_ollama_entry_with_null_details(self) -> None:
        models = ModelList.from_response(
            {"models": [{"name": "m", "size": 1, "details": None}]}
        )
        self.assertEqual(list(models), [Model("m", 0, None)])

    def test_missing_keys_returns_empty(self) -> None:
        self.assertEqual(list(ModelList.from_response({})), [])
        self.assertEqual(list(ModelList.from_response({"object": "list"})), [])

    def test_empty_lists_return_empty(self) -> None:
        self.assertEqual(list(ModelList.from_response({"models": []})), [])
        self.assertEqual(list(ModelList.from_response({"data": []})), [])

    def test_malformed_entries_are_skipped(self) -> None:
        models = ModelList.from_response({"models": [{"size": 1}, {"name": "ok"}]})
        self.assertEqual(models.names, ["ok"])
        models = ModelList.from_response({"data": [{"nope": 1}, {"id": "ok"}]})
        self.assertEqual(models.names, ["ok"])

    def test_non_dict_returns_empty(self) -> None:
        self.assertEqual(list(ModelList.from_response(None)), [])
        self.assertEqual(list(ModelList.from_response("models")), [])

    def test_empty_model_list_is_len_zero_and_empty_short(self) -> None:
        models = ModelList([])
        self.assertEqual(len(models), 0)
        self.assertEqual(models.short, "")
        self.assertEqual(models.names, [])

class TestModelListShort(unittest.TestCase):
    def test_openai_names_only(self) -> None:
        self.assertEqual(ModelList([Model("gpt-4o")]).short, "gpt-4o\n")

    def test_ollama_with_size(self) -> None:
        line = ModelList([Model("llama3.2", 2, "3B")]).short
        self.assertEqual(line, f"{'llama3.2':<20}\t 2 GB\t3B\n")

class TestGetModels(unittest.TestCase):
    def test_ollama_response(self) -> None:
        port, _ = _serve_once(
            _http_response(
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
        )
        result = Requester(port=port).get_models()
        self.assertEqual(list(result), [Model("llama3.2", 2, "3B")])
        self.assertIsNone(Requester(port=port).models_error)

    def test_openai_response(self) -> None:
        port, _ = _serve_once(_http_response({"data": [{"id": "gpt-4o"}]}))
        request = Requester(port=port, open_api=True)
        self.assertEqual(list(request.get_models()), [Model("gpt-4o")])
        self.assertIsNone(request.models_error)

    def test_no_server_returns_empty_and_reports_error(self) -> None:
        request = Requester(port=_closed_port())
        self.assertEqual(list(request.get_models()), [])
        self.assertIsNotNone(request.models_error)

    def test_wrong_content_type_reports_error(self) -> None:
        port, _ = _serve_once(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
        request = Requester(port=port)
        self.assertEqual(list(request.get_models()), [])
        self.assertIn("unexpected response", request.models_error)  # type: ignore[arg-type]

    def test_error_is_cleared_on_recovery(self) -> None:
        request = Requester(port=_closed_port())
        request.get_models()
        self.assertIsNotNone(request.models_error)
        port, _ = _serve_once(_http_response({"data": [{"id": "gpt-4o"}]}))
        request.set_endpoint(("127.0.0.1", port))
        request.get_models()
        self.assertIsNone(request.models_error)

class TestEndpoint(unittest.TestCase):
    def test_https_url_enables_tls_on_the_default_port(self) -> None:
        request = Requester()
        request.set_endpoint("https://api.example.com")
        self.assertEqual(request.host, "api.example.com")
        self.assertEqual(request.port, 443)
        self.assertTrue(request.tls)

    def test_http_url_defaults_to_port_eighty(self) -> None:
        request = Requester()
        request.set_endpoint("http://api.example.com")
        self.assertEqual(request.port, 80)
        self.assertFalse(request.tls)

    def test_url_with_explicit_port_keeps_it(self) -> None:
        request = Requester()
        request.set_endpoint("https://localhost:8443")
        self.assertEqual(request.host, "localhost")
        self.assertEqual(request.port, 8443)
        self.assertTrue(request.tls)

    def test_bare_host_and_port_stays_plain(self) -> None:
        request = Requester()
        request.set_endpoint("localhost:11434")
        self.assertEqual(request.host, "localhost")
        self.assertEqual(request.port, 11434)
        self.assertFalse(request.tls)

    def test_tuple_only_changes_host_and_port(self) -> None:
        request = Requester(tls=True)
        request.set_endpoint(("127.0.0.1", 11434))
        self.assertEqual(request.host, "127.0.0.1")
        self.assertEqual(request.port, 11434)
        self.assertTrue(request.tls)

class TestConnect(unittest.TestCase):
    def test_plain_connection_is_returned_as_is(self) -> None:
        request = Requester(host="127.0.0.1", port=11434)
        with mock.patch("ai_chat.requester.socket.socket") as socket_mock:
            raw = socket_mock.return_value
            self.assertIs(request._connect(), raw)
        raw.connect.assert_called_once_with(("127.0.0.1", 11434))

    def test_tls_connection_is_wrapped_with_the_server_name(self) -> None:
        request = Requester(host="api.example.com", port=443, tls=True)
        with mock.patch("ai_chat.requester.socket.socket") as socket_mock, mock.patch(
            "ai_chat.requester.ssl.create_default_context"
        ) as context_mock:
            raw = socket_mock.return_value
            wrapped = context_mock.return_value.wrap_socket.return_value
            self.assertIs(request._connect(), wrapped)
        context_mock.return_value.wrap_socket.assert_called_once_with(
            raw, server_hostname="api.example.com"
        )

    def test_tls_context_is_built_once(self) -> None:
        request = Requester(host="api.example.com", tls=True)
        with mock.patch("ai_chat.requester.socket.socket"), mock.patch(
            "ai_chat.requester.ssl.create_default_context"
        ) as context_mock:
            request._connect()
            request._connect()
        context_mock.assert_called_once()

    def test_raw_socket_is_closed_when_wrapping_fails(self) -> None:
        request = Requester(host="api.example.com", tls=True)
        with mock.patch("ai_chat.requester.socket.socket") as socket_mock, mock.patch(
            "ai_chat.requester.ssl.create_default_context"
        ) as context_mock:
            raw = socket_mock.return_value
            context_mock.return_value.wrap_socket.side_effect = OSError("tls failed")
            with self.assertRaises(OSError):
                request._connect()
        raw.close.assert_called_once()

@unittest.skipUnless(shutil.which("openssl"), "the openssl cli is required")
class TestTlsConnection(unittest.TestCase):
    def test_https_request_is_served_over_tls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            certfile, keyfile = _self_signed_cert(directory)
            port, _ = _serve_tls_once(
                _http_response({"data": [{"id": "gpt-4o"}]}), certfile, keyfile
            )
            request = Requester(host="127.0.0.1", port=port, tls=True, open_api=True)
            with mock.patch.object(
                Requester, "_ssl_context", return_value=_unverified_context()
            ):
                result = request.get_models()
        self.assertEqual(list(result), [Model("gpt-4o")])
        self.assertIsNone(request.models_error)

class TestSupervisorModelAttributes(unittest.TestCase):
    def test_load_models_on_unreachable_server_is_safe(self) -> None:
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.requester = Requester(port=_closed_port())
        supervisor.models = None  # type: ignore[assignment]
        supervisor.load_models()
        self.assertEqual(list(supervisor.models), [])
        self.assertEqual(supervisor.models.names, [])
        self.assertEqual(supervisor.models.short, "")

    def test_load_models_from_openai_names_only(self) -> None:
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.requester = Requester(port=_closed_port(), open_api=True)
        supervisor.requester.get_models = lambda: ModelList(  # type: ignore[method-assign]
            [Model("gpt-4o"), Model("o1")]
        )
        supervisor.load_models()
        self.assertEqual(supervisor.models.names, ["gpt-4o", "o1"])
        self.assertEqual(supervisor.models.short, "gpt-4o\no1\n")

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

    def test_cancel_interrupts_body_read(self) -> None:
        # the server accepts but never sends the body; cancelling the request
        # must wake the pending body read instead of blocking forever
        connected = threading.Event()
        port, _ = _serve_stall(connected)
        chat = Buffer()
        requester = Requester(port=port)
        requester._get_header = lambda *_: (  # type: ignore[method-assign]
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 10",
            b"",
        )
        result: list[str | None] = []
        thread = threading.Thread(
            target=lambda: result.append(requester.get_reply(chat)), daemon=True
        )
        thread.start()
        self.assertTrue(connected.wait(2), "server never accepted the connection")
        chat.cancel_request()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "get_reply did not return after cancel")
        self.assertEqual(result, [None])

    def test_socket_is_closed_when_connect_fails(self) -> None:
        chat = Buffer()
        request = Requester(port=_closed_port())
        with mock.patch("ai_chat.requester.socket.socket") as socket_mock:
            socket_mock.return_value.connect.side_effect = OSError("refused")
            error = request.get_reply(chat)
        self.assertIn("OSError", error)  # type: ignore[arg-type]
        socket_mock.return_value.close.assert_called()

    def test_idle_server_times_out(self) -> None:
        # a server that accepts but never replies must not block forever
        port, _ = _serve_stall()
        request = Requester(port=port)
        request.read_timeout = 0.2
        error = request.get_reply(Buffer())
        self.assertIn("TimeoutError", error)  # type: ignore[arg-type]

class TestGetHeader(unittest.TestCase):
    def test_body_after_header_terminator_is_kept(self) -> None:
        # _get_header must not drop bytes that followed the first \r\n\r\n
        client, peer = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(peer.close)
        payload = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"\r\n"
            b"data: one\r\n\r\n"
            b"data: two\r\n\r\n"
        )
        peer.sendall(payload)
        head, body = Requester()._get_header(client)  # type: ignore[misc]
        self.assertIn("HTTP/1.1 200 OK", head)
        self.assertEqual(body, b"data: one\r\n\r\ndata: two\r\n\r\n")

class TestStreamingUsage(unittest.TestCase):
    def test_usage_only_chunk_is_not_a_choice(self) -> None:
        # OpenAI sends the usage chunk after finish_reason: "stop", with an
        # empty choices array; it must not be parsed as a choice
        requester = Requester()
        chat = Buffer()
        chat.in_request = True
        chat.add_msg("assistant")
        done, call = requester._handle_msg(
            chat,
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 10,
                    "total_tokens": 29,
                },
            },
        )
        self.assertFalse(done)
        self.assertFalse(call)
        self.assertEqual(chat.tokens, 29)

    def test_streamed_reply_reports_server_token_usage(self) -> None:
        port, _ = _serve_once(
            _sse_response(
                'data: {"choices":[{"delta":{"content":"Hello"},'
                '"finish_reason":null}],"usage":null}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                'data: {"choices":[],"usage":{"prompt_tokens":19,'
                '"completion_tokens":10,"total_tokens":29}}\n\n',
                "data: [DONE]\n\n",
            )
        )
        chat = Buffer()
        error = Requester(port=port, open_api=True).get_reply(chat)
        self.assertIsNone(error)
        self.assertEqual(chat.tokens, 29)
        self.assertEqual(chat.chat_data["messages"][-2]["content"], "Hello")

    def test_openai_stream_requests_usage(self) -> None:
        chat = Buffer()
        chat.chat_data["stream"] = True
        self.assertEqual(chat.openai_data["stream_options"], {"include_usage": True})

    def test_openai_non_stream_omits_stream_options(self) -> None:
        chat = Buffer()
        chat.chat_data["stream"] = False
        self.assertNotIn("stream_options", chat.openai_data)

class TestOpenAiToolMessages(unittest.TestCase):
    def _chat(self) -> Buffer:
        chat = Buffer()
        chat.chat_data = {
            "model": "m",
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "index": 0,
                            "function": {"name": "echo", "arguments": '{"text": "a"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "id": "call_1",
                    "tool_name": "echo",
                    "content": "echoed a",
                },
            ],
        }
        return chat

    def test_assistant_tool_call_is_openai_shaped(self) -> None:
        # index must not leak, arguments stay a json string, type is added
        self.assertEqual(
            self._chat().openai_data["messages"][2],
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "a"}'},
                    }
                ],
            },
        )

    def test_tool_result_uses_tool_call_id(self) -> None:
        self.assertEqual(
            self._chat().openai_data["messages"][3],
            {"role": "tool", "tool_call_id": "call_1", "content": "echoed a"},
        )

    def test_plain_messages_are_unchanged(self) -> None:
        messages = self._chat().openai_data["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})

    def test_dict_arguments_are_serialized(self) -> None:
        chat = self._chat()
        chat.chat_data["messages"][2]["tool_calls"][0]["function"]["arguments"] = {
            "text": "a"
        }
        call = chat.openai_data["messages"][2]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], '{"text": "a"}')

class TestOpenAiToolStreaming(unittest.TestCase):
    def test_fragmented_arguments_are_accumulated(self) -> None:
        requester_ = Requester()
        collection: dict[Any, dict[str, Any]] = {}
        for frame in (
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": ""},
                }
            ],
            [{"index": 0, "function": {"arguments": '{"text":'}}],
            [{"index": 0, "function": {"arguments": '"a"}'}}],
        ):
            requester_._handle_tool_call(Buffer(), frame, collection)
        call = collection[0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "echo")
        self.assertEqual(call["function"]["arguments"], '{"text":"a"}')

    def test_first_delta_without_type_is_initialized(self) -> None:
        # some servers omit "type" on the first delta
        requester_ = Requester()
        collection: dict[Any, dict[str, Any]] = {}
        requester_._handle_tool_call(
            Buffer(),
            [
                {
                    "index": 0,
                    "id": "call_9",
                    "function": {"name": "echo", "arguments": ""},
                }
            ],
            collection,
        )
        requester_._handle_tool_call(
            Buffer(), [{"index": 0, "function": {"arguments": "{}"}}], collection
        )
        self.assertEqual(collection[0]["id"], "call_9")
        self.assertEqual(collection[0]["function"]["arguments"], "{}")

    def test_multiple_calls_in_one_delta(self) -> None:
        requester_ = Requester()
        collection: dict[Any, dict[str, Any]] = {}
        requester_._handle_tool_call(
            Buffer(),
            [
                {
                    "index": 0,
                    "id": "a",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                },
                {
                    "index": 1,
                    "id": "b",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                },
            ],
            collection,
        )
        self.assertEqual(sorted(collection), [0, 1])
        self.assertEqual([collection[i]["id"] for i in (0, 1)], ["a", "b"])

    def test_streamed_call_round_trips_to_openai(self) -> None:
        requester_ = Requester()
        collection: dict[Any, dict[str, Any]] = {}
        for frame in (
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": ""},
                }
            ],
            [{"index": 0, "function": {"arguments": '{"text":"a"}'}}],
        ):
            requester_._handle_tool_call(Buffer(), frame, collection)
        chat = Buffer()
        chat.chat_data = {
            "model": "m",
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": list(collection.values()),
                },
                {
                    "role": "tool",
                    "id": "call_1",
                    "tool_name": "echo",
                    "content": "echoed a",
                },
            ],
        }
        messages = chat.openai_data["messages"]
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"], '{"text":"a"}'
        )
        self.assertEqual(messages[1]["tool_call_id"], "call_1")

class TestNotification(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = Supervisor.__new__(Supervisor)
        self.supervisor.requester = Requester(port=_closed_port())
        self.supervisor.ui = UIState()
        self.supervisor.buffers = [Buffer()]
        self.supervisor.current_buffer_id = 0
        self.supervisor.models = ModelList([])

    def test_unreachable_server_is_reported(self) -> None:
        self.supervisor.requester.models_error = "refused"
        self.assertTrue(self.supervisor.report_models())
        self.assertIn("cannot reach", self.supervisor.ui.notification)
        self.assertIn("refused", self.supervisor.ui.notification)

    def test_no_models_is_reported(self) -> None:
        self.assertTrue(self.supervisor.report_models())
        self.assertIn("no models found", self.supervisor.ui.notification)

    def test_models_present_stays_silent(self) -> None:
        self.supervisor.models = ModelList([Model("llama3.2")])
        self.assertFalse(self.supervisor.report_models())
        self.assertEqual(self.supervisor.ui.notification, "")

    def test_reply_error_becomes_notification(self) -> None:
        self.supervisor.queue = None  # type: ignore[assignment]
        self.supervisor.requester.get_reply = lambda buffer: "boom"  # type: ignore
        self.supervisor.get_reply(self.supervisor.buffers[0])
        self.assertEqual(self.supervisor.ui.notification, "boom")

    def test_notification_marks_buffer_for_redraw(self) -> None:
        self.supervisor.show_notification("hello")
        self.assertTrue(self.supervisor.buffers[0].updated)
        self.assertEqual(self.supervisor.ui.scroll, 0)

class TestHelp(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = Supervisor.__new__(Supervisor)
        self.supervisor.buffers = [Buffer()]
        self.supervisor.current_buffer_id = 0
        self.supervisor.ui = UIState()
        self.commands = Commands(self.supervisor)

    def test_help_is_a_scrollable_view_not_a_notification(self) -> None:
        self.commands.execute(":h")
        self.assertEqual(self.supervisor.ui.mode.name, "HELP")
        self.assertEqual(self.supervisor.ui.scroll, 0)
        self.assertEqual(self.supervisor.ui.notification, "")

    def test_help_lists_every_command_with_a_description(self) -> None:
        text = self.commands.help_text()
        for name in self.commands.table:
            self.assertIn(f"{name}:", text)
        self.assertNotIn("None", text)

    def test_every_real_command_is_documented(self) -> None:
        undocumented = [n for n, c in self.commands.table.items() if not c.description]
        self.assertEqual(undocumented, [])

    def test_supervisor_owns_the_command_table(self) -> None:
        supervisor = Supervisor()
        self.assertIn("sm", supervisor.commands.table)

if __name__ == "__main__":
    unittest.main()
