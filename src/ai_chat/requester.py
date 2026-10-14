"""Raw-socket HTTP client for the Ollama and OpenAI-compatible chat APIs."""

import socket
import os
import re
import select
import json
import ssl
from typing import Any
from urllib.parse import urlsplit

from .buffer import Buffer
from .models import ModelList
from .tool import Tool
from .api_key import ApiKey
from . import toolchain


class Requester:
    """raw socket HTTP client for the Ollama and OpenAI chat APIs"""

    MAX_TOOL_DEPTH: int = 8

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        open_api: bool = False,
        tls: bool = False,
        tls_verify: bool = True,
        api_key: ApiKey | None = None,
    ) -> None:
        """configure the client for host and port, tls wraps the socket in TLS"""
        self.max_recv: int = 1024
        self.connect_timeout: int = 5
        self.read_timeout: float = 60
        self.host: str = host
        self.port: int = port
        self.open_api: bool = open_api
        self.tls: bool = tls
        self.tls_verify: bool = tls_verify
        self.api_key: ApiKey | None = api_key
        self._endpoint: str = f"{host}:{port}"
        self._tls_context: ssl.SSLContext | None = None
        self.models_error: str | None = None

    @property
    def endpoint(self) -> str:
        """the endpoint as it was entered, "https://local:99" or "127.0.0.1:11434" """
        return self._endpoint

    def set_endpoint(self, endpoint: tuple[str, int] | str) -> None:
        """point the client at a new host, accepting a URL or a (host, port) tuple

        a scheme only selects the transport: "https://..." turns TLS on and
        defaults to port 443, while "http://..." stays plain and defaults to port
        80. A token without a scheme keeps the current port and TLS, so
        "localhost:11434" behaves like the (host, port) tuple form.
        """
        if isinstance(endpoint, str):
            parsed = urlsplit(endpoint if "//" in endpoint else f"//{endpoint}")
            if parsed.scheme:
                self.tls = parsed.scheme == "https"
                default_port = 443 if self.tls else 80
            else:
                default_port = self.port
            try:
                port = parsed.port
            except ValueError as error:
                raise ValueError(f"invalid endpoint {endpoint!r}: {error}") from error
            self.host = parsed.hostname or self.host
            self.port = port or default_port
            self._endpoint = endpoint if parsed.scheme else f"{self.host}:{self.port}"
            return
        host, port = endpoint
        self.host = host
        self.port = port
        self._endpoint = f"{host}:{port}"

    def _ssl_context(self) -> ssl.SSLContext:
        """build and cache the default TLS context, reused across requests"""
        if self._tls_context is None:
            self._tls_context = ssl.create_default_context()
            if not self.tls_verify:
                self._tls_context.check_hostname = False
                self._tls_context.verify_mode = ssl.CERT_NONE
        return self._tls_context

    def _connect(self) -> socket.socket:
        """open a TCP connection, wrapped in TLS when the endpoint uses https"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        try:
            s.connect((self.host, self.port))
            if self.tls:
                s = self._ssl_context().wrap_socket(s, server_hostname=self.host)
        except OSError:
            s.close()
            raise
        s.settimeout(None)
        return s

    def _friendly_error(self, error: BaseException) -> str:
        """a short, actionable description of a connection failure"""
        text = str(error)
        if isinstance(error, ssl.SSLError):
            if "WRONG_VERSION_NUMBER" in text or "UNEXPECTED_EOF" in text:
                return "the server does not speak TLS - try http:// instead of https://"
            if "CERTIFICATE_VERIFY_FAILED" in text:
                return "the certificate is not trusted - use -k/--insecure to allow it"
            return f"TLS error: {text}"
        if isinstance(error, socket.gaierror):
            return f"could not resolve '{self.host}'"
        if isinstance(error, ConnectionRefusedError):
            return "connection refused"
        if isinstance(error, TimeoutError):
            return "the server sent no data (timed out)"
        return (
            f"{error.__class__.__name__}: {text}" if text else error.__class__.__name__
        )

    def _auth_headers(self) -> list[str]:
        """the bearer Authorization header for the API key, or nothing when unavailable"""
        key = self.api_key.get_key() if self.api_key is not None else None
        return [f"Authorization: Bearer {key}"] if key is not None else []

    def make_post(self, d: dict[str, Any]) -> str:
        """assemble a raw HTTP/1.1 POST request with a JSON body"""
        body = json.dumps(d)
        headers = [
            (
                "POST /v1/chat/completions HTTP/1.1"
                if self.open_api
                else "POST /api/chat HTTP/1.1"
            ),
            f"HOST: {self.host}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
        ] + self._auth_headers()

        request = "\r\n".join(headers) + "\r\n\r\n" + body
        return request

    def _recv(
        self,
        socket_: socket.socket,
        read_fd: int | None = None,
        max_recv: int | None = None,
    ) -> bytes | None:
        """receive data, return None on cancel, or time out when the server goes idle"""
        # a TLS socket can hold decrypted bytes that select() will not report
        # as readable, so drain those before waiting on the socket again
        pending = socket_.pending() if isinstance(socket_, ssl.SSLSocket) else 0
        if pending == 0:
            ready, _, _ = select.select(
                [x for x in [socket_, read_fd] if x], [], [], self.read_timeout
            )
            if not ready:
                raise TimeoutError(f"server sent no data within {self.read_timeout:g}s")
            if read_fd in ready:
                os.read(read_fd, 64)
                return None
        data = socket_.recv(max_recv or self.max_recv)
        if not data:
            raise ConnectionError("connection closed by server")
        return data

    def _get_header(
        self, socket_: socket.socket, read_fd: int | None = None
    ) -> tuple[str, bytes] | None:
        """read the status line and headers, keeping body bytes already received"""
        reply = self._recv(socket_, read_fd)
        if reply is None:
            return None
        while b"\r\n\r\n" not in reply:
            data = self._recv(socket_, read_fd)
            if data is None:
                return None
            reply += data
        head, _, body = reply.partition(b"\r\n\r\n")
        return head.decode("utf-8", errors="replace"), body

    def _get_all(
        self,
        final: bytes,
        size: int,
        socket_: socket.socket,
        read_fd: int | None = None,
    ) -> bytes:
        """keep receiving until size bytes have arrived"""
        while len(final) < size:
            data = self._recv(
                socket_, read_fd, max_recv=min(self.max_recv, size - len(final))
            )
            if data is None:
                raise ConnectionError("response cancelled")
            final += data
        return final

    def _handle_json(
        self,
        socket_: socket.socket,
        head: str,
        body: bytes,
        read_fd: int | None = None,
    ) -> Any:
        """read a whole JSON body, decoding chunked or content-length framing"""
        # backends send header names in mixed case, so match the lowercased head
        if "transfer-encoding: chunked" in head.lower():
            full = bytearray()
            while True:
                while b"\r\n" not in body:
                    data = self._recv(socket_, read_fd)
                    if data is None:
                        raise ConnectionError("response cancelled")
                    body += data
                size_line, _, body = body.partition(b"\r\n")
                size = int(size_line, 16)
                if size == 0:
                    break
                while len(body) < size:
                    data = self._recv(socket_, read_fd)
                    if data is None:
                        raise ConnectionError("response cancelled")
                    body += data
                full += body[:size]
                body = body[size:]
                if body[:2] == b"\r\n":
                    body = body[2:]
            ret = json.loads(full.decode("utf-8", errors="replace"))
        else:
            match = re.search(r"content-length:\s*(\d+)", head, re.IGNORECASE)
            if not match:
                raise ValueError("missing content-length in response headers")
            size = int(match.group(1))
            final = self._get_all(body, size, socket_, read_fd)
            ret = json.loads(final.decode("utf-8", errors="replace"))
        socket_.close()
        return ret

    def get_models(self) -> ModelList:
        """fetch the model list, errors are remembered in models_error"""
        self.models_error = None
        try:
            with self._connect() as s:
                path = "/v1/models" if self.open_api else "/api/tags"
                request = (
                    "\r\n".join(
                        [
                            f"GET {path} HTTP/1.1",
                            f"HOST: {self.host}",
                        ]
                        + self._auth_headers()
                    )
                    + "\r\n\r\n"
                )
                s.send(request.encode())

                raw = self._get_header(s)
                if raw is None:
                    raise ConnectionError("response cancelled")

                head, body = raw
                if "content-type: application/jso" not in head.lower():
                    self.models_error = f"unexpected response: {head}"
                    return ModelList([])
                return ModelList.from_response(self._handle_json(s, head, body))
        except (OSError, ValueError, IndexError, KeyError) as error:
            self.models_error = self._friendly_error(error)
            return ModelList([])

    def _handle_tool_call(
        self,
        chat: Buffer,
        tcalls: list[dict[str, Any]],
        collection: dict[Any, dict[str, Any]] | None = None,
    ) -> None:
        """collect streamed tool call fragments, or queue complete calls for approval"""
        if collection is not None:
            for tc in tcalls:
                index = tc.get("index", 0)
                entry = collection.setdefault(
                    index,
                    {
                        "id": tc.get("id"),
                        "index": index,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    entry["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    entry["function"]["name"] = function["name"]
                fragment = function.get("arguments")
                if isinstance(fragment, str):
                    entry["function"]["arguments"] += fragment
            return
        chat.add_ai_tool(tcalls)
        pending: list[dict[str, Any]] = []
        for tool_call in tcalls:
            name = tool_call["function"]["name"]
            _args = tool_call["function"]["arguments"]
            if isinstance(_args, str):
                try:
                    _args = json.loads(_args)
                except (json.JSONDecodeError, TypeError):
                    _args = {}
            if not isinstance(_args, dict):
                _args = {}
            known = self._find_tool(name) is not None
            pending.append(
                {
                    "id": tool_call.get("id"),
                    "name": name,
                    "args": _args,
                    "approved": None if known else False,
                    "result": None if known else f"tool {name!r} does not exist",
                }
            )
        chat.set_pending(pending)

    @staticmethod
    def _find_tool(name: str) -> Tool | None:
        """look up a registered tool by name"""
        return next((tool for tool in toolchain.toolchain if tool.name == name), None)

    def resume(self, chat: Buffer) -> str | None:
        """run the approved pending tool calls, then continue the conversation"""
        calls = chat.take_pending()
        try:
            for call in calls:
                if call["result"] is not None:
                    content = call["result"]
                elif call["approved"] is not True:
                    content = "user declined call"
                else:
                    try:
                        tool = self._find_tool(call["name"])
                        if tool is None:
                            raise LookupError(f"tool {call['name']!r} does not exist")
                        content = tool.function(**call["args"])
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        content = f"tool call failed: {error}"
                with chat.cond:
                    chat.chat_data["messages"].append(
                        {
                            "role": "tool",
                            "id": call["id"],
                            "tool_name": call["name"],
                            "content": content,
                        }
                    )
        except BaseException:
            with chat.cond:
                chat.in_request = False
            raise
        return self.get_reply(chat, wrapped=True)

    def _handle_msg(
        self,
        chat: Buffer,
        body: Any,
        tool_calls_collection: dict[Any, dict[str, Any]] | None = None,
    ) -> tuple[bool, bool]:
        """apply one streamed frame to the chat, returns done and whether a tool call arrived"""
        if "message" in body:  # ollama api
            msg = body["message"]["content"]
            if msg is not None:
                chat.add_ai(msg)
            if body["done"]:
                chat.set_tokens(body["prompt_eval_count"] + body["eval_count"])
            if body["message"].get("tool_calls"):
                self._handle_tool_call(chat, body["message"]["tool_calls"])
                return (body["done"], True)
            return (body["done"], False)
        # choices is empty on the trailing usage-only chunk sent when
        # stream_options: {"include_usage": true} is set
        choices = body.get("choices") or []
        _tool_calls = None
        done = False
        if choices:
            choice = choices[0]  # only the first choice is rendered
            if "message" in choice:
                msg = choice["message"]["content"]
                _tool_calls = choice["message"].get("tool_calls", None)
            else:
                msg = choice["delta"].get("content", None)
                _tool_calls = choice["delta"].get("tool_calls", None)
            done = choice["finish_reason"] == "stop"
            if msg is not None:
                chat.add_ai(msg)

        # "usage" is null on every chunk except the final one
        usage = body.get("usage")
        if usage:
            chat.set_tokens(usage["total_tokens"])
        elif done:
            chat.set_tokens(chat.estimate_tokens())

        if _tool_calls:
            self._handle_tool_call(chat, _tool_calls, tool_calls_collection)
            return (done, tool_calls_collection is None)
        return (done, False)

    def get_reply(self, chat: Buffer, wrapped: bool = False) -> str | None:
        """request a reply, returns an error message when something went wrong"""
        call = False
        s: socket.socket | None = None
        with chat.cond:
            chat.in_request = True
            if not wrapped:
                chat.clear_cancel()
        try:
            if wrapped:
                chat.tool_depth += 1
            else:
                chat.tool_depth = 1
            if chat.tool_depth > self.MAX_TOOL_DEPTH:
                chat.add_msg("user")
                return f"stopped: tool call chain exceeded {self.MAX_TOOL_DEPTH} rounds"

            s = self._connect()
            with chat.cond:
                request = self.make_post(
                    chat.openai_data if self.open_api else chat.chat_data
                )
            s.send(request.encode())

            chat.add_msg("assistant")
            raw = self._get_header(s, chat.pipe[0])
            if raw is None:
                s.close()
                chat.add_msg("user")
                chat.notify()
                return None

            head, body_bytes = raw
            if "HTTP/1.1 200 OK" not in head:
                s.close()
                return "\n".join([head, body_bytes.decode("utf-8", errors="replace")])

            if "content-type: application/x-ndjson" in head.lower():
                body = body_bytes.decode("utf-8", errors="replace").splitlines()
                done = False
                while not done and not chat.cancel:
                    body = [x for x in body if x != ""]
                    if len(body) == 0:
                        final = self._recv(s, chat.pipe[0])
                        if final is None:
                            break
                        body = final.decode("utf-8", errors="replace").splitlines()
                        if len(body) == 0:
                            continue
                    try:
                        size = int(body[0], 16)
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f"invalid chunk size frame: {body[:1]!r}") from exc
                    if len(body) < 2:
                        raise ValueError("incomplete ndjson frame")
                    if size != len(body[1].encode()) + 1:
                        final = self._get_all(
                            body[1].encode(), size + 2, s, chat.pipe[0]
                        )
                        body[1] = final.decode("utf-8", errors="replace").splitlines()[
                            0
                        ]
                    done, call = self._handle_msg(chat, json.loads(body[1]))
                    if call:
                        break
                    body.pop(0)
                    body.pop(0)

            if "content-type: text/event-stream" in head.lower():
                body = body_bytes.decode("utf-8", errors="replace").splitlines()
                finished = False
                tool_calls_collection: dict[Any, dict[str, Any]] = {}
                while not finished and not chat.cancel:
                    body = [x for x in body if x != ""]

                    if len(body) == 0:
                        final = self._recv(s, chat.pipe[0])
                        if final is None:
                            break
                        body = final.decode("utf-8", errors="replace").splitlines()
                        body = [x for x in body if x != ""]
                        if len(body) == 0:
                            continue
                    try:
                        size = int(body[0], 16)
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f"invalid chunk size frame: {body[:1]!r}") from exc

                    if len(body) >= 2 and size != len(body[1].encode()):
                        if len(body) >= 3:
                            final = "\n\n".join(body[1:]).encode()
                        else:
                            final = body[1].encode()
                        final = self._get_all(final, size - 2, s, chat.pipe[0])
                        body = final.decode("utf-8", errors="replace").splitlines()
                    else:
                        body.pop(0)
                    if len(body) == 0:
                        continue
                    while len(body) > 0 and "data:" in body[0]:
                        if body[0].startswith("data: [DONE]"):
                            finished = True
                            break
                        _, call = self._handle_msg(
                            chat, json.loads(body[0][6:]), tool_calls_collection
                        )
                        if call:
                            break
                        body.pop(0)
                    if call:
                        break

                if len(list(tool_calls_collection.values())) != 0:
                    self._handle_tool_call(chat, list(tool_calls_collection.values()))
                    call = True

            if "content-type: application/json" in head.lower():
                body = self._handle_json(s, head, body_bytes, chat.pipe[0])
                _, call = self._handle_msg(chat, body)
            s.close()
            if call:
                if chat.pending_resolved():  # nothing left to ask the user
                    return self.resume(chat)
                return None  # request ends here, user approval resumes it
            chat.add_msg("user")
            return None
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            IndexError,
            KeyError,
        ) as error:
            with chat.cond:
                last = chat.chat_data["messages"][-1]
                if (
                    last.get("role") == "assistant"
                    and not last.get("content", "")
                    and "tool_calls" not in last
                ):
                    chat.chat_data["messages"].pop()
                chat.updated = True
                chat.cond.notify()
            if chat.cancel:
                return None
            if isinstance(error, OSError):
                return f"cannot reach {self.endpoint} - {self._friendly_error(error)}"
            return self._friendly_error(error)
        finally:
            if s is not None:
                s.close()
            with chat.cond:
                chat.in_request = False
