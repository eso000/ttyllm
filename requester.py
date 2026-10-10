import socket
import os
import re
import select
import json
from typing import Any

from buffer import Buffer
from models import Model
from tool import Tool
import toolchain


def _parse_models(data: Any) -> list[Model]:
    """parse the model list out of an Ollama or OpenAI API response"""
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("data"), list):  # openai api: names only
        return [Model(name=model["id"]) for model in data["data"] if "id" in model]
    if isinstance(data.get("models"), list):  # ollama api
        parsed: list[Model] = []
        for model in data["models"]:
            if "name" not in model:
                continue
            size = model.get("size")
            details = model.get("details") or {}
            parsed.append(
                Model(
                    name=model["name"],
                    size_gb=size // 1073741824 if size is not None else None,
                    parameter_size=details.get("parameter_size"),
                )
            )
        return parsed
    return []


class Requester:
    """raw socket HTTP client for the Ollama and OpenAI chat APIs"""
    MAX_TOOL_DEPTH: int = 8

    def __init__(
        self, host: str = "127.0.0.1", port: int = 11434, open_api: bool = False
    ) -> None:
        """configure the client for host and port, open_api selects the OpenAI wire format"""
        self.max_recv: int = 1024
        self.connect_timeout: int = 5
        self.host: str = host
        self.port: int = port
        self.open_api: bool = open_api
        self.models_error: str | None = None

    def set_endpoint(self, endpoint: tuple[str, int]) -> None:
        """point the client at a different host and port"""
        host, port = endpoint
        self.host = host
        self.port = port

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
        ]

        request = "\r\n".join(headers) + "\r\n\r\n" + body
        return request

    def _recv(
        self, socket_: socket.socket, read_fd: int | None = None, max_recv: int | None = None
    ) -> bytes | None:
        """receive data, or return None when the cancel pipe fires instead"""
        r, _, _ = select.select([x for x in [socket_, read_fd] if x], [], [])
        if read_fd in r:
            os.read(read_fd, 64)
            return None
        data = socket_.recv(max_recv or self.max_recv)
        if not data:
            raise ConnectionError("connection closed by server")
        return data

    def _get_header(
        self, socket_: socket.socket, read_fd: int | None = None
    ) -> list[str] | None:
        """read up to the header terminator, returns headers and body or None on cancel"""
        reply = self._recv(socket_, read_fd)
        if reply is None:
            return None
        while b"\r\n\r\n" not in reply:
            data = self._recv(socket_, read_fd)
            if data is None:
                return None
            reply += data
        raw = reply.decode().split("\r\n\r\n")
        if len(raw) < 2:
            raw.append("")
        return raw

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
        existing: list[str],
        read_fd: int | None = None,
    ) -> Any:
        """read a whole JSON body, decoding chunked or content-length framing"""
        body = existing[1].encode()
        if "Transfer-Encoding: chunked" in existing[0]:
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
            match = re.search(
                r"content-length:\s*(\d+)", existing[0], re.IGNORECASE
            )
            if not match:
                raise ValueError("missing content-length in response headers")
            size = int(match.group(1))
            final = self._get_all(body, size, socket_, read_fd)
            ret = json.loads(final.decode("utf-8", errors="replace"))
        socket_.close()
        return ret

    def get_models(self) -> list[Model]:
        """fetch the model list, errors are remembered in models_error"""
        self.models_error = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.connect_timeout)
                s.connect((self.host, self.port))
                s.settimeout(None)
                path = "/v1/models" if self.open_api else "/api/tags"
                request = "\r\n".join(
                    [
                        f"GET {path} HTTP/1.1",
                        f"HOST: {self.host}",
                    ]
                ) + "\r\n\r\n"
                s.send(request.encode())

                raw = self._get_header(s)
                if raw is None:
                    raise ConnectionError("response cancelled")

                if "content-type: application/jso" not in raw[0].lower():
                    self.models_error = f"unexpected response: {raw[0]}"
                    return []
                return _parse_models(self._handle_json(s, raw))
        except (OSError, ValueError, IndexError, KeyError) as error:
            self.models_error = str(error) or error.__class__.__name__
            return []

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

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            s.settimeout(None)
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

            if "HTTP/1.1 200 OK" not in raw[0]:
                s.close()
                return "\n".join(raw)

            if "content-type: application/x-ndjson" in raw[0].lower():
                body = raw[1].splitlines()
                done = False
                while not done and not chat.cancel:
                    body = [x for x in body if x != ""]
                    if len(body) == 0:
                        final = self._recv(s, chat.pipe[0])
                        if final is None:
                            break
                        body = final.decode(
                            "utf-8", errors="replace"
                        ).splitlines()
                        if len(body) == 0:
                            continue
                    try:
                        size = int(body[0], 16)
                    except (ValueError, IndexError):
                        raise ValueError(
                            f"invalid chunk size frame: {body[:1]!r}"
                        )
                    if len(body) < 2:
                        raise ValueError("incomplete ndjson frame")
                    if size != len(body[1].encode()) + 1:
                        final = self._get_all(
                            body[1].encode(), size + 2, s, chat.pipe[0]
                        )
                        body[1] = final.decode(
                            "utf-8", errors="replace"
                        ).splitlines()[0]
                    done, call = self._handle_msg(chat, json.loads(body[1]))
                    if call:
                        break
                    body.pop(0)
                    body.pop(0)

            if "content-type: text/event-stream" in raw[0].lower():
                body = raw[1].splitlines()
                finished = False
                tool_calls_collection: dict[Any, dict[str, Any]] = {}
                while not finished and not chat.cancel:
                    body = [x for x in body if x != ""]

                    if len(body) == 0:
                        final = self._recv(s, chat.pipe[0])
                        if final is None:
                            break
                        body = final.decode(
                            "utf-8", errors="replace"
                        ).splitlines()
                        body = [x for x in body if x != ""]
                        if len(body) == 0:
                            continue
                    try:
                        size = int(body[0], 16)
                    except (ValueError, IndexError):
                        raise ValueError(
                            f"invalid chunk size frame: {body[:1]!r}"
                        )

                    if len(body) >= 2 and size != len(body[1].encode()):
                        if len(body) >= 3:
                            final = "\n\n".join(body[1:]).encode()
                        else:
                            final = body[1].encode()
                        final = self._get_all(final, size - 2, s, chat.pipe[0])
                        body = final.decode(
                            "utf-8", errors="replace"
                        ).splitlines()
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

            if "content-type: application/json" in raw[0].lower():
                body = self._handle_json(s, raw, chat.pipe[0])
                _, call = self._handle_msg(chat, body)
            s.close()
            if call:
                if chat.pending_resolved():  # nothing left to ask the user
                    return self.resume(chat)
                return None  # request ends here, user approval resumes it
            chat.add_msg("user")
            return None
        except (OSError, ValueError, json.JSONDecodeError, IndexError, KeyError) as error:
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
            return f"{error.__class__.__name__}: {error}"
        finally:
            if s is not None:
                s.close()
            with chat.cond:
                chat.in_request = False
