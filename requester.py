import socket
import os
import re
import select
import json

from buffer import Buffer
from dialog import Dialog
import toolchain


class Requester:
    MAX_TOOL_DEPTH = 8

    def __init__(self, host="127.0.0.1", port=11434, dialog=None, open_api=False):
        self.max_recv = 1024
        self.connect_timeout = 5
        self.host = host
        self.port = port
        self.dialog = dialog
        self.open_api = open_api
        self._depth = 0

    def set_endpoint(self, endpoint) -> None:
        host, port = endpoint
        self.host = host
        self.port = port

    def make_post(self, d) -> str:
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

    def _recv(self, socket_, read_fd=None, max_recv=None):
        r, _, _ = select.select([x for x in [socket_, read_fd] if x], [], [])
        if read_fd in r:
            os.read(read_fd, 2)
            return None
        data = socket_.recv(max_recv or self.max_recv)
        if not data:
            raise ConnectionError("connection closed by server")
        return data

    def _get_header(self, socket_, read_fd=None):
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

    def _get_all(self, final, size, socket_):
        while len(final) < size:
            data = self._recv(
                socket_, max_recv=min(self.max_recv, size - len(final))
            )
            if data is None:
                raise ConnectionError("response cancelled")
            final += data
        return final

    def _handle_json(self, socket_, existing):
        body = existing[1].encode()
        if "Transfer-Encoding: chunked" in existing[0]:
            full = bytearray()
            while True:
                while b"\r\n" not in body:
                    data = self._recv(socket_)
                    if data is None:
                        raise ConnectionError("response cancelled")
                    body += data
                size_line, _, body = body.partition(b"\r\n")
                size = int(size_line, 16)
                if size == 0:
                    break
                while len(body) < size:
                    data = self._recv(socket_)
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
            final = self._get_all(body, size, socket_)
            ret = json.loads(final.decode("utf-8", errors="replace"))
        socket_.close()
        return ret

    def get_models(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            s.settimeout(None)
            headers = [
                f"GET {"/v1/models" if self.open_api else "/api/tags"} HTTP/1.1",
                f"HOST: {self.host}",
            ]

            request = "\r\n".join(headers) + "\r\n\r\n"
            s.send(request.encode())

            raw = self._get_header(s)

            if "content-type: application/jso" in raw[0].lower():
                return self._handle_json(s, raw)
            return {}
        except (OSError, ValueError, IndexError, KeyError):
            return {}

    def _handle_tool_call(self, chat: Buffer, tcalls, collection=None):
        if collection is not None:
            _id = tcalls[0].get("id", None)
            _index = tcalls[0].get("index", None)
            _type = tcalls[0].get("type", None)

            if _type:
                collection[_index] = {
                    "id": _id,
                    "index": _index,
                    "function": tcalls[0]["function"],
                }
            else:
                _fragment = tcalls[0]["function"]["arguments"]
                if isinstance(_fragment, str):
                    collection[_index]["function"]["arguments"] += _fragment

            return
        chat.add_ai_tool(tcalls)
        for tool_call in tcalls:
            name = tool_call["function"]["name"]
            _args = tool_call["function"]["arguments"]
            if isinstance(_args, str):
                try:
                    _args = json.loads(_args)
                except (json.JSONDecodeError, TypeError):
                    _args = {}
            _tool = next(
                (__tool for __tool in toolchain.toolchain if __tool.name == name),
                None,
            )
            if _tool is None:
                content = f"tool {name!r} does not exist"
            else:
                _reply = (
                    self.dialog.ask(
                        (
                            f"function: {name}"
                            f" arguments: {_args}"
                        ),
                        chat.cond,
                    )
                    if self.dialog
                    else "\n"
                )
                if _reply != "\n":
                    content = "user declined call"
                else:
                    try:
                        content = _tool.function(**_args)
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        content = f"tool call failed: {error}"
            with chat.cond:
                chat.chat_data["messages"].append(
                    {
                        "role": "tool",
                        "id": tool_call["id"],
                        "tool_name": name,
                        "content": content,
                    }
                )

    def _handle_msg(
        self, chat: Buffer, body, tool_calls_collection=None
    ) -> tuple[bool, bool]:
        if "message" in body:  # ollama api
            msg = body["message"]["content"]
            if msg is not None:
                chat.add_ai(msg)
            if body["done"]:
                chat.set_tokens(body["prompt_eval_count"] + body["eval_count"])
            if "tool_calls" in body["message"]:
                self._handle_tool_call(chat, body["message"]["tool_calls"])
                return (body["done"], True)
            return (body["done"], False)
        if "message" in body["choices"][0]:
            msg = body["choices"][0]["message"]["content"]  # ignore choices
        else:
            msg = body["choices"][0]["delta"].get("content", None)  # ignore choices
        done = body["choices"][0]["finish_reason"] == "stop"
        if msg is not None:
            chat.add_ai(msg)
        if "usage" in body:
            chat.set_tokens(body["usage"]["total_tokens"])
        elif done:
            chat.set_tokens(chat.estimate_tokens())

        if "message" in body["choices"][0]:
            _tool_calls = body["choices"][0]["message"].get(
                "tool_calls", None
            )  # ignore choices
        else:
            _tool_calls = body["choices"][0]["delta"].get(
                "tool_calls", None
            )  # ignore choices
        if _tool_calls:
            self._handle_tool_call(chat, _tool_calls, tool_calls_collection)
            return (done, tool_calls_collection is None)
        return (done, False)

    def get_reply(self, chat: Buffer, wrapped=False) -> bool:
        call = False
        chat.cancel = False
        with chat.cond:
            chat.in_request = True
        try:
            if wrapped:
                self._depth += 1
            else:
                self._depth = 1
            if self._depth > self.MAX_TOOL_DEPTH:
                if self.dialog:
                    self.dialog.set_text(
                        "stopped: tool call chain exceeded "
                        f"{self.MAX_TOOL_DEPTH} rounds"
                    )
                return False

            r, _, _ = select.select([chat.pipe[0]], [], [], 0.5)
            if r:
                os.read(chat.pipe[0], 2)

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            s.settimeout(None)
            s.send(
                self.make_post(
                    chat.openai_data if self.open_api else chat.chat_data
                ).encode()
            )

            chat.add_msg("assistant")
            raw = self._get_header(s, chat.pipe[0])
            if raw is None:
                s.close()
                chat.add_msg("user")
                chat.notify()
                return False

            if "HTTP/1.1 200 OK" not in raw[0]:
                s.close()
                self.dialog.set_text("\n".join(raw))
                return True

            if "content-type: application/x-ndjson" in raw[0].lower():
                body = raw[1].splitlines()
                done = False
                while not done and not chat.cancel:
                    body = [x for x in body if x != ""]
                    if len(body) == 0:
                        final = self._recv(s)
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
                        final = self._get_all(body[1].encode(), size + 2, s)
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
                done = False
                tool_calls_collection = {}
                while not done and not chat.cancel:
                    body = [x for x in body if x != ""]

                    if len(body) == 0:
                        final = self._recv(s)
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
                        final = self._get_all(final, size - 2, s)
                        body = final.decode(
                            "utf-8", errors="replace"
                        ).splitlines()
                    else:
                        body.pop(0)
                    if len(body) == 0:
                        continue
                    while len(body) > 0 and "data:" in body[0]:
                        if body[0].startswith("data: [DONE]"):
                            done = True
                            break
                        done, call = self._handle_msg(
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
                body = self._handle_json(s, raw)
                _, call = self._handle_msg(chat, body)
            s.close()
            if call:
                self.get_reply(chat, wrapped=True)
            if not wrapped:
                chat.add_msg("user")
            return False
        except (OSError, ValueError, json.JSONDecodeError, IndexError, KeyError) as error:
            if not chat.cancel and self.dialog:
                self.dialog.set_text(str(error))
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
            return False
        finally:
            with chat.cond:
                chat.in_request = False
