import socket
import os
import select
import json

from buffer import Buffer
from dialog import Dialog
import toolchain


class Requester:
    def __init__(self, host="127.0.0.1", port=11434, dialog=None, open_api=False):
        self.max_recv = 1024
        self.host = host
        self.port = port
        self.dialog = dialog
        self.open_api = open_api

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

    def _get_header(self, socket_, read_fd=None):
        r, _, _ = select.select([x for x in [socket_, read_fd] if x], [], [])
        if read_fd in r:
            os.read(read_fd, 2)
            return None
        reply = socket_.recv(self.max_recv)
        while b"\r\n\r\n" not in reply:
            reply += socket_.recv(self.max_recv)
        raw = reply.decode().split("\r\n\r\n")
        if len(raw) < 2:
            raw.append("")
        return raw

    def _get_all(self, final, size, socket_):
        while len(final) < size:
            final += socket_.recv(min(self.max_recv, size - len(final)))
        return final

    def _handle_json(self, socket_, existing):
        if "Transfer-Encoding: chunked" in existing[0]:
            body = existing[1].encode()
            while b"\r\n0\r\n\r\n" not in body:
                body += socket_.recv(self.max_recv)
            ret = json.loads(body.decode().split("\r\n")[1])
        else:
            if "Content-Length:" in existing[0]:
                length = existing[0][existing[0].find("Content-Length: ") :]
            else:
                length = existing[0][existing[0].find("content-length: ") :]
            _find_c = length.find("\n")
            length = length[: len(length) if _find_c < 0 else _find_c]
            size = int(length[15::])
            final = self._get_all(existing[1].encode(), size, socket_)
            ret = json.loads(final.decode())
        socket_.close()
        return ret

    def get_models(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
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
                collection[_index]["function"]["arguments"] += tcalls[0]["function"][
                    "arguments"
                ]

            return
        chat.add_ai_tool(tcalls)
        for tool_call in tcalls:
            _args = tool_call["function"]["arguments"]
            if not isinstance(_args, dict):
                _arguments = json.loads(_args)

            _tool = [
                __tool
                for __tool in toolchain.toolchain
                if __tool.name == tool_call["function"]["name"]
            ][0]
            if isinstance(_args, str):
                _args = json.loads(_args)
            _reply = (
                self.dialog.ask(
                    (
                        f'function: {tool_call["function"]["name"]}'
                        f" arguments: {_args}"
                    ),
                    chat.cond,
                )
                if self.dialog
                else "\n"
            )
            chat.chat_data["messages"].append(
                {
                    "role": "tool",
                    "id": tool_call["id"],
                    "tool_name": tool_call["function"]["name"],
                    "content": (
                        "user declined call"
                        if _reply != "\n"
                        else _tool.function(**_args)
                    ),
                }
            )

    def _handle_msg(
        self, chat: Buffer, body, tool_calls_collection=None
    ) -> (bool, bool):
        if "message" in body:  # ollama api
            msg = body["message"]["content"]
            if msg is not None:
                chat.add_ai(msg)
            if body["done"]:
                chat.tokens = body["prompt_eval_count"] + body["eval_count"]
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
            chat.tokens = body["usage"]["total_tokens"]
        elif done:
            chat.tokens = chat.estimate_tokens()

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
        chat.in_request = True
        r, _, _ = select.select([chat.pipe[0]], [], [], 0.5)
        if r:
            os.read(chat.pipe[0], 2)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(
            self.make_post(
                chat.openai_data if self.open_api else chat.chat_data
            ).encode()
        )

        chat.add_msg("assistant")
        raw = self._get_header(s, chat.pipe[0])
        if raw is None:
            chat.in_request = False
            s.close()
            chat.add_msg("user")
            chat.notify()
            return False

        if "HTTP/1.1 200 OK" not in raw[0]:
            chat.in_request = False
            s.close()
            self.dialog.set_text("\n".join(raw))
            return True

        if "content-type: application/x-ndjson" in raw[0].lower():
            body = raw[1].splitlines()
            done = False
            while not done and not chat.cancel:
                body = [x for x in body if x != ""]
                if len(body) == 0:
                    body = s.recv(self.max_recv).decode().splitlines()
                size = int(body[0], 16)
                if size != len(body[1].encode()) + 1:
                    final = body[1].encode()
                    final = self._get_all(final, size + 2, s)
                    body[1] = final.decode()
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
                    final = s.recv(self.max_recv)
                    body = final.decode().splitlines()
                    body = [x for x in body if x != ""]
                size = int(body[0], 16)

                if size != len(body[1].encode()):
                    if len(body) >= 3:
                        final = "\n\n".join(body[1:]).encode()
                    else:
                        final = body[1].encode()
                    final = self._get_all(final, size - 2, s)
                    final = body[1].encode()
                    body[1] = final.decode().splitlines()[0]

                body.pop(0)
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
        chat.in_request = False
        if call:
            self.get_reply(chat, wrapped=True)
        if not wrapped:
            chat.add_msg("user")
        return False
