import socket
import json
import re
import sys
import termios
import threading
import dataclasses
import signal
import shutil
from glob import glob
from enum import Enum


class ColorCodeExeption(Exception):
    pass


@dataclasses.dataclass
class TextBox:
    lines: list = dataclasses.field(default_factory=list)
    last = -1


def text_to_lines(text, width, color=""):
    if text == "":
        return [""]
    res = []
    res_lines = text.replace("\t", "   ").split("\n")
    for line in res_lines:
        if len(line) <= width:
            res.append(f"{color}{line}")
        else:
            start = 0
            while start <= len(line) - 1:
                step = 0
                line_fitting_part = line[start : start + width]
                if len(line_fitting_part) + start < len(line):
                    last_space = line_fitting_part.rfind(" ")
                    if last_space != -1:
                        number_of_selected_chars = last_space
                        step = 1
                    else:
                        number_of_selected_chars = len(line_fitting_part)
                else:
                    number_of_selected_chars = width
                res += [f"{color}{line_fitting_part[0:number_of_selected_chars]}"]
                start += number_of_selected_chars + step

    return res


def clean_screan() -> None:
    print("\033[H\033[J\033[3J", end="", flush=True)


def make_text_box(text, width, last=-1, color="") -> TextBox:
    res = TextBox()
    res.last = last
    res.lines = text_to_lines(text, width, color)
    return res


def fit_to_box(
    box, width, height, start_x=1, start_y=1, background="", hide_cursor=False
):
    # for x in range(height):

    lines, last = (box.lines, box.last)

    start = 0
    if last < 0:
        last = len(lines)
    print("\033[?25l", end="")

    if len(lines) <= height:
        last = len(lines)
    last = max(last, height)
    if last >= height:
        start = max(0, last - height)
    print(f"\033[{start_x};{start_y}H", end="", flush=True)

    for x in range(max(0, height - len(lines))):
        print(f"{background}\033[{start_x+x+ len(lines)};{start_y}H", end="")
        print(width * " ", end="")

    for x, line in enumerate(lines[start:last]):
        print(f"\033[{start_x+x};{start_y}H", end="")
        print(f"{background}{width * ' '}", end="")
        print(f"\033[{start_x+x};{start_y}H", end="")
        print(
            line,
            end="",
        )

    print("\033[0m", end="", flush=True)

    if box.last != -1 or hide_cursor:
        print("\033[?25l", end="", flush=True)
    else:
        print("\033[?25h", end="", flush=True)

    if box.last == -1:
        return (len(lines), -1)
    return (len(lines), max(last, height))


def fit_text_to_box(
    text,
    width,
    height,
    start_x=1,
    start_y=1,
    color="",
    background="",
    hide_cursor=False,
):
    return fit_to_box(
        make_text_box(text, width, color=color),
        width,
        height,
        start_x,
        start_y,
        background,
        hide_cursor,
    )


class Buffer:
    TOOLS = False

    def __init__(self):
        self.ai_name = "AI"
        self.user_name = "You"
        self.ai_color = "\033[35m"
        self.user_color = "\033[36m"
        self.normal_color = "\033[37m"
        self.cancel = False
        self.updated = False
        self.show_data = False
        self.show_system = False
        self.chat_data = {
            "model": "llama3.2",  # mistral-nemo",
            "stream": True,
            "think": False,
            "messages": [],
        }
        if self.TOOLS:
            self.chat_data["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "description": "Get the current temperature for a city",
                        "parameters": {
                            "type": "object",
                            "required": ["city"],
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The name of the city",
                                }
                            },
                        },
                    },
                }
            ]

        self.tokens = -1
        self.cond = threading.Condition()
        self.in_request = False
        self.add_msg("system")

    def empty(self):
        with self.cond:
            self.chat_data["messages"] = []
            self.add_msg("system")
            self.updated = True
            self.cond.notify()

    def del_msg(self, number=1):
        with self.cond:
            for _ in range(min(number, len(self.chat_data["messages"]))):
                curent_role = self.chat_data["messages"][-1]["role"]
                initial_role = self.chat_data["messages"][-1]["role"]
                while curent_role in (initial_role, "tool"):
                    self.chat_data["messages"].pop()
                    if len(self.chat_data["messages"]) == 0:
                        break
                    curent_role = self.chat_data["messages"][-1]["role"]

            self.updated = True
            self.cond.notify()

    def add_msg(self, role):
        with self.cond:
            self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def change_msg(self, role):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1]["content"] == "":
                self.chat_data["messages"][-1]["role"] = role
            else:
                self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def add_open(self, ch):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1].get("role") in (
                "user",
                "system",
                "assistant",
            ):
                self.chat_data["messages"][-1]["content"] += ch
                self.show_data = False
                self.show_system = False
                self.updated = True
                self.cond.notify()

    def del_open(self):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1].get("role") in (
                "user",
                "system",
                "assistant",
            ):
                self.chat_data["messages"][-1]["content"] = self.chat_data["messages"][
                    -1
                ]["content"][:-1]
            self.updated = True
            self.cond.notify()

    def clear_msg(self):
        with self.cond:
            self.chat_data["messages"][-1]["content"] = ""
            self.updated = True
            self.cond.notify()

    def add_ai_tool(self, tool_calls):
        with self.cond:
            if self.in_request:
                if self.chat_data["messages"][-1].get("role") == "assistant":
                    self.chat_data["messages"][-1]["tool_calls"] = tool_calls

    def add_ai(self, ch):
        with self.cond:
            if self.in_request:
                if self.chat_data["messages"][-1].get("role") == "assistant":
                    self.chat_data["messages"][-1]["content"] += ch
                    self.updated = True
                    self.cond.notify()

    def _process_color(self, value: str) -> str:
        pattern_8bit = re.compile("^8b:[0-9]$")
        pattern_256bit = re.compile("^256b:[0-9]{1,3}$")
        pattern_rgb = re.compile("^rgb:[0-9]{1,3};[0-9]{1,3};[0-9]{1,3}$")
        if pattern_8bit.match(value):
            return f"\033[3{value[-1]}m"
        if pattern_256bit.match(value):
            return f"\033[38;5;{value[5:]}m"
        if pattern_rgb.match(value):
            return f"\033[38;2;{value[4:]}m"
        raise ColorCodeExeption(f"{value} does not fit acceptable formats")

    def set_normal_color(self, value: str) -> None:
        self.normal_color = self._process_color(value)

    def set_ai_color(self, value: str) -> None:
        self.ai_color = self._process_color(value)

    def set_user_color(self, value: str) -> None:
        self.user_color = self._process_color(value)

    def set_data(self, data):
        def _check_dict(_dict, _key, _instance):
            if _key not in _dict:
                return True
            if not isinstance(_dict[_key], _instance):
                return True
            return False

        if _check_dict(data, "model", str):
            return
        if _check_dict(data, "messages", list):
            return
        for msg in data["messages"]:
            if _check_dict(msg, "role", str):
                return
            if _check_dict(msg, "content", str) or _check_dict(msg, "tool_calls", list):
                return
        with self.cond:
            self.chat_data = data
            self.updated = True
            self.cond.notify()

    def notify(self):
        with self.cond:
            self.updated = True
            self.cond.notify()

    def get_text_box_system(self, width, last=-1) -> str:
        result = TextBox()
        res = []
        res += [f"{self.normal_color}system:"]
        msg_ = self.chat_data["messages"][0]
        if msg_["content"] == "":
            res += [""]
        else:
            res_lines = msg_["content"].split("\n")
            for line in res_lines:
                if line == "":
                    res.append("")
                else:
                    res += [
                        f"{self.normal_color}{line[i : i + width]}"
                        for i in range(0, len(line), width)
                    ]
        result.last = last
        result.lines = res
        return result

    def get_text_box_data(self, width, last=-1) -> str:
        return make_text_box(json.dumps(self.chat_data, indent=4), width, last=last)

    def get_text_box(self, width, last=-1) -> str:
        result = TextBox()
        res = []
        if (
            len(self.chat_data["messages"]) == 1
            and self.chat_data["messages"][0]["role"] == "system"
        ):
            res += [f"{self.normal_color}system:"]
            msg_ = self.chat_data["messages"][0]
            res += text_to_lines(msg_["content"], width, self.normal_color)
            result.last = last
            result.lines = res
            return result

        for msg_ in self.chat_data["messages"]:
            if msg_["role"] in ("system", "tool") or "tool_calls" in msg_:
                continue
            color = self.user_color if msg_["role"] == "user" else self.ai_color
            name = self.user_name if msg_["role"] == "user" else self.ai_name
            res += [f"{self.normal_color}{name}:"]
            res += text_to_lines(msg_["content"], width, color)
        result.last = last
        result.lines = res
        return result

    def get_string(self) -> str:
        res = ""
        for msg_ in self.chat_data["messages"]:
            if msg_["role"] == "user":
                res += f"{self.normal_color}{self.user_name}\n"
                res += f"{self.user_color}{msg_["content"]}\n"
            if msg_["role"] == "assistant":
                res += f"{self.normal_color}{self.ai_name}\n"
                res += f"{self.ai_color}{msg_["content"]}\n"
        return res


class Requester:
    def __init__(self, host="127.0.0.1", port=11434):
        self.max_recv = 1024
        self.host = host
        self.port = port

    def set_endpoint(self, endpoint) -> None:
        host, port = endpoint
        self.host = host
        self.port = port

    def make_post(self, d) -> str:
        body = json.dumps(d)
        headers = [
            "POST /api/chat HTTP/1.1",
            f"HOST: {self.host}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
        ]

        request = "\r\n".join(headers) + "\r\n\r\n" + body

        return request

    def _get_header(self, socket_):
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
            length = existing[0][existing[0].find("Content-Length: ") :]
            length = length[: max(len(length), length.find("\n"))]
            size = int(length[15::])
            final = self._get_all(existing[1].encode(), size, socket_)
            ret = json.loads(final.decode())
        socket_.close()
        return ret

    def get_models(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        headers = [
            "GET /api/tags HTTP/1.1",
            f"HOST: {self.host}",
        ]

        request = "\r\n".join(headers) + "\r\n\r\n"
        s.send(request.encode())

        raw = self._get_header(s)

        if "Content-Type: application/json;" in raw[0]:
            return self._handle_json(s, raw)
        return {}

    def _handle_msg(self, chat: Buffer, body) -> (bool, bool):
        call = False
        msg = body["message"]["content"]
        chat.add_ai(msg)
        if body["done"]:
            chat.tokens = body["prompt_eval_count"] + body["eval_count"]
        if "tool_calls" in body["message"]:
            call = True
            chat.add_ai_tool(body["message"]["tool_calls"])
            for tool_call in body["message"]["tool_calls"]:
                chat.chat_data["messages"].append(
                    {
                        "role": "tool",
                        "tool_name": tool_call["function"]["name"],
                        "content": "42 degree Celsius",
                    }
                )
        return (body["done"], call)

    def get_reply(self, chat: Buffer, wrapped=False) -> None:
        call = False
        chat.cancel = False
        chat.in_request = True

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(self.make_post(chat.chat_data).encode())

        chat.add_msg("assistant")
        raw = self._get_header(s)
        if "HTTP/1.1 200 OK" not in raw[0]:
            chat.in_request = False
            return "\n".join(raw)
        if "Content-Type: application/x-ndjson" in raw[0]:
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

        if "Content-Type: application/json;" in raw[1]:
            body = self._handle_json(s, raw)
            _, call = self._handle_msg(chat, body)
        s.close()
        chat.in_request = False
        if call:
            self.get_reply(chat, wrapped=True)
        if not wrapped:
            chat.add_msg("user")
        return None


class State(Enum):
    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4
    DIALOG = 5


class Supervisor:
    def __init__(self) -> None:
        self.buffers = [Buffer()]
        self.current_buffer_id = 0
        self.requester = Requester()

        self.run = True

        self.command = ""
        self.input_done = False
        self.input_cond = threading.Condition()
        self.buffer_last = -1
        self.buffer_lines = -1
        self.buffer_changed = False

        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        self.state = State.NORMAL
        signal.signal(signal.SIGWINCH, self.handle_sigwinch)

        self.models = self.requester.get_models()
        self.models_short = ""
        self.models_list = [x["name"] for x in self.models["models"]]
        for model in self.models["models"]:
            self.models_short += (
                f"{model["name"]:<20}"
                f"\t {model["size"] // 1073741824:} GB"
                f"\t{model["details"]["parameter_size"]}\n"
            )
        self.dialog = ""
        self.dialog_reply = ""

        self.commands = {
            "m": self.list_models,
            "h": self.help,
            "n": self.normal_view,
            "d": self.show_data,
            "S": self.show_system,
            "s": self.split,
            "r": self.redo,
            "R": self.start_over,
            "rr": self.undo,
            "bc": self.create_buffer,
            "bn": self.buffer_next,
            "bp": self.buffer_previous,
            "bd": self.buffer_delete,
            "sm": self.switch_model,
            "ai": self.change_ai,
            "us": self.change_user,
            "sys": self.change_system,
            "i": self.import_chat,
            "e": self.export_chat,
        }

    @property
    def current_buffer(self):
        return self.buffers[self.current_buffer_id]

    @property
    def current_data(self):
        return self.current_buffer.chat_data

    def create_dialog(self, message):
        self.dialog = f"{message}\n\n[ENTER] Dismiss"
        self.state = State.DIALOG
        self.current_buffer.notify()

    def handle_sigint(self):
        self.current_buffer.cancel = True

    def handle_sigwinch(self, signum, frame):
        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        clean_screan()
        self.update_command()
        self.current_buffer.notify()

    def get_reply(self, buffer):
        error = self.requester.get_reply(buffer)
        if error:
            self.create_dialog(error)

    def change_buffer(self, new_id):
        old_buffer = self.current_buffer
        self.current_buffer_id = new_id
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            with old_buffer.cond:
                old_buffer.cond.notify()
            self.current_buffer.cond.notify()

    ############# internal commands
    def list_models(self, *args):  # pylint: disable=unused-argument
        """list all models"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.MODELS
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def help(self, *args):  # pylint: disable=unused-argument
        """show help"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.HELP
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def normal_view(self, *args):  # pylint: disable=unused-argument
        """show chat"""
        with self.current_buffer.cond:
            self.state = State.NORMAL
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def show_data(self, *args):  # pylint: disable=unused-argument
        """show data"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.DATA
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def show_system(self, *args):  # pylint: disable=unused-argument
        """show system prompt"""
        with self.current_buffer.cond:
            self.state = State.SYSTEM
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def split(self, *args):  # pylint: disable=unused-argument
        """creates new buffer from current"""
        new = Buffer()
        new.chat_data = json.loads(json.dumps(self.current_data))
        self.buffers.append(new)
        self.change_buffer(len(self.buffers) - 1)

    def redo(self, *args):  # pylint: disable=unused-argument
        """regenerate last message"""
        self.current_buffer.del_msg(2)
        self.get_reply(self.current_buffer)

    def start_over(self, *args):  # pylint: disable=unused-argument
        """reset current buffer"""
        self.current_buffer.empty()

    def undo(self, *args):  # pylint: disable=unused-argument
        """revert last user message"""
        self.current_buffer.del_msg(3)
        self.current_buffer.add_msg("user")

    def create_buffer(self, *args):  # pylint: disable=unused-argument
        """create new blank buffer"""
        self.buffers.append(Buffer())
        self.change_buffer(len(self.buffers) - 1)

    def buffer_next(self, *args):  # pylint: disable=unused-argument
        """switch to next buffer"""
        if self.current_buffer_id < len(self.buffers) - 1:
            self.change_buffer(self.current_buffer_id + 1)

    def buffer_previous(self, *args):  # pylint: disable=unused-argument
        """switch to previous buffer"""
        if self.current_buffer_id > 0:
            self.change_buffer(self.current_buffer_id - 1)

    def buffer_delete(self, *args):  # pylint: disable=unused-argument
        """delete current buffer"""
        cond = self.current_buffer.cond
        self.buffers.pop(self.current_buffer_id)
        if len(self.buffers) == 0:
            self.buffers.append(Buffer())
        if self.current_buffer_id == len(self.buffers):
            self.current_buffer_id -= 1
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()
            with cond:
                cond.notify()

    def switch_model(self, *args):
        """show models"""
        if len(args) < 1:
            return
        self.current_data["model"] = args[0]
        self.current_buffer.notify()

    def change_ai(self, *args):  # pylint: disable=unused-argument
        """change current message to assistent"""
        self.current_buffer.change_msg("assistant")

    def change_user(self, *args):  # pylint: disable=unused-argument
        """change current message to user"""
        self.current_buffer.change_msg("user")

    def change_system(self, *args):  # pylint: disable=unused-argument
        """change current message to system"""
        self.current_buffer.change_msg("system")

    def import_chat(self, *args):
        """import chat from file -- argument: file name"""
        if len(args) < 1:
            return
        try:
            with open(args[0], "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    self.current_buffer.set_data(data)
                except json.JSONDecodeError:
                    pass
        except PermissionError:
            self.create_dialog(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.create_dialog(f"File {args[0]} not found")

    def export_chat(self, *args):
        """export chat from file -- argument: file name"""
        if len(args) < 1:
            return
        try:
            with open(args[0], "w", encoding="utf-8") as file:
                json.dump(self.current_data, file, indent=4)
        except PermissionError:
            self.create_dialog(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.create_dialog(f"File {args[0]} not found")

    def process_command(self) -> bool:
        command = self.command.split(" ")
        if command[0][0] not in (":", "/"):
            return True
        command[0] = command[0][1:]
        self.command = ""
        self.update_command()
        if command[0] == "q":
            return False
        if command[0] in self.commands:
            self.commands[command[0]](*command[1:])
        return True

    def handle_new_message(self):
        content = self.current_data["messages"][-1]["content"].replace("\\n", "\n")
        self.current_data["messages"][-1]["content"] = content
        if self.current_data["messages"][-1]["role"] != "user":
            self.current_buffer.change_msg("user")
        else:
            self.get_reply(self.current_buffer)

    def watcher(self):
        while self.run:
            with self.input_cond:
                while not self.input_done and self.run:
                    self.input_cond.wait()
                self.input_done = False
            if not self.run:
                return
            if len(self.command) > 0:
                self.process_command()
            else:
                self.handle_new_message()

    def handle_escape_sequence(self):
        buffer = ""
        ch = 27
        while ch not in ("A", "B", "C", "D", "~"):
            ch = sys.stdin.read(1)
            if ord(ch) == 27:
                return
            buffer += ch

        if buffer == "[A" or buffer.startswith("OA"):
            if self.command != "":
                pass
            else:
                if self.buffer_last < 0:
                    self.buffer_last = self.buffer_lines - 1
                else:
                    if self.buffer_last:
                        self.buffer_last -= 1
        if buffer == "[B" or buffer.startswith("OB"):
            if self.command != "":
                pass
            else:
                if 0 <= self.buffer_last < self.buffer_lines:
                    self.buffer_last += 1
                else:
                    self.buffer_last = -1
        if buffer == "[D":
            self.buffer_last = -1
        if buffer == "[C":
            self.buffer_last = 1

        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def update_command(self):
        if self.command == "":
            fit_text_to_box(
                f"    model: {self.current_data['model']}"
                f" -- tokens: {self.current_buffer.tokens}"
                f"   buffer: {self.current_buffer_id + 1} of {len(self.buffers)}",
                self.columns,
                1,
                self.lines,
                1,
                "\033[32m",
                "\033[0m",
                hide_cursor=True,
            )
        else:
            fit_text_to_box(
                self.command, self.columns, 1, self.lines, 1, "\033[32m", "\033[41m"
            )

    def get_input(self):
        input_length = 0
        is_command = False
        while self.run:
            ch = sys.stdin.read(1)
            with self.input_cond:
                if ord(ch) == 21:
                    input_length = 0
                    if self.command:
                        self.command = ""
                        is_command = False
                        with self.current_buffer.cond:
                            self.current_buffer.updated = True
                            self.current_buffer.cond.notify()
                    else:
                        self.current_buffer.clear_msg()
                if ord(ch) == 27:
                    self.handle_escape_sequence()
                    continue
                if ord(ch) == 3:
                    self.handle_sigint()
                if not ch or ch == "\r" or ch == "\n":
                    if is_command:
                        if self.command in ("/q", ":q"):
                            self.run = False
                            self.handle_sigint()
                            with self.current_buffer.cond:
                                self.current_buffer.cond.notify()
                        is_command = False
                        self.update_command()
                        self.input_done = True
                        self.input_cond.notify()
                    else:
                        if self.state == State.NORMAL:
                            self.input_done = True
                            self.input_cond.notify()
                        else:
                            self.state = State.NORMAL
                            self.current_buffer.notify()
                    input_length = 0
                if ord(ch) == 127:
                    if input_length <= 0:
                        continue
                    input_length -= 1
                    if is_command:
                        self.command = self.command[0:-1]
                        self.update_command()
                        if input_length == 0:
                            is_command = False
                            with self.current_buffer.cond:
                                self.current_buffer.updated = True
                                self.current_buffer.cond.notify()
                    else:
                        self.buffer_last = -1
                        self.state = State.NORMAL
                        self.current_buffer.del_open()
                    continue
                if 32 <= ord(ch) <= 126:
                    if input_length == 0 and ch in ("/", ":"):
                        is_command = True
                        self.command = ch
                        self.update_command()
                    elif not self.current_buffer.in_request and not is_command:
                        if self.state == State.DIALOG:
                            self.state = State.NORMAL
                            self.dialog_reply = ch
                            self.current_buffer.notify()
                        else:
                            self.buffer_last = -1
                            self.state = State.NORMAL
                            self.current_buffer.add_open(ch)
                    else:
                        self.command += ch
                        self.update_command()
                    input_length += 1
                if ord(ch) == 9:
                    if is_command:
                        command_tokens = self.command.split(" ")
                        if len(command_tokens) < 2:
                            continue
                        match command_tokens[0]:
                            case ":sm" | "/sm":
                                possible_models = [
                                    x
                                    for x in self.models_list
                                    if x.startswith(command_tokens[1])
                                ]
                                if len(possible_models) == 1:
                                    self.command += possible_models[0][
                                        len(command_tokens) - 1 :
                                    ]
                                    self.update_command()

    def update_buffer(self):
        while self.run:
            with self.current_buffer.cond:
                while (
                    not self.current_buffer.updated
                    and self.run
                    and not self.buffer_changed
                ):
                    self.current_buffer.cond.wait()
                self.update_command()
                if self.buffer_changed:
                    self.buffer_changed = False
                    continue
                if not self.run:
                    return
                if len(self.current_data["messages"]) == 0:
                    self.current_buffer.add_msg("system")
                match self.state:
                    case State.NORMAL:
                        box = self.current_buffer.get_text_box(
                            self.columns - 4, self.buffer_last
                        )
                        hide_cursor = False
                    case State.SYSTEM:
                        box = self.current_buffer.get_text_box_system(
                            self.columns - 4, self.buffer_last
                        )
                        hide_cursor = True
                    case State.DATA:
                        box = self.current_buffer.get_text_box_data(
                            self.columns - 4, self.buffer_last
                        )
                        hide_cursor = True
                    case State.HELP:
                        help_string = ""
                        for name, function in self.commands.items():
                            help_string += f"{name}:\t{function.__doc__}\n"
                        box = make_text_box(
                            help_string, self.columns - 4, self.buffer_last
                        )
                    case State.MODELS:
                        box = make_text_box(
                            # json.dumps(self.models, indent=4),
                            self.models_short,
                            self.columns - 4,
                            self.buffer_last,
                        )
                    case State.DIALOG:
                        box = make_text_box(
                            self.dialog,
                            self.columns - 4,
                            self.buffer_last,
                        )
                        hide_cursor = True

                (self.buffer_lines, self.buffer_last) = fit_to_box(
                    box, self.columns - 4, self.lines - 5, 2, 3, hide_cursor=hide_cursor
                )
                self.current_buffer.updated = False

    def run_chat(self) -> None:
        print("\033[?1049h", end="", flush=True)
        clean_screan()
        self.run = True
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)

        threads = [
            threading.Thread(target=self.get_input),
            threading.Thread(target=self.update_buffer),
            threading.Thread(target=self.watcher),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        clean_screan()
        print("\033[?25h", end="", flush=True)
        print("\033[?1049l", end="", flush=True)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    if "-t" in sys.argv:
        Buffer.TOOLS = True
    Supervisor().run_chat()
