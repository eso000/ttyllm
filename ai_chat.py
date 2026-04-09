import socket
import json
import re
import sys
import termios
import threading
import dataclasses
import signal
import shutil


class ColorCodeExeption(Exception):
    pass


@dataclasses.dataclass
class TextBox:
    lines: list = dataclasses.field(default_factory=list)
    last = -1


def clean_screan() -> None:
    print("\033[H\033[J\033[3J", end="", flush=True)


def make_text_box(text, width, last=-1, color="") -> TextBox:
    res = TextBox()
    res.last = last
    if text == "":
        res.lines = [""]
        return res
    res_lines = text.split("\n")
    for part in res_lines:
        res.lines += [
            f"{color}{part[i : i + width]}" for i in range(0, len(part), width)
        ]
    return res


def fit_to_box(box, width, height, start_x=1, start_y=1, background=""):
    # clean box
    print(f"\033[{start_x};{start_y}H", end="")
    for x in range(height):
        print(f"{background}\033[{start_x+x};{start_y}H", end="")
        print(width * " ", end="")

    lines, last = (box.lines, box.last)

    start = 0
    if last < 0:
        last = len(lines)
    if len(lines) <= height:
        last = len(lines)
    last = max(last, height)
    if last >= height:
        start = max(0, last - height)
    print(f"\033[{start_x};{start_y}H", end="", flush=True)
    for x, line in enumerate(lines[start:last]):
        print(f"\033[{start_x+x};{start_y}H", end="")
        print(
            line,
            end="",
        )
    print("\033[0m", end="", flush=True)
    return len(lines)


def fit_text_to_box(text, width, height, start_x=1, start_y=1, color="", background=""):
    fit_to_box(
        make_text_box(text, width, color=color),
        width,
        height,
        start_x,
        start_y,
        background,
    )


class Buffer:
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
            "model": "gemma2:2b",  # "mistral-nemo",
            "stream": True,
            "think": False,
            "messages": [],
        }
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
            for _ in range(number):
                self.chat_data["messages"].pop()
            self.updated = True
            self.cond.notify()

    def add_msg(self, role):
        with self.cond:
            self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def add_user(self, ch):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1].get("role") in ("user", "system"):
                self.chat_data["messages"][-1]["content"] += ch
                self.show_data = False
                self.show_system = False
                self.updated = True
                self.cond.notify()

    def del_user(self):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1].get("role") in ("user", "system"):
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

    def print_normal(self, string: str, end="\n\r"):
        print(
            f"{self.normal_color}{string}",
            end=end,
            flush=True,
        )

    def get_text_box(self, width, last=-1) -> str:
        if self.show_data:
            return self._get_text_box_data(width, last)
        if self.show_system:
            return self._get_text_box_system(width, last)
        return self._get_text_box_normal(width, last)

    def _get_text_box_system(self, width, last=-1) -> str:
        result = TextBox()
        res = []
        res += [f"{self.normal_color}system:"]
        msg_ = self.chat_data["messages"][0]
        if msg_["content"] == "":
            res += [""]
        else:
            res_lines = msg_["content"].split("\n")
            for line in res_lines:
                res += [
                    f"{self.normal_color}{line[i : i + width]}"
                    for i in range(0, len(line), width)
                ]
        result.last = last
        result.lines = res
        return result

    def _get_text_box_data(self, width, last=-1) -> str:
        return make_text_box(json.dumps(self.chat_data, indent=4), width, last=last)

    def _get_text_box_normal(self, width, last=-1) -> str:
        result = TextBox()
        res = []
        if len(self.chat_data["messages"]) == 1:
            res += [f"{self.normal_color}system:"]
            msg_ = self.chat_data["messages"][0]
            if msg_["content"] == "":
                res += [""]
            else:
                res_lines = msg_["content"].split("\n")
                for line in res_lines:
                    res += [
                        f"{self.normal_color}{line[i : i + width]}"
                        for i in range(0, len(line), width)
                    ]
            result.last = last
            result.lines = res
            return result

        for msg_ in self.chat_data["messages"]:
            if msg_["role"] == "system":
                continue
            color = self.user_color if msg_["role"] == "user" else self.ai_color
            name = self.user_name if msg_["role"] == "user" else self.ai_name
            res += [f"{self.normal_color}{name}:"]
            if msg_["content"] == "":
                res += [""]
            else:
                res_lines = msg_["content"].split("\n")
                for line in res_lines:
                    res += [
                        f"{color}{line[i : i + width]}"
                        for i in range(0, len(line), width)
                    ]
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

    def print_data(self) -> None:
        for key, value in self.chat_data.items():
            if key != "messages":
                self.print_normal(f"{key}: {value}")

    def print_messages(self) -> None:
        clean_screan()
        for _id_, _msg_ in enumerate(self.chat_data["messages"]):
            self.print_normal(_id_)
            self.print_normal(_msg_)

    def print_token(self):
        self.print_normal(f"tokes: {self.tokens if self.tokens>-1 else '-'}")


class Requester:
    def __init__(self, host="127.0.0.1", port=11434):
        self.max_recv = 124
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

    def _get_all(self, final, size, socket_):
        while len(final) < size:
            final += socket_.recv(min(self.max_recv, size - len(final)))
        return final

    def get_reply(self, chat: Buffer) -> None:
        chat.cancel = False
        chat.in_request = True

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(self.make_post(chat.chat_data).encode())

        chat.add_msg("assistant")
        reply = s.recv(self.max_recv)
        raw = reply.decode().split("\r\n\r\n")
        if "HTTP/1.1 200 OK" not in raw[0]:
            for raw_ in raw:
                print(raw_, end="\r\n")
            return
        if "Content-Type: application/x-ndjson" in raw[0]:
            body = raw[1].split("\n")
            size = int(body[0], 16)
            final = body[1].encode()
            final = self._get_all(final, size + 2, s)
            body[1] = final.decode()
            done = json.loads(body[1])["done"]
            msg = json.loads(body[1])["message"]["content"]
            msg = msg.replace("\n", "\r\n")
            chat.add_ai(msg)
            while not done and not chat.cancel:
                body = s.recv(self.max_recv).decode().split("\n")
                size = int(body[0], 16)
                final = body[1].encode()
                final = self._get_all(final, size + 2, s)
                body[1] = final.decode()
                done = json.loads(body[1])["done"]
                msg = json.loads(body[1])["message"]["content"]
                msg = msg.replace("\n", "\r\n")
                chat.add_ai(msg)
                if done:
                    chat.tokens = (
                        json.loads(final.decode())["prompt_eval_count"]
                        + json.loads(final.decode())["eval_count"]
                    )
        if "Content-Type: application/json;" in raw[0]:
            length = raw[0][raw[0].find("Content-Length: ") :]
            length = length[: max(len(length), length.find("\n"))]
            size = int(length[15::])
            final = raw[1].encode()
            final += s.recv(min(self.max_recv, size - len(final)))
            msg = json.loads(final.decode())["message"]["content"]
            msg = msg.replace("\n", "\n\r")
            chat.add_ai(msg)
            chat.tokens = (
                json.loads(final.decode())["prompt_eval_count"]
                + json.loads(final.decode())["eval_count"]
            )
        s.close()
        chat.add_msg("user")
        chat.in_request = False


class Supervisor:
    def __init__(self) -> None:
        self.buffers = [Buffer()]
        self.current_buffer_id = 0
        self.requester = Requester()

        self.run = True

        self.command = ""
        self.input_done = False
        self.input_cond = threading.Condition()
        self.buffer_start = -1
        self.buffer_lines = -1
        self.buffer_changed = False

        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        signal.signal(signal.SIGWINCH, self.handle_sigwinch)

    @property
    def current_buffer(self):
        return self.buffers[self.current_buffer_id]

    @property
    def current_data(self):
        return self.current_buffer.chat_data

    def handle_sigint(self):
        self.current_buffer.cancel = True

    def handle_sigwinch(self, signum, frame):
        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        clean_screan()
        self.update_command()
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def change_buffer(self, new_id):
        old_buffer = self.current_buffer
        self.current_buffer_id = new_id
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            with old_buffer.cond:
                old_buffer.cond.notify()
            self.current_buffer.cond.notify()

    def process_command(self) -> bool:
        command = self.command.split(" ")
        self.command = ""
        self.update_command()
        if command[0] == "/q":
            return False
        match command[0]:
            case "/n":
                with self.current_buffer.cond:
                    self.current_buffer.show_data = False
                    self.current_buffer.show_system = False
                    self.current_buffer.updated = True
                    self.current_buffer.cond.notify()
            case "/d":
                with self.current_buffer.cond:
                    self.current_buffer.show_data = True
                    self.current_buffer.show_system = False
                    self.current_buffer.updated = True
                    self.current_buffer.cond.notify()
            case "/S":
                with self.current_buffer.cond:
                    self.current_buffer.show_system = True
                    self.current_buffer.show_data = False
                    self.current_buffer.updated = True
                    self.current_buffer.cond.notify()
            case "/s":
                new = Buffer()
                new.chat_data = json.loads(json.dumps(self.current_data))
                self.buffers.append(new)
                self.change_buffer(len(self.buffers) - 1)
            case "/r":
                self.current_buffer.del_msg(2)
                self.requester.get_reply(self.current_buffer)
            case "/R":
                self.current_buffer.empty()
            case "/rr":
                self.current_buffer.del_msg(3)
                self.current_buffer.add_msg("user")
            case "/bc":
                self.buffers.append(Buffer())
                self.change_buffer(len(self.buffers) - 1)
            case "/bn":
                if self.current_buffer_id < len(self.buffers) - 1:
                    self.change_buffer(self.current_buffer_id + 1)
            case "/bp":
                if self.current_buffer_id > 0:
                    self.change_buffer(self.current_buffer_id - 1)
            case "/bd":
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

            case "/sm":
                if len(command) >= 2:
                    self.current_data["model"] = command[1]
                else:
                    self.current_buffer.print_normal("no model given")
            case _:
                self.current_buffer.print_normal(f"Unknown command: {command[0]}")
        return True

    def handle_new_message(self):
        content = self.current_data["messages"][-1]["content"].replace("\\n", "\n")
        self.current_data["messages"][-1]["content"] = content
        if (
            len(self.current_data["messages"]) == 1
            and self.current_data["messages"][0]["role"] == "system"
        ):
            self.current_buffer.add_msg("user")
        else:
            self.requester.get_reply(self.current_buffer)

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
        while ch not in ["A", "B", "C", "D", "~"]:
            ch = sys.stdin.read(1)
            buffer += ch
        if buffer == "[A" or buffer.startswith("OA"):
            if self.buffer_start < 0:
                self.buffer_start = self.buffer_lines - 1
            else:
                if self.buffer_start:
                    self.buffer_start -= 1
        if buffer == "[B" or buffer.startswith("OB"):
            if 0 <= self.buffer_start < self.buffer_lines:
                self.buffer_start += 1
            else:
                self.buffer_start = -1
        if buffer == "[D":
            self.buffer_start = -1
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def update_command(self):
        if self.command == "":
            fit_text_to_box(
                self.columns * " ", self.columns, 1, self.lines, 1, "\033[0m"
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
                    if self.command:
                        self.command = ""
                        self.update_command()
                        is_command = False
                    else:
                        self.current_buffer.clear_msg()
                if ord(ch) == 27:
                    self.handle_escape_sequence()
                    continue
                if ord(ch) == 3:
                    self.handle_sigint()
                if not ch or ch == "\r" or ch == "\n":
                    if is_command:
                        if self.command == "/q":
                            self.run = False
                            self.handle_sigint()
                            with self.current_buffer.cond:
                                self.current_buffer.cond.notify()
                        is_command = False
                        self.update_command()
                        self.input_done = True
                        self.input_cond.notify()
                    else:
                        self.input_done = True
                        self.input_cond.notify()
                    input_length = 0
                if ord(ch) == 127:
                    if input_length <= 0:
                        continue
                    input_length -= 1
                    if is_command:
                        self.command = self.command[0:-1]
                        if input_length == 0:
                            is_command = False
                        self.update_command()
                    else:
                        self.current_buffer.del_user()
                    continue
                if 32 <= ord(ch) <= 126:
                    if input_length == 0 and ch == "/":
                        is_command = True
                        self.command = "/"
                        self.update_command()
                    elif not self.current_buffer.in_request and not is_command:
                        self.current_buffer.add_user(ch)
                    else:
                        self.command += ch
                        self.update_command()
                    input_length += 1

    def update_buffer(self):
        while self.run:
            with self.current_buffer.cond:
                while (
                    not self.current_buffer.updated
                    and self.run
                    and not self.buffer_changed
                ):
                    self.current_buffer.cond.wait()
                if self.buffer_changed:
                    self.buffer_changed = False
                    continue
                if not self.run:
                    return
                if len(self.current_data["messages"]) == 0:
                    self.current_buffer.add_msg("system")
                self.buffer_lines = fit_to_box(
                    self.current_buffer.get_text_box(
                        self.columns - 4, self.buffer_start
                    ),
                    self.columns - 4,
                    self.lines - 5,
                    2,
                    3,
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
        print("\033[?1049l", end="", flush=True)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    Supervisor().run_chat()
