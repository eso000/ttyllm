import socket
import json
import os
import re
import sys
import tty
import termios


class ColorCodeExeption(Exception):
    pass


def clean_screan() -> None:
    print("\033[H\033[J\033[3J", end="", flush=True)


class Buffer:
    def __init__(self):
        self.ai_name = "AI"
        self.user_name = "You"
        self.ai_color = "\033[35m"
        self.user_color = "\033[36m"
        self.normal_color = "\033[37m"

        self.chat_data = {
            "model": "mistral-nemo",
            "stream": True,
            "think": False,
            "messages": [],
        }

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

    def print_ai(self, string: str, end="\n\r"):
        print(
            f"{self.ai_color}{string}",
            end=end,
            flush=True,
        )

    def print_user(self, string: str, end="\n\r"):
        print(
            f"{self.user_color}{string}",
            end=end,
            flush=True,
        )

    def reprint(self) -> None:
        clean_screan()
        for msg_ in self.chat_data["messages"]:
            if msg_["role"] == "user":
                self.print_normal(f"{self.user_name}:")
                self.print_user(msg_["content"])
            if msg_["role"] == "assistant":
                self.print_normal(f"{self.ai_name}:")
                self.print_ai(msg_["content"])

    def print_data(self) -> None:
        for key, value in self.chat_data.items():
            if key != "messages":
                self.print_normal(f"{key}: {value}")

    def print_messages(self) -> None:
        clean_screan()
        for _id_, _msg_ in enumerate(self.chat_data["messages"]):
            self.print_normal(_id_)
            self.print_normal(_msg_)


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
        chat.print_normal(f"{chat.ai_name}:")
        chat.print_ai("", end="")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(self.make_post(chat.chat_data).encode())

        chat.chat_data["messages"].append({"role": "assistant", "content": ""})
        reply = s.recv(self.max_recv)
        raw = reply.decode().split("\r\n\r\n")
        try:
            if "HTTP/1.1 200 OK" not in raw[0]:
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
                chat.chat_data["messages"][-1]["content"] += msg
                chat.print_ai(msg, end="")
                while not done:
                    body = s.recv(self.max_recv).decode().split("\n")
                    size = int(body[0], 16)
                    final = body[1].encode()
                    final = self._get_all(final, size + 2, s)
                    body[1] = final.decode()
                    done = json.loads(body[1])["done"]
                    msg = json.loads(body[1])["message"]["content"]
                    msg = msg.replace("\n", "\r\n")
                    chat.chat_data["messages"][-1]["content"] += msg
                    chat.print_ai(msg, end="")
            if "Content-Type: application/json;" in raw[0]:
                length = raw[0][raw[0].find("Content-Length: ") :]
                length = length[: max(len(length), length.find("\n"))]
                size = int(length[15::])
                final = raw[1].encode()
                final += s.recv(min(self.max_recv, size - len(final)))
                msg = json.loads(final.decode())["message"]["content"]
                msg = msg.replace("\n", "\n\r")
                chat.chat_data["messages"][-1]["content"] += msg
                chat.print_ai(msg, end="")
        except KeyboardInterrupt:
            pass
        print("", end="\n\r")
        s.close()


class Supervisor:
    def __init__(self) -> None:
        self.buffers = [Buffer()]
        self.current_buffer_id = 0
        self.requester = Requester()

    @property
    def current_buffer(self):
        return self.buffers[self.current_buffer_id]

    @property
    def current_data(self):
        return self.current_buffer.chat_data

    def add_massage(self, msg, type_) -> None:
        self.current_data["messages"].append({"role": type_, "content": msg})

    def process_command(self, command: str) -> bool:
        command = command.split(" ")
        if command[0] == "/q":
            return False
        match command[0]:
            case "/d":
                self.current_buffer.print_data()
            case "/m":
                self.current_buffer.print_messages()
            case "/p":
                self.current_buffer.reprint()
            case "/r":
                if self.current_data["messages"][-1]["role"] == "assistant":
                    self.current_data["messages"].pop()
                    self.current_buffer.reprint()
                    self.requester.get_reply(self.current_buffer)
            case "/R":
                self.current_data["messages"] = []
                self.current_buffer.reprint()
            case "/rr":
                if self.current_data["messages"][-1]["role"] == "assistant":
                    self.current_data["messages"].pop()
                    self.current_data["messages"].pop()
                    self.current_buffer.reprint()
            case "/bc":
                self.buffers.append(Buffer())
                self.current_buffer_id = len(self.buffers) - 1
                self.current_buffer.reprint()
            case "/bn":
                if self.current_buffer_id < len(self.buffers) - 1:
                    self.current_buffer_id += 1
                self.current_buffer.reprint()
            case "/bp":
                if self.current_buffer_id > 0:
                    self.current_buffer_id -= 1
                self.current_buffer.reprint()
            case "/bd":
                self.buffers.pop(self.current_buffer_id)
                if len(self.buffers) == 0:
                    self.buffers.append(Buffer())
                    self.current_buffer.reprint()
                if self.current_buffer_id == len(self.buffers):
                    self.current_buffer_id -= 1
                    self.buffers[self.current_buffer_id].reprint()
            case "/sm":
                self.current_data["model"] = command[1]
            case _:
                self.current_buffer.print_normal(f"Unknown command: {command[0]}")
        return True

    def get_input(self):
        new_input = []
        while True:
            ch = sys.stdin.read(1)
            if not ch or ch == "\r" or ch == "\n":
                print("", end="\n\r")
                return "".join(new_input)
            if ord(ch) == 127:
                print("\b", end="", flush=True)
                new_input.pop(-1)
                continue
            new_input.append(ch)
            print(ch, end="", flush=True)
        print("", end="\n\r")
        return "".join(new_input)

    def handle_empy_buffer(self):
        self.current_buffer.print_normal("system message")
        input_ = self.get_input().replace("\\n", "\n\r")
        if input_[0:2] == "/o":
            if os.path.isfile(input_[2::]):
                with open(input_[2::], encoding="utf-8") as file:
                    string = file.read()
                    self.add_massage(string, "system")
            else:
                self.current_buffer.print_normal("no such file")
        else:
            self.add_massage(input_, "system")
            self.current_buffer.reprint()

    def run_chat(self) -> None:
        clean_screan()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)  # raw mode (no echo, no line buffering)
        run = True
        while run:
            try:
                if len(self.current_data["messages"]) == 0:
                    self.handle_empy_buffer()
                    continue

                self.current_buffer.print_normal(f"{self.current_buffer.user_name}:")
                self.current_buffer.print_user("", end="")
                input_ = self.get_input().replace("\\n", "\n\r")

                if len(input_) == 0:
                    self.current_buffer.reprint()
                    continue
                if input_[0] == "/":
                    run = self.process_command(input_)
                else:
                    self.add_massage(input_, "user")
                    self.requester.get_reply(self.current_buffer)
            except (KeyboardInterrupt, UnicodeDecodeError) as e:
                if isinstance(e, UnicodeDecodeError):
                    self.current_buffer.print_normal("input too large")
                continue
        clean_screan()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    Supervisor().run_chat()
