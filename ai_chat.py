import socket
import json
import os
import re


class ColorCodeExeption(Exception):
    pass


class Chat:
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

    def get_normal_color(self) -> str:
        return self.normal_color

    def get_ai_color(self) -> str:
        return self.ai_color

    def get_user_color(self) -> str:
        return self.user_color

    def get_user_name(self) -> str:
        return self.user_name

    def get_ai_name(self) -> str:
        return self.ai_name

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

    def set_user_name(self, value) -> None:
        self.user_name = value

    def set_ai_name(self, value: str) -> None:
        self.ai_name = value


class Print:
    DO_CLEAN = "\033[H\033[J\033[3J"

    @staticmethod
    def clean_screan() -> None:
        print(Print.DO_CLEAN, end="", flush=True)

    @staticmethod
    def print_normal(chat: Chat, string: str, end="\n"):
        print(
            f"{chat.get_normal_color()}{string}",
            end=end,
            flush=True,
        )

    @staticmethod
    def print_ai(chat: Chat, string: str, end="\n"):
        print(
            f"{chat.get_ai_color()}{string}",
            end=end,
            flush=True,
        )

    @staticmethod
    def print_user(chat: Chat, string: str, end="\n"):
        print(
            f"{chat.get_user_color()}{string}",
            end=end,
            flush=True,
        )

    @staticmethod
    def reprint(chat) -> None:
        Print.clean_screan()
        for msg_ in chat.chat_data["messages"]:
            if msg_["role"] == "user":
                Print.print_normal(chat, f"{chat.get_user_name()}:")
                Print.print_user(chat, msg_["content"])
            if msg_["role"] == "assistant":
                Print.print_normal(chat, f"{chat.get_ai_name()}:")
                Print.print_ai(chat, msg_["content"])

    @staticmethod
    def print_data(chat: Chat) -> None:
        for key in chat.chat_data:
            if key != "messages":
                Print.print_normal(chat, f"{key}: {chat.chat_data[key]}")

    @staticmethod
    def print_messages(chat: Chat) -> None:
        Print.clean_screan()
        for _id_, _msg_ in enumerate(chat.chat_data["messages"]):
            Print.print_normal(chat, _id_)
            Print.print_normal(chat, _msg_)


class ChatList:
    def __init__(self):
        self.chats = [Chat()]
        self.current_chat_index = 0

    def get_current_chat(self):
        return self.chats[self.current_chat_index]

    def get_current_data(self):
        return self.chats[self.current_chat_index].chat_data

    def creat_new_chat(self) -> None:
        self.chats.append(Chat())
        self.current_chat_index = len(self.chats) - 1

    def previous_chat(self) -> None:
        if self.current_chat_index > 0:
            self.current_chat_index -= 1

    def next_chat(self) -> None:
        if self.current_chat_index < len(self.chats) - 1:
            self.current_chat_index += 1

    def delete_current_chat(self) -> None:
        self.chats.pop(self.current_chat_index)
        if len(self.chats) == 0:
            self.creat_new_chat()
        if self.current_chat_index == len(self.chats):
            self.current_chat_index -= 1


class Supervisor:
    def __init__(self) -> None:
        self.chat_list = ChatList()
        self.commands = {
            "/d": self.print_data,
            "/m": self.print_messages,
            "/p": self.reprint,
            "/r": self.redo,
            "/R": self.restart,
            "/rr": self.rewrite,
            "/bc": self.buffer_create,
            "/bn": self.buffer_next,
            "/bp": self.buffer_previous,
            "/bd": self.buffer_delete,
            "/sm": self.set_model,
        }
        self.host = "127.0.0.1"
        self.port = 11434

    def reprint(self) -> None:
        Print.reprint(self.chat_list.get_current_chat())

    def print_data(self) -> None:
        Print.print_data(self.chat_list.get_current_chat())

    def print_messages(self) -> None:
        Print.print_messages(self.chat_list.get_current_chat())

    def redo(self) -> None:
        if self.chat_list.get_current_data()["messages"][-1]["role"] == "assistant":
            self.chat_list.get_current_data()["messages"].pop()
            self.reprint()
            self.get_reply(self.chat_list.get_current_chat())

    def rewrite(self) -> None:
        if self.chat_list.get_current_data()["messages"][-1]["role"] == "assistant":
            self.chat_list.get_current_data()["messages"].pop()
            self.chat_list.get_current_data()["messages"].pop()
            self.reprint()
            self.get_reply(self.chat_list.get_current_chat())

    def restart(self) -> None:
        self.chat_list.get_current_data()["messages"] = []
        self.reprint()

    def buffer_create(self) -> None:
        self.chat_list.creat_new_chat()
        Print.reprint(self.chat_list.get_current_chat())

    def buffer_previous(self) -> None:
        self.chat_list.previous_chat()
        Print.reprint(self.chat_list.get_current_chat())

    def buffer_next(self) -> None:
        self.chat_list.next_chat()
        Print.reprint(self.chat_list.get_current_chat())

    def buffer_delete(self) -> None:
        self.chat_list.delete_current_chat()
        Print.reprint(self.chat_list.get_current_chat())

    def set_model(self, model: str) -> None:
        self.chat_list.get_current_data()["model"] = model

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

    def add_massage(self, msg, type_) -> None:
        self.chat_list.get_current_data()["messages"].append(
            {"role": type_, "content": msg}
        )

    def get_reply(self, chat: Chat) -> None:
        Print.print_normal(chat, f"{chat.get_ai_name()}:")
        Print.print_ai(chat, "", end="")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(self.make_post(chat.chat_data).encode())

        chat.chat_data["messages"].append({"role": "assistant", "content": ""})
        max_recv = 124
        reply = s.recv(max_recv)
        raw = reply.decode().split("\r\n\r\n")
        try:
            if "HTTP/1.1 200 OK" not in raw[0]:
                return
            if "Content-Type: application/x-ndjson" in raw[0]:
                body = raw[1].split("\n")
                size = int(body[0], 16)
                final = body[1].encode()
                while len(final) < size:
                    final += s.recv(min(max_recv, size - len(final) + 2))
                body[1] = final.decode()
                done = json.loads(body[1])["done"]
                msg = json.loads(body[1])["message"]["content"]
                chat.chat_data["messages"][-1]["content"] += msg
                Print.print_ai(chat, msg, end="")
                while not done:
                    body = s.recv(max_recv).decode().split("\n")
                    size = int(body[0], 16)
                    final = body[1].encode()
                    while len(final) < size:
                        final += s.recv(min(max_recv, size - len(final) + 2))
                    body[1] = final.decode()
                    done = json.loads(body[1])["done"]
                    msg = json.loads(body[1])["message"]["content"]
                    chat.chat_data["messages"][-1]["content"] += msg
                    Print.print_ai(chat, msg, end="")
            if "Content-Type: application/json;" in raw[0]:
                length = raw[0][raw[0].find("Content-Length: ") :]
                length = length[: max(len(length), length.find("\n"))]
                size = int(length[15::])
                final = raw[1].encode()
                while len(final) < size:
                    final += s.recv(min(max_recv, size - len(final)))
                msg = json.loads(final.decode())["message"]["content"]
                chat.chat_data["messages"][-1]["content"] += msg
                Print.print_ai(chat, msg, end="")
        except KeyboardInterrupt:
            pass
        print("")
        s.close()

    def process_command(self, command: str) -> bool:
        command = command.split(" ")
        if command[0] == "/q":
            return True
        if command[0] not in self.commands:
            Print.print_normal(
                self.chat_list.get_current_chat(), f"Unknown command: {command[0]}"
            )
            return False
        self.commands[command[0]](*command[1:])
        return False

    def run_chat(self) -> None:
        Print.clean_screan()
        while True:
            try:
                if len(self.chat_list.get_current_data()["messages"]) == 0:
                    Print.print_normal(
                        self.chat_list.get_current_chat(), "system message"
                    )
                    input_ = input().replace("\\n", "\n")
                    if input_[0:2] == "/o":
                        if os.path.isfile(input_[2::]):
                            with open(input_[2::], encoding="utf-8") as file:
                                string = file.read()
                                self.add_massage(string, "system")
                        else:
                            Print.print_normal(
                                self.chat_list.get_current_chat(), "no such file"
                            )
                    else:
                        self.add_massage(input_, "system")
                        self.reprint()
                    continue

                Print.print_normal(
                    self.chat_list.get_current_chat(),
                    f"{self.chat_list.get_current_chat().get_user_name()}:",
                )
                Print.print_user(self.chat_list.get_current_chat(), "", end="")
                input_ = input().replace("\\n", "\n")

                if input_[0] == "/":
                    if self.process_command(input_):
                        break
                else:
                    self.add_massage(input_, "user")
                    self.get_reply(self.chat_list.get_current_chat())
            except (KeyboardInterrupt, UnicodeDecodeError) as e:
                if isinstance(e, UnicodeDecodeError):
                    Print.print_normal(
                        self.chat_list.get_current_chat(), "input too large"
                    )
                continue

        Print.clean_screan()


if __name__ == "__main__":
    Supervisor().run_chat()
