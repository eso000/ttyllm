import socket
import json
import os


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

    def get_normal_color(self):
        return self.normal_color

    def get_ai_color(self):
        return self.ai_color

    def get_user_color(self):
        return self.user_color

    def get_user_name(self):
        return self.user_name

    def get_ai_name(self):
        return self.ai_name


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

    # TODO refactor this nightmare
    def get_reply(self, chat: Chat) -> None:
        Print.print_normal(chat, f"{chat.get_ai_name()}:")
        Print.print_ai(chat, "", end="")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(self.make_post(chat.chat_data).encode())

        size = -1
        final = ""
        done = False
        type_ = 0
        chat.chat_data["messages"].append({"role": "assistant", "content": ""})
        try:
            while not done:
                reply = s.recv(1024)
                lines = reply.decode().split("\n")
                if lines[0][0] == "0":
                    continue

                if type_ != 0:
                    final = lines[1]
                else:
                    for line in lines:
                        if "application/json" in line:
                            type_ = 0
                        if "application/x-ndjson" in line:
                            type_ = 1
                        if line == "":
                            final += lines[5]
                            break
                        if "Content-Length:" in line:
                            size = int(line[15::])
                            final = lines[5]
                            while len(final) < size:
                                final += s.recv(1024).decode()
                        if "Transfer-Encoding: chunked" in line:
                            final += lines[6]
                            if type_ == 1:
                                break
                            while True:
                                lines = s.recv(1024).decode().split("\n")
                                final += lines[0]
                                if len(lines) > 1 and lines[1][0] == "0":
                                    break
                done = json.loads(final)["done"]
                msg = json.loads(final)["message"]["content"]
                chat.chat_data["messages"][-1]["content"] += msg
                print(msg, end="", flush=True)
            print("")
        except KeyboardInterrupt:
            pass
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
