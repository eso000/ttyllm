import json
import sys
import termios
import threading
import signal
import shutil
from glob import glob
from enum import Enum

from box_drawing import make_text_box, clean_screan, fit_text_to_box, fit_to_box
from buffer import Buffer, ColorCodeExeption
from requester import Requester


class State(Enum):
    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4
    DIALOG = 5


class Queue:
    def __init__(self):
        self.cond = threading.Condition()
        self.qu = []

    def scheduel(self, chat):
        if chat.queued or chat.in_request:
            return
        with self.cond:
            self.qu.append(chat)
            chat.queued = True
            self.cond.notify()
            chat.notify()

    def descheduel(self, chat):
        with self.cond:
            for index, scheduled_chat in enumerate(self.qu):
                if scheduled_chat == chat:
                    chat.queued = False
                    self.qu.pop(index)
                    chat.notify()
                    return

    @property
    def next(self):
        with self.cond:
            chat = self.qu.pop(0)
            chat.queued = False
            chat.notify()
            return chat

    @property
    def has_scheduled(self):
        with self.cond:
            return len(self.qu) != 0


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

        self.queue = Queue()

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
            "v": self.visual,
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
        self.current_buffer.cancel_request()
        self.queue.descheduel(self.current_buffer)

    def handle_sigwinch(self, signum, frame):  # pylint: disable=unused-argument
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
            help_string = ""
            for name, function in self.commands.items():
                help_string += f"{name}:\t{function.__doc__}\n"
            self.create_dialog(help_string)

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
        self.queue.scheduel(self.current_buffer)

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
                    self.create_dialog(f"File {args[0]} is corrupted")
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
                json.dump(
                    {
                        "visual": self.current_buffer.visual_data,
                        "chat": self.current_data,
                    },
                    file,
                    indent=4,
                )
        except PermissionError:
            self.create_dialog(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.create_dialog(f"File {args[0]} not found")

    def visual(self, *args):
        if len(args) == 0:
            self.create_dialog(json.dumps(self.current_buffer.visual_data, indent=3))
        elif len(args) == 1:
            self.create_dialog("no data provided")
        else:
            try:
                match args[0]:
                    case "un":
                        self.current_buffer.user_name = args[1]
                    case "an":
                        self.current_buffer.ai_name = args[1]
                    case "uc":
                        self.current_buffer.user_color = args[1]
                    case "ac":
                        self.current_buffer.ai_color = args[1]
                    case "nc":
                        self.current_buffer.normal_color = args[1]
                    case _:
                        self.create_dialog(f"comand {args[0]} not found")
            except ColorCodeExeption:
                self.create_dialog(f"invalid color code :{args[1]}")

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
            self.queue.scheduel(self.current_buffer)

    def queue_manager(self):
        while self.run:
            with self.queue.cond:
                while not self.queue.has_scheduled and self.run:
                    self.queue.cond.wait()
                if not self.run:
                    return
                chat = self.queue.next
            self.get_reply(chat)

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
                f"   buffer: {self.current_buffer_id + 1} of {len(self.buffers)}"
                f" {'in_request' if self.current_buffer.in_request else
                    "queued" if self.current_buffer.queued else ""} ",
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
        while self.run:
            ch = sys.stdin.read(1)
            with self.input_cond:
                if ord(ch) == 2:
                    self.create_buffer()
                if ord(ch) == 4:
                    self.buffer_delete()
                if ord(ch) == 14:
                    self.buffer_next()
                if ord(ch) == 16:
                    self.buffer_previous()
                if ord(ch) == 18:
                    self.redo()
                if ord(ch) == 5:
                    self.undo()
                if ord(ch) == 1:
                    if self.command == "":
                        self.command = "/"
                    else:
                        self.command = ""
                    self.update_command()

                if ord(ch) == 21:
                    if self.command:
                        self.command = ""
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
                    if self.command != "":
                        if self.command in ("/q", ":q"):
                            self.run = False
                            for buffer in self.buffers:
                                buffer.cancel_request()
                            with self.current_buffer.cond:
                                self.current_buffer.cond.notify()
                            with self.queue.cond:
                                self.queue.cond.notify()
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
                if ord(ch) == 127:
                    if self.command != "":
                        self.command = self.command[:-1]
                        self.update_command()
                        if self.command == "":
                            with self.current_buffer.cond:
                                self.current_buffer.updated = True
                                self.current_buffer.cond.notify()
                    else:
                        self.buffer_last = -1
                        self.state = State.NORMAL
                        self.current_buffer.del_open()
                    continue
                if 32 <= ord(ch) <= 126:
                    if len(
                        self.current_data["messages"][-1]["content"]
                    ) == 0 and ch in (
                        "/",
                        ":",
                    ):
                        self.command = ch
                        self.update_command()
                    elif (
                        not self.current_buffer.in_request
                        and not self.current_buffer.queued
                        and not self.command != ""
                    ):
                        if self.state == State.DIALOG:
                            self.state = State.NORMAL
                            self.dialog_reply = ch
                            self.current_buffer.notify()
                        else:
                            self.buffer_last = -1
                            self.state = State.NORMAL
                            self.current_buffer.add_open(ch)
                    elif len(self.command) > 0:
                        self.command += ch
                        self.update_command()
                    else:
                        continue
                if ord(ch) == 9:
                    if self.command != "":
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
                                        len(command_tokens[1]) :
                                    ]
                                    self.update_command()
                            case ":e" | "/e" | ":i" | "/i":
                                possible_completions = glob(f"{command_tokens[1]}*")
                                if len(possible_completions) == 1:
                                    self.command += possible_completions[0][
                                        len(command_tokens[1]) :
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
            threading.Thread(target=self.queue_manager),
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
