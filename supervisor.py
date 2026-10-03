import codecs
import json
import sys
import termios
import threading
import time
import signal
import shutil
import select
import os
import base64
from glob import glob
from enum import Enum

from box_drawing import make_text_box, clean_screen, fit_text_to_box, fit_to_box
from buffer import Buffer, ColorCodeException
from requester import Requester


class State(Enum):
    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4


class Queue:
    def __init__(self):
        self.cond = threading.Condition()
        self.qu = []

    def schedule(self, chat):
        with self.cond:
            if chat.queued or chat.in_request:
                return
            self.qu.append(chat)
            chat.queued = True
            self.cond.notify()
            chat.notify()

    def deschedule(self, chat):
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
    def __init__(self, open_api=False) -> None:

        self.open_api = open_api

        self.buffers = [Buffer()]
        self.current_buffer_id = 0
        self.notification = ""
        self.requester = Requester(open_api=open_api)

        self.run = True

        self.command = ""
        self.input_done = False
        self.input_cond = threading.Condition()
        self.buffer_last = -1
        self.buffer_lines = -1

        self.queue = Queue()

        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        self.state = State.NORMAL
        signal.signal(signal.SIGWINCH, self.handle_sigwinch)

        self.models = []
        self.models_list = []
        self.models_short = ""
        self.load_models()
        self._report_models()

        self.commands = {
            "m": self.list_models,
            "mr": self.refresh_models,
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
            "o": self.options,
            "image": self.add_image,
            "rmi": self.remove_image,
            "-": self.correct,
        }

    @property
    def current_buffer(self):
        return self.buffers[self.current_buffer_id]

    @property
    def current_data(self):
        return self.current_buffer.chat_data

    def show_notification(self, message):
        """shown over the chat until the user presses a key"""
        self.notification = message
        self.buffer_last = 0
        self.current_buffer.notify()

    def handle_sigint(self):
        buffer = self.current_buffer
        self.queue.deschedule(buffer)
        if not buffer.abort_pending():
            buffer.cancel_request()

    def decide_tool_call(self, approved):
        buffer = self.current_buffer
        if buffer.decide(approved):
            self.queue.schedule(buffer)

    def handle_sigwinch(self, signum, frame):  # pylint: disable=unused-argument
        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        clean_screen()
        self.update_command()
        self.current_buffer.notify()

    def get_reply(self, buffer):
        if buffer.pending_resolved():
            error = self.requester.resume(buffer)
        else:
            error = self.requester.get_reply(buffer)
        if error:
            self.show_notification(error)
        else:
            self.current_buffer.notify()  # status line may list new approvals

    def change_buffer(self, new_id):
        old_buffer = self.current_buffer
        self.current_buffer_id = new_id
        with old_buffer.cond:
            old_buffer.cond.notify()
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    ############# internal commands
    def load_models(self):
        self.models = self.requester.get_models()
        self.models_list = [model.name for model in self.models]
        self.models_short = "".join(
            self._format_model(model) for model in self.models
        )

    @staticmethod
    def _format_model(model) -> str:
        if model.size_gb is None:  # openai api provides names only
            return f"{model.name}\n"
        return (
            f"{model.name:<20}"
            f"\t {model.size_gb} GB"
            f"\t{model.parameter_size}\n"
        )

    def _models_problem(self):
        endpoint = f"{self.requester.host}:{self.requester.port}"
        if self.requester.models_error:
            return f"cannot reach {endpoint} - {self.requester.models_error}"
        if not self.models_list:
            return f"no models found on {endpoint}"
        return None

    def _report_models(self):
        """notify about a failed or empty model list, True if there was a problem"""
        problem = self._models_problem()
        if problem:
            self.show_notification(problem)
        return problem is not None

    def remove_image(self, *args):  # pylint: disable=unused-argument
        """remove images from current message"""
        with self.current_buffer.cond:
            if "images" in self.current_data["messages"][-1]:
                self.current_data["messages"][-1].pop("images")

    def add_image(self, *args):  # pylint: disable=unused-argument
        """add image to message"""
        if len(args) < 1:
            return
        if self.current_data["messages"][-1]["role"] != "user":
            return

        try:
            with open(args[0], "rb") as file:
                img = base64.b64encode(file.read()).decode("utf-8")
            with self.current_buffer.cond:
                if "images" not in self.current_data["messages"][-1]:
                    self.current_data["messages"][-1]["images"] = []
                self.current_data["messages"][-1]["images"].append(img)

        except PermissionError:
            self.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.show_notification(f"File {args[0]} not found")

    def list_models(self, *args):  # pylint: disable=unused-argument
        """list all models"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.MODELS
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def refresh_models(self, *args):  # pylint: disable=unused-argument
        """fetch the model list from the server again"""
        self.load_models()
        if not self._report_models():
            self.show_notification(f"loaded {len(self.models_list)} models")

    def help(self, *args):  # pylint: disable=unused-argument
        """show help"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.HELP
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def _help_text(self):
        lines = ["commands  (up/down: scroll, enter: back to chat)", ""]
        for name, function in self.commands.items():
            lines.append(f"{name}:\t{function.__doc__ or ''}")
        return "\n".join(lines)

    def normal_view(self, *args):  # pylint: disable=unused-argument
        """show chat"""
        with self.current_buffer.cond:
            self.buffer_last = -1
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
        if self.current_buffer.pending_calls:
            self.show_notification("answer the pending tool calls before splitting")
            return
        new = Buffer()
        with self.current_buffer.cond:
            new.chat_data = json.loads(json.dumps(self.current_data))
            new.visual_data = json.loads(json.dumps(self.current_buffer.visual_data))
            new.tokens = self.current_buffer.tokens
        self.buffers.append(new)
        self.change_buffer(len(self.buffers) - 1)

    def correct(self, *args):  # pylint: disable=unused-argument
        """removes last message"""
        self.current_buffer.del_msg(1)

    def redo(self, *args):  # pylint: disable=unused-argument
        """regenerate last message"""
        self.current_buffer.del_msg(2)
        self.queue.schedule(self.current_buffer)

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
        old_buffer = self.current_buffer
        self.queue.deschedule(old_buffer)
        old_buffer.abort_pending()
        self.buffers.pop(self.current_buffer_id)
        if len(self.buffers) == 0:
            self.buffers.append(Buffer())
        if self.current_buffer_id == len(self.buffers):
            self.current_buffer_id -= 1
        with old_buffer.cond:
            old_buffer.cond.notify()
        self.current_buffer.notify()

    def switch_model(self, *args):
        """switch model -- argument: model name"""
        if len(args) < 1:
            return
        with self.current_buffer.cond:
            self.current_data["model"] = args[0]
        self.current_buffer.notify()

    def change_ai(self, *args):  # pylint: disable=unused-argument
        """change current message to assistant"""
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
                    self.show_notification(f"File {args[0]} is corrupted")
        except PermissionError:
            self.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.show_notification(f"File {args[0]} not found")

    def export_chat(self, *args):
        """export chat from file -- argument: file name"""
        if len(args) < 1:
            return
        with self.current_buffer.cond:
            data = json.dumps(
                {
                    "visual": self.current_buffer.visual_data,
                    "chat": self.current_data,
                },
                indent=4,
            )
        try:
            with open(args[0], "w", encoding="utf-8") as file:
                file.write(data)
        except PermissionError:
            self.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.show_notification(f"File {args[0]} not found")

    def options(self, *args):
        """show or set options -- arguments: k|p|t|pen|stream value"""
        if len(args) == 0:
            self.show_notification(json.dumps(self.current_buffer.options, indent=3))
        elif len(args) == 1:
            self.show_notification("no data provided")
        else:
            match args[0]:
                case "k":
                    self.current_buffer.top_k = float(args[1])
                case "p":
                    self.current_buffer.top_p = float(args[1])
                case "t":
                    self.current_buffer.temperature = int(args[1])
                case "pen":
                    self.current_buffer.repeat_penalty = float(args[1])
                case "stream":
                    self.current_data["stream"] = int(args[1]) & 1 == 1
                case _:
                    self.show_notification(f"command {args[0]} not found")

    def visual(self, *args):
        """show or set looks -- arguments: un|an|uc|ac|nc value"""
        if len(args) == 0:
            self.show_notification(
                json.dumps(self.current_buffer.visual_data, indent=3)
            )
        elif len(args) == 1:
            self.show_notification("no data provided")
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
                        self.show_notification(f"command {args[0]} not found")
            except ColorCodeException:
                self.show_notification(f"invalid color code :{args[1]}")

    def process_command(self, command) -> bool:
        command = command.split(" ")
        if command[0][0] not in (":", "/"):
            return True
        command[0] = command[0][1:]
        if command[0] == "q":
            return False
        if command[0] in self.commands:
            self.commands[command[0]](*command[1:])
        return True

    def handle_new_message(self):
        buffer = self.current_buffer
        with buffer.cond:
            last = buffer.chat_data["messages"][-1]
            last["content"] = last["content"].replace("\\n", "\n")
            if last["role"] != "user":
                buffer.change_msg("user")
                return
            buffer.updated = True
            buffer.cond.notify()
        self.queue.schedule(buffer)

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
                command, self.command = self.command, ""
            if not self.run:
                return
            self.update_command()
            if len(command) > 0:
                self.process_command(command)
            else:
                self.handle_new_message()

    def _handle_escape_sequence(self):
        fd = sys.stdin.fileno()
        buffer = ""
        ch = 27
        r, _, _ = select.select([fd], [], [], 0.0)
        if len(r) == 0:
            return
        while ch not in ("A", "B", "C", "D", "~"):
            ch = os.read(fd, 1).decode()
            if ord(ch) == 27:
                return
            buffer += ch

        if buffer == "[A" or buffer.startswith("OA"):
            if self.command == "":
                if self.buffer_last < 0:
                    self.buffer_last = self.buffer_lines - 1
                else:
                    if self.buffer_last:
                        self.buffer_last -= 1
        if buffer == "[B" or buffer.startswith("OB"):
            if self.command == "":
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
            waiting = [
                str(number)
                for number, buffer in enumerate(self.buffers, 1)
                if buffer.awaiting_approval
            ]
            fit_text_to_box(
                f"    model: {self.current_data['model']}"
                f" -- tokens: {self.current_buffer.tokens}"
                f"   buffer: {self.current_buffer_id + 1} of {len(self.buffers)}"
                f" {'in_request' if self.current_buffer.in_request else
                    "queued" if self.current_buffer.queued else ""} "
                f"{'approval needed in: ' + ', '.join(waiting) if waiting else ''}",
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

    def _auto_complete(self):
        if self.command != "":
            command_tokens = self.command.split(" ")
            if len(command_tokens) < 2:
                return
            possible_completions = []
            match command_tokens[0]:
                case ":sm" | "/sm":
                    possible_completions = [
                        x for x in self.models_list if x.startswith(command_tokens[1])
                    ]
                case ":e" | "/e" | ":i" | "/i" | ":image" | "/image":

                    def _dir_completion(path):
                        if not os.path.isdir(path):
                            return path
                        if path[-1] == "/":
                            return path
                        return path + "/"

                    possible_completions = [
                        _dir_completion(path) for path in glob(f"{command_tokens[1]}*")
                    ]

            if len(possible_completions) == 0:
                return
            if len(possible_completions) == 1:
                self.command += possible_completions[0][len(command_tokens[1]) :]
                self.update_command()
                return
            common_prefix = possible_completions[0]
            for string in possible_completions:
                while not string.startswith(common_prefix):
                    common_prefix = common_prefix[:-1]
                if common_prefix == "":
                    return
            self.command += common_prefix[len(command_tokens[1]) :]
            self.update_command()

    def stop(self):
        self.run = False
        for buffer in self.buffers:
            buffer.cancel_request()
        with self.current_buffer.cond:
            self.current_buffer.cond.notify()
        with self.queue.cond:
            self.queue.cond.notify()
        with self.input_cond:
            self.input_cond.notify()

    def get_input(self):
        fd = sys.stdin.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while self.run:
            data = os.read(fd, 1)
            if not data:
                self.stop()
                return
            ch = decoder.decode(data)
            if not ch:
                continue
            if self.notification:
                self.notification = ""
                self.current_buffer.notify()
                continue
            if (
                ch in ("y", "n")
                and self.state == State.NORMAL
                and self.command == ""
                and self.current_buffer.awaiting_approval
            ):
                self.decide_tool_call(ch == "y")
                continue
            with self.input_cond:
                match ord(ch):
                    case 19:
                        self.split()
                    case 2:
                        self.create_buffer()
                    case 4:
                        self.buffer_delete()
                    case 14:
                        self.buffer_next()
                    case 16:
                        self.buffer_previous()
                    case 18:
                        self.redo()
                    case 5:
                        self.undo()
                    case 17:
                        if self.command == "":
                            self.command = "/"
                        else:
                            self.command = ""
                        self.update_command()
                    case 21:
                        if self.command:
                            self.command = ""
                            self.current_buffer.notify()
                        else:
                            self.current_buffer.clear_msg()
                    case 27:
                        self._handle_escape_sequence()
                        continue
                    case 3:
                        self.handle_sigint()
                    case 127:
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
                    case 9:
                        self._auto_complete()

                if not ch or ch == "\r" or ch == "\n":
                    if self.command != "":
                        if self.command in ("/q", ":q"):
                            self.stop()
                        self.update_command()
                        self.input_done = True
                        self.input_cond.notify()
                    else:
                        if self.state == State.NORMAL:
                            self.input_done = True
                            self.input_cond.notify()
                        else:
                            self.buffer_last = -1
                            self.state = State.NORMAL
                            self.current_buffer.notify()
                if ch.isprintable():
                    idle, last_empty = self.current_buffer.currently_typed()
                    if (last_empty or not idle) and self.command == "" and ch in (
                        "/",
                        ":",
                    ):
                        self.command = ch
                        self.update_command()
                    elif idle and self.command == "":
                        self.buffer_last = -1
                        self.state = State.NORMAL
                        self.current_buffer.add_open(ch)
                    elif len(self.command) > 0:
                        self.command += ch
                        self.update_command()
                    else:
                        continue

    def update_buffer(self):
        while self.run:
            buffer = self.current_buffer
            with buffer.cond:
                while not buffer.updated and self.run and buffer is self.current_buffer:
                    buffer.cond.wait()
                if not self.run:
                    return
                if buffer is not self.current_buffer:
                    continue
                if len(buffer.chat_data["messages"]) == 0:
                    buffer.add_msg("system")
                last = self.buffer_last
                match self.state:
                    case _ if self.notification:
                        box = make_text_box(self.notification, self.columns - 4, last)
                        hide_cursor = True
                    case State.NORMAL:
                        box = buffer.get_text_box(self.columns - 4, last)
                        hide_cursor = False
                    case State.SYSTEM:
                        box = buffer.get_text_box_system(self.columns - 4, last)
                        hide_cursor = True
                    case State.DATA:
                        box = buffer.get_text_box_data(self.columns - 4, last)
                        hide_cursor = True
                    case State.HELP:
                        box = make_text_box(self._help_text(), self.columns - 4, last)
                        hide_cursor = True
                    case State.MODELS:
                        box = make_text_box(
                            # json.dumps(self.models, indent=4),
                            self.models_short,
                            self.columns - 4,
                            last,
                        )
                        hide_cursor = True
                buffer.updated = False

            self.update_command()
            lines, new_last = fit_to_box(
                box, self.columns - 4, self.lines - 5, 2, 3, hide_cursor=hide_cursor
            )
            with self.input_cond:
                self.buffer_lines = lines
                if self.buffer_last == last:  # not scrolled while drawing
                    self.buffer_last = new_last

    def run_chat(self) -> None:
        print("\033[?1049h", end="", flush=True)
        clean_screen()
        self.run = True
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[0] &= ~(termios.IXON)
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)

        threads = [
            threading.Thread(target=self._guarded(self.get_input)),
            threading.Thread(target=self._guarded(self.update_buffer)),
            threading.Thread(target=self._guarded(self.watcher)),
            threading.Thread(target=self._guarded(self.queue_manager)),
        ]

        try:
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
        finally:
            clean_screen()
            print("\033[?25h", end="", flush=True)
            print("\033[?1049l", end="", flush=True)
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _guarded(self, target):
        def run():
            while self.run:
                try:
                    target()
                    return
                except Exception as error:  # pylint: disable=broad-exception-caught
                    self.show_notification(f"{target.__name__} failed: {error!r}")
                    time.sleep(1)

        return run
