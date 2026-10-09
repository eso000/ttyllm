import codecs
import json
import sys
import threading
import time
import signal
import shutil
import select
import os
import base64
import types
from collections.abc import Callable
from glob import glob
from enum import Enum
from typing import Any

from box_drawing import make_text_box, clean_screen, fit_text_to_box, fit_to_box
from buffer import Buffer, ColorCodeException
from requester import Model, Requester
from terminal import raw_terminal


class State(Enum):
    """which view is currently displayed"""
    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4


class Queue:
    """serialises reply requests, one buffer at a time"""
    def __init__(self) -> None:
        """create an empty queue"""
        self.cond: threading.Condition = threading.Condition()
        self.qu: list[Buffer] = []

    def schedule(self, chat: Buffer) -> None:
        """add a chat to the queue unless it is already queued or running"""
        with self.cond:
            if chat.queued or chat.in_request:
                return
            self.qu.append(chat)
            chat.queued = True
            self.cond.notify()
            chat.notify()

    def deschedule(self, chat: Buffer) -> None:
        """remove a chat from the queue"""
        with self.cond:
            for index, scheduled_chat in enumerate(self.qu):
                if scheduled_chat == chat:
                    chat.queued = False
                    self.qu.pop(index)
                    chat.notify()
                    return

    @property
    def next(self) -> Buffer:
        """pop the oldest queued chat"""
        with self.cond:
            chat = self.qu.pop(0)
            chat.queued = False
            chat.notify()
            return chat

    @property
    def has_scheduled(self) -> bool:
        """True when at least one chat waits for a reply"""
        with self.cond:
            return len(self.qu) != 0


class Supervisor:
    """owns the buffers, threads and terminal, and drives the whole chat"""
    def __init__(self, open_api: bool = False) -> None:

        """set up buffers, commands, request queue and the model list"""
        self.open_api: bool = open_api

        self.buffers: list[Buffer] = [Buffer()]
        self.current_buffer_id: int = 0
        self.notification: str = ""
        self.requester: Requester = Requester(open_api=open_api)

        self.run: bool = True

        self.command: str = ""
        self.input_done: bool = False
        self.input_cond: threading.Condition = threading.Condition()
        self.buffer_last: int = -1
        self.buffer_lines: int = -1

        self.queue: Queue = Queue()

        size = shutil.get_terminal_size()
        self.columns: int = size.columns
        self.lines: int = size.lines
        self.state: State = State.NORMAL
        signal.signal(signal.SIGWINCH, self.handle_sigwinch)

        self.models: list[Model] = []
        self.models_list: list[str] = []
        self.models_short: str = ""
        self.load_models()
        self._report_models()

        self.commands: dict[str, Callable[..., None]] = {
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
    def current_buffer(self) -> Buffer:
        """the buffer currently shown"""
        return self.buffers[self.current_buffer_id]

    @property
    def current_data(self) -> dict[str, Any]:
        """chat data of the buffer currently shown"""
        return self.current_buffer.chat_data

    def show_notification(self, message: str) -> None:
        """shown over the chat until the user presses a key"""
        self.notification = message
        self.buffer_last = 0
        self.current_buffer.notify()

    def handle_sigint(self) -> None:
        """decline pending tool calls or cancel the running request"""
        buffer = self.current_buffer
        self.queue.deschedule(buffer)
        if not buffer.abort_pending():
            buffer.cancel_request()

    def decide_tool_call(self, approved: bool) -> None:
        """answer a pending tool call and resume the chat once all are settled"""
        buffer = self.current_buffer
        if buffer.decide(approved):
            self.queue.schedule(buffer)

    def handle_sigwinch(
        self, signum: int, frame: types.FrameType | None
    ) -> None:  # pylint: disable=unused-argument
        """re-read the terminal size and redraw after a resize"""
        size = shutil.get_terminal_size()
        self.columns = size.columns
        self.lines = size.lines
        clean_screen()
        self.update_command()
        self.current_buffer.notify()

    def get_reply(self, buffer: Buffer) -> None:
        """run a request for the buffer and surface errors as a notification"""
        if buffer.pending_resolved():
            error = self.requester.resume(buffer)
        else:
            error = self.requester.get_reply(buffer)
        if error:
            self.show_notification(error)
        else:
            self.current_buffer.notify()  # status line may list new approvals

    def change_buffer(self, new_id: int) -> None:
        """switch to another buffer and wake both views"""
        old_buffer = self.current_buffer
        self.current_buffer_id = new_id
        with old_buffer.cond:
            old_buffer.cond.notify()
        with self.current_buffer.cond:
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    ############# internal commands
    def load_models(self) -> None:
        """refresh the model list and its display text"""
        self.models = self.requester.get_models()
        self.models_list = [model.name for model in self.models]
        self.models_short = "".join(
            self._format_model(model) for model in self.models
        )

    @staticmethod
    def _format_model(model: Model) -> str:
        """one line of the model list, with size and parameters when known"""
        if model.size_gb is None:  # openai api provides names only
            return f"{model.name}\n"
        return (
            f"{model.name:<20}"
            f"\t {model.size_gb} GB"
            f"\t{model.parameter_size}\n"
        )

    def _models_problem(self) -> str | None:
        """describe why the model list is unusable, None when it is fine"""
        endpoint = f"{self.requester.host}:{self.requester.port}"
        if self.requester.models_error:
            return f"cannot reach {endpoint} - {self.requester.models_error}"
        if not self.models_list:
            return f"no models found on {endpoint}"
        return None

    def _report_models(self) -> bool:
        """notify about a failed or empty model list, True if there was a problem"""
        problem = self._models_problem()
        if problem:
            self.show_notification(problem)
        return problem is not None

    def remove_image(self, *args: str) -> None:  # pylint: disable=unused-argument
        """remove images from current message"""
        with self.current_buffer.cond:
            if "images" in self.current_data["messages"][-1]:
                self.current_data["messages"][-1].pop("images")

    def add_image(self, *args: str) -> None:  # pylint: disable=unused-argument
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

    def list_models(self, *args: str) -> None:  # pylint: disable=unused-argument
        """list all models"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.MODELS
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def refresh_models(self, *args: str) -> None:  # pylint: disable=unused-argument
        """fetch the model list from the server again"""
        self.load_models()
        if not self._report_models():
            self.show_notification(f"loaded {len(self.models_list)} models")

    def help(self, *args: str) -> None:  # pylint: disable=unused-argument
        """show help"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.HELP
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def _help_text(self) -> str:
        """build the help screen from the command docstrings"""
        lines = ["commands  (up/down: scroll, enter: back to chat)", ""]
        for name, function in self.commands.items():
            lines.append(f"{name}:\t{function.__doc__ or ''}")
        return "\n".join(lines)

    def normal_view(self, *args: str) -> None:  # pylint: disable=unused-argument
        """show chat"""
        with self.current_buffer.cond:
            self.buffer_last = -1
            self.state = State.NORMAL
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def show_data(self, *args: str) -> None:  # pylint: disable=unused-argument
        """show data"""
        with self.current_buffer.cond:
            self.buffer_last = 0
            self.state = State.DATA
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def show_system(self, *args: str) -> None:  # pylint: disable=unused-argument
        """show system prompt"""
        with self.current_buffer.cond:
            self.state = State.SYSTEM
            self.current_buffer.updated = True
            self.current_buffer.cond.notify()

    def split(self, *args: str) -> None:  # pylint: disable=unused-argument
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

    def correct(self, *args: str) -> None:  # pylint: disable=unused-argument
        """removes last message"""
        self.current_buffer.del_msg(1)

    def redo(self, *args: str) -> None:  # pylint: disable=unused-argument
        """regenerate last message"""
        self.current_buffer.del_msg(2)
        self.queue.schedule(self.current_buffer)

    def start_over(self, *args: str) -> None:  # pylint: disable=unused-argument
        """reset current buffer"""
        self.current_buffer.empty()

    def undo(self, *args: str) -> None:  # pylint: disable=unused-argument
        """revert last user message"""
        self.current_buffer.del_msg(3)
        self.current_buffer.add_msg("user")

    def create_buffer(self, *args: str) -> None:  # pylint: disable=unused-argument
        """create new blank buffer"""
        self.buffers.append(Buffer())
        self.change_buffer(len(self.buffers) - 1)

    def buffer_next(self, *args: str) -> None:  # pylint: disable=unused-argument
        """switch to next buffer"""
        if self.current_buffer_id < len(self.buffers) - 1:
            self.change_buffer(self.current_buffer_id + 1)

    def buffer_previous(self, *args: str) -> None:  # pylint: disable=unused-argument
        """switch to previous buffer"""
        if self.current_buffer_id > 0:
            self.change_buffer(self.current_buffer_id - 1)

    def buffer_delete(self, *args: str) -> None:  # pylint: disable=unused-argument
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

    def switch_model(self, *args: str) -> None:
        """switch model -- argument: model name"""
        if len(args) < 1:
            return
        with self.current_buffer.cond:
            self.current_data["model"] = args[0]
        self.current_buffer.notify()

    def change_ai(self, *args: str) -> None:  # pylint: disable=unused-argument
        """change current message to assistant"""
        self.current_buffer.change_msg("assistant")

    def change_user(self, *args: str) -> None:  # pylint: disable=unused-argument
        """change current message to user"""
        self.current_buffer.change_msg("user")

    def change_system(self, *args: str) -> None:  # pylint: disable=unused-argument
        """change current message to system"""
        self.current_buffer.change_msg("system")

    def import_chat(self, *args: str) -> None:
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

    def export_chat(self, *args: str) -> None:
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

    def options(self, *args: str) -> None:
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

    def visual(self, *args: str) -> None:
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

    def process_command(self, command: str) -> bool:
        """run a slash or colon command, False when the user wants to quit"""
        parts = command.split(" ")
        if parts[0][0] not in (":", "/"):
            return True
        parts[0] = parts[0][1:]
        if parts[0] == "q":
            return False
        if parts[0] in self.commands:
            self.commands[parts[0]](*parts[1:])
        return True

    def handle_new_message(self) -> None:
        """schedule the current buffer after the user submitted a message"""
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

    def queue_manager(self) -> None:
        """worker thread: pull queued chats and request their replies"""
        while self.run:
            with self.queue.cond:
                while not self.queue.has_scheduled and self.run:
                    self.queue.cond.wait()
                if not self.run:
                    return
                chat = self.queue.next
            self.get_reply(chat)

    def watcher(self) -> None:
        """worker thread: dispatch submitted input to commands or to the chat"""
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

    def _handle_escape_sequence(self) -> None:
        """read the rest of an escape sequence and scroll with the arrow keys"""
        fd = sys.stdin.fileno()
        buffer = ""
        ch: int | str = 27
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

    def update_command(self) -> None:
        """draw the status line, or the command line while a command is typed"""
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

    def _auto_complete(self) -> None:
        """complete model names and file paths when tab is pressed"""
        if self.command != "":
            command_tokens = self.command.split(" ")
            if len(command_tokens) < 2:
                return
            possible_completions: list[str] = []
            match command_tokens[0]:
                case ":sm" | "/sm":
                    possible_completions = [
                        x for x in self.models_list if x.startswith(command_tokens[1])
                    ]
                case ":e" | "/e" | ":i" | "/i" | ":image" | "/image":

                    def _dir_completion(path: str) -> str:
                        """append a slash so directories complete to a prefix"""
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

    def stop(self) -> None:
        """end the chat: cancel requests and wake every waiting thread"""
        self.run = False
        for buffer in self.buffers:
            buffer.cancel_request()
        with self.current_buffer.cond:
            self.current_buffer.cond.notify()
        with self.queue.cond:
            self.queue.cond.notify()
        with self.input_cond:
            self.input_cond.notify()

    def get_input(self) -> None:
        """input thread: turn raw keystrokes into commands, edits and approvals"""
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

    def update_buffer(self) -> None:
        """render thread: redraw the view whenever the buffer changes"""
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
        """set up the terminal, run the worker threads and restore the terminal on exit"""
        self.run = True
        threads = [
            threading.Thread(target=self._guarded(self.get_input)),
            threading.Thread(target=self._guarded(self.update_buffer)),
            threading.Thread(target=self._guarded(self.watcher)),
            threading.Thread(target=self._guarded(self.queue_manager)),
        ]

        with raw_terminal():
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

    def _guarded(self, target: Callable[[], None]) -> Callable[[], None]:
        """wrap a thread target so an error notifies instead of killing the thread"""
        def run() -> None:
            """call the target, restarting it after errors until the chat stops"""
            while self.run:
                try:
                    target()
                    return
                except Exception as error:  # pylint: disable=broad-exception-caught
                    self.show_notification(f"{target.__name__} failed: {error!r}")
                    time.sleep(1)

        return run
