import codecs
import sys
import threading
import time
import signal
import shutil
import select
import os
import types
from collections.abc import Callable
from glob import glob
from typing import Any

from box_drawing import make_text_box, clean_screen, fit_text_to_box, fit_to_box
from buffer import Buffer
from commands import Commands
from request_queue import RequestQueue
from requester import Model, Requester
from state import State
from terminal import raw_terminal


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

        self.queue: RequestQueue = RequestQueue()

        size = shutil.get_terminal_size()
        self.columns: int = size.columns
        self.lines: int = size.lines
        self.state: State = State.NORMAL
        signal.signal(signal.SIGWINCH, self.handle_sigwinch)

        self.models: list[Model] = []
        self.models_list: list[str] = []
        self.models_short: str = ""
        self.load_models()
        self.report_models()

        self.commands: Commands = Commands(self)

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

    def report_models(self) -> bool:
        """notify about a failed or empty model list, True if there was a problem"""
        problem = self._models_problem()
        if problem:
            self.show_notification(problem)
        return problem is not None

    def process_command(self, command: str) -> bool:
        """run a slash or colon command, False when the user wants to quit"""
        return self.commands.execute(command)

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
                        self.commands.split()
                    case 2:
                        self.commands.create_buffer()
                    case 4:
                        self.commands.buffer_delete()
                    case 14:
                        self.commands.buffer_next()
                    case 16:
                        self.commands.buffer_previous()
                    case 18:
                        self.commands.redo()
                    case 5:
                        self.commands.undo()
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
                        box = make_text_box(
                            self.commands.help_text(), self.columns - 4, last
                        )
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
