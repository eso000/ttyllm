import json
import threading
import time
import signal
import types
from collections.abc import Callable
from typing import Any

from box_drawing import clean_screen
from buffer import Buffer
from commands import Commands
from input_handler import InputHandler
from request_queue import RequestQueue
from requester import Model, Requester
from state import UIState
from terminal import raw_terminal
from view import View


class Supervisor:
    """owns the buffers, threads and terminal, and drives the whole chat"""
    def __init__(self, open_api: bool = False) -> None:

        """set up buffers, commands and request queue, run_chat loads the models"""
        self.open_api: bool = open_api

        self.buffers: list[Buffer] = [Buffer()]
        self.current_buffer_id: int = 0
        self.ui: UIState = UIState.from_terminal()
        self.requester: Requester = Requester(open_api=open_api)

        self.run: bool = True

        self.queue: RequestQueue = RequestQueue()

        self.models: list[Model] = []
        self.models_list: list[str] = []
        self.models_short: str = ""

        self.commands: Commands = Commands(self)
        self.keyboard: InputHandler = InputHandler(self)
        self.view: View = View(self)

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
        self.ui.set_notification(message)
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
        self.ui.resize()
        clean_screen()
        self.view.draw_command_line()
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

    def split(self) -> None:
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

    def correct(self) -> None:
        """removes last message"""
        self.current_buffer.del_msg(1)

    def redo(self) -> None:
        """regenerate last message"""
        self.current_buffer.del_msg(2)
        self.queue.schedule(self.current_buffer)

    def start_over(self) -> None:
        """reset current buffer"""
        self.current_buffer.empty()

    def undo(self) -> None:
        """revert last user message"""
        self.current_buffer.del_msg(3)
        self.current_buffer.add_msg("user")

    def create_buffer(self) -> None:
        """create new blank buffer"""
        self.buffers.append(Buffer())
        self.change_buffer(len(self.buffers) - 1)

    def buffer_next(self) -> None:
        """switch to next buffer"""
        if self.current_buffer_id < len(self.buffers) - 1:
            self.change_buffer(self.current_buffer_id + 1)

    def buffer_previous(self) -> None:
        """switch to previous buffer"""
        if self.current_buffer_id > 0:
            self.change_buffer(self.current_buffer_id - 1)

    def buffer_delete(self) -> None:
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
            with self.ui.cond:
                while not self.ui.submitted and self.run:
                    self.ui.cond.wait()
                command = self.ui.take_command()
            if not self.run:
                return
            self.view.draw_command_line()
            if len(command) > 0:
                self.commands.execute(command)
            else:
                self.handle_new_message()

    def stop(self) -> None:
        """end the chat: cancel requests and wake every waiting thread"""
        self.run = False
        for buffer in self.buffers:
            buffer.cancel_request()
        with self.current_buffer.cond:
            self.current_buffer.cond.notify()
        with self.queue.cond:
            self.queue.cond.notify()
        with self.ui.cond:
            self.ui.cond.notify()

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
                scroll = self.ui.scroll
                box, hide_cursor = self.view.compose(buffer, scroll)
                buffer.updated = False

            self.view.draw_command_line()
            lines, new_scroll = self.view.draw(box, hide_cursor)
            with self.ui.cond:
                self.ui.scroll_max = lines
                if self.ui.scroll == scroll:  # not scrolled while drawing
                    self.ui.scroll = new_scroll

    def run_chat(self) -> None:
        """load the models, run the worker threads in the terminal and restore it on exit"""
        self.load_models()
        self.report_models()
        self.run = True
        threads = [
            threading.Thread(target=self._guarded(name, target), name=name)
            for name, target in (
                ("input", self.keyboard.run),
                ("render", self.update_buffer),
                ("dispatch", self.watcher),
                ("replies", self.queue_manager),
            )
        ]

        with raw_terminal():
            previous = signal.signal(signal.SIGWINCH, self.handle_sigwinch)
            try:
                for thread in threads:
                    thread.start()

                for thread in threads:
                    thread.join()
            finally:
                signal.signal(signal.SIGWINCH, previous or signal.SIG_DFL)

    def _guarded(self, name: str, target: Callable[[], None]) -> Callable[[], None]:
        """wrap a thread target so an error notifies instead of killing the thread"""
        def run() -> None:
            """call the target, restarting it after errors until the chat stops"""
            while self.run:
                try:
                    target()
                    return
                except Exception as error:  # pylint: disable=broad-exception-caught
                    self.show_notification(f"{name} thread failed: {error!r}")
                    time.sleep(1)

        return run
