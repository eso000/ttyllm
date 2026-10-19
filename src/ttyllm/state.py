"""UIState: the view mode, scroll position, command line and notification."""

import dataclasses
import shutil
import threading
from enum import Enum


class ViewMode(Enum):
    """which view is currently displayed"""

    NORMAL = 0
    SYSTEM = 1
    DATA = 2
    HELP = 3
    MODELS = 4


@dataclasses.dataclass
class UIState:
    """what the user currently sees and types, shared by all threads

    scroll is the index of the last visible line, -1 follows the end of the
    text. scroll_max is the line count of the view drawn last.

    cond is a reentrant lock guarding every change made here; the methods
    take it themselves, so callers must not hold it while doing slow work.
    submitted is the flag the watcher waits for and take_command clears.
    """

    mode: ViewMode = ViewMode.NORMAL
    scroll: int = -1
    scroll_max: int = -1
    command: str = ""
    notification: str = ""
    columns: int = 80
    lines: int = 24
    submitted: bool = False
    cond: threading.Condition = dataclasses.field(
        default_factory=threading.Condition, compare=False, repr=False
    )

    @classmethod
    def from_terminal(cls) -> "UIState":
        """a fresh state sized like the terminal"""
        state = cls()
        state.resize()
        return state

    def resize(self) -> None:
        """re-read the terminal size"""
        size = shutil.get_terminal_size()
        with self.cond:
            self.columns = size.columns
            self.lines = size.lines

    def set_mode(self, mode: ViewMode, scroll: int | None = None) -> None:
        """switch the view, scroll is where it starts and None keeps the position"""
        with self.cond:
            if scroll is not None:
                self.scroll = scroll
            self.mode = mode

    def show_chat(self) -> None:
        """back to the chat, following its end"""
        self.set_mode(ViewMode.NORMAL, -1)

    def scroll_up(self) -> None:
        """one line up, from the end of the text to its last full page"""
        with self.cond:
            if self.command != "":
                return
            if self.scroll < 0:
                self.scroll = self.scroll_max - 1
            elif self.scroll:
                self.scroll -= 1

    def scroll_down(self) -> None:
        """one line down, past the last line the view follows the end again"""
        with self.cond:
            if self.command != "":
                return
            if 0 <= self.scroll < self.scroll_max:
                self.scroll += 1
            else:
                self.scroll = -1

    def scroll_to_end(self) -> None:
        """follow the end of the text"""
        with self.cond:
            self.scroll = -1

    def scroll_to_top(self) -> None:
        """show the beginning of the text"""
        with self.cond:
            self.scroll = 1

    def set_notification(self, message: str) -> None:
        """show a message over the chat, scrolled to its start"""
        with self.cond:
            self.notification = message
            self.scroll = 0

    def dismiss_notification(self) -> bool:
        """hide the message, True if there was one"""
        with self.cond:
            had_message = self.notification != ""
            self.notification = ""
            return had_message

    def take_command(self) -> str:
        """return the typed command, clear the line and mark it as consumed"""
        with self.cond:
            command, self.command = self.command, ""
            self.submitted = False
            return command

    def submit(self) -> None:
        """tell the watcher that a command or a message was submitted"""
        with self.cond:
            self.submitted = True
            self.cond.notify()
