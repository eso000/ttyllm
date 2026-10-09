import codecs
import enum
import os
import select
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from completion import complete
from state import ViewMode

if TYPE_CHECKING:
    from supervisor import Supervisor

ENTER_KEYS = ("\r", "\n")
ESCAPE_SEQUENCE_END = ("A", "B", "C", "D", "~")
QUIT_COMMANDS = ("/q", ":q")


class Key(enum.IntEnum):
    """control keys the chat reacts to"""
    CTRL_B = 2
    CTRL_C = 3
    CTRL_D = 4
    CTRL_E = 5
    TAB = 9
    CTRL_N = 14
    CTRL_P = 16
    CTRL_Q = 17
    CTRL_R = 18
    CTRL_S = 19
    CTRL_U = 21
    ESCAPE = 27
    BACKSPACE = 127


class InputHandler:
    """turns keystrokes into commands, line edits, scrolling and tool approvals"""
    def __init__(self, supervisor: "Supervisor") -> None:
        """bind every control key to its action"""
        self.sv: "Supervisor" = supervisor
        commands = supervisor.commands
        self.keys: dict[int, Callable[[], None]] = {
            Key.CTRL_S: commands.split,
            Key.CTRL_B: commands.create_buffer,
            Key.CTRL_D: commands.buffer_delete,
            Key.CTRL_N: commands.buffer_next,
            Key.CTRL_P: commands.buffer_previous,
            Key.CTRL_R: commands.redo,
            Key.CTRL_E: commands.undo,
            Key.CTRL_Q: self.toggle_command_line,
            Key.CTRL_U: self.clear_line,
            Key.ESCAPE: self.escape,
            Key.CTRL_C: supervisor.handle_sigint,
            Key.BACKSPACE: self.backspace,
            Key.TAB: self.tab_complete,
        }

    def run(self) -> None:
        """input thread: read stdin one character at a time until the app stops"""
        fd = sys.stdin.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while self.sv.run:
            data = os.read(fd, 1)
            if not data:
                self.sv.stop()
                return
            ch = decoder.decode(data)
            if ch:
                self.handle_key(ch)

    def handle_key(self, ch: str) -> None:
        """react to one typed character"""
        sv, ui = self.sv, self.sv.ui
        if ui.dismiss_notification():
            sv.current_buffer.notify()
            return
        if (
            ch in ("y", "n")
            and ui.mode is ViewMode.NORMAL
            and ui.command == ""
            and sv.current_buffer.awaiting_approval
        ):
            sv.decide_tool_call(ch == "y")
            return
        with sv.input_cond:
            action = self.keys.get(ord(ch))
            if action:
                action()
            elif ch in ENTER_KEYS:
                self.enter()
            elif ch.isprintable():
                self.type_char(ch)

    def toggle_command_line(self) -> None:
        """open the command line, or close it again"""
        self.sv.ui.command = "/" if self.sv.ui.command == "" else ""
        self.sv.update_command()

    def clear_line(self) -> None:
        """clear the command line, or the message when there is no command"""
        if self.sv.ui.command:
            self.sv.ui.command = ""
            self.sv.current_buffer.notify()
        else:
            self.sv.current_buffer.clear_msg()

    def backspace(self) -> None:
        """delete the last character of the command, or of the message"""
        sv, ui = self.sv, self.sv.ui
        if ui.command != "":
            ui.command = ui.command[:-1]
            sv.update_command()
            if ui.command == "":
                sv.current_buffer.notify()
        else:
            ui.show_chat()
            sv.current_buffer.del_open()

    def tab_complete(self) -> None:
        """complete model names and file paths in the command line"""
        sv, ui = self.sv, self.sv.ui
        if ui.command == "":
            return
        suffix = complete(ui.command, sv.models_list)
        if suffix:
            ui.command += suffix
            sv.update_command()

    def enter(self) -> None:
        """submit the command or message, or leave the current view"""
        sv, ui = self.sv, self.sv.ui
        if ui.command != "":
            if ui.command in QUIT_COMMANDS:
                sv.stop()
            sv.update_command()
            sv.submit_input()
        elif ui.mode is ViewMode.NORMAL:
            sv.submit_input()
        else:
            ui.show_chat()
            sv.current_buffer.notify()

    def type_char(self, ch: str) -> None:
        """add a printable character to the command line or to the message"""
        sv, ui = self.sv, self.sv.ui
        idle, last_empty = sv.current_buffer.currently_typed()
        if (last_empty or not idle) and ui.command == "" and ch in ("/", ":"):
            ui.command = ch
            sv.update_command()
        elif idle and ui.command == "":
            ui.show_chat()
            sv.current_buffer.add_open(ch)
        elif ui.command:
            ui.command += ch
            sv.update_command()

    def escape(self) -> None:
        """handle an escape sequence, the arrow keys scroll"""
        sequence = self.read_escape()
        if sequence is None:
            return
        self.apply_escape(sequence)
        self.sv.current_buffer.notify()

    @staticmethod
    def read_escape() -> str | None:
        """read the rest of an escape sequence, None if nothing or another escape follows"""
        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], 0.0)
        if not ready:
            return None
        sequence = ""
        while not sequence or sequence[-1] not in ESCAPE_SEQUENCE_END:
            data = os.read(fd, 1)
            if not data:
                return None
            ch = data.decode("utf-8", errors="replace")
            if ch == "\x1b":
                return None
            sequence += ch
        return sequence

    def apply_escape(self, sequence: str) -> None:
        """scroll for an arrow key sequence, anything else is ignored"""
        ui = self.sv.ui
        if sequence == "[A" or sequence.startswith("OA"):
            ui.scroll_up()
        if sequence == "[B" or sequence.startswith("OB"):
            ui.scroll_down()
        if sequence == "[D":
            ui.scroll_to_end()
        if sequence == "[C":
            ui.scroll_to_top()
