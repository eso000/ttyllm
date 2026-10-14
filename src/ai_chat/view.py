"""Renders the current buffer for the current UIState; never mutates state."""

from typing import TYPE_CHECKING

from .box_drawing import TextBox, fit_text_to_box, fit_to_box, make_text_box
from .buffer import Buffer
from .state import UIState, ViewMode

if TYPE_CHECKING:
    from .supervisor import Supervisor

STATUS_COLOR = "\033[32m"
STATUS_BACKGROUND = "\033[0m"
COMMAND_BACKGROUND = "\033[41m"


class View:
    """draws the supervisor's buffers according to the UI state and never changes that state"""
    def __init__(self, supervisor: "Supervisor") -> None:
        """the view reads the buffers, models and help text from the supervisor"""
        self.sv: "Supervisor" = supervisor

    @property
    def ui(self) -> UIState:
        """the state that decides what is drawn"""
        return self.sv.ui

    @property
    def box_width(self) -> int:
        """width of the text area, the border takes the rest"""
        return self.ui.columns - 4

    def compose(self, buffer: Buffer, scroll: int) -> tuple[TextBox, bool]:
        """choose and lay out what to show, returns the box and whether to hide the cursor"""
        width = self.box_width
        if self.ui.notification:
            return make_text_box(self.ui.notification, width, scroll), True
        match self.ui.mode:
            case ViewMode.NORMAL:
                return buffer.get_text_box(width, scroll), False
            case ViewMode.SYSTEM:
                return buffer.get_text_box_system(width, scroll), True
            case ViewMode.DATA:
                return buffer.get_text_box_data(width, scroll), True
            case ViewMode.HELP:
                help_text = (
                    f"{self.sv.commands.help_text()}\n\n"
                    f"{self.sv.keyboard.help_text()}"
                )
                return make_text_box(help_text, width, scroll), True
            case ViewMode.MODELS:
                return make_text_box(self.sv.models.short, width, scroll), True

    def draw(self, box: TextBox, hide_cursor: bool) -> tuple[int, int]:
        """draw the box, returns its line count and the scroll position it ended at"""
        return fit_to_box(
            box, self.box_width, self.ui.lines - 5, 2, 3, hide_cursor=hide_cursor
        )

    def status_text(self) -> str:
        """the status line of the current buffer"""
        buffers = self.sv.buffers
        index = self.sv.current_buffer_id
        buffer = buffers[index]
        waiting = [
            str(number)
            for number, other in enumerate(buffers, 1)
            if other.awaiting_approval
        ]
        if buffer.in_request:
            request = "in_request"
        elif buffer.queued:
            request = "queued"
        else:
            request = ""
        approvals = "approval needed in: " + ", ".join(waiting) if waiting else ""
        return (
            f"    model: {buffer.chat_data['model']}"
            f" -- tokens: {buffer.tokens}"
            f"   buffer: {index + 1} of {len(buffers)}"
            f" {request} "
            f"{approvals}"
        )

    def draw_command_line(self) -> None:
        """draw the status line, or the command line while a command is typed"""
        ui = self.ui
        if ui.command == "":
            fit_text_to_box(
                self.status_text(),
                ui.columns,
                1,
                ui.lines,
                1,
                STATUS_COLOR,
                STATUS_BACKGROUND,
                hide_cursor=True,
            )
        else:
            fit_text_to_box(
                ui.command, ui.columns, 1, ui.lines, 1, STATUS_COLOR, COMMAND_BACKGROUND
            )
