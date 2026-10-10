import base64
import dataclasses
import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from buffer import Buffer, ColorCodeException
from state import ViewMode

if TYPE_CHECKING:
    from supervisor import Supervisor


@dataclasses.dataclass(frozen=True)
class Command:
    """a colon command, its one line description and whether it takes arguments"""
    handler: Callable[..., None]
    description: str
    takes_args: bool = False


class Commands:
    """the colon commands of the chat, buffer commands are the supervisor's own methods"""
    def __init__(self, supervisor: "Supervisor") -> None:
        """build the command table for the given supervisor"""
        self.sv: "Supervisor" = supervisor
        self.table: dict[str, Command] = {
            "m": Command(partial(self.set_view, ViewMode.MODELS, 0), "list all models"),
            "mr": self._command(self.refresh_models),
            "h": Command(partial(self.set_view, ViewMode.HELP, 0), "show help"),
            "n": Command(partial(self.set_view, ViewMode.NORMAL, -1), "show chat"),
            "d": Command(partial(self.set_view, ViewMode.DATA, 0), "show data"),
            "S": Command(
                partial(self.set_view, ViewMode.SYSTEM, None), "show system prompt"
            ),
            "s": self._command(supervisor.split),
            "r": self._command(supervisor.redo),
            "R": self._command(supervisor.start_over),
            "rr": self._command(supervisor.undo),
            "bc": self._command(supervisor.create_buffer),
            "bn": self._command(supervisor.buffer_next),
            "bp": self._command(supervisor.buffer_previous),
            "bd": self._command(supervisor.buffer_delete),
            "sm": self._command(self.switch_model, takes_args=True),
            "ai": Command(
                partial(self.change_role, "assistant"),
                "change current message to assistant",
            ),
            "us": Command(
                partial(self.change_role, "user"), "change current message to user"
            ),
            "sys": Command(
                partial(self.change_role, "system"),
                "change current message to system",
            ),
            "i": self._command(self.import_chat, takes_args=True),
            "e": self._command(self.export_chat, takes_args=True),
            "v": self._command(self.visual, takes_args=True),
            "o": self._command(self.options, takes_args=True),
            "image": self._command(self.add_image, takes_args=True),
            "rmi": self._command(self.remove_image),
            "-": self._command(supervisor.correct),
        }

    @staticmethod
    def _command(handler: Callable[..., None], takes_args: bool = False) -> Command:
        """a command described by the docstring of its handler"""
        return Command(handler, handler.__doc__ or "", takes_args)

    @property
    def _buffer(self) -> Buffer:
        """the buffer currently shown"""
        return self.sv.current_buffer

    @property
    def _data(self) -> dict[str, Any]:
        """chat data of the buffer currently shown"""
        return self.sv.current_data

    def execute(self, command: str) -> bool:
        """run a slash or colon command, False when the user wants to quit"""
        parts = command.split(" ")
        if parts[0][0] not in (":", "/"):
            return True
        parts[0] = parts[0][1:]
        if parts[0] == "q":
            return False
        entry = self.table.get(parts[0])
        if entry is None:
            return True
        if entry.takes_args:
            entry.handler(*parts[1:])
        else:
            entry.handler()
        return True

    def help_text(self) -> str:
        """build the help screen from the command descriptions"""
        lines = ["commands  (up/down: scroll, enter: back to chat)", ""]
        for name, command in self.table.items():
            lines.append(f"{name}:\t{command.description}")
        return "\n".join(lines)

    def set_view(self, view: ViewMode, scroll: int | None) -> None:
        """show another view, scroll is where it starts and None keeps the position"""
        self.sv.ui.set_mode(view, scroll)
        self._buffer.notify()

    def change_role(self, role: str) -> None:
        """change the role of the current message"""
        self._buffer.change_msg(role)

    def remove_image(self) -> None:
        """remove images from current message"""
        with self._buffer.cond:
            if "images" in self._data["messages"][-1]:
                self._data["messages"][-1].pop("images")

    def add_image(self, *args: str) -> None:
        """add image to message -- argument: file name"""
        if len(args) < 1:
            return
        if self._data["messages"][-1]["role"] != "user":
            return

        try:
            with open(args[0], "rb") as file:
                img = base64.b64encode(file.read()).decode("utf-8")
            with self._buffer.cond:
                if "images" not in self._data["messages"][-1]:
                    self._data["messages"][-1]["images"] = []
                self._data["messages"][-1]["images"].append(img)

        except PermissionError:
            self.sv.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.sv.show_notification(f"File {args[0]} not found")

    def refresh_models(self) -> None:
        """fetch the model list from the server again"""
        self.sv.load_models()
        if not self.sv.report_models():
            self.sv.show_notification(f"loaded {len(self.sv.models_list)} models")

    def switch_model(self, *args: str) -> None:
        """switch model -- argument: model name"""
        if len(args) < 1:
            return
        with self._buffer.cond:
            self._data["model"] = args[0]
        self._buffer.notify()

    def import_chat(self, *args: str) -> None:
        """import chat from file -- argument: file name"""
        if len(args) < 1:
            return
        try:
            with open(args[0], "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    self._buffer.set_data(data)
                except json.JSONDecodeError:
                    self.sv.show_notification(f"File {args[0]} is corrupted")
        except PermissionError:
            self.sv.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.sv.show_notification(f"File {args[0]} not found")

    def export_chat(self, *args: str) -> None:
        """export chat from file -- argument: file name"""
        if len(args) < 1:
            return
        with self._buffer.cond:
            data = json.dumps(
                {
                    "visual": self._buffer.visual_data,
                    "chat": self._data,
                },
                indent=4,
            )
        try:
            with open(args[0], "w", encoding="utf-8") as file:
                file.write(data)
        except PermissionError:
            self.sv.show_notification(f"insufficient permissions for {args[0]}")
        except FileNotFoundError:
            self.sv.show_notification(f"File {args[0]} not found")

    def options(self, *args: str) -> None:
        """show or set options -- arguments: k|p|t|pen|stream value"""
        if len(args) == 0:
            self.sv.show_notification(json.dumps(self._buffer.options, indent=3))
        elif len(args) == 1:
            self.sv.show_notification("no data provided")
        else:
            match args[0]:
                case "k":
                    self._buffer.top_k = float(args[1])
                case "p":
                    self._buffer.top_p = float(args[1])
                case "t":
                    self._buffer.temperature = int(args[1])
                case "pen":
                    self._buffer.repeat_penalty = float(args[1])
                case "stream":
                    self._data["stream"] = int(args[1]) & 1 == 1
                case _:
                    self.sv.show_notification(f"command {args[0]} not found")

    def visual(self, *args: str) -> None:
        """show or set looks -- arguments: un|an|uc|ac|nc value"""
        if len(args) == 0:
            self.sv.show_notification(json.dumps(self._buffer.visual_data, indent=3))
        elif len(args) == 1:
            self.sv.show_notification("no data provided")
        else:
            try:
                match args[0]:
                    case "un":
                        self._buffer.user_name = args[1]
                    case "an":
                        self._buffer.ai_name = args[1]
                    case "uc":
                        self._buffer.user_color = args[1]
                    case "ac":
                        self._buffer.ai_color = args[1]
                    case "nc":
                        self._buffer.normal_color = args[1]
                    case _:
                        self.sv.show_notification(f"command {args[0]} not found")
            except ColorCodeException:
                self.sv.show_notification(f"invalid color code :{args[1]}")
