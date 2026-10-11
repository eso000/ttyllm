import base64
import dataclasses
import inspect
import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from .buffer import Buffer, ColorCodeException
from .state import ViewMode

if TYPE_CHECKING:
    from supervisor import Supervisor


@dataclasses.dataclass(frozen=True)
class Command:
    """a colon command, its one line description, arguments and extra detail"""
    handler: Callable[..., None]
    description: str
    takes_args: bool = False
    detail: str | None = None


def _first_line(doc: str | None) -> str:
    """the summary line of a docstring, empty when there is none"""
    lines = inspect.cleandoc(doc or "").splitlines()
    return lines[0].strip() if lines else ""


def _rest_lines(doc: str | None) -> str | None:
    """the lines after the docstring summary, None when there are none"""
    rest = "\n".join(inspect.cleandoc(doc or "").splitlines()[1:]).strip()
    return rest or None


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
            "s": self._command(self.sv.split),
            "r": self._command(self.sv.redo),
            "R": self._command(self.sv.start_over),
            "rr": self._command(self.sv.undo),
            "bc": self._command(self.sv.create_buffer),
            "bn": self._command(self.sv.buffer_next),
            "bp": self._command(self.sv.buffer_previous),
            "bd": self._command(self.sv.buffer_delete),
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
            "endp": self._command(self.sv.set_endpoint, takes_args=True),
            "v": self._command(self.visual, takes_args=True),
            "o": self._command(self.options, takes_args=True),
            "image": self._command(self.add_image, takes_args=True),
            "rmi": self._command(self.remove_image),
            "-": self._command(self.sv.correct),
        }

    @staticmethod
    def _command(handler: Callable[..., None], takes_args: bool = False) -> Command:
        """a command described by the docstring of its handler"""
        return Command(
            handler,
            _first_line(handler.__doc__),
            takes_args,
            _rest_lines(handler.__doc__),
        )

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
        """build the help screen from the command descriptions and details"""
        lines = [
            "commands  (up/down: scroll, enter: back to chat)",
            "",
            "q:\tquit",  # handled by the input handler, not a table command
        ]
        details = []
        for name, command in self.table.items():
            lines.append(f"{name}:\t{command.description}")
            if command.detail is not None:
                details.append(f"{name}:\n{command.detail}")
        if details:
            lines.extend(["", "details:", "", "\n\n".join(details)])
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
            self.sv.show_notification(f"loaded {len(self.sv.models)} models")

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

    def _number(self, value: str, convert: Callable[[str], Any]) -> Any:
        """parse a numeric value, show a notification and give None when invalid"""
        try:
            return convert(value)
        except ValueError:
            self.sv.show_notification(f"invalid value: {value}")
            return None

    def options(self, *args: str) -> None:
        """show or set sampling options

        k: top_k (int)
        p: top_p (float)
        t: temperature (float)
        pen: repeat_penalty (float)
        stream: 0 or 1
        with no argument the current options are shown
        """
        if len(args) == 0:
            self.sv.show_notification(json.dumps(self._buffer.options, indent=3))
        elif len(args) == 1:
            self.sv.show_notification("no data provided")
        else:
            match args[0]:
                case "k":
                    if (top_k := self._number(args[1], int)) is not None:
                        self._buffer.top_k = top_k
                case "p":
                    if (top_p := self._number(args[1], float)) is not None:
                        self._buffer.top_p = top_p
                case "t":
                    if (temperature := self._number(args[1], float)) is not None:
                        self._buffer.temperature = temperature
                case "pen":
                    if (penalty := self._number(args[1], float)) is not None:
                        self._buffer.repeat_penalty = penalty
                case "stream":
                    if (stream := self._number(args[1], int)) is not None:
                        self._data["stream"] = stream & 1 == 1
                case _:
                    self.sv.show_notification(f"command {args[0]} not found")

    def visual(self, *args: str) -> None:
        """show or set names and colors

        un: user name
        an: assistant name
        uc: user color
        ac: assistant color
        nc: normal color
        color formats: 8b:N | 256b:N | rgb:R;G;B
        with no argument the current values are shown
        """
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
