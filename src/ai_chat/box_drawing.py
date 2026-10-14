"""Text box layout, word wrapping and the ANSI primitives used to draw them."""

import dataclasses
import threading

_print_lock = threading.Lock()


@dataclasses.dataclass
class TextBox:
    """a block of wrapped lines together with a scroll offset"""
    lines: list[str] = dataclasses.field(default_factory=list)
    last: int = -1


def text_to_lines(text: str, width: int, color: str = "") -> list[str]:
    """wrap text into lines of at most width characters, breaking at spaces,
    each prefixed with color"""
    if text == "" or width <= 0:
        return [""]
    res: list[str] = []
    res_lines = text.replace("\t", "   ").split("\n")
    for line in res_lines:
        if len(line) <= width:
            res.append(f"{color}{line}")
        else:
            start = 0
            while start <= len(line) - 1:
                step = 0
                line_fitting_part = line[start : start + width]
                if start > 0:
                    spaces = len(line_fitting_part) - len(line_fitting_part.lstrip(" "))
                    start += spaces
                    line_fitting_part = line[start : start + width]
                if len(line_fitting_part) + start < len(line):
                    last_space = line_fitting_part.rfind(" ")
                    if last_space != -1:
                        number_of_selected_chars = last_space
                        step = 1
                    else:
                        number_of_selected_chars = len(line_fitting_part)
                else:
                    number_of_selected_chars = width
                res += [f"{color}{line_fitting_part[0:number_of_selected_chars]}"]
                start += number_of_selected_chars + step

    return res


def clean_screen() -> None:
    """clear the screen, the scrollback and park the cursor"""
    with _print_lock:
        print("\033[H\033[J\033[3J", end="", flush=True)


def make_text_box(text: str, width: int, last: int = -1, color: str = "") -> TextBox:
    """wrap text into a TextBox of the given width"""
    res = TextBox()
    res.last = last
    res.lines = text_to_lines(text, width, color)
    return res


def fit_to_box(
    box: TextBox,
    width: int,
    height: int,
    start_x: int = 1,
    start_y: int = 1,
    background: str = "",
    hide_cursor: bool = False,
) -> tuple[int, int]:
    """draw a TextBox inside a width x height region at the given offset, scrolling to last"""
    with _print_lock:
        # for x in range(height):

        lines, last = (box.lines, box.last)

        start = 0
        if last < 0:
            last = len(lines)
        print("\033[?25l", end="")

        if len(lines) <= height:
            last = len(lines)
        last = max(last, height)
        if last >= height:
            start = max(0, last - height)
        print(f"\033[{start_x};{start_y}H", end="", flush=True)

        for x in range(max(0, height - len(lines))):
            print(f"\033[{start_x+x+ len(lines)};{start_y}H", end="")
            print(width * " ", end="")

        for x, line in enumerate(lines[start:last]):
            print(f"\033[{start_x+x};{start_y}H", end="")
            print(f"{background}{width * ' '}", end="")
            print(f"\033[{start_x+x};{start_y}H", end="")
            print(
                line,
                end="",
            )

        print("\033[0m", end="", flush=True)

        if box.last != -1 or hide_cursor:
            print("\033[?25l", end="", flush=True)
        else:
            print("\033[?25h", end="", flush=True)

        if box.last == -1:
            return (len(lines), -1)
        return (len(lines), max(last, height))


def fit_text_to_box(
    text: str,
    width: int,
    height: int,
    start_x: int = 1,
    start_y: int = 1,
    color: str = "",
    background: str = "",
    hide_cursor: bool = False,
) -> tuple[int, int]:
    """wrap text into a box and draw it in one call"""
    return fit_to_box(
        make_text_box(text, width, color=color),
        width,
        height,
        start_x,
        start_y,
        background,
        hide_cursor,
    )
