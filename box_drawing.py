import dataclasses


@dataclasses.dataclass
class TextBox:
    lines: list = dataclasses.field(default_factory=list)
    last = -1


def text_to_lines(text, width, color=""):
    if text == "":
        return [""]
    res = []
    res_lines = text.replace("\t", "   ").split("\n")
    for line in res_lines:
        if len(line) <= width:
            res.append(f"{color}{line}")
        else:
            start = 0
            while start <= len(line) - 1:
                step = 0
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
    print("\033[H\033[J\033[3J", end="", flush=True)


def make_text_box(text, width, last=-1, color="") -> TextBox:
    res = TextBox()
    res.last = last
    res.lines = text_to_lines(text, width, color)
    return res


def fit_to_box(
    box, width, height, start_x=1, start_y=1, background="", hide_cursor=False
):
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
        print(f"{background}\033[{start_x+x+ len(lines)};{start_y}H", end="")
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
    text,
    width,
    height,
    start_x=1,
    start_y=1,
    color="",
    background="",
    hide_cursor=False,
):
    return fit_to_box(
        make_text_box(text, width, color=color),
        width,
        height,
        start_x,
        start_y,
        background,
        hide_cursor,
    )
