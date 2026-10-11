import contextlib
import sys
import termios
from collections.abc import Iterator

from .box_drawing import clean_screen


@contextlib.contextmanager
def raw_terminal() -> Iterator[None]:
    """switch to the alternate screen and raw input, restore both on exit"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[0] &= ~(termios.IXON)
    new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    print("\033[?1049h", end="", flush=True)
    clean_screen()
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        clean_screen()
        print("\033[?25h", end="", flush=True)
        print("\033[?1049l", end="", flush=True)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
