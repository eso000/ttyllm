"""Tab completion for command names, model names and file paths."""

import os
from collections.abc import Sequence
from glob import glob

MODEL_COMMANDS = (":sm", "/sm")
PATH_COMMANDS = (":e", "/e", ":i", "/i", ":image", "/image")


def _mark_directory(path: str) -> str:
    """append a slash to a directory so it completes to a prefix"""
    if not os.path.isdir(path) or path.endswith("/"):
        return path
    return path + "/"


def candidates(command: str, models: Sequence[str]) -> tuple[str, list[str]]:
    """the argument typed so far and everything it could be completed to"""
    tokens = command.split(" ")
    if len(tokens) < 2:
        return "", []
    typed = tokens[1]
    if tokens[0] in MODEL_COMMANDS:
        return typed, [name for name in models if name.startswith(typed)]
    if tokens[0] in PATH_COMMANDS:
        return typed, [_mark_directory(path) for path in glob(f"{typed}*")]
    return typed, []


def complete(command: str, models: Sequence[str]) -> str:
    """the text to append to the command line, empty when nothing can be completed"""
    typed, options = candidates(command, models)
    if not options:
        return ""
    return os.path.commonprefix(options)[len(typed) :]
