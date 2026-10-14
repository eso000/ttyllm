"""The built-in tools and the registry the model can call."""

import subprocess

from .tool import Tool, Parameter

toolchain: list[Tool] = []


def read_file(file_path: str) -> str:
    """return the contents of a file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def execute_shell_command(command: str) -> str:
    """run a shell command and return its stdout"""
    return subprocess.run(
        [command], shell=True, capture_output=True, text=True, check=False
    ).stdout


toolchain.append(
    Tool(
        "read_file",
        read_file,
        "reads the content of the file",
        [Parameter("file_path", "string", "Path to the file to be read.")],
    )
)
toolchain.append(
    Tool(
        "run_shell_command",
        execute_shell_command,
        "executes a Linux-shell command on the users machine and returns the output",
        [
            Parameter(
                "command", "string", "the full command to be executed on the machine"
            )
        ],
    )
)
