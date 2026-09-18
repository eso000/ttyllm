import subprocess
from tool import Tool, Parameter

toolchain = []


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def execute_shell_command(command):
    return subprocess.run([command], shell=True, capture_output=True, text=True).stdout


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
