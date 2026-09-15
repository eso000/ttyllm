import subprocess
from tool import Tool, Parameter

toolchain = []


def temp(*args):
    return "19 degrees Celsius"


def execute_shell_command(command):
    return subprocess.run([command], shell=True, capture_output=True, text=True).stdout


toolchain.append(
    Tool(
        "get_temperature",
        temp,
        "Get the current temperature for a city",
        [Parameter("city", "string", "The name of the city")],
    )
)
toolchain.append(
    Tool(
        "run_shell_command",
        execute_shell_command,
        "executes a Linux-shell command on the users machine and returns the output",
        [
            Parameter(
                "command", "string", "the full comand to be executed on the machine"
            )
        ],
    )
)
