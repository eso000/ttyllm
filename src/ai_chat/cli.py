"""Command-line entry point: parse arguments and launch the chat."""

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """build the argument parser for the ``ai-chat`` command"""
    parser = argparse.ArgumentParser(
        prog="ai-chat",
        description="A minimalist terminal chat client for local LLMs.",
    )
    parser.add_argument(
        "-t",
        "--tools",
        action="store_true",
        help="enable tool calling (each call must be approved before it runs)",
    )
    parser.add_argument(
        "-o",
        "--open-api",
        action="store_true",
        help="use an OpenAI-compatible API (/v1/chat/completions)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """parse the command line and run the chat until the user quits"""
    args = build_parser().parse_args(argv)

    # imported lazily so --help/--version never touch the terminal layer
    from .buffer import Buffer
    from .supervisor import Supervisor

    if args.tools:
        Buffer.TOOLS = True

    Supervisor(open_api=args.open_api).run_chat()
    return 0
