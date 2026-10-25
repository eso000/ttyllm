"""Command-line entry point: parse arguments and launch the chat."""

import argparse
from collections.abc import Sequence

from . import __doc__ as DESCRIPTION, __version__
from .api_key import ApiKey, ApiKeyType


def build_parser() -> argparse.ArgumentParser:
    """build the argument parser for the ``ttyllm`` command"""
    parser = argparse.ArgumentParser(
        prog="ttyllm",
        description=DESCRIPTION,
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
        help="use an OpenAI-compatible API; its base path defaults to /v1",
    )
    parser.add_argument(
        "-H",
        "--endpoint",
        metavar="BASE_URL",
        help=(
            "server base URL, e.g. 127.0.0.1:11434 or "
            "https://openrouter.ai/api/v1 (https selects TLS; the path is kept)"
        ),
    )
    parser.add_argument(
        "-k",
        "--insecure",
        action="store_true",
        help="skip TLS certificate verification (for self-signed local servers)",
    )
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--api-key",
        metavar="KEY",
        help=(
            "API key sent in the Authorization header. Visible in process "
            "listings and shell history; prefer --api-key-env or --api-key-file."
        ),
    )
    key_group.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="name of an environment variable holding the API key",
    )
    key_group.add_argument(
        "--api-key-file",
        metavar="PATH",
        help="path to a file whose contents are the API key",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """parse the command line and run the chat until the user quits"""
    parser = build_parser()
    args = parser.parse_args(argv)

    # imported lazily so --help/--version never touch the terminal layer
    from .buffer import Buffer
    from .supervisor import Supervisor

    if args.tools:
        Buffer.TOOLS = True

    if args.api_key is not None:
        api_key: ApiKey | None = ApiKey(args.api_key, ApiKeyType.RAW)
    elif args.api_key_env is not None:
        api_key = ApiKey(args.api_key_env, ApiKeyType.ENV)
    elif args.api_key_file is not None:
        api_key = ApiKey(args.api_key_file, ApiKeyType.FILE)
    else:
        api_key = None

    try:
        supervisor = Supervisor(
            open_api=args.open_api,
            endpoint=args.endpoint,
            insecure=args.insecure,
            api_key=api_key,
        )
    except ValueError as error:
        parser.error(str(error))

    supervisor.run_chat()
    return 0
