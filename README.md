# ai_chat

A minimalist terminal chat client for local LLMs, built with **zero external
dependencies** — only the Python standard library.

I created this project as a goal to see how far a chat client can be pushed
without pulling in a single third-party package. To keep everything
transparent, the HTTP layer is implemented on **raw sockets**: requests are
assembled and responses (headers, chunked transfer encoding, NDJSON and SSE
streams) are parsed by hand in
[`src/ai_chat/requester.py`](src/ai_chat/requester.py), so there is no
HTTP library standing between you and the wire protocol.

## Features

- **No dependencies** — runs on a stock Python 3.10+ installation.
- **Raw socket HTTP** — hand-rolled requests, header parsing, chunked
  decoding, and streaming for both Ollama (`application/x-ndjson`) and
  OpenAI-compatible (`text/event-stream`) APIs.
- **Streaming replies** rendered live in a box-drawn TUI.
- **Multiple buffers** — split, create, switch, and delete independent chats;
  requests are queued and processed one at a time.
- **Tool calls** (opt-in) with per-buffer `y`/`n` approval before execution —
  including shell commands, so see the [security note](#usage) under Usage.
- **Images** — attach base64-encoded images to the current message.
- **Model management** — list models from the server, switch models per
  buffer, tune sampling options.
- **Configurable endpoint** — point the client at any host with the
  `-H/--endpoint` flag or the `:endp` command, including `https://` URLs. The
  scheme is kept, `https://` selects TLS, and `-k/--insecure` allows a
  self-signed local certificate.
- **Import / export** chats as JSON.
- **Vim-style command mode** with tab completion (`:`, `/`).

## Requirements

- Python **3.10+**
- A POSIX terminal (uses `termios`, ANSI escape codes, and the alternate
  screen buffer)
- A running LLM server, e.g. [Ollama](https://ollama.com) on
  `127.0.0.1:11434` (the default)

## Usage

Run straight from a checkout (no install needed) with `python3 .`:

```sh
python3 .            # chat against the default Ollama endpoint
python3 . -o         # use an OpenAI-compatible API (/v1/chat/completions)
python3 . -t         # enable tool calling (asks for approval before running)
python3 . -H http://127.0.0.1:8080   # point at another server (https selects TLS)
python3 . -H https://local:8443 -k   # https with a self-signed certificate
python3 . --version  # print the version
python3 . --help     # show all options
```

Optionally install it to get the `ai-chat` command (and `python -m ai_chat`):

```sh
pip install -e .
ai-chat
```

Type a message and press <kbd>Enter</kbd> to send. Start a line with `:` or
`/` to enter command mode; `:q` quits, `:h` shows the in-app help.

> **⚠️ Shell command execution:** When tool calling is enabled (`-t`), the
> built-in `run_shell_command` tool can execute arbitrary commands on your
> machine with your user's permissions. Every tool call is shown with its
> arguments and must be explicitly approved with <kbd>y</kbd> before it runs,
> but you should read each command carefully before approving it. Enable `-t`
> only with models and prompts you trust, and ideally run it inside a sandbox
> or container.

### Key bindings

| Key                    | Action                       |
| ---------------------- | ---------------------------- |
| <kbd>Ctrl+S</kbd>      | split: fork current buffer   |
| <kbd>Ctrl+B</kbd>      | create new blank buffer      |
| <kbd>Ctrl+D</kbd>      | delete current buffer        |
| <kbd>Ctrl+N</kbd> / <kbd>Ctrl+P</kbd> | next / previous buffer |
| <kbd>Ctrl+R</kbd>      | regenerate last reply        |
| <kbd>Ctrl+E</kbd>      | undo last user message       |
| <kbd>Ctrl+Q</kbd>      | toggle command mode          |
| <kbd>Ctrl+H</kbd>      | show help                    |
| <kbd>Ctrl+U</kbd>      | clear input                  |
| <kbd>Ctrl+C</kbd>      | interrupt request            |
| <kbd>Tab</kbd>         | complete command / path      |
| <kbd>↑</kbd> / <kbd>↓</kbd> | scroll view             |
| <kbd>y</kbd> / <kbd>n</kbd> | approve / deny a tool call |

### Commands

Run with `:h` (or see the table below) inside the app:

| Command | Description |
| ------- | ----------- |
| `:m` / `:mr` | list models / refresh model list |
| `:sm <model>` | switch model |
| `:endp <host>[:port] [port]` | show the current endpoint, or set it (`host:port`, two args, or an `http(s)://` URL) |
| `:n` / `:d` / `:S` | show chat / raw data / system prompt |
| `:s` / `:bc` / `:bn` / `:bp` / `:bd` | split / create / next / previous / delete buffer |
| `:r` / `:R` / `:rr` / `-` | regenerate / start over / revert last user message / remove last message |
| `:ai` / `:us` / `:sys` | change current message role |
| `:o k\|p\|t\|pen\|stream <value>` | set sampling options (top_k, top_p, temperature, repeat_penalty, stream) |
| `:v ...` | show or set names and colors |
| `:image <file>` / `:rmi` | attach / remove an image |
| `:i <file>` / `:e <file>` | import / export chat (JSON) |
| `:h` | help (commands and key bindings) |
| `:q` | quit |

## Development

Run the test suite:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests
```

### Project layout

| File                     | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| `__main__.py`            | runs `ai-chat` from a checkout (`python3 .`)          |
| `pyproject.toml`         | package metadata, build config, `ai-chat` entry point |
| `src/ai_chat/cli.py`     | command-line arguments and the `main()` entry point   |
| `src/ai_chat/__main__.py`| runs `main()` for `python -m ai_chat`                 |
| `src/ai_chat/supervisor.py` | buffers and their operations, request scheduling, threads |
| `src/ai_chat/input_handler.py` | keystrokes to commands, line edits, scrolling, approvals |
| `src/ai_chat/completion.py` | tab completion of model names and file paths      |
| `src/ai_chat/commands.py` | the `:` / `/` command table and the help screen     |
| `src/ai_chat/request_queue.py` | serialises reply requests, one buffer at a time |
| `src/ai_chat/terminal.py` | raw input mode and alternate screen handling        |
| `src/ai_chat/state.py`   | `UIState`: locked view mode, scroll, command line, notification |
| `src/ai_chat/view.py`    | renders the buffer for the current `UIState`         |
| `src/ai_chat/buffer.py`  | per-chat state, messages, and rendering              |
| `src/ai_chat/requester.py` | raw-socket HTTP client and streaming parsers       |
| `src/ai_chat/models.py`  | `Model` entries and `ModelList`, the parsed and formatted model list |
| `src/ai_chat/box_drawing.py` | text box layout and wrapping                     |
| `src/ai_chat/tool.py` / `src/ai_chat/toolchain.py` | tool definitions and available tools |
| `tests/`                 | unit tests (unittest, no extra dependencies)         |
