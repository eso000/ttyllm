# ai-chat

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

A terminal chat client for Ollama and OpenAI-compatible servers, using nothing
but the Python standard library.

I decided to build an AI chat TUI with no dependencies. With no HTTP library,
[`requester.py`](src/ai_chat/requester.py) builds requests and parses the
responses by hand: headers, chunked bodies, Ollama's NDJSON and OpenAI's SSE
streams, TLS. The TUI is a handful of threads talking through condition
variables, and if you enable tools the model can read files or run shell
commands, but only after you look at the call and press `y`.

```sh
pip install -e .
ai-chat                 # chat with a local Ollama server on 127.0.0.1:11434
```

## Features

- **Streaming replies** drawn live in a box-drawn TUI.
- **Multiple buffers.** Fork, create, switch and delete independent chats.
  Requests are queued and run one at a time.
- **Revert and regenerate.** Undo your last message, regenerate the last reply,
  or remove the last message.
- **Approval-gated tool calls** (opt-in with `-t`): read a file or run a shell
  command, each only after you press `y`. See the
  [security note](#tool-calling-and-safety).
- **Ollama and OpenAI-compatible backends** (`-o`), including `https://`
  endpoints and self-signed certificates (`-k`).
- **API keys** sent as a bearer token, read from an environment variable, a file
  or the command line.
- **Models and sampling.** List models from the server, switch per buffer, and
  tune `top_k`, `top_p`, temperature and repeat penalty.
- **Images.** Attach a base64-encoded image to the current message.
- **Import and export** a chat as JSON.
- **Vim-style command mode** (`:` or `/`) with tab completion of model names and
  file paths.

## Requirements

- Python **3.10+**
- A POSIX terminal (uses `termios`, ANSI escape codes and the alternate screen)
- A running LLM server, for example [Ollama](https://ollama.com) on
  `127.0.0.1:11434`, which is the default

## Usage

```sh
ai-chat                          # Ollama on the default endpoint
ai-chat -o                       # OpenAI-compatible API (/v1/chat/completions)
ai-chat -t                       # enable tool calling (asks before running anything)
ai-chat -H http://127.0.0.1:8080 # another server; https:// selects TLS
ai-chat -H https://local:8443 -k # https with a self-signed certificate
ai-chat -o -H https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY
ai-chat --help                   # every option
```

No install needed: `python3 .` from a checkout takes the same options.

Type a message and press <kbd>Enter</kbd>. Start a line with `:` or `/` for
command mode. `:q` quits and `:h` opens the in-app help.

The API key goes in an `Authorization: Bearer` header. Use `--api-key-env` or
`--api-key-file` rather than `--api-key`, which is visible in process listings
and shell history. If the key cannot be resolved, no header is sent and the
server will answer 401.

### Tool calling and safety

With `-t`, the model can call two built-in tools: `read_file` and
`run_shell_command`. The shell tool runs arbitrary commands with your user's
permissions.

> **⚠️ Every call is shown with its arguments and runs only after you press
> <kbd>y</kbd>.** Read each command before approving it. Use `-t` only with
> models and prompts you trust, ideally inside a sandbox or container.

### Key bindings

| Key | Action |
| --- | ------ |
| <kbd>Enter</kbd> | send message or command, or leave a view |
| <kbd>Ctrl+S</kbd> | fork the current buffer |
| <kbd>Ctrl+B</kbd> | new blank buffer |
| <kbd>Ctrl+D</kbd> | delete current buffer |
| <kbd>Ctrl+N</kbd> / <kbd>Ctrl+P</kbd> | next / previous buffer |
| <kbd>Ctrl+R</kbd> | regenerate last reply |
| <kbd>Ctrl+E</kbd> | undo last user message |
| <kbd>Ctrl+Q</kbd> | open or close the command line |
| <kbd>Ctrl+H</kbd> | help |
| <kbd>Ctrl+U</kbd> | clear the input |
| <kbd>Ctrl+C</kbd> | stop the running reply, or decline pending tool calls |
| <kbd>Tab</kbd> | complete a model name (`:sm`) or a file path (`:i`, `:e`, `:image`) |
| <kbd>↑</kbd> / <kbd>↓</kbd> | scroll |
| <kbd>←</kbd> / <kbd>→</kbd> | jump to the end / top |
| <kbd>y</kbd> / <kbd>n</kbd> | approve / deny a tool call |

### Common commands

`:h` lists every command with its details. The ones you will use most:

| Command | Description |
| ------- | ----------- |
| `:m` / `:mr` | list models / refresh the list |
| `:sm <model>` | switch model for this buffer |
| `:endp <url or host[:port]>` | show or set the server endpoint |
| `:o k\|p\|t\|pen\|stream <value>` | sampling options (top_k, top_p, temperature, repeat_penalty, stream) |
| `:s` / `:bc` / `:bn` / `:bp` / `:bd` | fork / create / next / previous / delete buffer |
| `:r` / `:rr` / `:R` / `-` | regenerate / undo / start over / remove last message |
| `:image <file>` / `:rmi` | attach / remove an image |
| `:i <file>` / `:e <file>` | import / export the chat as JSON |
| `:q` | quit |

## How it works

Four threads share the work: one reads the keyboard, one turns submitted lines
into commands or messages, one talks to the server, and one redraws the screen.
They hand off through condition variables, and each buffer owns a small pipe
that the socket reader `select`s on alongside the socket, which is how `Ctrl+C`
stops a reply mid-stream. Every chat is kept in the Ollama request format and
converted on the way out for OpenAI-compatible servers, so rendering, import and
export only ever deal with one shape. Tool calls are reassembled from the
stream, shown to you, and only the ones you approve get run before the
conversation continues.

The details, including the raw HTTP layer and the locking rules, are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Status and limitations

This is an alpha-quality tool for local use, and its narrow scope is deliberate.

- It targets **Ollama** and **OpenAI-compatible** servers only. It is not a
  general HTTP client: no redirects, no `Content-Encoding` such as gzip, no
  keep-alive, no chunk extensions.
- Success is detected from the exact status line `HTTP/1.1 200 OK` rather than
  the numeric code, so a server that answers differently is treated as an error.
- Connections are **IPv4 only**.
- **POSIX only.** It needs `termios` and ANSI escape codes.
- Tool calling is opt-in and manually approved, but the shell tool is not
  sandboxed.

## Development

```sh
pip install -e ".[dev]"
black --check src tests
mypy
python3 -m unittest discover -s tests
```

The tests use only loopback sockets and need no network. Without installing, run
`PYTHONPATH=src python3 -m unittest discover -s tests`. CI runs the same three
checks on every push.

Source files are listed by layer in the
[component map](ARCHITECTURE.md#component-map).

## License

[MIT](LICENSE)
