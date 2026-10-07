import re
import os
import json
import threading
from typing import Any

from box_drawing import TextBox, text_to_lines, make_text_box
import toolchain


class ColorCodeException(Exception):
    pass


class Buffer:  # pylint: disable=R0902, R0904
    TOOLS: bool = False
    OLLAMA: bool = True

    def __init__(self) -> None:
        self.visual_data: dict[str, str] = {
            "ai_name": "AI",
            "user_name": "You",
            "ai_color": "\033[35m",
            "user_color": "\033[36m",
            "normal_color": "\033[37m",
        }
        self.in_request: bool = False
        self.queued: bool = False
        self.cancel: bool = False
        self.pending_calls: list[dict[str, Any]] = []
        self.tool_depth: int = 0
        self.updated: bool = False
        self.pipe: tuple[int, int] = os.pipe()
        os.set_blocking(self.pipe[0], False)

        self.tokens: int = -1
        self.cond: threading.Condition = threading.Condition()

        self.chat_data: dict[str, Any] = {
            "model": "llama3.2",  # mistral-nemo",
            "stream": True,
            "think": False,
            "messages": [],
            "options": {
                "top_k": 30,
                "top_p": 0.9,
                "temperature": 0.8,
                "repeat_penalty": 1.1,
            },
        }
        if self.TOOLS:
            self.chat_data["tool_choice"] = "auto"
            self.chat_data["tools"] = [tool.dict for tool in toolchain.toolchain]

        self.add_msg("system")

    def __del__(self) -> None:
        for fd in self.pipe:
            os.close(fd)

    @property
    def openai_data(self) -> dict[str, Any]:

        def to_open_ai_msg(m: dict[str, Any]) -> Any:
            non_text_content: list[dict[str, Any]] = []
            if "images" in m:
                for img in m["images"]:
                    non_text_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            if len(non_text_content) != 0:
                non_text_content.append(
                    {"type": "text", "text": m["content"]},
                )
                return non_text_content
            return m["content"]

        data: dict[str, Any] = {
            "model": self.chat_data["model"],
            "stream": self.chat_data["stream"],
            "think": self.chat_data["think"],
            "messages": [
                {"role": msg["role"], "content": to_open_ai_msg(msg)}
                for msg in self.chat_data["messages"]
            ],
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "frequency_penalty": self.repeat_penalty,
            "stream_options": {"inclide_usage": True},
        }
        if self.TOOLS:
            data["tool_choice"] = "auto"
            data["tools"] = self.chat_data["tools"]
        return data

    @property
    def ai_name(self) -> str:
        return self.visual_data["ai_name"]

    @ai_name.setter
    def ai_name(self, new_name: str) -> None:
        self.visual_data["ai_name"] = new_name
        self.notify()

    @property
    def user_name(self) -> str:
        return self.visual_data["user_name"]

    @user_name.setter
    def user_name(self, new_name: str) -> None:
        self.visual_data["user_name"] = new_name
        self.notify()

    @property
    def ai_color(self) -> str:
        return self.visual_data["ai_color"]

    @ai_color.setter
    def ai_color(self, new_color: str) -> None:
        self.visual_data["ai_color"] = self._process_color(new_color)
        self.notify()

    @property
    def user_color(self) -> str:
        return self.visual_data["user_color"]

    @user_color.setter
    def user_color(self, new_color: str) -> None:
        self.visual_data["user_color"] = self._process_color(new_color)
        self.notify()

    @property
    def normal_color(self) -> str:
        return self.visual_data["normal_color"]

    @normal_color.setter
    def normal_color(self, new_color: str) -> None:
        self.visual_data["normal_color"] = self._process_color(new_color)
        self.notify()

    @property
    def options(self) -> dict[str, Any]:
        if "options" in self.chat_data:
            return self.chat_data["options"]
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
        }

    @property
    def top_k(self) -> Any:
        if "options" in self.chat_data:
            return self.chat_data["options"].get("top_k", 0)
        return self.chat_data.get("top_k", 0)

    @top_k.setter
    def top_k(self, top_k: Any) -> None:
        with self.cond:
            if "options" in self.chat_data:
                self.chat_data["options"]["top_k"] = top_k
            self.chat_data["top_k"] = top_k

    @property
    def top_p(self) -> Any:
        if "options" in self.chat_data:
            return self.chat_data["options"].get("top_p", 0)
        return self.chat_data.get("top_p", 0)

    @top_p.setter
    def top_p(self, top_p: Any) -> None:
        with self.cond:
            if "options" in self.chat_data:
                self.chat_data["options"]["top_p"] = top_p
            self.chat_data["top_p"] = top_p

    @property
    def temperature(self) -> Any:
        if "options" in self.chat_data:
            return self.chat_data["options"].get("temperature", 0)
        return self.chat_data.get("temperature", 0)

    @temperature.setter
    def temperature(self, temperature: Any) -> None:
        with self.cond:
            if "options" in self.chat_data:
                self.chat_data["options"]["temperature"] = temperature
            self.chat_data["temperature"] = temperature

    @property
    def repeat_penalty(self) -> Any:
        if "options" in self.chat_data:
            return self.chat_data["options"].get("repeat_penalty", 0)
        return self.chat_data.get("frequency_penalty", 0)

    @repeat_penalty.setter
    def repeat_penalty(self, repeat_penalty: Any) -> None:
        with self.cond:
            if "options" in self.chat_data:
                self.chat_data["options"]["repeat_penalty"] = repeat_penalty
            self.chat_data["repeat_penalty"] = repeat_penalty

    def cancel_request(self) -> None:
        with self.cond:
            if not self.in_request:
                return
            self.cancel = True
            os.write(self.pipe[1], b"1")

    def clear_cancel(self) -> None:
        with self.cond:
            self.cancel = False
            try:
                while os.read(self.pipe[0], 64):
                    pass
            except BlockingIOError:
                pass

    @property
    def awaiting_approval(self) -> bool:
        with self.cond:
            return any(call["approved"] is None for call in self.pending_calls)

    def set_pending(self, calls: list[dict[str, Any]]) -> None:
        """calls: dicts with id, name, args, approved (None = undecided), result"""
        with self.cond:
            self.pending_calls = calls
            self.updated = True
            self.cond.notify()

    def pending_resolved(self) -> bool:
        with self.cond:
            return len(self.pending_calls) != 0 and not self.awaiting_approval

    def decide(self, approved: bool) -> bool:
        """settle the first undecided call, True when none are left undecided"""
        with self.cond:
            for call in self.pending_calls:
                if call["approved"] is None:
                    call["approved"] = approved
                    break
            self.updated = True
            self.cond.notify()
            return not self.awaiting_approval

    def take_pending(self) -> list[dict[str, Any]]:
        with self.cond:
            calls = self.pending_calls
            self.pending_calls = []
            self.in_request = True
            return calls

    def abort_pending(self) -> bool:
        """decline everything that is still pending, True if there was any"""
        with self.cond:
            if not self.pending_calls or self.in_request:
                return False
            for call in self.pending_calls:
                self.chat_data["messages"].append(
                    {
                        "role": "tool",
                        "id": call["id"],
                        "tool_name": call["name"],
                        "content": call["result"] or "user declined call",
                    }
                )
            self.pending_calls = []
            self.add_msg("user")
            return True

    def empty(self) -> None:
        with self.cond:
            self.pending_calls = []
            self.chat_data["messages"] = []
            self.add_msg("system")
            self.updated = True
            self.cond.notify()

    def del_msg(self, number: int = 1) -> None:
        with self.cond:
            self.pending_calls = []
            for _ in range(min(number, len(self.chat_data["messages"]))):
                current_role = self.chat_data["messages"][-1]["role"]
                initial_role = self.chat_data["messages"][-1]["role"]
                while current_role in (initial_role, "tool"):
                    self.chat_data["messages"].pop()
                    if len(self.chat_data["messages"]) == 0:
                        break
                    current_role = self.chat_data["messages"][-1]["role"]

            self.updated = True
            self.cond.notify()

    def add_msg(self, role: str) -> None:
        with self.cond:
            self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def change_msg(self, role: str) -> None:
        with self.cond:
            if self.in_request or self.pending_calls:
                return
            if self.chat_data["messages"][-1]["content"] == "":
                self.chat_data["messages"][-1]["role"] = role
            else:
                self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def currently_typed(self) -> tuple[bool, bool]:
        with self.cond:
            last = self.chat_data["messages"][-1]
            return (
                not (self.in_request or self.queued or self.pending_calls),
                len(last.get("content", "")) == 0,
            )

    def add_open(self, ch: str) -> None:
        with self.cond:
            if self.in_request or self.pending_calls:
                return
            if self.chat_data["messages"][-1].get("role") in (
                "user",
                "system",
                "assistant",
            ):
                self.chat_data["messages"][-1]["content"] += ch
                self.updated = True
                self.cond.notify()

    def del_open(self) -> None:
        with self.cond:
            if self.in_request or self.pending_calls:
                return
            if self.chat_data["messages"][-1].get("role") in (
                "user",
                "system",
                "assistant",
            ):
                self.chat_data["messages"][-1]["content"] = self.chat_data["messages"][
                    -1
                ]["content"][:-1]
            self.updated = True
            self.cond.notify()

    def clear_msg(self) -> None:
        with self.cond:
            self.chat_data["messages"][-1]["content"] = ""
            self.updated = True
            self.cond.notify()

    def add_ai_tool(self, tool_calls: list[dict[str, Any]]) -> None:
        with self.cond:
            if self.in_request:
                if self.chat_data["messages"][-1].get("role") == "assistant":
                    self.chat_data["messages"][-1]["tool_calls"] = tool_calls

    def add_ai(self, ch: str) -> None:
        with self.cond:
            if self.in_request:
                if self.chat_data["messages"][-1].get("role") == "assistant":
                    self.chat_data["messages"][-1]["content"] += ch
                    self.updated = True
                    self.cond.notify()

    def _process_color(self, value: str) -> str:
        pattern_8bit = re.compile("^8b:[0-9]$")
        pattern_256bit = re.compile("^256b:[0-9]{1,3}$")
        pattern_rgb = re.compile("^rgb:[1-9]{1,3};[0-9]{1,3};[0-9]{1,3}$")
        if pattern_8bit.match(value):
            return f"\033[3{value[-1]}m"
        if pattern_256bit.match(value):
            return f"\033[38;5;{value[5:]}m"
        if pattern_rgb.match(value):
            return f"\033[38;2;{value[4:]}m"
        raise ColorCodeException(f"{value} does not fit acceptable formats")

    def set_normal_color(self, value: str) -> None:
        self.normal_color = self._process_color(value)

    def set_ai_color(self, value: str) -> None:
        self.ai_color = self._process_color(value)

    def set_user_color(self, value: str) -> None:
        self.user_color = self._process_color(value)

    def set_data(self, data: dict[str, Any]) -> None:
        def _check_dict(_dict: dict[str, Any], _key: str, _instance: type) -> bool:
            if _key not in _dict:
                return True
            if not isinstance(_dict[_key], _instance):
                return True
            return False

        if "visual" in data:
            self.visual_data.update(data["visual"])
        if "chat" in data:
            data = data["chat"]
        if _check_dict(data, "model", str):
            return
        if _check_dict(data, "messages", list):
            return
        for msg in data["messages"]:
            if _check_dict(msg, "role", str):
                return
            if not _check_dict(msg, "content", str) and not _check_dict(
                msg, "tool_calls", list
            ):
                return
        with self.cond:
            self.chat_data = data
            self.pending_calls = []
            self.updated = True
            self.cond.notify()

    def notify(self) -> None:
        with self.cond:
            self.updated = True
            self.cond.notify()

    def get_text_box_system(self, width: int, last: int = -1) -> TextBox:
        result = TextBox()
        res: list[str] = []
        res += [f"{self.normal_color}system:"]
        msg_ = self.chat_data["messages"][0]
        if msg_["content"] == "" or width <= 0:
            res += [""]
        else:
            res_lines = msg_["content"].split("\n")
            for line in res_lines:
                if line == "":
                    res.append("")
                else:
                    res += [
                        f"{self.normal_color}{line[i : i + width]}"
                        for i in range(0, len(line), width)
                    ]
        result.last = last
        result.lines = res
        return result

    def get_text_box_data(self, width: int, last: int = -1) -> TextBox:
        return make_text_box(json.dumps(self.chat_data, indent=4), width, last=last)

    def get_text_box(self, width: int, last: int = -1) -> TextBox:
        result = TextBox()
        res: list[str] = []
        if (
            len(self.chat_data["messages"]) == 1
            and self.chat_data["messages"][0]["role"] == "system"
        ):
            res += [f"{self.normal_color}system:"]
            msg_ = self.chat_data["messages"][0]
            res += text_to_lines(msg_["content"], width, self.normal_color)
            result.last = last
            result.lines = res
            return result

        for msg_ in self.chat_data["messages"]:
            if msg_["role"] in ("system", "tool") or "tool_calls" in msg_:
                continue
            color = self.user_color if msg_["role"] == "user" else self.ai_color
            name = self.user_name if msg_["role"] == "user" else self.ai_name
            res += [f"{self.normal_color}{name}:"]
            res += text_to_lines(msg_["content"], width, color)
        res += self._pending_lines(width)
        result.last = last
        result.lines = res
        return result

    def _pending_lines(self, width: int) -> list[str]:
        warn = "\033[33m"
        res: list[str] = []
        with self.cond:
            total = len(self.pending_calls)
            for number, call in enumerate(self.pending_calls, 1):
                if call["approved"] is None:
                    status = "approval required"
                elif call["result"]:
                    status = "rejected"
                else:
                    status = "approved" if call["approved"] else "declined"
                res += text_to_lines(
                    f"tool call {number} of {total}: {call['name']} ({status})",
                    width,
                    warn,
                )
                for key, value in call["args"].items():
                    # ensure_ascii escapes control characters, no terminal tricks
                    res += text_to_lines(
                        f"  {key}: {json.dumps(value)}", width, self.normal_color
                    )
                if call["approved"] is None:
                    break
            if self.awaiting_approval:
                res += text_to_lines(
                    "[y] approve  [n] decline  [ctrl-c] decline all", width, warn
                )
        return res

    def get_string(self) -> str:
        res = ""
        for msg_ in self.chat_data["messages"]:
            if msg_["role"] == "user":
                res += f"{self.normal_color}{self.user_name}\n"
                res += f"{self.user_color}{msg_["content"]}\n"
            if msg_["role"] == "assistant":
                res += f"{self.normal_color}{self.ai_name}\n"
                res += f"{self.ai_color}{msg_["content"]}\n"
        return res

    def estimate_tokens(self) -> int:
        with self.cond:
            estimate = 0
            for msg in self.chat_data["messages"]:
                estimate += len(msg.get("content", ""))
        return int(estimate / 4)

    def set_tokens(self, tokens: int) -> None:
        with self.cond:
            self.tokens = tokens
