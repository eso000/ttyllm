import re
import os
import json
import threading

from box_drawing import TextBox, text_to_lines, make_text_box


class ColorCodeExeption(Exception):
    pass


class Buffer:  # pylint: disable=R0902, R0904
    TOOLS = False
    OLLAMA = True

    def __init__(self):
        self.visual_data = {
            "ai_name": "AI",
            "user_name": "You",
            "ai_color": "\033[35m",
            "user_color": "\033[36m",
            "normal_color": "\033[37m",
        }
        self.in_request = False
        self.queued = False
        self.cancel = False
        self.updated = False
        self.pipe = os.pipe()

        self.tokens = -1
        self.cond = threading.Condition()

        self.chat_data = {
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
            self.chat_data["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "description": "Get the current temperature for a city",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["city"],
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The name of the city",
                                }
                            },
                        },
                    },
                }
            ]

        self.add_msg("system")

    @property
    def openai_data(self):

        def to_open_ai_msg(m):
            non_text_content = []
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

        data = {
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
    def ai_name(self):
        return self.visual_data["ai_name"]

    @property
    def user_name(self):
        return self.visual_data["user_name"]

    @property
    def ai_color(self):
        return self.visual_data["ai_color"]

    @property
    def user_color(self):
        return self.visual_data["user_color"]

    @property
    def normal_color(self):
        return self.visual_data["normal_color"]

    @property
    def options(self):
        if "options" in self.chat_data:
            return self.chat_data["options"]
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
        }

    @property
    def top_k(self):
        if "options" in self.chat_data:
            return self.chat_data["options"].get("top_k", 0)
        return self.chat_data.get("top_k", 0)

    @property
    def top_p(self):
        if "options" in self.chat_data:
            return self.chat_data["options"].get("top_p", 0)
        return self.chat_data.get("top_p", 0)

    @property
    def temperature(self):
        if "options" in self.chat_data:
            return self.chat_data["options"].get("temperature", 0)
        return self.chat_data.get("temperature", 0)

    @property
    def repeat_penalty(self):
        if "options" in self.chat_data:
            return self.chat_data["options"].get("repeat_penalty", 0)
        return self.chat_data.get("frequency_penalty", 0)

    @top_k.setter
    def top_k(self, top_k):
        if "options" in self.chat_data:
            self.chat_data["options"]["top_k"] = top_k
        self.chat_data["top_k"] = top_k

    @top_p.setter
    def top_p(self, top_p):
        if "options" in self.chat_data:
            self.chat_data["options"]["top_p"] = top_p
        self.chat_data["top_p"] = top_p

    @temperature.setter
    def temperature(self, temperature):
        if "options" in self.chat_data:
            self.chat_data["options"]["temperature"] = temperature
        self.chat_data["temperature"] = temperature

    @repeat_penalty.setter
    def repeat_penalty(self, repeat_penalty):
        if "options" in self.chat_data:
            self.chat_data["options"]["repeat_penalty"] = repeat_penalty
        self.chat_data["repeat_penalty"] = repeat_penalty

    @ai_name.setter
    def ai_name(self, new_name):
        self.visual_data["ai_name"] = new_name
        self.notify()

    @user_name.setter
    def user_name(self, new_name):
        self.visual_data["user_name"] = new_name
        self.notify()

    @ai_color.setter
    def ai_color(self, new_color):
        self.visual_data["ai_color"] = self._process_color(new_color)
        self.notify()

    @user_color.setter
    def user_color(self, new_color):
        self.visual_data["user_color"] = self._process_color(new_color)
        self.notify()

    @normal_color.setter
    def normal_color(self, new_color):
        self.visual_data["normal_color"] = self._process_color(new_color)
        self.notify()

    def cancel_request(self):
        self.cancel = True
        os.write(self.pipe[1], b"1")

    def empty(self):
        with self.cond:
            self.chat_data["messages"] = []
            self.add_msg("system")
            self.updated = True
            self.cond.notify()

    def del_msg(self, number=1):
        with self.cond:
            for _ in range(min(number, len(self.chat_data["messages"]))):
                curent_role = self.chat_data["messages"][-1]["role"]
                initial_role = self.chat_data["messages"][-1]["role"]
                while curent_role in (initial_role, "tool"):
                    self.chat_data["messages"].pop()
                    if len(self.chat_data["messages"]) == 0:
                        break
                    curent_role = self.chat_data["messages"][-1]["role"]

            self.updated = True
            self.cond.notify()

    def add_msg(self, role):
        with self.cond:
            self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def change_msg(self, role):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1]["content"] == "":
                self.chat_data["messages"][-1]["role"] = role
            else:
                self.chat_data["messages"].append({"role": role, "content": ""})
            self.updated = True
            self.cond.notify()

    def add_open(self, ch):
        with self.cond:
            if self.in_request:
                return
            if self.chat_data["messages"][-1].get("role") in (
                "user",
                "system",
                "assistant",
            ):
                self.chat_data["messages"][-1]["content"] += ch
                self.updated = True
                self.cond.notify()

    def del_open(self):
        with self.cond:
            if self.in_request:
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

    def clear_msg(self):
        with self.cond:
            self.chat_data["messages"][-1]["content"] = ""
            self.updated = True
            self.cond.notify()

    def add_ai_tool(self, tool_calls):
        with self.cond:
            if self.in_request:
                if self.chat_data["messages"][-1].get("role") == "assistant":
                    self.chat_data["messages"][-1]["tool_calls"] = tool_calls

    def add_ai(self, ch):
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
        raise ColorCodeExeption(f"{value} does not fit acceptable formats")

    def set_normal_color(self, value: str) -> None:
        self.normal_color = self._process_color(value)

    def set_ai_color(self, value: str) -> None:
        self.ai_color = self._process_color(value)

    def set_user_color(self, value: str) -> None:
        self.user_color = self._process_color(value)

    def set_data(self, data):
        def _check_dict(_dict, _key, _instance):
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
            self.updated = True
            self.cond.notify()

    def notify(self):
        with self.cond:
            self.updated = True
            self.cond.notify()

    def get_text_box_system(self, width, last=-1) -> str:
        result = TextBox()
        res = []
        res += [f"{self.normal_color}system:"]
        msg_ = self.chat_data["messages"][0]
        if msg_["content"] == "":
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

    def get_text_box_data(self, width, last=-1) -> str:
        return make_text_box(json.dumps(self.chat_data, indent=4), width, last=last)

    def get_text_box(self, width, last=-1) -> str:
        result = TextBox()
        res = []
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
        result.last = last
        result.lines = res
        return result

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

    def estimate_tokkens(self):
        estimate = 0
        for msg in self.chat_data["messages"]:
            estimate += len(msg.get("content", ""))
        return int(estimate / 4)
