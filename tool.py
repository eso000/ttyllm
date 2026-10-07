import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class Parameter:
    name: str
    type: str
    description: str
    required: bool = False

    @property
    def dict(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description}


class Tool:
    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        parameters: list[Parameter],
    ) -> None:
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters

    @property
    def dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [param for param in self.parameters if param.required],
                    "properties": {param.name: param.dict for param in self.parameters},
                },
            },
        }
