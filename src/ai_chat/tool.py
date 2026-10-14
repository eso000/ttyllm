"""Tool and parameter definitions sent to the model for tool calling."""

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class Parameter:
    """one argument of a tool function"""
    name: str
    type: str
    description: str
    required: bool = False

    @property
    def dict(self) -> dict[str, Any]:
        """JSON schema fragment describing this parameter"""
        return {"type": self.type, "description": self.description}


@dataclasses.dataclass
class Tool:
    """a callable exposed to the model, with schema metadata"""
    name: str
    function: Callable[..., Any]
    description: str
    parameters: list[Parameter]

    @property
    def dict(self) -> dict[str, Any]:
        """full tool definition in the OpenAI function calling format"""
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
