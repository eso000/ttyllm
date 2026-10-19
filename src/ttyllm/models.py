"""The model list returned by the server and how it is displayed."""

import dataclasses
from collections.abc import Iterator
from typing import Any

GIB = 1073741824


@dataclasses.dataclass
class Model:
    """a model as reported by the server"""

    name: str
    size_gb: int | None = None
    parameter_size: str | None = None


class ModelList:
    """the models a server offers, parsed from a response and formatted for display"""

    def __init__(self, models: list[Model]) -> None:
        """hold the parsed models"""
        self._models: list[Model] = models

    @classmethod
    def from_response(cls, data: Any) -> "ModelList":
        """build the list from an already decoded Ollama or OpenAI API response"""
        if not isinstance(data, dict):
            return cls([])
        if isinstance(data.get("data"), list):  # openai api: names only
            return cls(
                [Model(name=model["id"]) for model in data["data"] if "id" in model]
            )
        if isinstance(data.get("models"), list):  # ollama api
            parsed: list[Model] = []
            for model in data["models"]:
                if "name" not in model:
                    continue
                size = model.get("size")
                details = model.get("details") or {}
                parsed.append(
                    Model(
                        name=model["name"],
                        size_gb=size // GIB if size is not None else None,
                        parameter_size=details.get("parameter_size"),
                    )
                )
            return cls(parsed)
        return cls([])

    def __iter__(self) -> Iterator[Model]:
        """iterate over the models in the order the server listed them"""
        return iter(self._models)

    def __len__(self) -> int:
        """the number of models"""
        return len(self._models)

    @property
    def names(self) -> list[str]:
        """the bare model names, for tab completion and switching"""
        return [model.name for model in self._models]

    @property
    def short(self) -> str:
        """the model list as a printable block, one line per model"""
        if not self._models:
            return "no models available\n"
        return "".join(self._line(model) for model in self._models)

    @staticmethod
    def _line(model: Model) -> str:
        """one line of the model list, with size and parameters when known"""
        if model.size_gb is None:  # openai api provides names only
            return f"{model.name}\n"
        return (
            f"{model.name:<20}" f"\t {model.size_gb} GB" f"\t{model.parameter_size}\n"
        )
