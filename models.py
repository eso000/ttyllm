import dataclasses


@dataclasses.dataclass
class Model:
    """a model as reported by the server"""
    name: str
    size_gb: int | None = None
    parameter_size: str | None = None
