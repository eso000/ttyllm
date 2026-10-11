"""API key handling: where the client gets the bearer token for the Authorization header."""

import os
from dataclasses import dataclass, field
from enum import Enum


class ApiKeyType(Enum):
    """where the API key comes from"""

    ENV = "env"  # the key names an environment variable holding the secret
    RAW = "raw"  # the key is the secret itself
    FILE = "file"  # the key is a path to a file containing the secret


@dataclass(frozen=True)
class ApiKey:
    """a named key source; get_key() returns the secret or None when unavailable"""

    key: str | None = field(repr=False)
    type: ApiKeyType

    def get_key(self) -> str | None:
        """return the API key for the Authorization header, or None when unavailable

        RAW keys are the secret itself, ENV keys name an environment variable
        and FILE keys name a file to read. Missing variables, missing or
        unreadable files, whitespace-only or empty sources, and keys with an
        embedded newline all resolve to None. The value is stripped so a
        trailing newline in a key file is harmless.
        """
        if self.key is None:
            return None
        match self.type:
            case ApiKeyType.RAW:
                value: str | None = self.key
            case ApiKeyType.ENV:
                value = os.environ.get(self.key)
            case ApiKeyType.FILE:
                try:
                    with open(os.path.expanduser(self.key), encoding="utf-8") as file:
                        value = file.read()
                except (OSError, UnicodeDecodeError):
                    value = None
        value = (value or "").strip()
        return (
            value
            if value and "\r" not in value and "\n" not in value
            else None
        )