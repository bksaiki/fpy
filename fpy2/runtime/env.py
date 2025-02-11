
from dataclasses import dataclass
from typing import Any

@dataclass
class PythonEnv:
    """Python environment of an FPy function."""
    globals: dict[str, Any]
    nonlocals: dict[str, Any]

    @staticmethod
    def empty():
        return PythonEnv({}, {})

    def __contains__(self, key) -> bool:
        return key in self.globals or key in self.nonlocals
