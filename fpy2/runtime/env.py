from dataclasses import dataclass
from typing import Any

@dataclass
class ForeignEnv:
    """Python environment of an FPy function."""
    globals: dict[str, Any]
    nonlocals: dict[str, Any]
    builtins: dict[str, Any]

    @staticmethod
    def empty():
        return ForeignEnv({}, {}, {})

    def __contains__(self, key) -> bool:
        return key in self.globals or key in self.nonlocals or key in self.builtins

    def __getitem__(self, key) -> Any:
        if key in self.nonlocals:
            return self.nonlocals[key].cell_contents
        if key in self.globals:
            return self.globals[key]
        if key in self.builtins:
            return self.builtins[key]
        raise KeyError(key)

    def get(self, key, default=None) -> Any:
        """Like `get()` for `dict` instances."""
        if key in self.nonlocals:
            return self.nonlocals[key].cell_contents
        if key in self.globals:
            return self.globals[key]
        if key in self.builtins:
            return self.builtins[key]
        return default
