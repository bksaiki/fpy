from dataclasses import dataclass
from typing import Any, Iterable
from types import CellType

@dataclass
class ForeignEnv:
    """Python environment of an FPy function."""
    globals: dict[str, Any]
    nonlocals: dict[str, CellType]
    builtins: dict[str, Any]

    @staticmethod
    def default():
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

    def copy(self) -> 'ForeignEnv':
        return ForeignEnv(self.globals.copy(), self.nonlocals.copy(), self.builtins.copy())

    def get(self, key, default=None) -> Any:
        """Like `get()` for `dict` instances."""
        if key in self.nonlocals:
            return self.nonlocals[key].cell_contents
        if key in self.globals:
            return self.globals[key]
        if key in self.builtins:
            return self.builtins[key]
        return default

    def merge(self, other: 'ForeignEnv', keys: Iterable[str] | None = None) -> 'ForeignEnv':
        """
        Merge two environments.

        Optionally, specify `keys` to restrict which keys to merge.
        """
        if keys is not None:
            keys = set(keys)

        globals = self.globals.copy()
        if keys is None:
            globals.update(other.globals)
        else:
            globals.update({k: v for k, v in other.globals.items() if k in keys})

        nonlocals = self.nonlocals.copy()
        if keys is None:
            nonlocals.update(other.nonlocals)
        else:
            nonlocals.update({k: v for k, v in other.nonlocals.items() if k in keys})

        builtins = self.builtins.copy()
        if keys is None:
            builtins.update(other.builtins)
        else:
            builtins.update({k: v for k, v in other.builtins.items() if k in keys})

        return ForeignEnv(globals, nonlocals, builtins)
