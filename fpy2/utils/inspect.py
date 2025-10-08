"""
Python object inspection.

Extends the standard `inspect` module with additional utilities.
"""

import inspect

from typing import Callable

from .loader import get_module_source

def source_indent(lines: list[str]):
    return min(len(line) - len(line.lstrip()) for line in lines if line.strip() != '')

def getfunclines(func: Callable):
    """Get the source lines for a function `func`."""
    src_name = inspect.getabsfile(func)
    mod = inspect.getmodule(func)
    if mod is None:
        raise ValueError(f"cannot determine source module for function: {func}")

    lines = get_module_source(mod)
    if lines is None:
        # caching loader did not capture source
        # fallback to using `inspect`
        lines, start_line = inspect.getsourcelines(func)
    else:
        # find the function definition line
        start_line = func.__code__.co_firstlineno - 1
        lines = inspect.getblock(lines[start_line:])

    col_offset = source_indent(lines)
    return lines, src_name, start_line, col_offset


def has_keyword(func: Callable, name: str) -> bool:
    """Check if a function `func` has a keyword argument `name`."""
    sig = inspect.signature(func)
    return (
        name in sig.parameters or
        any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )
