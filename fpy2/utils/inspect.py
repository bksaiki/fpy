"""
Python object inspection
"""

from typing import Callable

import inspect

def has_keyword(func: Callable, name: str) -> bool:
    """Check if a function `func` has a keyword argument `name`."""
    sig = inspect.signature(func)
    return (
        name in sig.parameters or
        any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )
