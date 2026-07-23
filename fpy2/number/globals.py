"""
Global variables for the `fpy2.number` module.
"""

from collections.abc import Callable
from typing import TypeAlias, Union

from . import number

# avoids circular dependency issues (useful for type checking)
Float: TypeAlias = 'number.Float'
RealFloat: TypeAlias = 'number.RealFloat'

# type of `Float` (or `RealFloat`) to `float` conversions.
# `Float`/`RealFloat` are runtime string forward-refs, so these aliases must use
# `Union[...]` (which accepts strings); PEP 604 `|` would raise at import time.
_FloatCvt: TypeAlias = Callable[[Union[Float, RealFloat]], float]  # noqa: UP007
_StrCvt: TypeAlias = Callable[[Union[Float, RealFloat]], str]  # noqa: UP007

_current_float_converter: _FloatCvt | None = None
_current_str_converter: _StrCvt | None = None


def get_current_float_converter() -> _FloatCvt:
    """Gets the current `__float__` implementation for `Float`."""
    if _current_float_converter is None:
        raise RuntimeError('float converter not set')
    return _current_float_converter

def set_current_float_converter(cvt: _FloatCvt):
    """Sets the current `__float__` implementation for `Float`."""
    global _current_float_converter
    _current_float_converter = cvt


def get_current_str_converter() -> _StrCvt:
    """Gets the current `__str__` implementation for `Float`."""
    if _current_str_converter is None:
        raise RuntimeError('str converter not set')
    return _current_str_converter

def set_current_str_converter(cvt: _StrCvt):
    """Sets the current `__str__` implementation for `Float`."""
    global _current_str_converter
    _current_str_converter = cvt
