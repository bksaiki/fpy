"""
Global variables for the `fpy2.number` module.
"""

from collections.abc import Callable

from . import number

# `Float` (or `RealFloat`) to `float`/`str` conversions; `type` is evaluated
# lazily, so `number.Float` need not exist yet when this module loads
type _FloatCvt = Callable[[number.Float | number.RealFloat], float]
type _StrCvt = Callable[[number.Float | number.RealFloat], str]

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
