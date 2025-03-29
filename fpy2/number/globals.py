"""
Global variables for the `fpy2.number` module.
"""

from . import number
from . import real

from typing import Callable, Optional, TypeAlias, Union

# avoids circular dependency issues (useful for type checking)
Float: TypeAlias = 'number.Float'
RealFloat: TypeAlias = 'real.RealFloat'

# type of `Float` (or `RealFloat`) to `float` conversions
_FloatCvt: TypeAlias = Callable[[Union[Float, RealFloat]], float]

_current_float_converter: Optional[_FloatCvt] = None

def get_current_float_converter() -> _FloatCvt:
    """Gets the current `__float__` implementation for `Float`."""
    global _current_float_converter
    if _current_float_converter is None:
        raise RuntimeError('float converter not set')
    return _current_float_converter

def set_current_float_converter(cvt: _FloatCvt):
    """Sets the current `__float__` implementation for `Float`."""
    global _current_float_converter
    _current_float_converter = cvt
