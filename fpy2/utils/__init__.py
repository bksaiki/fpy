"""Common utilities for the FPy infrastructure."""

from .compare import CompareOp
from .defaults import default_repr, rcomparable
from .error import FPySyntaxError, raise_type_error
from .fraction import digits_to_fraction, hexnum_to_fraction
from .gensym import Gensym
from .identifier import Id, NamedId, UnderscoreId, SourceId
from .location import Location
from .ordering import Ordering
from .string import pythonize_id

def bitmask(k: int) -> int:
    """Return a bitmask of `k` bits."""
    return (1 << k) - 1
