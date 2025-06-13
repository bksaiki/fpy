"""Common utilities for the FPy infrastructure."""

from .bits import bitmask, float_to_bits, bits_to_float
from .compare import CompareOp
from .decorator import default_repr, rcomparable, enum_repr
from .error import FPySyntaxError, raise_type_error

from .float_params import (
    FP64_NBITS,
    FP64_ES,
    FP64_P,
    FP64_M,
    FP64_EMAX,
    FP64_EMIN,
    FP64_EXPMAX,
    FP64_EXPMIN,
    FP64_BIAS,
)

from .fractions import fraction, digits_to_fraction, decnum_to_fraction, hexnum_to_fraction
from .gensym import Gensym
from .identifier import Id, NamedId, UnderscoreId, SourceId
from .inspect import has_keyword
from .iterator import sliding_window
from .location import Location
from .ordering import Ordering
from .string import pythonize_id
