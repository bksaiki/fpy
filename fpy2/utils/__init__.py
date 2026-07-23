"""Common utilities for the FPy infrastructure."""

from .bits import bitmask, bits_to_float, float_to_bits, is_power_of_two, trailing_zeros
from .compare import CompareOp
from .decorator import default_repr, enum_repr, rcomparable
from .default import DEFAULT, DefaultOr
from .float_params import (
    FP64_BIAS,
    FP64_EMASK,
    FP64_EMAX,
    FP64_EMIN,
    FP64_EONES,
    FP64_ES,
    FP64_EXPMAX,
    FP64_EXPMIN,
    FP64_IMPLICIT1,
    FP64_M,
    FP64_MMASK,
    FP64_NBITS,
    FP64_P,
    FP64_SMASK,
)
from .fractions import (
    decnum_to_fraction,
    digits_to_fraction,
    hexnum_to_fraction,
    is_dyadic,
)
from .gensym import Gensym
from .identifier import Id, NamedId, SourceId, UnderscoreId
from .inspect import getfunclines, has_keyword
from .iterator import sliding_window
from .loader import get_module_source, install_caching_loader
from .location import Location
from .ordering import Ordering
from .string import pythonize_id
from .uninit import UNINIT
from .unionfind import Unionfind

# install custom loader
install_caching_loader()
