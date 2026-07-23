"""
This module defines number types.
"""

from fractions import Fraction
from typing import TypeAlias

from .floats import Float
from .reals import RNG, RealFloat

__all__ = [
    'RNG',
    'Float',
    'Real',
    'RealFloat'
]

Real: TypeAlias = int | float | Float | Fraction
