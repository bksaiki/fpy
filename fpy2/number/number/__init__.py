"""
This module defines number types.
"""

from fractions import Fraction

from .floats import Float, same_value
from .reals import RNG, RealFloat

__all__ = [
    'RNG',
    'Float',
    'Real',
    'RealFloat'
]

Real = int | float | Float | Fraction
