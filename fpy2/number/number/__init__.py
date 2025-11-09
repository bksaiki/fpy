"""
This module defines number types.
"""

from fractions import Fraction
from typing import TypeAlias

from .reals import RealFloat
from .floats import Float

__all__ = [
    'Real',
    'Float',
    'RealFloat'
]

Real: TypeAlias = int | float | Float | Fraction
