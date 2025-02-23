"""
This module defines a partial order type.
"""

from enum import IntEnum

class Ordering(IntEnum):
    """
    An enumeration to represent the result of a comparison.
    """
    LESS = -1
    EQUAL = 0
    GREATER = 1
