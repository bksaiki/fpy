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


    @staticmethod
    def from_compare(x, y):
        """
        Convert the result of a comparison to an Ordering.
        """
        if x < y:
            return Ordering.LESS
        elif x > y:
            return Ordering.GREATER
        elif x == y:
            return Ordering.EQUAL
        else:
            raise ValueError(f"cannot compare {x} and {y}")

    def reverse(self):
        """
        Reverse the ordering.
        """
        if self == Ordering.LESS:
            return Ordering.GREATER
        elif self == Ordering.GREATER:
            return Ordering.LESS
        else:
            return Ordering.EQUAL
