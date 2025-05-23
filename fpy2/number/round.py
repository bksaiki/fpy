"""
This module defines rounding utilities.
"""

from enum import Enum

from ..utils import enum_repr

@enum_repr
class RoundingDirection(Enum):
    """Rounding directions for finite-precision operations."""
    RTZ = 0
    """rounding towards zero"""
    RAZ = 1
    """rounding away from zero"""
    RTE = 2
    """rounding towards even"""
    RTO = 3
    """rounding towards odd"""

@enum_repr
class RoundingMode(Enum):
    """Rounding modes for finite-precision operations."""
    RNE = 0
    """round to nearest, ties to even"""
    RNA = 1
    """round to nearest, ties away from zero"""
    RTP = 2
    """round towards positive infinity"""
    RTN = 3
    """round towards negative infinity"""
    RTZ = 4
    """round towards zero"""
    RAZ = 5
    """round away from zero"""
    RTO = 6
    """round towards odd"""
    RTE = 7
    """round towards even"""

    def to_direction(self, s: bool) -> tuple[bool, RoundingDirection]:
        """Converts to a `nearest` flag and `RoundingDirection`."""
        match (s, self):
            case (_, RoundingMode.RNE):
                return True, RoundingDirection.RTE
            case (_, RoundingMode.RNA):
                return True, RoundingDirection.RAZ
            case (True, RoundingMode.RTP):
                return False, RoundingDirection.RTZ
            case (False, RoundingMode.RTP):
                return False, RoundingDirection.RAZ
            case (True, RoundingMode.RTN):
                return False, RoundingDirection.RAZ
            case (False, RoundingMode.RTN):
                return False, RoundingDirection.RTZ
            case (_, RoundingMode.RTZ):
                return False, RoundingDirection.RTZ
            case (_, RoundingMode.RAZ):
                return False, RoundingDirection.RAZ
            case (_, RoundingMode.RTO):
                return False, RoundingDirection.RTO
            case (_, RoundingMode.RTE):
                return False, RoundingDirection.RTE
            case _:
                raise ValueError('unsupported rounding mode', self)
