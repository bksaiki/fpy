"""
This module defines intervals for the `RealInterpreter`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Self

from titanfp.titanic.digital import Digital

class Interval(ABC):
    """Abstract base class for intervals."""

    @abstractmethod
    def union(self, other: Self):
        """Union of two intervals."""
        raise NotImplementedError('virtual method')


class BoolInterval(Interval):
    """Boolean interval."""
    lo: bool
    hi: bool

    def __init__(self, lo: bool, hi: bool):
        self.lo = lo
        self.hi = hi

    @staticmethod
    def from_val(val: bool):
        return BoolInterval(val, val)

    def union(self, other: Self):
        if not isinstance(other, BoolInterval):
            raise TypeError(f'expected BoolInterval, got {other}')
        lo = self.lo and other.lo
        hi = self.hi or other.hi
        return BoolInterval(lo, hi)

    def as_bool(self) -> Optional[bool]:
        if self.lo == self.hi:
            return self.lo
        return None


class RealInterval(Interval):
    """Real interval."""
    lo: Digital
    hi: Digital

    def __init__(self, lo: Digital, hi: Digital):
        self.lo = lo
        self.hi = hi

    @staticmethod
    def from_val(val: Digital):
        return RealInterval(val, val)

    def union(self, other: Self):
        if not isinstance(other, RealInterval):
            raise TypeError(f'expected RealInterval, got {other}')
        lo = min(self.lo, other.lo)
        hi = max(self.hi, other.hi)
        return RealInterval(lo, hi)


