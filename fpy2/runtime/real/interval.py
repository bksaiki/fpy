"""
This module defines intervals for the `RealInterpreter`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Self

from ...utils import default_repr

class Interval(ABC):
    """Abstract base class for intervals."""

    @abstractmethod
    def union(self, other: Self):
        """Union of two intervals."""
        raise NotImplementedError('virtual method')

@default_repr
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

@default_repr
class RealInterval(Interval):
    """Real interval."""
    lo: str
    hi: str
    prec: int

    def __init__(self, lo: str, hi: str, prec: int):
        self.lo = lo
        self.hi = hi
        self.prec = prec

    @staticmethod
    def from_val(val: str):
        return RealInterval(val, val, 53)

    def union(self, other: Self):
        raise NotImplementedError('unimplemented')
