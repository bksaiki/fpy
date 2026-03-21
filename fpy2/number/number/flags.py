"""
This module defines the `Flags` class which encapsulates
exceptional events when rounding.
"""

__all__ = [
    'Flags',
    'NO_FLAGS',
]

class Flags:
    """
    The `Flags` class represents the status flags for arithmetic operations.
    - `inexact`: the rounded result is not the same as the exact result
    - `away_zero`: the rounded result has greater magnitude than the exact result
    - `carry`: the rounded result has a different exponent than the exact result
    """

    __slots__ = ("_flags",)

    _flags: int

    # bit positions for each flag
    _INEXACT = 1 << 0
    _AWAY_ZERO = 1 << 1
    _CARRY = 1 << 2

    def __init__(self, *, x=None, inexact=False, away_zero=False, carry=False):
        if x is not None and not isinstance(x, Flags):
            raise TypeError("x must be a Flags instance")
        self._flags = 0

        if inexact:
            self._flags |= self._INEXACT
        elif x is not None:
            self._flags = x._flags & self._INEXACT

        if away_zero:
            self._flags |= self._AWAY_ZERO
        elif x is not None:
            self._flags |= x._flags & self._AWAY_ZERO

        if carry:
            self._flags |= self._CARRY
        elif x is not None:
            self._flags |= x._flags & self._CARRY

    def __repr__(self) -> str:
        flag_strs: list[str] = []
        if self.inexact:
            flag_strs.append("inexact=True")
        if self.away_zero:
            flag_strs.append("away_zero=True")
        if self.carry:
            flag_strs.append("carry=True")
        return f"{self.__class__.__name__}({', '.join(flag_strs)})"

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return bool(self._flags & self._INEXACT)

    @property
    def away_zero(self) -> bool:
        """Away-from-zero flag: the rounded result has greater magnitude than the exact result."""
        return bool(self._flags & self._AWAY_ZERO)

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return bool(self._flags & self._CARRY)

NO_FLAGS = Flags()
"""singleton instance representing no flags set"""
