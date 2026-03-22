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
    - `carry`: the rounded result has a different exponent than the exact result
    """

    __slots__ = ("_flags",)

    _flags: int

    # bit positions for each flag
    _INEXACT = 1 << 0
    _CARRY = 1 << 1

    def __init__(self, *, x=None, inexact=False, carry=False):
        if x is not None and not isinstance(x, Flags):
            raise TypeError("x must be a Flags instance")
        self._flags = 0

        if inexact:
            self._flags |= self._INEXACT
        elif x is not None:
            self._flags = x._flags & self._INEXACT

        if carry:
            self._flags |= self._CARRY
        elif x is not None:
            self._flags |= x._flags & self._CARRY

    def __repr__(self) -> str:
        flag_strs: list[str] = []
        if self.inexact:
            flag_strs.append("inexact=True")
        if self.carry:
            flag_strs.append("carry=True")
        return f"{self.__class__.__name__}({', '.join(flag_strs)})"

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return bool(self._flags & self._INEXACT)

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return bool(self._flags & self._CARRY)

NO_FLAGS = Flags()
"""singleton instance representing no flags set"""
