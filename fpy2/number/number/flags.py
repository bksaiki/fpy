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
    - `overflow`: the result exceeded the representable range
    - `inexact`: the rounded result is not the same as the exact result
    - `carry`: the rounded result has a different exponent than the exact result
    """

    __slots__ = ("_flags",)

    _flags: int

    # bit positions for each flag
    _OVERFLOW = 1 << 0
    _INEXACT = 1 << 1
    _CARRY = 1 << 2

    def __init__(self, *, x=None, overflow=False, inexact=False, carry=False):
        if x is not None and not isinstance(x, Flags):
            raise TypeError("x must be a Flags instance")
        self._flags = 0

        if overflow:
            self._flags |= self._OVERFLOW
        elif x is not None:
            self._flags = x._flags & self._OVERFLOW

        if inexact:
            self._flags |= self._INEXACT
        elif x is not None:
            self._flags |= x._flags & self._INEXACT

        if carry:
            self._flags |= self._CARRY
        elif x is not None:
            self._flags |= x._flags & self._CARRY

    def __repr__(self) -> str:
        flag_strs: list[str] = []
        if self.overflow:
            flag_strs.append("overflow=True")
        if self.inexact:
            flag_strs.append("inexact=True")
        if self.carry:
            flag_strs.append("carry=True")
        return f"{self.__class__.__name__}({', '.join(flag_strs)})"

    @property
    def overflow(self) -> bool:
        """Overflow flag: the result exceeded the representable range."""
        return bool(self._flags & self._OVERFLOW)

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return bool(self._flags & self._INEXACT)

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return bool(self._flags & self._CARRY)

    def _set_overflow(self, value: bool) -> None:
            """Unsafe setter for overflow flag."""
            if value:
                self._flags |= self._OVERFLOW
            else:
                self._flags &= ~self._OVERFLOW

    def _set_inexact(self, value: bool) -> None:
        """Unsafe setter for inexact flag."""
        if value:
            self._flags |= self._INEXACT
        else:
            self._flags &= ~self._INEXACT

    def _set_carry(self, value: bool) -> None:
        """Unsafe setter for carry flag."""
        if value:
            self._flags |= self._CARRY
        else:
            self._flags &= ~self._CARRY

NO_FLAGS = Flags()
"""singleton instance representing no flags set"""
