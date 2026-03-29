"""
This module defines the `Flags` class which encapsulates
exceptional events when rounding.
"""

from typing import Self

__all__ = [
    'Flags',
]


# bit positions for each flag
_OVERFLOW = 1 << 0
_INEXACT = 1 << 1
_CARRY = 1 << 2


class Flags:
    """
    The `Flags` class represents the status flags for arithmetic operations.
    - `overflow`: the result exceeded the representable range
    - `inexact`: the rounded result is not the same as the exact result
    - `carry`: the rounded result has a different exponent than the exact result
    """

    __slots__ = ("_flags",)

    _flags: int

    def __init__(
        self, *,
        x: Self | None = None,
        overflow: bool | None = None,
        inexact: bool | None = None,
        carry: bool | None = None
    ):
        # if `x` is provided, copy flags from `x`
        if x is not None:
            if overflow is None:
                overflow = bool(x._flags & _OVERFLOW)
            if inexact is None:
                inexact = bool(x._flags & _INEXACT)
            if carry is None:
                carry = bool(x._flags & _CARRY)

        # set flags
        self._flags = 0
        if overflow:
            self._flags |= _OVERFLOW
        if inexact:
            self._flags |= _INEXACT
        if carry:
            self._flags |= _CARRY

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
        return bool(self._flags & _OVERFLOW)

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return bool(self._flags & _INEXACT)

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return bool(self._flags & _CARRY)

    def _set_overflow(self, value: bool) -> None:
            """Unsafe setter for overflow flag."""
            if value:
                self._flags |= _OVERFLOW
            else:
                self._flags &= ~_OVERFLOW

    def _set_inexact(self, value: bool) -> None:
        """Unsafe setter for inexact flag."""
        if value:
            self._flags |= _INEXACT
        else:
            self._flags &= ~_INEXACT

    def _set_carry(self, value: bool) -> None:
        """Unsafe setter for carry flag."""
        if value:
            self._flags |= _CARRY
        else:
            self._flags &= ~_CARRY
