"""
This module defines the `Flags` class which encapsulates
exceptional events when rounding.
"""

from typing import Self

__all__ = [
    'Flags',
]


# bit positions for each flag
_INVALID = 1 << 0
_DIVZERO = 1 << 1
_OVERFLOW = 1 << 2
_TINY_PRE = 1 << 3
_TINY_POST = 1 << 4
_INEXACT = 1 << 5
_CARRY = 1 << 6


class Flags:
    """
    The `Flags` class represents the status flags for arithmetic operations.
    - `invalid`: the operation produced an invalid result
    - `divzero`: the operation divided by zero
    - `overflow`: the result exceeded the representable range
    - `tiny_pre`: the result before rounding satisfies `|x| < 2^emin`
    - `tiny_post`: the result after rounding (without subnormalization) satisfies `|x| < 2^emin`
    - `inexact`: the rounded result is not the same as the exact result
    - `carry`: the rounded result has a different exponent than the exact result
    """

    __slots__ = ("_flags",)

    _flags: int

    def __init__(
        self, *,
        x: Self | None = None,
        invalid: bool | None = None,
        divzero: bool | None = None,
        overflow: bool | None = None,
        tiny_pre: bool | None = None,
        tiny_post: bool | None = None,
        inexact: bool | None = None,
        carry: bool | None = None
    ):
        # if `x` is provided, copy flags from `x`
        if x is not None:
            if invalid is None:
                invalid = bool(x._flags & _INVALID)
            if divzero is None:
                divzero = bool(x._flags & _DIVZERO)
            if overflow is None:
                overflow = bool(x._flags & _OVERFLOW)
            if tiny_pre is None:
                tiny_pre = bool(x._flags & _TINY_PRE)
            if tiny_post is None:
                tiny_post = bool(x._flags & _TINY_POST)
            if inexact is None:
                inexact = bool(x._flags & _INEXACT)
            if carry is None:
                carry = bool(x._flags & _CARRY)

        # set flags
        self._flags = 0
        if invalid:
            self._flags |= _INVALID
        if divzero:
            self._flags |= _DIVZERO
        if overflow:
            self._flags |= _OVERFLOW
        if tiny_pre:
            self._flags |= _TINY_PRE
        if tiny_post:
            self._flags |= _TINY_POST
        if inexact:
            self._flags |= _INEXACT
        if carry:
            self._flags |= _CARRY

    def __repr__(self) -> str:
        flag_strs: list[str] = []
        if self.invalid:
            flag_strs.append("invalid=True")
        if self.divzero:
            flag_strs.append("divzero=True")
        if self.overflow:
            flag_strs.append("overflow=True")
        if self.tiny_pre:
            flag_strs.append("tiny_pre=True")
        if self.tiny_post:
            flag_strs.append("tiny_post=True")
        if self.inexact:
            flag_strs.append("inexact=True")
        if self.carry:
            flag_strs.append("carry=True")
        return f"{self.__class__.__name__}({', '.join(flag_strs)})"

    @property
    def invalid(self) -> bool:
        """Invalid operation flag: the operation produced an invalid result."""
        return bool(self._flags & _INVALID)

    @property
    def divzero(self) -> bool:
        """Division by zero flag: the operation divided by zero."""
        return bool(self._flags & _DIVZERO)

    @property
    def overflow(self) -> bool:
        """Overflow flag: the result exceeded the representable range."""
        return bool(self._flags & _OVERFLOW)

    @property
    def tiny_pre(self) -> bool:
        """Tiny before rounding flag: the result before rounding satisfies `|x| < 2^emin`."""
        return bool(self._flags & _TINY_PRE)

    @property
    def tiny_post(self) -> bool:
        """
        Tiny after rounding flag: the result after rounding
        (without subnormalization) satisfies `|x| < 2^emin`.
        """
        return bool(self._flags & _TINY_POST)

    @property
    def inexact(self) -> bool:
        """Inexact flag: the rounded result is not the same as the exact result."""
        return bool(self._flags & _INEXACT)

    @property
    def carry(self) -> bool:
        """Carry flag: the rounded result has a different exponent than the exact result."""
        return bool(self._flags & _CARRY)

    @property
    def underflow_pre(self) -> bool:
        """Underflow before rounding flag: `self.tiny_pre and self.inexact`."""
        return self.tiny_pre and self.inexact

    @property
    def underflow_post(self) -> bool:
        """Underflow after rounding flag: `self.tiny_post and self.inexact`."""
        return self.tiny_post and self.inexact


    def _set_invalid(self, value: bool) -> None:
        """Unsafe setter for invalid operation flag."""
        if value:
            self._flags |= _INVALID
        else:
            self._flags &= ~_INVALID

    def _set_divzero(self, value: bool) -> None:
        """Unsafe setter for division by zero flag."""
        if value:
            self._flags |= _DIVZERO
        else:
            self._flags &= ~_DIVZERO

    def _set_overflow(self, value: bool) -> None:
        """Unsafe setter for overflow flag."""
        if value:
            self._flags |= _OVERFLOW
        else:
            self._flags &= ~_OVERFLOW

    def _set_tiny_pre(self, value: bool) -> None:
        """Unsafe setter for the tiny before rounding flag."""
        if value:
            self._flags |= _TINY_PRE
        else:
            self._flags &= ~_TINY_PRE

    def _set_tiny_post(self, value: bool) -> None:
        """Unsafe setter for the tiny after rounding flag."""
        if value:
            self._flags |= _TINY_POST
        else:
            self._flags &= ~_TINY_POST

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
