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

_NAMES = (
    ('invalid', _INVALID),
    ('divzero', _DIVZERO),
    ('overflow', _OVERFLOW),
    ('tiny_pre', _TINY_PRE),
    ('tiny_post', _TINY_POST),
    ('inexact', _INEXACT),
    ('carry', _CARRY),
)


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
    ) -> None:
        self._flags = 0 if x is None else x._flags
        values = (invalid, divzero, overflow, tiny_pre, tiny_post, inexact, carry)
        for (_, mask), value in zip(_NAMES, values, strict=True):
            if value is not None:
                self._set(mask, value)

    def __repr__(self) -> str:
        set_flags = ', '.join(f'{name}=True' for name, mask in _NAMES if self._flags & mask)
        return f'{self.__class__.__name__}({set_flags})'

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

    def _set(self, mask: int, value: bool) -> None:
        """Unsafe setter for the flag at *mask*."""
        if value:
            self._flags |= mask
        else:
            self._flags &= ~mask

    def _set_invalid(self, value: bool) -> None:
        self._set(_INVALID, value)

    def _set_divzero(self, value: bool) -> None:
        self._set(_DIVZERO, value)

    def _set_overflow(self, value: bool) -> None:
        self._set(_OVERFLOW, value)

    def _set_tiny_pre(self, value: bool) -> None:
        self._set(_TINY_PRE, value)

    def _set_tiny_post(self, value: bool) -> None:
        self._set(_TINY_POST, value)

    def _set_inexact(self, value: bool) -> None:
        self._set(_INEXACT, value)
