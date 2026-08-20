"""
Runtime values of the FPy interpreter.

An FPy value is one of::

    value ::= bool                     # boolean
            | Float | Fraction        # real number
            | Context                  # rounding context
            | tuple[value, ...]        # tuple
            | list[value]              # list
            | Foreign                  # opaque Python value

:func:`to_value` classifies a Python object into this ADT at the
Python boundary; :func:`from_value` converts back.
"""

from fractions import Fraction
from typing import Any, TypeAlias

from ..number import FP64, INTEGER, REAL, Context, Float, RealFloat
from ..utils import UNINIT, is_dyadic

__all__ = [
    'Foreign',
    'RealValue',
    'ScalarValue',
    'Value',
    'from_value',
    'to_value',
]


class Foreign:
    """
    An opaque Python value inside an FPy program.

    FPy cannot operate on such a value: it may only be passed along,
    given to a rounding-context constructor, or read with `.attr`.
    """

    __slots__ = ('val',)
    val: Any

    def __init__(self, val: Any):
        self.val = val

    def __repr__(self):
        return f'Foreign({self.val!r})'

    # identity, not `==`: a foreign `__eq__` may be arbitrary
    # (e.g. numpy returns an array), and identity is total
    def __eq__(self, other):
        return isinstance(other, Foreign) and self.val is other.val

    def __hash__(self):
        return id(self.val)


RealValue: TypeAlias = Float | Fraction
"""Type of real values in FPy programs."""
ScalarValue: TypeAlias = bool | Context | RealValue
"""Type of scalar values in FPy programs."""
Value: TypeAlias = ScalarValue | list['Value'] | tuple['Value', ...] | Foreign
"""Type of values in FPy programs."""


def to_value(arg: Any) -> Value:
    """
    Converts a Python object crossing into FPy to a :data:`Value`.

    Idempotent: FPy values pass through unchanged. Containers are
    rebuilt; anything with no FPy form is wrapped as :class:`Foreign`.
    """
    match arg:
        case bool() | Float() | Fraction() | Context() | Foreign():
            return arg
        case RealFloat():
            return Float.from_real(arg, ctx=REAL)
        case int():
            return Float.from_int(arg, ctx=INTEGER, checked=False)
        case float():
            return Float.from_float(arg, ctx=FP64, checked=False)
        case tuple():
            return tuple(to_value(x) for x in arg)
        case list():
            return [to_value(x) for x in arg]
        case _ if arg is UNINIT:
            # `empty` placeholder: interpreter-internal, never foreign
            return arg
        case _:
            return Foreign(arg)


def _is_boundary_value(x) -> bool:
    match x:
        case Fraction():
            return not is_dyadic(x)
        case Foreign():
            return False
        case tuple() | list():
            return all(_is_boundary_value(v) for v in x)
        case _:
            return True


def _cvt_boundary(x: Value):
    match x:
        case Fraction():
            return Float.from_rational(x) if is_dyadic(x) else x
        case Foreign():
            return x.val
        case tuple():
            return tuple(from_value(v) for v in x)
        case list():
            return [from_value(v) for v in x]
        case _:
            return x


def from_value(x: Value):
    """
    Converts a :data:`Value` crossing out of FPy to a Python object.

    Dyadic rationals fold to :class:`Float`, :class:`Foreign` unwraps
    to its payload; containers are rebuilt only when needed.
    """
    return x if _is_boundary_value(x) else _cvt_boundary(x)
