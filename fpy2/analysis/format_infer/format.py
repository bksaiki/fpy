"""
Abstract number system.
"""

import math

from typing import TypeAlias

from ...number import RealFloat
from ...number.context.efloat import EFloatFormat
from ...number.context.exponential import ExpFormat
from ...number.context.fixed import FixedFormat
from ...number.context.format import Format
from ...number.context.mp_fixed import MPFixedFormat
from ...number.context.mp_float import MPFloatFormat
from ...number.context.mpb_fixed import MPBFixedFormat
from ...number.context.mpb_float import MPBFloatFormat
from ...number.context.mps_float import MPSFloatFormat
from ...number.context.real import REAL_FORMAT, RealFormat
from ...utils import default_repr

__all__ = [
    'AbstractFormat',
    'AbstractableFormat',
]

AbstractableFormat: TypeAlias = (
    RealFormat
    | MPFixedFormat | MPBFixedFormat
    | ExpFormat
    | MPFloatFormat | MPSFloatFormat | MPBFloatFormat
    | EFloatFormat
)
"""Union of :class:`Format` subclasses supported by :meth:`AbstractFormat.from_format`."""


def _maxval_precision(bound: RealFloat, exp: int) -> int:
    """
    Computes the precision of `bound` when represented
    with the exponent `exp`, i.e., the number of bits
    required to represent `c` where `bound = c * 2**exp`.
    """
    n = exp - 1
    bound = bound.normalize(n=n)
    return bound.c.bit_length()


@default_repr
class AbstractFormat:
    """
    Abstract number system.
    - `prec`: maximum precision (use float('inf') for unbounded)
    - `exp`: minimum unnormalized exponent (use float('-inf') for unbounded)
    - `pos_bound`: largest positive representable number (use float('inf') for unbounded)
    - `neg_bound`: largest negative representable number (use float('inf') for unbounded magnitude)
    """

    prec: int | float
    exp: int | float
    pos_bound: RealFloat | float
    neg_bound: RealFloat | float

    def __init__(
        self,
        prec: int | float,
        exp: int | float,
        bound: RealFloat | float,
        *,
        neg_bound: RealFloat | float | None = None,
    ):
        if prec <= 0:
            raise ValueError("`prec` must be positive.")

        self.prec = prec
        self.exp = exp
        self.pos_bound = bound
        self.neg_bound = -bound if neg_bound is None else neg_bound

    def __hash__(self):
        return hash((self.prec, self.exp, self.pos_bound, self.neg_bound))

    def __eq__(self, other):
        return (
            isinstance(other, AbstractFormat)
            and self.prec == other.prec
            and self.exp == other.exp
            and self.pos_bound == other.pos_bound
            and self.neg_bound == other.neg_bound
        )

    def __str__(self) -> str:
        return f'A({self.prec}, {self.exp}, +{str(self.pos_bound)}, {str(self.neg_bound)})'

    def __pos__(self) -> 'AbstractFormat':
        """Identity of the format."""
        return AbstractFormat(self.prec, self.exp, self.pos_bound, neg_bound=self.neg_bound)

    def __neg__(self) -> 'AbstractFormat':
        """Negation of the format (swaps positive and negative bounds)."""
        return AbstractFormat(self.prec, self.exp, -self.neg_bound, neg_bound=-self.pos_bound)

    def __abs__(self) -> 'AbstractFormat':
        """Absolute value of the format (clamps the negative bound to zero)."""
        return AbstractFormat(self.prec, self.exp, self.pos_bound, neg_bound=RealFloat.from_int(0))

    def __add__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Addition of two formats.

        Produces a format that can represent the sum of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # exponent: min(e1, e2)
        # bounds: b1 + b2
        exp = min(self.exp, other.exp)
        pos_bound = self.pos_bound + other.pos_bound
        neg_bound = self.neg_bound + other.neg_bound

        # compute precision based on bounds and exponent
        if isinstance(pos_bound, float) or isinstance(neg_bound, float):
            # precision must be unbounded since we need to represent
            # any sum of the form `+HUGE - quantum`
            prec = float('inf')
        elif isinstance(exp, float):
            # no subnormalization point means we need to represent
            # any sum of the form `+x - SMALL`
            prec = float('inf')
        else:
            # compute the magnitude of the largest bound
            max_bound = max(pos_bound, abs(neg_bound))

            # normalize the largest bound with the desired quantum
            # its precision is the required precision
            max_bound = max_bound.normalize(n=exp - 1)
            prec = max_bound.p

        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __sub__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Subtraction of two formats.

        Produces a format that can represent the difference of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # exponent: min(e1, e2)
        # bounds: b1 + b2
        exp = min(self.exp, other.exp)
        pos_bound = self.pos_bound - other.neg_bound
        neg_bound = self.neg_bound - other.pos_bound
        if isinstance(pos_bound, float) and math.isnan(pos_bound):
            pos_bound = float('inf')
        if isinstance(neg_bound, float) and math.isnan(neg_bound):
            neg_bound = float('-inf')

        # compute precision based on bounds and exponent
        if isinstance(pos_bound, float) or isinstance(neg_bound, float):
            # precision must be unbounded since we need to represent
            # any difference of the form `+HUGE - quantum`
            prec = float('inf')
        elif isinstance(exp, float):
            # no subnormalization point means we need to represent
            # any difference of the form `+x - SMALL`
            prec = float('inf')
        else:
            # compute the magnitude of the largest bound
            max_bound = max(pos_bound, abs(neg_bound))

            # normalize the largest bound with the desired quantum
            # its precision is the required precision
            max_bound = max_bound.normalize(n=exp - 1)
            prec = max_bound.p

        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)


    def __mul__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Multiplication of two formats.

        Produces a format that can represent the product of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # precision: p1 + p2
        # exponent: e1 + e2
        # bounds: b1 * b2
        prec = self.effective_prec() + other.effective_prec()
        exp = self.exp + other.exp
        pos_bound = max(self.pos_bound * other.pos_bound, self.neg_bound * other.neg_bound)
        neg_bound = max(self.pos_bound * other.neg_bound, self.neg_bound * other.pos_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __and__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Intersection of two formats."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec = min(self.prec, other.prec)
        exp = max(self.exp, other.exp)
        pos_bound = min(self.pos_bound, other.pos_bound)
        neg_bound = max(self.neg_bound, other.neg_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __or__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Union of two formats."""
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        prec = max(self.prec, other.prec)
        exp = min(self.exp, other.exp)
        pos_bound = max(self.pos_bound, other.pos_bound)
        neg_bound = min(self.neg_bound, other.neg_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __le__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        return self._is_contained_in(other)

    def __ge__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        return other._is_contained_in(self)

    @property
    def bound(self) -> RealFloat | float:
        """Maximum magnitude bound (pos or neg)."""
        return max(self.pos_bound, abs(self.neg_bound))

    @staticmethod
    def from_format(fmt: AbstractableFormat) -> 'AbstractFormat':
        """
        Constructs an :class:`AbstractFormat` that represents the same set of
        values as *fmt*.

        Partial: raises :class:`ValueError` when *fmt* is not one of the
        :class:`Format` subclasses listed in :data:`AbstractableFormat`.
        Callers should gate with ``isinstance(fmt, AbstractableFormat)``.
        """
        match fmt:
            case RealFormat():
                return AbstractFormat(
                    float('inf'),
                    float('-inf'),
                    float('inf'),
                    neg_bound=float('-inf'),
                )
            case FixedFormat() if not fmt.signed:
                neg_maxval = RealFloat.from_int(0)
                return AbstractFormat(
                    float('inf'), fmt.expmin, fmt.pos_maxval, neg_bound=neg_maxval
                )
            case MPBFixedFormat():
                return AbstractFormat(
                    float('inf'), fmt.expmin, fmt.pos_maxval, neg_bound=fmt.neg_maxval
                )
            case MPFixedFormat():
                return AbstractFormat(float('inf'), fmt.expmin, float('inf'))
            case ExpFormat():
                pos_maxval = fmt.maxval().as_real()
                neg_maxval = RealFloat.from_int(0)
                expmin = fmt.minval().exp
                return AbstractFormat(1, expmin, pos_maxval, neg_bound=neg_maxval)
            case EFloatFormat():
                return AbstractFormat(
                    fmt.pmax,
                    fmt.expmin,
                    fmt._mpb_fmt.pos_maxval,
                    neg_bound=fmt._mpb_fmt.neg_maxval,
                )
            case MPBFloatFormat():
                return AbstractFormat(
                    fmt.pmax, fmt.expmin, fmt.pos_maxval, neg_bound=fmt.neg_maxval
                )
            case MPSFloatFormat():
                return AbstractFormat(fmt.pmax, fmt.expmin, float('inf'))
            case MPFloatFormat():
                return AbstractFormat(fmt.pmax, float('-inf'), float('inf'))
            case _:
                raise ValueError(f'format is not abstractable: {fmt!r}')

    def format(self) -> Format:
        """
        Returns a :class:`Format` whose representable set is a (sound)
        superset of ``self``'s representable set.

        The mapping from abstract parameters to a concrete :class:`Format`
        is not unique; this method picks a canonical choice by parameter
        shape.  Fully-saturated abstract formats (all four parameters
        unbounded) collapse to ``REAL_FORMAT``.  When the parameter shape
        does not correspond cleanly to one of the supported :class:`Format`
        subclasses, ``REAL_FORMAT`` is returned as a sound fall-back.
        """
        prec_inf = isinstance(self.prec, float)
        exp_inf = isinstance(self.exp, float)
        pos_inf = isinstance(self.pos_bound, float)
        neg_inf = isinstance(self.neg_bound, float)
        bounds_bounded = not pos_inf and not neg_inf
        bounds_unbounded = pos_inf and neg_inf

        if prec_inf and exp_inf and bounds_unbounded:
            return REAL_FORMAT

        if not prec_inf and not exp_inf:
            assert isinstance(self.prec, int) and isinstance(self.exp, int)
            emin = self.exp + self.prec - 1
            if bounds_bounded:
                assert isinstance(self.pos_bound, RealFloat)
                assert isinstance(self.neg_bound, RealFloat)
                neg_maxval = self.neg_bound
                if not neg_maxval.s:
                    # MPBFloatFormat requires a strictly-negative neg_maxval;
                    # widen symmetrically (sound over-approximation).
                    neg_maxval = RealFloat(s=True, x=self.pos_bound)
                return MPBFloatFormat(self.prec, emin, self.pos_bound, neg_maxval)
            if bounds_unbounded:
                return MPSFloatFormat(self.prec, emin)

        if not prec_inf and exp_inf and bounds_unbounded:
            assert isinstance(self.prec, int)
            return MPFloatFormat(self.prec)

        if prec_inf and not exp_inf:
            assert isinstance(self.exp, int)
            nmin = self.exp - 1
            if bounds_bounded:
                assert isinstance(self.pos_bound, RealFloat)
                assert isinstance(self.neg_bound, RealFloat)
                return MPBFixedFormat(nmin, self.pos_bound, self.neg_bound)
            if bounds_unbounded:
                return MPFixedFormat(nmin)

        return REAL_FORMAT

    def effective_prec(self):
        """Effective maximum precision."""
        if isinstance(self.prec, float) and not isinstance(self.bound, float):
            # bounded fixed-point format
            assert not isinstance(self.exp, float)
            return _maxval_precision(self.bound, self.exp)

        if not isinstance(self.prec, float) and not isinstance(self.bound, float) and not isinstance(self.exp, float):
            # bounded floating-point format
            # check against the cutoff value
            cutoff = RealFloat(False, self.exp, 1 << self.prec)
            if self.bound < cutoff:
                # format acts like a fixed-point format
                return _maxval_precision(self.bound, self.exp)

        # everything else
        return self.prec

    def _is_contained_in(self, other: 'AbstractFormat') -> bool:
        """Return True iff every value representable by `self` is also representable by `other`.

        The three necessary and sufficient conditions are:
          1. Quantum: other.exp <= self.exp  (other is at least as fine-grained)
          2. Bounds:  other.pos_bound >= self.pos_bound  and  other.neg_bound <= self.neg_bound
          3. Precision: either other.prec >= self.prec, *or* self's entire range lies within
             other's subnormal region (pos_bound <= 2^(other.exp + other.prec)), in which case
             the floating-point precision of other is irrelevant — all values fit exactly.
        """

        # 1. quantum
        if other.exp > self.exp:
            return False
        # 2. bounds
        if other.pos_bound < self.pos_bound:
            return False
        if other.neg_bound > self.neg_bound:
            return False
        # 3. precision — only constraining when other has a finite normal region
        if not isinstance(other.prec, float) and not isinstance(other.exp, float):
            if self.prec > other.prec:
                # easy check failed: other's spacing in its normal region widens faster.
                # Containment still holds if self's bound stays within the region where
                # other's effective quantum is <= self's quantum 2^self.exp, i.e.,
                # pos_bound1 <= 2^(self.exp + other.prec).
                if not isinstance(self.exp, int):
                    return False
                cutoff = RealFloat(False, self.exp, 1 << other.prec)
                if isinstance(self.pos_bound, float) or self.pos_bound > cutoff:
                    return False
                if isinstance(self.neg_bound, float) or abs(self.neg_bound) > cutoff:
                    return False
        return True

    def contained_in(self, other: 'AbstractFormat') -> bool:
        """Check if this format is contained in another format."""
        return self._is_contained_in(other)

    def with_prec_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with precision adjusted by delta.

        Args:
            delta: Amount to add to precision (can be negative).
        Returns:
            New AbstractFormat with adjusted precision.
        """
        new_prec = self.prec + delta
        if new_prec < 1:
            raise ValueError("resulting precision must be at least 1")
        return AbstractFormat(new_prec, self.exp, self.pos_bound, neg_bound=self.neg_bound)

    def with_exp_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with exponent adjusted by delta.

        Args:
            delta: Amount to add to exponent (can be negative).
        Returns:
            New AbstractFormat with adjusted exponent.
        """
        new_exp = self.exp + delta
        return AbstractFormat(self.prec, new_exp, self.pos_bound, neg_bound=self.neg_bound)

    def with_bounds_scale(self, factor: RealFloat) -> 'AbstractFormat':
        """
        Return a new format with bounds scaled by factor.

        Args:
            factor: Factor to multiply bounds by (must be positive).
        Returns:
            New AbstractFormat with scaled bounds.
        """
        if factor <= 0:
            raise ValueError("Factor must be positive")

        # inf * positive = inf, so no need to check
        new_pos_bound = self.pos_bound * factor
        new_neg_bound = self.neg_bound * factor
        return AbstractFormat(self.prec, self.exp, new_pos_bound, neg_bound=new_neg_bound)
