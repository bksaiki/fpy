"""
Abstract number system.
"""

import math
from typing import TypeAlias

from ...number import Float, RealFloat
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
    - `pos_bound`: largest positive *finite* representable number (use float('inf') for unbounded)
    - `neg_bound`: largest negative *finite* representable number (use float('inf') for unbounded magnitude)
    - `has_pos_inf`: whether `+inf` is a representable value
    - `has_neg_inf`: whether `-inf` is a representable value
    - `has_nan`: whether `NaN` is a representable value
    - `has_neg_zero`: whether `-0.0` is a representable value

    A special-value flag is independent of the corresponding bound: e.g.
    `pos_bound = inf` means the finite values are unbounded, not that `+inf` is a
    member (that is `has_pos_inf`).
    """

    prec: int | float
    exp: int | float
    pos_bound: RealFloat | float
    neg_bound: RealFloat | float
    has_pos_inf: bool
    has_neg_inf: bool
    has_nan: bool
    has_neg_zero: bool

    def __init__(
        self,
        prec: float,
        exp: float,
        bound: RealFloat | float,
        *,
        neg_bound: RealFloat | float | None = None,
        has_pos_inf: bool = False,
        has_neg_inf: bool = False,
        has_nan: bool = False,
        has_neg_zero: bool = False,
    ):
        if prec <= 0:
            raise ValueError("`prec` must be positive.")

        self.prec = prec
        self.exp = exp
        self.pos_bound = bound
        self.neg_bound = -bound if neg_bound is None else neg_bound
        self.has_pos_inf = has_pos_inf
        self.has_neg_inf = has_neg_inf
        self.has_nan = has_nan
        self.has_neg_zero = has_neg_zero

    def __hash__(self):
        return hash((
            self.prec, self.exp, self.pos_bound, self.neg_bound,
            self.has_pos_inf, self.has_neg_inf, self.has_nan, self.has_neg_zero,
        ))

    def __eq__(self, other):
        return (
            isinstance(other, AbstractFormat)
            and self.prec == other.prec
            and self.exp == other.exp
            and self.pos_bound == other.pos_bound
            and self.neg_bound == other.neg_bound
            and self.has_pos_inf == other.has_pos_inf
            and self.has_neg_inf == other.has_neg_inf
            and self.has_nan == other.has_nan
            and self.has_neg_zero == other.has_neg_zero
        )

    def __str__(self) -> str:
        specials = ','.join(
            tag for flag, tag in (
                (self.has_pos_inf, '+inf'),
                (self.has_neg_inf, '-inf'),
                (self.has_nan, 'nan'),
                (self.has_neg_zero, '-0'),
            ) if flag
        )
        base = f'A({self.prec}, {self.exp}, +{self.pos_bound}, {self.neg_bound}'
        return f'{base}, S={{{specials}}})' if specials else f'{base})'

    def __pos__(self) -> 'AbstractFormat':
        """Identity of the format."""
        return AbstractFormat(
            self.prec, self.exp, self.pos_bound, neg_bound=self.neg_bound,
            has_pos_inf=self.has_pos_inf, has_neg_inf=self.has_neg_inf, has_nan=self.has_nan,
            has_neg_zero=self.has_neg_zero,
        )

    def __neg__(self) -> 'AbstractFormat':
        """Negation of the format (swaps positive and negative bounds).

        ``has_neg_zero`` carries over unchanged.  Negation maps ``+0.0`` to
        ``-0.0``, and every format represents a ``+0.0`` -- ``pos_bound >= 0 >=
        neg_bound`` holds by convention and nothing excludes zero -- so the image
        holds a ``-0.0`` exactly when this number system has one at all.  A
        system without: ``-(0)`` under ``SINT8`` is ``+0.0``, since
        two's-complement has a single zero.
        """
        # negation maps +inf <-> -inf; NaN is unsigned so it is preserved
        return AbstractFormat(
            self.prec, self.exp, -self.neg_bound, neg_bound=-self.pos_bound,
            has_pos_inf=self.has_neg_inf, has_neg_inf=self.has_pos_inf, has_nan=self.has_nan,
            has_neg_zero=self.has_neg_zero,
        )

    def __abs__(self) -> 'AbstractFormat':
        """Absolute value of the format (clamps the negative bound to zero)."""
        # abs maps -inf to +inf, so +inf is present if either infinity was.
        # `has_neg_zero` is left at its default: `abs` never yields a negative
        # zero, so false is the derived answer here, not an omission.
        return AbstractFormat(
            self.prec, self.exp, self.pos_bound, neg_bound=RealFloat.from_int(0),
            has_pos_inf=self.has_pos_inf or self.has_neg_inf, has_neg_inf=False, has_nan=self.has_nan,
        )

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
            # ``bit_length()`` is 0 for an exactly-zero ``max_bound``
            # (both pos and neg bounds are 0).  Clamp to the minimum
            # positive precision; the resulting format represents only
            # ``{0}``, but ``prec`` itself must be valid.
            prec = max(max_bound.p, 1)

        # special values: +inf + x = +inf, -inf + x = -inf, +inf + -inf = NaN
        has_pos_inf = self.has_pos_inf or other.has_pos_inf
        has_neg_inf = self.has_neg_inf or other.has_neg_inf
        has_nan = (
            self.has_nan or other.has_nan
            or (self.has_pos_inf and other.has_neg_inf)
            or (self.has_neg_inf and other.has_pos_inf)
        )
        # A sum is `-0.0` only when *both* addends are: IEEE-754 gives a
        # like-signed sum that sign, and every other zero sum is `+0.0`.
        has_neg_zero = self.has_neg_zero and other.has_neg_zero

        return AbstractFormat(
            prec, exp, pos_bound, neg_bound=neg_bound,
            has_pos_inf=has_pos_inf, has_neg_inf=has_neg_inf, has_nan=has_nan,
            has_neg_zero=has_neg_zero,
        )

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
            # its precision is the required precision; clamp to >=1
            # so an exactly-zero difference still has valid ``prec``.
            max_bound = max_bound.normalize(n=exp - 1)
            prec = max(max_bound.p, 1)

        # special values: subtraction is addition with `other` negated, so
        # other's +inf/-inf swap roles; +inf - +inf and -inf - -inf give NaN
        has_pos_inf = self.has_pos_inf or other.has_neg_inf
        has_neg_inf = self.has_neg_inf or other.has_pos_inf
        has_nan = (
            self.has_nan or other.has_nan
            or (self.has_pos_inf and other.has_pos_inf)
            or (self.has_neg_inf and other.has_neg_inf)
        )
        # `a - b` is `a + (-b)`, so by the sum rule both addends must be able to
        # be `-0.0`: `a` directly, and `-b` when *b* can be `+0.0`.  Every format
        # represents a `+0.0` -- `pos_bound >= 0 >= neg_bound` holds by convention
        # and nothing excludes zero -- so only `a` is in question.
        has_neg_zero = self.has_neg_zero

        return AbstractFormat(
            prec, exp, pos_bound, neg_bound=neg_bound,
            has_pos_inf=has_pos_inf, has_neg_inf=has_neg_inf, has_nan=has_nan,
            has_neg_zero=has_neg_zero,
        )


    def __mul__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Multiplication of two formats.

        Produces a format that can represent the product of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # precision: p1 + p2 (clamped to >=1 so an exactly-zero
        # product still has valid ``prec``)
        # exponent: e1 + e2
        # bounds: b1 * b2
        p_self, p_other = self.effective_prec(), other.effective_prec()
        if p_self == 1 or p_other == 1:
            # A single-precision-bit format holds nothing but powers of two
            # (`±1 * 2**e`), so multiplying by one only shifts an exponent: each
            # product keeps its operand's significand rather than widening it.
            # Summing here would charge a scale-in or scale-out a bit it never
            # spends -- which is what `rescale_fixed` emits on every rounding.
            prec = max(p_self, p_other)
        else:
            prec = max(p_self + p_other, 1)
        exp = self.exp + other.exp
        # every format straddles zero (`pos_bound >= 0 >= neg_bound`), so the
        # two like-sign corners give the maximum and the two cross corners the
        # minimum -- `max` on the latter would claim the *tighter* of the two
        # and miss the product it names: `[-1,1] * [-2,1]` reaches -2
        pos_bound = max(self.pos_bound * other.pos_bound, self.neg_bound * other.neg_bound)
        neg_bound = min(self.pos_bound * other.neg_bound, self.neg_bound * other.pos_bound)

        # special values: 0 is representable everywhere, so `inf * 0 = NaN` is
        # reachable whenever either operand has an infinity -- the NaN result is
        # exact.  The infinity outputs are conservative: we set both signs
        # rather than tracking sign ranges (exact for symmetric operands like
        # IEEE x IEEE; over-approximate for asymmetric or {0}-only operands).
        self_inf = self.has_pos_inf or self.has_neg_inf
        other_inf = other.has_pos_inf or other.has_neg_inf
        inf_out = self_inf or other_inf
        has_nan = self.has_nan or other.has_nan or inf_out
        # A zero product takes the XOR of the operand signs (IEEE-754 §6.3, in
        # every rounding mode), so a `-0.0` needs a sign disagreement -- and one
        # is always available once *either* number system has a signed zero:
        # every format represents a `+0.0`, and a negative times that zero is
        # `-0.0`.  When neither system has one, no product can be one:
        # two's-complement has a single zero, and `(-2) * 0` under `SINT8` is
        # `+0.0`.
        has_neg_zero = self.has_neg_zero or other.has_neg_zero
        return AbstractFormat(
            prec, exp, pos_bound, neg_bound=neg_bound,
            has_pos_inf=inf_out, has_neg_inf=inf_out, has_nan=has_nan,
            has_neg_zero=has_neg_zero,
        )

    def __and__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Intersection of two formats."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec = min(self.prec, other.prec)
        exp = max(self.exp, other.exp)
        pos_bound = min(self.pos_bound, other.pos_bound)
        neg_bound = max(self.neg_bound, other.neg_bound)
        # a special value is in the intersection iff it is in both operands --
        # the negative zero included, or `x & x` would not be `x`
        return AbstractFormat(
            prec, exp, pos_bound, neg_bound=neg_bound,
            has_pos_inf=self.has_pos_inf and other.has_pos_inf,
            has_neg_inf=self.has_neg_inf and other.has_neg_inf,
            has_nan=self.has_nan and other.has_nan,
            has_neg_zero=self.has_neg_zero and other.has_neg_zero,
        )

    def __or__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Union of two formats."""
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        prec = max(self.prec, other.prec)
        exp = min(self.exp, other.exp)
        pos_bound = max(self.pos_bound, other.pos_bound)
        neg_bound = min(self.neg_bound, other.neg_bound)
        # a special value is in the union iff it is in either operand -- the
        # negative zero included, or the join would not contain its operands
        return AbstractFormat(
            prec, exp, pos_bound, neg_bound=neg_bound,
            has_pos_inf=self.has_pos_inf or other.has_pos_inf,
            has_neg_inf=self.has_neg_inf or other.has_neg_inf,
            has_nan=self.has_nan or other.has_nan,
            has_neg_zero=self.has_neg_zero or other.has_neg_zero,
        )

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

        Special-value membership is derived by probing *fmt*'s
        :meth:`representable_in` with each signed infinity, NaN, and a negative
        zero.  Probing ``+inf``/``-inf`` separately captures per-sign asymmetry a
        single ``enable_inf`` flag cannot (e.g. a positive-only format has no
        -inf).  This round-trips with :meth:`format`, which sets the
        ``enable_*`` flags.

        ``has_neg_zero`` round-trips the same way, via each format's
        ``enable_neg_zero``.  That flag exists for this: without it
        :meth:`format` could not express "no negative zero", so every
        integer-valued bound materialized on the way to storage selection would
        acquire one and lose its integer storage — ``int8 + int8`` lands on a
        *float*-shaped ``MPBFloatFormat``, whose value set legitimately includes
        a ``-0.0`` unless told otherwise.  Every probe is believed, so a format
        that does claim a negative zero is kept off the integer rungs of the C++
        storage ladder by containment — correctly, as no C++ integer type has one.
        """
        if not isinstance(fmt, Format):
            raise TypeError(f'Expected \'Format\', got {fmt}')

        # finite values: quantum, precision, and bounds
        match fmt:
            case RealFormat():
                af = AbstractFormat(
                    float('inf'),
                    float('-inf'),
                    float('inf'),
                    neg_bound=float('-inf'),
                )
            case FixedFormat() if not fmt.signed:
                neg_maxval = RealFloat.from_int(0)
                af = AbstractFormat(
                    float('inf'), fmt.expmin, fmt.pos_maxval, neg_bound=neg_maxval
                )
            case MPBFixedFormat():
                af = AbstractFormat(
                    float('inf'), fmt.expmin, fmt.pos_maxval, neg_bound=fmt.neg_maxval
                )
            case MPFixedFormat():
                af = AbstractFormat(float('inf'), fmt.expmin, float('inf'))
            case ExpFormat():
                pos_maxval = fmt.maxval().as_real()
                neg_maxval = RealFloat.from_int(0)
                expmin = fmt.minval().exp
                af = AbstractFormat(1, expmin, pos_maxval, neg_bound=neg_maxval)
            case EFloatFormat():
                af = AbstractFormat(
                    fmt.pmax,
                    fmt.expmin,
                    fmt._mpb_fmt.pos_maxval,
                    neg_bound=fmt._mpb_fmt.neg_maxval,
                )
            case MPBFloatFormat():
                af = AbstractFormat(
                    fmt.pmax, fmt.expmin, fmt.pos_maxval, neg_bound=fmt.neg_maxval
                )
            case MPSFloatFormat():
                af = AbstractFormat(fmt.pmax, fmt.expmin, float('inf'))
            case MPFloatFormat():
                af = AbstractFormat(fmt.pmax, float('-inf'), float('inf'))
            case _:
                raise ValueError(f'format is not abstractable: {fmt!r}')

        # special values: probe the format's representable set directly.
        af.has_pos_inf = fmt.representable_in(Float.inf(s=False))
        af.has_neg_inf = fmt.representable_in(Float.inf(s=True))
        af.has_nan = fmt.representable_in(Float.nan())
        af.has_neg_zero = fmt.representable_in(Float(s=True, exp=0, c=0))
        return af

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

        Special values: each non-``REAL_FORMAT`` branch is constructed with
        ``enable_nan``/``enable_inf`` matching ``self`` so the flags round-trip
        through :meth:`from_format` (``REAL_FORMAT`` already represents them
        all).  ``enable_inf`` is a single flag for both signs, so ``has_pos_inf
        or has_neg_inf`` may add the opposite-signed infinity — a sound
        over-approximation.
        """
        enable_nan = self.has_nan
        enable_inf = self.has_pos_inf or self.has_neg_inf
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
                if not self._prec_constrains():
                    # Only integers: describe it as fixed-point.  Materializing
                    # as a float would be correct but perverse — every value would
                    # be subnormal — and, more to the point, a float format has a
                    # signed zero.  `int8 + int8` lands here, and describing it as a
                    # float is what cost it its `int16_t` storage.
                    return MPBFixedFormat(
                        self.exp - 1, self.pos_bound, self.neg_bound,
                        enable_nan=enable_nan, enable_inf=enable_inf,
                        enable_neg_zero=self.has_neg_zero,
                    )
                neg_maxval = self.neg_bound
                if not neg_maxval.s:
                    # MPBFloatFormat requires a strictly-negative neg_maxval;
                    # widen symmetrically (sound over-approximation).
                    neg_maxval = RealFloat(s=True, x=self.pos_bound)
                return MPBFloatFormat(
                    self.prec, emin, self.pos_bound, neg_maxval,
                    enable_nan=enable_nan, enable_inf=enable_inf,
                )
            if bounds_unbounded:
                return MPSFloatFormat(
                    self.prec, emin, enable_nan=enable_nan, enable_inf=enable_inf,
                )

        if not prec_inf and exp_inf and bounds_unbounded:
            assert isinstance(self.prec, int)
            return MPFloatFormat(self.prec, enable_nan=enable_nan, enable_inf=enable_inf)

        if prec_inf and not exp_inf:
            assert isinstance(self.exp, int)
            nmin = self.exp - 1
            if bounds_bounded:
                assert isinstance(self.pos_bound, RealFloat)
                assert isinstance(self.neg_bound, RealFloat)
                # A bound short of the finest representable digit leaves only
                # zero -- an empty intersection, e.g. a multiple of `2 ** 24`
                # bounded by `1024`.  Clamping to zero states that exactly: no
                # representable value lies in between.
                pos_bound, neg_bound = self.pos_bound, self.neg_bound
                quantum = RealFloat(exp=self.exp, c=1)
                if pos_bound < quantum and RealFloat(s=False, x=neg_bound) < quantum:
                    pos_bound, neg_bound = RealFloat(), RealFloat(s=True)
                return MPBFixedFormat(
                    nmin, pos_bound, neg_bound,
                    enable_nan=enable_nan, enable_inf=enable_inf,
                    enable_neg_zero=self.has_neg_zero,
                )
            if bounds_unbounded:
                return MPFixedFormat(nmin, enable_nan=enable_nan, enable_inf=enable_inf,
                    enable_neg_zero=self.has_neg_zero)

        return REAL_FORMAT

    def _prec_constrains(self) -> bool:
        """Does ``prec`` actually thin the values inside the bounds?

        Values sit ``2**exp`` apart; spanning the bounds at that spacing needs
        some number of significand bits.  If ``prec`` supplies at least that many
        it removes nothing and every such value is representable —
        which is exactly what a fixed-point format describes.  If ``prec`` is
        smaller, values thin out as the magnitude grows, and only a floating-point
        format can say that.

        So this is the float/fixed discriminator, and it is about the precision
        constraint alone — not about whether the values happen to be integers.
        ``A(24, -149, ±3.4e38)`` (FP32's own shape) needs 278 bits to span its
        range at quantum ``2**-149`` and has 24, so it is genuinely floating.
        ``A(9, 0, +254, -256)`` — the sum of two ``int8`` formats — needs 9 and
        has 9, so it is genuinely fixed.

        Only meaningful with a finite ``prec``/``exp`` and finite bounds; callers
        check that first.
        """
        if not isinstance(self.prec, int) or not isinstance(self.exp, int):
            return True
        if isinstance(self.pos_bound, float) or isinstance(self.neg_bound, float):
            return True
        needed = max(
            _maxval_precision(self.pos_bound, self.exp),
            _maxval_precision(RealFloat(s=False, x=self.neg_bound), self.exp),
        )
        return self.prec < needed

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

    def specials_contained_in(self, other: 'AbstractFormat') -> bool:
        """Is every special value of ``self`` also one of ``other``?

        Condition 4 of :meth:`_is_contained_in`, factored out because storage
        selection needs it *without* the magnitude conditions: its fallback for an
        unbounded integer deliberately ignores the bounds, but must not ignore
        these.  Keeping one implementation is the point -- ``has_neg_zero`` was
        added to the containment check and missed in that fallback, which let a
        bound carrying a ``-0.0`` reach an integer storage.
        """
        return not (
            (self.has_pos_inf and not other.has_pos_inf)
            or (self.has_neg_inf and not other.has_neg_inf)
            or (self.has_nan and not other.has_nan)
            or (self.has_neg_zero and not other.has_neg_zero)
        )

    def _is_contained_in(self, other: 'AbstractFormat') -> bool:
        """Return True iff every value representable by `self` is also representable by `other`.

        The conditions are:
          1. Quantum: other.exp <= self.exp  (other is at least as fine-grained)
          2. Bounds:  other.pos_bound >= self.pos_bound  and  other.neg_bound <= self.neg_bound
          3. Precision: either other.prec >= self.prec, *or* self's entire range lies within
             other's subnormal region (pos_bound <= 2^(other.exp + other.prec)), in which case
             the floating-point precision of other is irrelevant — all values fit exactly.
          4. Special values: every special value in self must also be in other
             (a member of self that other lacks breaks containment).  This
             includes ``-0.0``, which conditions 1-3 cannot see: they compare
             ``RealFloat`` bounds by magnitude, so the two zeros are
             indistinguishable there.

        Under-reporting a flag is not merely imprecise.  It accepts containments
        that should fail, and storage selection reads that as permission -- a
        bound that can hold a ``-0.0`` reaching an integer type is a wrong
        answer, not a loose one.  Every operator derives ``has_neg_zero`` for
        this reason.
        """

        # 4. special values
        if not self.specials_contained_in(other):
            return False

        # 1. quantum
        if other.exp > self.exp:
            return False
        # 2. bounds
        if other.pos_bound < self.pos_bound:
            return False
        if other.neg_bound > self.neg_bound:
            return False
        # 3. precision
        if not isinstance(other.prec, float) and self.prec > other.prec:
            # Easy check failed: other's spacing in its normal region widens
            # faster.  Containment still holds if self's bound stays within the
            # region where other's effective quantum is <= self's quantum
            # 2^self.exp, i.e. pos_bound1 <= 2^(self.exp + other.prec) -- but
            # that region is other's *subnormal* one, which exists only where
            # its exponent is finite.  Unbounded below, precision binds
            # everywhere and less of it is less.
            if not isinstance(self.exp, int) or not isinstance(other.exp, int):
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

    def next_bound(self) -> 'AbstractFormat':
        """
        Figure 8's ``next(b)``: both bounds one step away from zero, in *this*
        format's grid.  A caller wanting a wider grid extends the format first.
        An unbounded side is left alone.
        """
        return AbstractFormat(
            self.prec, self.exp,
            self._next_away(self.pos_bound), neg_bound=self._next_away(self.neg_bound),
            has_pos_inf=self.has_pos_inf, has_neg_inf=self.has_neg_inf,
            has_nan=self.has_nan, has_neg_zero=self.has_neg_zero,
        )

    def _next_away(self, b: RealFloat | float) -> RealFloat | float:
        """One step outward from *b* in this format's grid.

        `RealFloat`'s `n` is the first *unrepresentable* digit, one below the
        minimum representable exponent, so `exp - 1` is this format's grid.
        """
        if not isinstance(b, RealFloat) or b.is_zero():
            return b        # unbounded, or a zero with no direction to step
        p = self.prec if isinstance(self.prec, int) else None
        n = (self.exp - 1) if isinstance(self.exp, int) else None
        return b.next_away_zero(p, n)

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
        return AbstractFormat(
            new_prec, self.exp, self.pos_bound, neg_bound=self.neg_bound,
            has_pos_inf=self.has_pos_inf, has_neg_inf=self.has_neg_inf,
            has_nan=self.has_nan, has_neg_zero=self.has_neg_zero,
        )

    def with_exp_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with exponent adjusted by delta.

        Args:
            delta: Amount to add to exponent (can be negative).
        Returns:
            New AbstractFormat with adjusted exponent.
        """
        new_exp = self.exp + delta
        return AbstractFormat(
            self.prec, new_exp, self.pos_bound, neg_bound=self.neg_bound,
            has_pos_inf=self.has_pos_inf, has_neg_inf=self.has_neg_inf,
            has_nan=self.has_nan, has_neg_zero=self.has_neg_zero,
        )

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
        # scaling by a positive factor preserves special-value membership
        return AbstractFormat(
            self.prec, self.exp, new_pos_bound, neg_bound=new_neg_bound,
            has_pos_inf=self.has_pos_inf, has_neg_inf=self.has_neg_inf,
            has_nan=self.has_nan, has_neg_zero=self.has_neg_zero,
        )
