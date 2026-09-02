"""
`AbstractFormat.from_format` and `AbstractFormat.format` against each other.

`format()` promises a *superset*, not an identity, so the round-trip is an
inequality in general.  What matters for a release is narrower and checkable:

- starting from a concrete format, the values must survive
  (`from_format` then `format()` represents everything the original did), and
- `format()` must return a `Format` for every `AbstractFormat` that can be
  built -- it is total, and a shape it cannot name cleanly falls back to
  `REAL_FORMAT` rather than raising.
"""

import itertools

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import AbstractableFormat, AbstractFormat
from fpy2.number import (
    EFloatContext,
    EFloatNanKind,
    Float,
    RealFloat,
    RoundingMode,
)

_R = RealFloat.from_int

CONTEXTS = [
    ('FP16', fp.FP16),
    ('FP32', fp.FP32),
    ('FP64', fp.FP64),
    ('bfloat16', fp.IEEEContext(8, 16)),
    ('ieee_4_8', fp.IEEEContext(4, 8)),
    ('MX_E5M2', fp.MX_E5M2),
    ('MX_E4M3', fp.MX_E4M3),
    ('MX_E2M1', fp.MX_E2M1),
    ('saturating', fp.IEEEContext(5, 16, RoundingMode.RNE,
                                  fp.OverflowMode.SATURATE)),
    ('SINT8', fp.SINT8),
    ('SINT16', fp.SINT16),
    ('SINT32', fp.SINT32),
    ('SINT64', fp.SINT64),
    ('UINT8', fp.UINT8),
    ('UINT32', fp.UINT32),
    ('INTEGER', fp.INTEGER),
    ('MPFloat', fp.MPFloatContext(24)),
    ('MPSFloat', fp.MPSFloatContext(24, -10)),
    ('MPBFloat', fp.MPBFloatContext(24, -10, _R(1))),
    ('MPFixed', fp.MPFixedContext(-8)),
    ('MPBFixed', fp.MPBFixedContext(-8, _R(1))),
    ('MPBFixed_int', fp.MPBFixedContext(-1, _R(1024))),
    ('REAL', fp.REAL),
]
CONTEXT_IDS = [name for name, _ in CONTEXTS]


def _probes():
    """Values spanning every class the formats distinguish."""
    out = [
        Float(c=0), Float(s=True, exp=0, c=0), Float(isnan=True),
        Float(isinf=True), Float(isinf=True, s=True),
    ]
    for exp in (-1074, -200, -149, -30, -24, -11, -4, -1, 0, 1, 3, 8,
                10, 16, 20, 63, 128, 1000):
        for c in (1, 2, 3, 5, 127, 255, 1023, 2047,
                  (1 << 24) - 1, (1 << 53) - 1):
            out.append(Float(exp=exp, c=c))
            out.append(Float(s=True, exp=exp, c=c))
    return out


PROBES = _probes()


class TestFromAConcreteFormat:
    @pytest.mark.parametrize('_name,ctx', CONTEXTS, ids=CONTEXT_IDS)
    def test_no_value_is_lost(self, _name, ctx):
        """The direction that has to hold: `format()` is a superset, so nothing
        the original represented may go missing."""
        f0 = ctx.format()
        assert isinstance(f0, AbstractableFormat)
        f1 = AbstractFormat.from_format(f0).format()
        lost = [v for v in PROBES
                if f0.representable_in(v) and not f1.representable_in(v)]
        assert not lost, [str(v) for v in lost[:5]]

    @pytest.mark.parametrize('_name,ctx', CONTEXTS, ids=CONTEXT_IDS)
    def test_the_abstraction_is_a_fixpoint(self, _name, ctx):
        """A second trip changes nothing, so whatever `format()` widened it
        widened once."""
        af = AbstractFormat.from_format(ctx.format())   # type: ignore[arg-type]
        f1 = af.format()
        assert isinstance(f1, AbstractableFormat)
        af2 = AbstractFormat.from_format(f1)
        assert AbstractFormat.from_format(af2.format()) == af2  # type: ignore[arg-type]

    @pytest.mark.parametrize('_name,ctx', CONTEXTS, ids=CONTEXT_IDS)
    def test_the_set_survives_exactly(self, _name, ctx):
        """Stronger than *no value lost*, and it holds for every context the
        library ships: the round-tripped format represents neither more nor
        less.  `TestKnownWidenings` has the shapes it does not."""
        f0 = ctx.format()
        f1 = AbstractFormat.from_format(f0).format()   # type: ignore[arg-type]
        differ = [v for v in PROBES
                  if f0.representable_in(v) != f1.representable_in(v)]
        assert not differ, [str(v) for v in differ[:5]]


class TestKnownWidenings:
    """Where `format()` cannot say what the abstraction knows.  Each is a sound
    superset, and each is a missing knob rather than a bug in the mapping."""

    def test_a_float_shape_cannot_refuse_a_negative_zero(self):
        """`enable_neg_zero` exists on `MPFixedFormat` and `MPBFixedFormat` and
        on none of the float formats, so ``has_neg_zero=False`` survives a
        fixed-point round trip and not a floating-point one.

        The witness is a format whose NaN *is* the negative-zero encoding, so
        `-0.0` is not one of its values.
        """
        ctx = EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0)
        f0 = ctx.format()
        neg_zero = Float(s=True, exp=0, c=0)
        assert not f0.representable_in(neg_zero)

        af = AbstractFormat.from_format(f0)   # type: ignore[arg-type]
        assert not af.has_neg_zero            # the abstraction knows
        assert af.format().representable_in(neg_zero)   # the format cannot say

    def test_a_fixed_shape_can(self):
        af = AbstractFormat(float('inf'), 0, _R(127), neg_bound=-_R(128))
        assert not af.has_neg_zero
        assert not af.format().representable_in(Float(s=True, exp=0, c=0))

    def test_one_flag_covers_both_infinities(self):
        """`enable_inf` is a single flag, so an asymmetric infinity gains the
        other sign."""
        af = AbstractFormat(11, -24, _R(1024), has_pos_inf=True)
        assert not af.has_neg_inf
        assert af.format().representable_in(Float(isinf=True, s=True))


class TestFormatIsTotal:
    """`format()` returns a `Format` for every `AbstractFormat` that can be
    built.  It used to raise for a bound off the grid `exp` defines, which an
    ordinary meet produces."""

    EXPS = (-1074, -149, -24, -1, 0, 5, float('-inf'))
    BOUNDS = (_R(0), _R(1), _R(254), _R(1024), RealFloat(exp=-4, c=3),
              float('inf'))

    @pytest.mark.parametrize('prec', [1, 2, 9, 11, 24, 53, float('inf')])
    def test_over_every_shape(self, prec):
        for exp, pos in itertools.product(self.EXPS, self.BOUNDS):
            for negk in ('sym', 'zero', 'neg1', 'inf'):
                if negk == 'sym':
                    neg = float('-inf') if isinstance(pos, float) else -pos
                elif negk == 'zero':
                    neg = _R(0)
                elif negk == 'neg1':
                    neg = RealFloat(s=True, x=_R(1))
                else:
                    neg = float('-inf')
                for flags in itertools.product([False, True], repeat=4):
                    af = AbstractFormat(
                        prec, exp, pos, neg_bound=neg,
                        has_pos_inf=flags[0], has_neg_inf=flags[1],
                        has_nan=flags[2], has_neg_zero=flags[3],
                    )
                    af.format()      # must not raise

    def test_a_meet_of_a_float_and_a_coarse_fixed_format(self):
        for coarse in (0, 5, 24):
            a = AbstractFormat.from_format(fp.FP16.format())   # type: ignore[arg-type]
            b = AbstractFormat.from_format(
                fp.MPFixedContext(coarse).format())            # type: ignore[arg-type]
            (a & b).format()
            (a | b).format()


class TestTheBoundIsReducedExactly:
    """A concrete format's bound has to be one of its own values, and an
    abstract one's need not be -- ``prec``, ``exp`` and the bounds are
    independent.  Reducing it to the largest value that *is* representable keeps
    the set unchanged, where rounding outward would admit values the format does
    not hold."""

    def test_the_meet_keeps_its_own_maximum(self):
        """``FP16 & MPFixedContext(5)``: multiples of 64 bounded by 65504.  The
        largest is 65472; 65536 is a multiple of 64 that `FP16` has not got."""
        af = (AbstractFormat.from_format(fp.FP16.format())        # type: ignore[arg-type]
              & AbstractFormat.from_format(fp.MPFixedContext(5).format()))  # type: ignore[arg-type]
        fmt = af.format()
        holds = lambda v: fmt.representable_in(
            Float(x=RealFloat.from_int(v), ctx=fp.FP64))
        assert holds(65472)
        assert not holds(65504)      # not a multiple of 64
        assert not holds(65536)      # a multiple of 64, but not an `FP16` value

    def test_precision_reduces_it_too_not_just_the_grid(self):
        """Two bits of significand over the integers: the largest value at most
        255 is 192, not 255 and not 256."""
        af = AbstractFormat(2, 0, _R(255), neg_bound=-_R(255))
        assert float(af._max_representable(_R(255))) == 192

    @pytest.mark.parametrize('prec', [1, 2, 9, 11, 24, 53, float('inf')])
    def test_nothing_representable_is_skipped(self, prec):
        """The reduction is at most the bound, and the next value away from zero
        is past it -- so no representable value was dropped."""
        bounds = (_R(0), _R(1), _R(127), _R(254), _R(1024), _R(65504),
                  RealFloat(exp=-4, c=3), RealFloat(exp=6, c=1023))
        for exp in (-1074, -149, -24, -1, 0, 5, 24, float('-inf')):
            for b in bounds:
                af = AbstractFormat(prec, exp, b, neg_bound=-b)
                red = af._max_representable(b)
                assert red <= b, (af, red)
                if isinstance(exp, float):
                    continue
                if red.is_zero():
                    nxt = RealFloat(exp=exp, c=1)
                else:
                    nxt = red.next_away_zero(
                        p=None if isinstance(prec, float) else prec,
                        n=exp - 1,
                    )
                assert nxt > b, (af, red, nxt)
