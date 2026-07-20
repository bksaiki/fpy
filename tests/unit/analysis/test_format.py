"""
Test for the abstract number format `A(p, exp, bound)`.
"""

import fpy2 as fp
import itertools
import math

from hypothesis import given, settings, strategies as st
from fractions import Fraction
from typing import Generator

from fpy2.analysis.format_infer import AbstractFormat

def is_power_of_two(v: Fraction) -> bool:
    assert v > 0
    n, d = v.numerator, v.denominator
    return (n & (n - 1) == 0) and (d & (d - 1) == 0)

def is_dyadic(b: Fraction) -> bool:
    d = b.denominator
    return d & (d - 1) == 0

def generate(p: int | float, q: Fraction, b: Fraction) -> Generator[Fraction, None, None]:
    assert (isinstance(p, int) and p >= 1) or p == math.inf
    assert is_power_of_two(q)
    assert is_dyadic(b)
    v_cut = q * Fraction(2) ** p if p != math.inf else math.inf
    v = Fraction(0)
    yield v
    while v + q <= b:
        v = v + q
        yield +v
        yield -v
        if is_power_of_two(v) and v >= v_cut:
            q = 2 * q


class TestAbstractFormat():

    def test_construct(self):
        """Testing construction of AbstractFormat."""
        # MPFloatContext(24)
        fmt = AbstractFormat.from_format((fp.MPFloatContext(24).format()))
        assert fmt.prec == 24
        assert fmt.exp == float('-inf')
        assert fmt.bound == float('inf')
        # MPSFloatContext(24, -10)
        fmt = AbstractFormat.from_format((fp.MPSFloatContext(24, -10).format()))
        assert fmt.prec == 24
        assert fmt.exp == -33
        assert fmt.bound == float('inf')
        # MPBFloatContext(24, -10, 1.0)
        fmt = AbstractFormat.from_format(fp.MPBFloatContext(24, -10, fp.RealFloat.from_int(1)).format())
        assert fmt.prec == 24
        assert fmt.exp == -33
        assert fmt.bound == fp.RealFloat.from_int(1)
        # FP64
        fmt = AbstractFormat.from_format((fp.FP64).format())
        assert fmt.prec == 53
        assert fmt.exp == -1074
        assert fmt.bound == fp.RealFloat(exp=971, c=(1 << 53) - 1)
        # FP32
        fmt = AbstractFormat.from_format((fp.FP32).format())
        assert fmt.prec == 24
        assert fmt.exp == -149
        assert fmt.bound == fp.RealFloat(exp=104, c=(1 << 24) - 1)
        # MPFixedContext(-8)
        fmt = AbstractFormat.from_format((fp.MPFixedContext(-8).format()))
        assert fmt.prec == float('inf')
        assert fmt.exp == -7
        assert fmt.bound == float('inf')
        # MPBFixedContext(-8, 1.0)
        fmt = AbstractFormat.from_format(fp.MPBFixedContext(-8, fp.RealFloat.from_int(1)).format())
        assert fmt.prec == float('inf')
        assert fmt.exp == -7
        assert fmt.bound == fp.RealFloat.from_int(1)
        # INT8
        fmt = AbstractFormat.from_format((fp.SINT8).format())
        assert fmt.prec == float('inf')
        assert fmt.exp == 0
        assert fmt.bound == fp.RealFloat.from_int(128)

    def test_contains_sandbox(self):
        A1 = AbstractFormat(5, -3, fp.RealFloat.from_int(1))
        A2 = AbstractFormat(5, -4, fp.RealFloat.from_int(1))

        # A1 = AbstractFormat(1, 0, fp.RealFloat.from_int(2))
        # A2 = AbstractFormat(1, 0, fp.RealFloat.from_int(4))

        print(A1 <= A2)
        print(list(generate(A1.prec, Fraction(2) ** A1.exp, A1.bound)))
        print(list(generate(A2.prec, Fraction(2) ** A2.exp, A2.bound)))

    def test_contains_examples(self):
        """Testing containment check."""
        # FP32 \subseteq FP64
        CTX1 = AbstractFormat.from_format((fp.FP32).format())
        CTX2 = AbstractFormat.from_format((fp.FP64).format())
        assert CTX1 <= CTX2, "Expected FP32 to be contained in FP64."

        # MX_E5M2 \subseteq FP32
        CTX1 = AbstractFormat.from_format((fp.MX_E5M2).format())
        CTX2 = AbstractFormat.from_format((fp.FP32).format())
        assert CTX1 <= CTX2, "Expected MX_E5M2 to be contained in FP32."

        # FP64 ⊄ FP32
        CTX1 = AbstractFormat.from_format((fp.FP64).format())
        CTX2 = AbstractFormat.from_format((fp.FP32).format())
        assert not CTX1 <= CTX2, "Expected FP64 to not be contained in FP32."

        # MX_E4M3 ⊄ MX_E5M2
        CTX1 = AbstractFormat.from_format((fp.MX_E4M3).format())
        CTX2 = AbstractFormat.from_format((fp.MX_E5M2).format())
        assert not CTX1 <= CTX2, "Expected MX_E4M3 to not be contained in MX_E5M2."

        # MX_E4M3 ⊄ fixed<-9, 32>: E4M3 represents NaN (nan_kind=MAX_VAL) but
        # fixed-point does not, so the NaN member breaks containment even
        # though every finite E4M3 value fits in fixed<-9, 32>.
        CTX1 = AbstractFormat.from_format((fp.MX_E4M3).format())
        CTX2 = AbstractFormat.from_format((fp.FixedContext(True, -9, 32).format()))
        assert not CTX1 <= CTX2, "Expected MX_E4M3 to not be contained in fixed<-9, 32> (NaN not representable)."

        # MX_E5M2 ⊄ fixed<-9, 32>
        CTX1 = AbstractFormat.from_format((fp.MX_E5M2).format())
        CTX2 = AbstractFormat.from_format((fp.FixedContext(True, -9, 32).format()))
        assert not CTX1 <= CTX2, "Expected MX_E5M2 to not be contained in fixed<-9, 32>."

        # INT8 \subseteq FP32
        CTX1 = AbstractFormat.from_format((fp.SINT8).format())
        CTX2 = AbstractFormat.from_format((fp.FP32).format())
        assert CTX1 <= CTX2, "Expected INT8 to be contained in FP32."

        # INT4 \subseteq A(3, 0, 4)
        CTX1 = AbstractFormat.from_format((fp.FixedContext(True, 0, 4).format()))
        CTX2 = AbstractFormat(3, 0, fp.RealFloat.from_int(8))
        assert CTX1 <= CTX2, "Expected INT4 to be contained in A(3, 0, 4)."

        # INT4 \subseteq A(4, 0, 12)
        CTX1 = AbstractFormat.from_format((fp.FixedContext(True, 0, 4).format()))
        CTX2 = AbstractFormat(4, 0, fp.RealFloat.from_int(12))
        assert CTX1 <= CTX2, "Expected INT4 to be contained in A(4, 0, 12)."

    # ------------------------------------------------------------------
    # Subnormal-region containment: the asymmetric branch of
    # ``_is_contained_in`` where ``self.prec > other.prec`` but every
    # value in ``self`` lives within ``other``'s subnormal region.

    def test_contains_high_prec_but_small_bound_in_subnormal_region(self):
        """
        ``self`` has more precision than ``other``, but every value of
        ``self`` fits within ``other``'s subnormal region (where the
        effective quantum is ``2^other.exp`` and is finer than the
        normal-region spacing implied by ``other.prec``).  Containment
        should hold via the cutoff check.

        cutoff = ``2^(self.exp + other.prec)``; if ``self.bound <= cutoff``,
        ``self`` is contained.
        """
        # self: 10 bits of precision, quantum=1, bound=4. Represents the
        # integers {-4, ..., 4}.
        # other: 2 bits of precision, quantum=1, bound=4. Normal-region
        # spacing widens past 4 but every value <= 4 is exactly
        # representable in the subnormal region.
        # cutoff = 2^(0 + 2) = 4. self.bound = 4 <= 4 → contained.
        SELF = AbstractFormat(10, 0, fp.RealFloat.from_int(4))
        OTHER = AbstractFormat(2, 0, fp.RealFloat.from_int(4))
        assert SELF <= OTHER, \
            'high-prec self with bound at cutoff should be contained'

    def test_contains_high_prec_bound_just_above_cutoff(self):
        """
        Same parameters as the previous test except ``self``'s bound is
        one unit above the subnormal-region cutoff.  Containment must
        fail — values just above the cutoff land in ``other``'s normal
        region where the spacing is too coarse for ``self``'s precision.
        """
        # cutoff = 2^(0 + 2) = 4. self.bound = 8 > 4.  Even though
        # other.bound (16) covers self.bound (8), the precision check
        # rejects.
        SELF = AbstractFormat(10, 0, fp.RealFloat.from_int(8))
        OTHER = AbstractFormat(2, 0, fp.RealFloat.from_int(16))
        assert not SELF <= OTHER, \
            'high-prec self with bound past cutoff must not be contained'

    def test_contains_high_prec_neg_bound_above_cutoff(self):
        """
        Symmetric counter-test: positive bound stays at the cutoff but
        the *negative* bound exceeds it.  Containment must still fail.
        """
        SELF = AbstractFormat(
            10, 0,
            fp.RealFloat.from_int(4),
            neg_bound=fp.RealFloat(s=True, exp=0, c=8),  # -8
        )
        OTHER = AbstractFormat(
            2, 0,
            fp.RealFloat.from_int(16),
            neg_bound=fp.RealFloat(s=True, exp=0, c=16),  # -16
        )
        assert not SELF <= OTHER, \
            'asymmetric subnormal check should also examine neg_bound'

    def test_contains_high_prec_other_unbounded_prec(self):
        """
        When ``other.prec`` is unbounded (float('inf')), the precision
        check is skipped entirely — any finite-prec ``self`` with bounds
        and quantum that fit is contained regardless of its precision.
        """
        # MPFixedFormat-shaped: prec=inf, finite exp, unbounded bound.
        OTHER = AbstractFormat(float('inf'), -10, float('inf'))
        SELF = AbstractFormat(53, -10, fp.RealFloat.from_int(1024))
        assert SELF <= OTHER

    # ------------------------------------------------------------------
    # __abs__ returns a RealFloat-typed neg_bound

    def test_abs_neg_bound_is_realfloat(self):
        """
        Regression: a previous version of ``__abs__`` stored the integer
        literal ``0`` as ``neg_bound``, violating the documented
        ``RealFloat | float`` typing and tripping ``format()``'s
        ``isinstance(self.neg_bound, RealFloat)`` assertion.
        """
        af = AbstractFormat.from_format((fp.FP32).format())
        absolute = abs(af)
        assert isinstance(absolute.neg_bound, fp.RealFloat)
        assert absolute.neg_bound == fp.RealFloat.from_int(0)
        # The result must round-trip through .format() without tripping
        # the bounded-float assertion.
        result = absolute.format()
        assert result is not None


    @given(
        st.one_of(st.integers(1, 6), st.just(float('inf'))),  # p1: precision (inf = fixed-point)
        st.integers(-8, 0),       # e1: minimum exponent of fmt1
        st.integers(1, 128),      # k1: bound of fmt1 is k1 * 2^e1
        st.one_of(st.integers(1, 6), st.just(float('inf'))),  # p2: precision (inf = fixed-point)
        st.integers(-8, 0),       # e2: minimum exponent of fmt2
        st.integers(1, 128),      # k2: bound of fmt2 is k2 * 2^e2
    )
    @settings(max_examples=500)
    def test_contains_exhaustive(self, p1, e1, k1, p2, e2, k2):
        """Check <= against exhaustive enumeration of representable values."""
        q1 = Fraction(2) ** e1
        q2 = Fraction(2) ** e2

        b1_raw = fp.RealFloat(exp=e1, c=k1)
        if p1 == float('inf'):
            b1 = b1_raw.round(min_n=e1 - 1, rm=fp.RM.RTZ)
        else:
            b1 = b1_raw.round(p1, e1 - 1, rm=fp.RM.RTZ)
        
        b2_raw = fp.RealFloat(exp=e2, c=k2)
        if p2 == float('inf'):
            b2 = b2_raw.round(min_n=e2 - 1, rm=fp.RM.RTZ)
        else:
            b2 = b2_raw.round(p2, e2 - 1, rm=fp.RM.RTZ)

        fmt1 = AbstractFormat(p1, e1, b1)
        fmt2 = AbstractFormat(p2, e2, b2)

        # Exhaustive oracle: check if every value in fmt1 is also in fmt2
        vals2 = set(generate(p2, q2, b2.as_rational()))
        all_contained = all(v in vals2 for v in generate(p1, q1, b1.as_rational()))

        assert (fmt1 <= fmt2) == all_contained, f"format containment mismatch: {fmt1} <= {fmt2} should be {all_contained}"


    def test_effective_prec(self):
        """Testing effective precision calculation."""
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        for p, e, b in itertools.product(precs, exps, bounds):
            if p == float('inf') and e == float('-inf'):
                continue  # skip invalid format
            fmt = AbstractFormat(p, e, b)
            assert fmt.effective_prec() <= p

    def test_pos(self):
        """Testing positive bound calculation."""
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        for p, e, b in itertools.product(precs, exps, bounds):
            if p == float('inf') and e == float('-inf'):
                continue  # skip invalid format
            fmt1 = AbstractFormat(p, e, b)
            fmt = +fmt1

            # should be an exact copy
            assert fmt.prec == p
            assert fmt.exp == e
            assert fmt.pos_bound == b
            assert fmt.neg_bound == -b

    def test_neg(self):
        """Testing negative bound calculation."""
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        for p, e, b in itertools.product(precs, exps, bounds):
            if p == float('inf') and e == float('-inf'):
                continue  # skip invalid format
            fmt1 = AbstractFormat(p, e, b)
            fmt = -fmt1

            # should be an exact copy
            assert fmt.prec == p
            assert fmt.exp == e
            assert fmt.pos_bound == b
            assert fmt.neg_bound == -b

    def test_abs(self):
        """Testing absolute bound calculation."""
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        for p, e, b in itertools.product(precs, exps, bounds):
            if p == float('inf') and e == float('-inf'):
                continue  # skip invalid format
            fmt1 = AbstractFormat(p, e, b)
            fmt = abs(fmt1)

            # should be an exact copy
            assert fmt.prec == p
            assert fmt.exp == e
            assert fmt.pos_bound == b
            assert fmt.neg_bound == 0


    def test_add(self, logging: bool = False):
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        # iterator over all combinations
        for p1, e1, b1 in itertools.product(precs, exps, bounds):
            if p1 == float('inf') and e1 == float('-inf'):
                continue  # skip invalid format

            fmt1 = AbstractFormat(p1, e1, b1)
            for p2, e2, b2 in itertools.product(precs, exps, bounds):
                if p2 == float('inf') and e2 == float('-inf'):
                    continue  # skip invalid format

                fmt2 = AbstractFormat(p2, e2, b2)
                fmt = fmt1 + fmt2

                if logging:
                    fmt1_str = f"A({fmt1.prec}, {fmt1.exp}, {float(fmt1.pos_bound)})"
                    fmt2_str = f"A({fmt2.prec}, {fmt2.exp}, {float(fmt2.pos_bound)})"
                    fmt_str = f"A({fmt.prec}, {fmt.exp}, {float(fmt.pos_bound)})"
                    print(f"{fmt1_str} + {fmt2_str} = {fmt_str}")

                assert fmt.exp == min(e1, e2)
                assert fmt.pos_bound == b1 + b2


    # ------------------------------------------------------------------
    # Special-value membership (has_pos_inf / has_neg_inf / has_nan)

    def test_specials_from_format(self):
        """from_format lifts special-value membership from concrete formats."""
        # IEEE formats (E5M2/FP32) represent +inf, -inf, and NaN
        e5m2 = AbstractFormat.from_format(fp.MX_E5M2.format())
        assert e5m2.has_pos_inf and e5m2.has_neg_inf and e5m2.has_nan
        # E4M3 represents NaN but no infinities (nan_kind=MAX_VAL, enable_inf=False)
        e4m3 = AbstractFormat.from_format(fp.MX_E4M3.format())
        assert e4m3.has_nan
        assert not e4m3.has_pos_inf and not e4m3.has_neg_inf
        # fixed-point / integer formats have no special values
        sint8 = AbstractFormat.from_format(fp.SINT8.format())
        assert not sint8.has_pos_inf and not sint8.has_neg_inf and not sint8.has_nan
        # a plain MPBFloatContext admits NaN/inf (enable_* default True),
        # matching its pass-through/overflow rounding
        mpb = AbstractFormat.from_format(
            fp.MPBFloatContext(24, -10, fp.RealFloat.from_int(1)).format()
        )
        assert mpb.has_pos_inf and mpb.has_neg_inf and mpb.has_nan
        # but a float format constructed with special values disabled reports none
        # (this is what AbstractFormat.format() emits for e.g. sint8 + sint8)
        mpb_none = AbstractFormat.from_format(
            fp.number.context.mpb_float.MPBFloatFormat(
                24, -10, fp.RealFloat.from_int(1),
                enable_nan=False, enable_inf=False,
            )
        )
        assert not mpb_none.has_pos_inf and not mpb_none.has_neg_inf and not mpb_none.has_nan

    def test_specials_default_absent(self):
        """Special values default to absent, preserving legacy semantics."""
        fmt = AbstractFormat(5, -3, fp.RealFloat.from_int(1))
        assert not fmt.has_pos_inf
        assert not fmt.has_neg_inf
        assert not fmt.has_nan

    def test_specials_distinguish_eq_and_hash(self):
        """Flags participate in equality and hashing."""
        base = AbstractFormat(5, -3, fp.RealFloat.from_int(1))
        with_nan = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_nan=True)
        assert base != with_nan
        assert hash(base) != hash(with_nan)
        # same flags => equal and hash-equal
        again = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_nan=True)
        assert with_nan == again
        assert hash(with_nan) == hash(again)

    def test_specials_neg_swaps_infinities(self):
        """Negation swaps +/-inf and preserves NaN."""
        fmt = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True, has_nan=True)
        neg = -fmt
        assert neg.has_neg_inf and not neg.has_pos_inf
        assert neg.has_nan

    def test_specials_abs_folds_infinities(self):
        """abs maps -inf to +inf and clears -inf."""
        fmt = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_neg_inf=True, has_nan=True)
        result = abs(fmt)
        assert result.has_pos_inf and not result.has_neg_inf
        assert result.has_nan

    def test_specials_containment_implication(self):
        """A member special value in self must exist in other."""
        no_specials = AbstractFormat(5, -3, fp.RealFloat.from_int(1))
        with_nan = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_nan=True)
        # a format with NaN is not contained in one without it
        assert not (with_nan <= no_specials)
        # but the reverse holds (dropping a special is fine)
        assert no_specials <= with_nan

    def test_specials_add_inf_minus_inf_is_nan(self):
        """+inf + -inf produces NaN in the covering format."""
        pos = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True)
        neg = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_neg_inf=True)
        result = pos + neg
        assert result.has_pos_inf and result.has_neg_inf and result.has_nan

    def test_specials_sub_inf_minus_inf_is_nan(self):
        """+inf - +inf produces NaN in the covering format."""
        a = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True)
        b = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True)
        result = a - b
        # a.+inf - b.+inf = NaN; a.+inf - b.finite = +inf; finite - b.+inf = -inf
        assert result.has_nan and result.has_pos_inf and result.has_neg_inf

    def test_specials_mul_inf_times_zero_is_nan(self):
        """inf * 0 is reachable (0 is always representable), forcing NaN."""
        inf_fmt = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True)
        finite = AbstractFormat(5, -3, fp.RealFloat.from_int(1))
        result = inf_fmt * finite
        assert result.has_nan
        assert result.has_pos_inf and result.has_neg_inf

    def test_specials_union_and_intersection(self):
        """Union OR-s flags; intersection AND-s them."""
        with_nan = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_nan=True)
        with_inf = AbstractFormat(5, -3, fp.RealFloat.from_int(1), has_pos_inf=True)
        union = with_nan | with_inf
        assert union.has_nan and union.has_pos_inf
        inter = with_nan & with_inf
        assert not inter.has_nan and not inter.has_pos_inf

    def test_mul(self):
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        # iterator over all combinations
        params1 = itertools.product(precs, exps, bounds)
        params2 = itertools.product(precs, exps, bounds)

        for p1, e1, b1 in params1:
            if p1 == float('inf') and e1 == float('-inf'):
                continue  # skip invalid format

            fmt1 = AbstractFormat(p1, e1, b1)
            for p2, e2, b2 in params2:
                if p2 == float('inf') and e2 == float('-inf'):
                    continue  # skip invalid format

                fmt2 = AbstractFormat(p2, e2, b2)
                fmt = fmt1 * fmt2

                assert fmt.effective_prec() == fmt1.effective_prec() + fmt2.effective_prec()
                assert fmt.exp == e1 + e2
                assert fmt.pos_bound == b1 * b2
