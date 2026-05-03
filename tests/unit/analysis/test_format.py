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
        fmt = AbstractFormat.from_context(fp.MPFloatContext(24))
        assert fmt.prec == 24
        assert fmt.exp == float('-inf')
        assert fmt.bound == float('inf')
        # MPSFloatContext(24, -10)
        fmt = AbstractFormat.from_context(fp.MPSFloatContext(24, -10))
        assert fmt.prec == 24
        assert fmt.exp == -33
        assert fmt.bound == float('inf')
        # MPBFloatContext(24, -10, 1.0)
        fmt = AbstractFormat.from_context(fp.MPBFloatContext(24, -10, fp.RealFloat.from_int(1)))
        assert fmt.prec == 24
        assert fmt.exp == -33
        assert fmt.bound == fp.RealFloat.from_int(1)
        # FP64
        fmt = AbstractFormat.from_context(fp.FP64)
        assert fmt.prec == 53
        assert fmt.exp == -1074
        assert fmt.bound == fp.RealFloat(exp=971, c=(1 << 53) - 1)
        # FP32
        fmt = AbstractFormat.from_context(fp.FP32)
        assert fmt.prec == 24
        assert fmt.exp == -149
        assert fmt.bound == fp.RealFloat(exp=104, c=(1 << 24) - 1)
        # MPFixedContext(-8)
        fmt = AbstractFormat.from_context(fp.MPFixedContext(-8))
        assert fmt.prec == float('inf')
        assert fmt.exp == -7
        assert fmt.bound == float('inf')
        # MPBFixedContext(-8, 1.0)
        fmt = AbstractFormat.from_context(fp.MPBFixedContext(-8, fp.RealFloat.from_int(1)))
        assert fmt.prec == float('inf')
        assert fmt.exp == -7
        assert fmt.bound == fp.RealFloat.from_int(1)
        # INT8
        fmt = AbstractFormat.from_context(fp.SINT8)
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
        CTX1 = AbstractFormat.from_context(fp.FP32)
        CTX2 = AbstractFormat.from_context(fp.FP64)
        assert CTX1 <= CTX2, "Expected FP32 to be contained in FP64."

        # MX_E5M2 \subseteq FP32
        CTX1 = AbstractFormat.from_context(fp.MX_E5M2)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        assert CTX1 <= CTX2, "Expected MX_E5M2 to be contained in FP32."

        # FP64 ⊄ FP32
        CTX1 = AbstractFormat.from_context(fp.FP64)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        assert not CTX1 <= CTX2, "Expected FP64 to not be contained in FP32."

        # MX_E4M3 ⊄ MX_E5M2
        CTX1 = AbstractFormat.from_context(fp.MX_E4M3)
        CTX2 = AbstractFormat.from_context(fp.MX_E5M2)
        assert not CTX1 <= CTX2, "Expected MX_E4M3 to not be contained in MX_E5M2."

        # MX_E4M3 \subseteq fixed<-9, 32>
        CTX1 = AbstractFormat.from_context(fp.MX_E4M3)
        CTX2 = AbstractFormat.from_context(fp.FixedContext(True, -9, 32))
        assert CTX1 <= CTX2, "Expected MX_E4M3 to be contained in fixed<-9, 32>."

        # MX_E5M2 ⊄ fixed<-9, 32>
        CTX1 = AbstractFormat.from_context(fp.MX_E5M2)
        CTX2 = AbstractFormat.from_context(fp.FixedContext(True, -9, 32))
        assert not CTX1 <= CTX2, "Expected MX_E5M2 to not be contained in fixed<-9, 32>."

        # INT8 \subseteq FP32
        CTX1 = AbstractFormat.from_context(fp.SINT8)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        assert CTX1 <= CTX2, "Expected INT8 to be contained in FP32."

        # INT4 \subseteq A(3, 0, 4)
        CTX1 = AbstractFormat.from_context(fp.FixedContext(True, 0, 4))
        CTX2 = AbstractFormat(3, 0, fp.RealFloat.from_int(8))
        assert CTX1 <= CTX2, "Expected INT4 to be contained in A(3, 0, 4)."

        # INT4 \subseteq A(4, 0, 12)
        CTX1 = AbstractFormat.from_context(fp.FixedContext(True, 0, 4))
        CTX2 = AbstractFormat(4, 0, fp.RealFloat.from_int(12))
        assert CTX1 <= CTX2, "Expected INT4 to be contained in A(4, 0, 12)."


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
