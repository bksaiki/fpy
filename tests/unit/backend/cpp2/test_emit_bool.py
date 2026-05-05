"""
Phase 3b tests for the cpp2 emitter — booleans and comparisons.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


class TestBoolAndCompare:
    """Phase 3b — bool literals and comparison expressions."""

    def test_bool_literal_true(self):
        @fp.fpy
        def f() -> bool:
            return True

        out = Cpp2Compiler().compile(f)
        assert out == 'bool f() {\n    return true;\n}\n'

    def test_bool_literal_false(self):
        @fp.fpy
        def f() -> bool:
            return False

        out = Cpp2Compiler().compile(f)
        assert out == 'bool f() {\n    return false;\n}\n'

    def test_pairwise_lt(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            with fp.FP64:
                return x < y

        cc = Cpp2Compiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'return (x < y);' in out
        assert out.startswith('bool f(double x, double y)')

    def test_all_six_comparison_ops(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            with fp.FP64:
                a = x < y
                b = x <= y
                c = x > y
                d = x >= y
                e = x == y
                g = x != y
                return a

        cc = Cpp2Compiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'a = (x < y);' in out
        assert 'b = (x <= y);' in out
        assert 'c = (x > y);' in out
        assert 'd = (x >= y);' in out
        assert 'e = (x == y);' in out
        assert 'g = (x != y);' in out

    def test_chained_comparison(self):
        """``x < y < z`` expands to a conjunction."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> bool:
            with fp.FP64:
                return x < y < z

        cc = Cpp2Compiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)] * 3,
        )
        assert '((x < y) && (y < z))' in out
