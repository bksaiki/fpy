"""
Phase 4f tests for the cpp2 emitter — tuple literals.

FPy tuples are accessed via tuple-binding destructuring (handled in a
later phase together with ``IndexedAssign``), not via ``t[i]``
subscripting — so this phase only covers tuple *construction*.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType, TupleType


class TestTupleExpr:
    """Phase 4f — tuple construction with ``std::make_tuple``."""

    def test_pair_return(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> tuple[fp.Real, fp.Real]:
            with fp.FP64:
                return (x + 1, y - 1)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert out.startswith('std::tuple<double, double> f(double x, double y)')
        assert 'return std::make_tuple((x + 1), (y - 1));' in out

    def test_tuple_passed_through_arg(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> tuple[fp.Real, fp.Real]:
            with fp.FP64:
                return p

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::tuple<double, double>' in out
        assert 'return p;' in out

    def test_heterogeneous_tuple(self):
        """Storage selection picks per-element types independently."""

        @fp.fpy
        def f() -> tuple[fp.Real, bool]:
            with fp.FP64:
                return (1.5, True)

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'std::tuple<' in out
        assert 'bool' in out
        assert 'std::make_tuple(' in out
