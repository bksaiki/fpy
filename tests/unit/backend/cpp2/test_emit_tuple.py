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


class TestTupleDestructure:
    """Phase 4f — tuple-binding destructuring.

    Covers ``(a, b) = expr``, the matching ``ForStmt`` /
    ``ListComp`` shapes, ``UnderscoreId`` skips, and nested
    bindings.  Each ``NamedId`` in the binding has its own SSA def
    (and so its own storage class) registered at the enclosing
    statement.
    """

    def test_assign_pair(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> fp.Real:
            with fp.FP64:
                a, b = p
                return a + b

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        # The rhs binds to a temp once, then each NamedId is extracted
        # with ``std::get<i>``.
        assert 'auto __cpp2_tmp1 = p;' in out
        assert 'double a = std::get<0>(__cpp2_tmp1);' in out
        assert 'double b = std::get<1>(__cpp2_tmp1);' in out
        assert 'return (a + b);' in out

    def test_assign_with_underscore(self):
        """``_`` positions are skipped — no extraction emitted."""

        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> fp.Real:
            with fp.FP64:
                _, b = p
                return b

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::get<0>' not in out
        assert 'double b = std::get<1>(__cpp2_tmp1);' in out

    def test_nested_destructure(self):
        """Nested tuple bindings recurse via fresh temps."""

        @fp.fpy
        def f(p: tuple[tuple[fp.Real, fp.Real], fp.Real]) -> fp.Real:
            with fp.FP64:
                (a, b), c = p
                return a + b + c

        inner = TupleType(RealType(fp.FP64), RealType(fp.FP64))
        outer = TupleType(inner, RealType(fp.FP64))
        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[outer])
        # Outer temp binds the rhs; a fresh inner temp captures the
        # nested tuple slot.
        assert 'auto __cpp2_tmp1 = p;' in out
        assert 'auto __cpp2_tmp2 = std::get<0>(__cpp2_tmp1);' in out
        assert 'double a = std::get<0>(__cpp2_tmp2);' in out
        assert 'double b = std::get<1>(__cpp2_tmp2);' in out
        assert 'double c = std::get<1>(__cpp2_tmp1);' in out

    def test_for_over_list_of_tuples(self):
        @fp.fpy
        def f(xs: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b in xs:
                    acc = acc + a * b
                return acc

        from fpy2.types import ListType
        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(TupleType(RealType(fp.FP64), RealType(fp.FP64)))],
        )
        assert 'for (auto __cpp2_tmp1 : xs) {' in out
        assert '        double a = std::get<0>(__cpp2_tmp1);' in out
        assert '        double b = std::get<1>(__cpp2_tmp1);' in out
        assert 'acc = (acc + (a * b));' in out

    def test_comp_tuple_target(self):
        @fp.fpy
        def f(xs: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                ys = [a + b for (a, b) in xs]
                return ys[0]

        from fpy2.types import ListType
        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(TupleType(RealType(fp.FP64), RealType(fp.FP64)))],
        )
        # Comprehension iterates a tuple-typed temp, destructures, then
        # ``push_back``s the element expression.
        assert 'for (auto __cpp2_tmp2 : xs) {' in out
        assert '        double a = std::get<0>(__cpp2_tmp2);' in out
        assert '        double b = std::get<1>(__cpp2_tmp2);' in out
        assert '.push_back((a + b));' in out

