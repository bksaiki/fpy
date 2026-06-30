"""
Phase 4f tests for the cpp emitter — tuple literals.

FPy tuples are accessed via tuple-binding destructuring (handled in a
later phase together with ``IndexedAssign``), not via ``t[i]``
subscripting — so this phase only covers tuple *construction*.
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType, TupleType


class TestTupleExpr:
    """Phase 4f — tuple construction with ``std::make_tuple``."""

    def test_pair_return(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> tuple[fp.Real, fp.Real]:
            with fp.FP64:
                return (x + 1, y - 1)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert out.startswith('std::tuple<double, double> f(double x, double y)')
        assert (
            'return std::make_tuple('
            '(x + static_cast<double>(1)), '
            '(y - static_cast<double>(1)));'
        ) in out

    def test_tuple_passed_through_arg(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> tuple[fp.Real, fp.Real]:
            with fp.FP64:
                return p

        out = CppCompiler().compile(
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

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
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

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        # The rhs binds to a temp once, then each NamedId is extracted
        # with ``std::get<i>``.
        assert 'auto __cpp_tmp1 = p;' in out
        assert 'double a = std::get<0>(__cpp_tmp1);' in out
        assert 'double b = std::get<1>(__cpp_tmp1);' in out
        assert 'return (a + b);' in out

    def test_assign_with_underscore(self):
        """``_`` positions are skipped — no extraction emitted."""

        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> fp.Real:
            with fp.FP64:
                _, b = p
                return b

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::get<0>' not in out
        assert 'double b = std::get<1>(__cpp_tmp1);' in out

    def test_nested_destructure(self):
        """Nested tuple bindings recurse via fresh temps."""

        @fp.fpy
        def f(p: tuple[tuple[fp.Real, fp.Real], fp.Real]) -> fp.Real:
            with fp.FP64:
                (a, b), c = p
                return a + b + c

        inner = TupleType(RealType(fp.FP64), RealType(fp.FP64))
        outer = TupleType(inner, RealType(fp.FP64))
        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[outer])
        # Outer temp binds the rhs; a fresh inner temp captures the
        # nested tuple slot.
        assert 'auto __cpp_tmp1 = p;' in out
        assert 'auto __cpp_tmp2 = std::get<0>(__cpp_tmp1);' in out
        assert 'double a = std::get<0>(__cpp_tmp2);' in out
        assert 'double b = std::get<1>(__cpp_tmp2);' in out
        assert 'double c = std::get<1>(__cpp_tmp1);' in out

    def test_for_over_list_of_tuples(self):
        @fp.fpy
        def f(xs: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b in xs:
                    acc = acc + a * b
                return acc

        from fpy2.types import ListType
        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(TupleType(RealType(fp.FP64), RealType(fp.FP64)))],
        )
        assert 'for (auto __cpp_tmp1 : xs) {' in out
        assert '        double a = std::get<0>(__cpp_tmp1);' in out
        assert '        double b = std::get<1>(__cpp_tmp1);' in out
        assert 'acc = (acc + (a * b));' in out

    def test_comp_tuple_target(self):
        @fp.fpy
        def f(xs: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                ys = [a + b for (a, b) in xs]
                return ys[0]

        from fpy2.types import ListType
        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(TupleType(RealType(fp.FP64), RealType(fp.FP64)))],
        )
        # Comprehension iterates a tuple-typed temp, destructures, then
        # ``push_back``s the element expression.
        assert 'for (auto __cpp_tmp2 : xs) {' in out
        assert '        double a = std::get<0>(__cpp_tmp2);' in out
        assert '        double b = std::get<1>(__cpp_tmp2);' in out
        assert '.push_back((a + b));' in out



class TestTupleAccessors:
    """``fst`` / ``snd`` lower to ``std::get`` (and ``std::make_tuple`` for
    the tail of a tuple longer than a pair)."""

    def test_fst_emits_get0(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(p)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::get<0>(p)' in out

    def test_snd_pair_emits_get1(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.snd(p)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::get<1>(p)' in out

    def test_snd_longer_emits_make_tuple(self):
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real]) -> tuple[fp.Real, fp.Real]:
            return fp.snd(p)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64), RealType(fp.FP64))],
        )
        # tail of a 3-tuple -> a new tuple of elements 1 and 2
        assert 'std::make_tuple(' in out
        assert 'std::get<1>(' in out
        assert 'std::get<2>(' in out

    def test_fst_snd_chain_folds_to_single_get(self):
        """``fst(snd(p))`` over a 3-tuple reads element 1 directly — one
        ``std::get<1>``, with no intermediate ``std::make_tuple``."""
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(fp.snd(p))

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[TupleType(RealType(fp.FP64), RealType(fp.FP64), RealType(fp.FP64))],
        )
        assert 'std::get<1>(p)' in out
        assert 'make_tuple' not in out

    def test_deep_chain_folds_to_single_get(self):
        """``fst(snd(snd(p)))`` over a 4-tuple folds to ``std::get<2>``."""
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(fp.snd(fp.snd(p)))

        R = RealType(fp.FP64)
        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[TupleType(R, R, R, R)])
        assert 'std::get<2>(p)' in out
        assert 'make_tuple' not in out

    def test_all_snd_chain_to_bare_element_folds(self):
        """``snd(snd(p))`` over a 3-tuple is the bare last element —
        ``std::get<2>``, no ``make_tuple``."""
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.snd(fp.snd(p))

        R = RealType(fp.FP64)
        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[TupleType(R, R, R)])
        assert 'std::get<2>(p)' in out
        assert 'make_tuple' not in out

    def test_unconsumed_tail_still_materializes(self):
        """A ``snd`` whose multi-element tail is the actual result (not
        consumed by an outer ``fst``) still builds a ``std::make_tuple``."""
        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real, fp.Real]) -> tuple[fp.Real, fp.Real, fp.Real]:
            return fp.snd(p)

        R = RealType(fp.FP64)
        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[TupleType(R, R, R, R)])
        assert 'std::make_tuple(' in out
        assert 'std::get<3>(' in out

    def test_nested_element_tuple_rebases(self):
        """When ``snd`` yields a bare element that is itself a tuple, an
        outer ``fst`` indexes into it (a fresh base)."""
        @fp.fpy
        def f(p: tuple[fp.Real, tuple[fp.Real, fp.Real]]) -> fp.Real:
            return fp.fst(fp.snd(p))

        R = RealType(fp.FP64)
        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[TupleType(R, TupleType(R, R))],
        )
        assert 'std::get<0>(std::get<1>(p))' in out
