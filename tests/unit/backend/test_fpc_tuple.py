"""
FPCore lowering of the ``fst`` / ``snd`` tuple accessors.

``fst``/``snd`` are pair projections; tuples compile to FPCore arrays, so
they lower to ``(ref t 0)`` / ``(ref t 1)``.  A chain over a nested pair
lowers to nested refs.  The FPCore backend has no tuple-typed arguments, so
the tuples here are built from scalar arguments inside the body.
"""

import fpy2 as fp
import pytest

from fpy2 import FPCoreCompiler
from fpy2.backend.fpc import FPCoreCompileError


def _compile(f) -> str:
    return str(FPCoreCompiler().compile(f))


class TestTupleAccessors:
    def test_fst_emits_ref0(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = (a, b)
            return fp.fst(t)

        assert '(ref t 0)' in _compile(f)

    def test_snd_pair_emits_ref1(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = (a, b)
            return fp.snd(t)

        assert '(ref t 1)' in _compile(f)

    def test_chain_over_nested_pair(self):
        """``fst(snd(t))`` over a nested pair lowers to nested refs."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            t = (a, (b, c))
            return fp.fst(fp.snd(t))

        assert '(ref (ref t 1) 0)' in _compile(f)


class TestComparisonOperands:
    """FPCore compares numbers, so an aggregate -- well-typed under FPy's
    ``a -> a -> bool`` -- has to be refused here."""

    def test_a_tuple_comparison_is_refused(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            a = (x, y)
            b = (y, x)
            return 1.0 if a == b else 0.0

        with pytest.raises(FPCoreCompileError, match='compares numbers only'):
            FPCoreCompiler().compile(f)

    def test_a_list_comparison_is_refused(self):
        @fp.fpy
        def f() -> fp.Real:
            a = [1.0, 2.0]
            b = [1.0, 2.0]
            return 1.0 if a == b else 0.0

        with pytest.raises(FPCoreCompileError, match='compares numbers only'):
            FPCoreCompiler().compile(f)

    def test_a_real_comparison_still_compiles(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            return x == y

        assert '==' in str(FPCoreCompiler().compile(f))


class TestComparisonChains:
    """FPCore's comparisons are n-ary, so a run of one operator is a single
    call and a change of operator starts a new one at the shared operand."""

    def test_a_run_of_one_operator_is_one_call(self):
        @fp.fpy
        def lt(x: fp.Real, y: fp.Real, z: fp.Real) -> bool:
            return x < y < z

        @fp.fpy
        def le(x: fp.Real, y: fp.Real, z: fp.Real) -> bool:
            return x <= y <= z

        assert '(< x y z)' in str(FPCoreCompiler().compile(lt))
        assert '(<= x y z)' in str(FPCoreCompiler().compile(le))

    def test_a_change_of_operator_starts_a_new_call(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> bool:
            return x < y == z

        assert '(and (< x y) (== y z))' in str(FPCoreCompiler().compile(f))

    def test_three_runs(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real, w: fp.Real) -> bool:
            return x < y == z > w

        out = str(FPCoreCompiler().compile(f))
        assert '(and (< x y) (== y z) (> z w))' in out, out

    def test_a_two_operand_comparison_is_unchanged(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            return x < y

        assert '(< x y)' in str(FPCoreCompiler().compile(f))
