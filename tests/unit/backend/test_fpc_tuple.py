"""
FPCore lowering of the ``fst`` / ``snd`` tuple accessors.

``fst``/``snd`` are pair projections; tuples compile to FPCore arrays, so
they lower to ``(ref t 0)`` / ``(ref t 1)``.  A chain over a nested pair
lowers to nested refs.  The FPCore backend has no tuple-typed arguments, so
the tuples here are built from scalar arguments inside the body.
"""

import fpy2 as fp

from fpy2 import FPCoreCompiler


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
