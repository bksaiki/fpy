"""
FPCore lowering of the ``fst`` / ``snd`` tuple accessors.

Tuples compile to FPCore arrays, so ``fst``/``snd`` lower to ``ref`` (and,
for the tail of a tuple longer than a pair, an ``array`` of the remaining
refs).  The FPCore backend has no tuple-typed arguments, so the tuples here
are built from scalar arguments inside the body.
"""

import fpy2 as fp

from fpy2 import FPCoreCompiler


def _compile(f) -> str:
    return str(FPCoreCompiler().compile(f))


class TestTupleAccessors:
    def test_fst_emits_ref0(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            t = (a, b, c)
            return fp.fst(t)

        assert '(ref t 0)' in _compile(f)

    def test_snd_pair_emits_ref1(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = (a, b)
            return fp.snd(t)

        assert '(ref t 1)' in _compile(f)

    def test_snd_longer_emits_array_of_refs(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real):
            t = (a, b, c)
            return fp.snd(t)

        # tail of a 3-tuple -> an array of the last two elements
        assert '(array (ref' in _compile(f)
