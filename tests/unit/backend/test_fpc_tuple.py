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

    def test_fst_snd_chain_folds_to_single_ref(self):
        """``fst(snd(t))`` over a 3-tuple folds to ``(ref t 1)`` — no
        intermediate tail array."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            t = (a, b, c)
            return fp.fst(fp.snd(t))

        s = _compile(f)
        assert '(ref t 1)' in s
        assert 'array (ref' not in s

    def test_deep_chain_folds_to_single_ref(self):
        """``fst(snd(snd(t)))`` over a 4-tuple folds to ``(ref t 2)``."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real) -> fp.Real:
            t = (a, b, c, d)
            return fp.fst(fp.snd(fp.snd(t)))

        s = _compile(f)
        assert '(ref t 2)' in s
        assert 'array (ref' not in s

    def test_all_snd_chain_to_bare_element_folds(self):
        """``snd(snd(t))`` over a 3-tuple is the bare last element ``(ref t 2)``."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            t = (a, b, c)
            return fp.snd(fp.snd(t))

        s = _compile(f)
        assert '(ref t 2)' in s
        assert 'array (ref' not in s

    def test_unconsumed_tail_still_materializes(self):
        """A multi-element tail that is the result still builds an array."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real, d: fp.Real):
            t = (a, b, c, d)
            return fp.snd(t)

        s = _compile(f)
        assert '(array (ref' in s
