"""
Interpreter behaviour of the ``fst`` / ``snd`` tuple accessors.

``fst`` returns the head; ``snd`` returns the tail — the bare second
element for a pair, else the tuple of the remaining elements.
"""

import fpy2 as fp


class TestTupleAccessors:

    def test_fst(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(t)

        assert float(f((1.0, 2.0), ctx=fp.FP64)) == 1.0

    def test_snd(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.snd(t)

        assert float(f((1.0, 2.0), ctx=fp.FP64)) == 2.0
