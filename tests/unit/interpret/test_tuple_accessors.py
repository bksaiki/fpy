"""
Interpreter behaviour of the ``fst`` / ``snd`` tuple accessors.

``fst`` returns the head; ``snd`` returns the tail — the bare second
element for a pair, else the tuple of the remaining elements.
"""

import fpy2 as fp


class TestTupleAccessors:
    def test_fst_head(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(t)

        assert float(f((1.0, 2.0, 3.0), ctx=fp.FP64)) == 1.0

    def test_fst_one_tuple(self):
        @fp.fpy
        def f(t: tuple[fp.Real]) -> fp.Real:
            return fp.fst(t)

        assert float(f((7.0,), ctx=fp.FP64)) == 7.0

    def test_snd_pair_is_bare(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.snd(t)

        assert float(f((4.0, 5.0), ctx=fp.FP64)) == 5.0

    def test_snd_longer_is_tuple(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            a, b = fp.snd(t)        # snd((x, y, z)) == (y, z)
            with fp.FP64:
                return a + b

        assert float(f((1.0, 2.0, 3.0), ctx=fp.FP64)) == 5.0

    def test_chained_fst_snd(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(fp.snd(t))

        assert float(f((1.0, 2.0, 3.0), ctx=fp.FP64)) == 2.0
