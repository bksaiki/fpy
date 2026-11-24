"""Unit tests for :class:`fpy2.transform.CopyPropagate`."""

import fpy2 as fp
from fpy2.transform import CopyPropagate


class TestCopyPropagate:

    def test_a_copy_of_a_name_defined_once_propagates(self):
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real) -> fp.Real:
            x = y
            return x + 1

        @fp.fpy(ctx=fp.FP64)
        def f_expect(y: fp.Real) -> fp.Real:
            x = y
            return y + 1

        g = CopyPropagate.apply(f.ast)
        g.name = f_expect.name
        assert g.is_equiv(f_expect.ast)

    def test_a_copy_of_a_rebound_name_is_kept(self):
        """`x` holds the old `y`; substituting `y` would read the new one."""
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real) -> fp.Real:
            x = y
            y = y + 1
            return x

        assert CopyPropagate.apply(f.ast).is_equiv(f.ast)

    def test_a_source_redefined_after_the_last_use_still_propagates(self):
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real) -> fp.Real:
            x = y
            z = x + 1
            y = y * 2
            return z + y

        @fp.fpy(ctx=fp.FP64)
        def f_expect(y: fp.Real) -> fp.Real:
            x = y
            z = y + 1
            y = y * 2
            return z + y

        g = CopyPropagate.apply(f.ast)
        g.name = f_expect.name
        assert g.is_equiv(f_expect.ast)

    def test_a_source_changed_by_a_loop_is_kept(self):
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real, n: fp.Real) -> fp.Real:
            x = y
            i = 0
            while i < n:
                y = y + 1
                i = i + x
            return i

        assert CopyPropagate.apply(f.ast).is_equiv(f.ast)

    def test_a_source_changed_before_a_while_condition_is_kept(self):
        """The condition re-runs each iteration, against the loop's `y`."""
        @fp.fpy(ctx=fp.FP64)
        def f(y: fp.Real, n: fp.Real) -> fp.Real:
            x = y
            i = 0
            while x + i < n:
                y = y + 1
                i = i + 1
            return i

        assert CopyPropagate.apply(f.ast).is_equiv(f.ast)
