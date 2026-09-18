"""
Interpreter checks for what a ``with`` block does to the active context.

`fpy2/interpret/byte.py:_visit_context` lowers ``with e: body`` to

.. code-block:: python

    tmp = __ctx__
    __ctx__ = __fpy_real      # the context expression is evaluated under REAL
    target = __ctx__ = <e>    # full replacement, not a merge
    <body>
    finally: __ctx__ = tmp

Four properties follow, and passes that move ``with`` blocks around depend on
every one of them.  They are pinned here so that a change to the lowering
breaks a test that names the reason rather than silently invalidating a
transform:

1. A nested ``with`` does not inherit from the block around it.
2. A context expression evaluates the same however deeply it is nested.
3. A name bound inside a block is readable after it.
4. A target bound by an outer block is usable inside a nested one.

Values are chosen to straddle FP16 subnormals and overflow, where a context
mix-up shows up immediately instead of being masked by FP64's range.
"""

import fpy2 as fp

# spans FP16 subnormals (1e-8), the normal range, and overflow (65600.0)
_VALUES = (1.1, 3.7, 1e-5, 65600.0, 1e-8)


class TestNoInheritance:
    """A nested ``with`` replaces the active context; it does not refine the
    one around it.  This is what lets a transform move an inner block out of
    its parent."""

    def test_inner_block_matches_the_same_block_unnested(self):
        @fp.fpy
        def nested(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    y = x * 3.0
            return y

        @fp.fpy
        def alone(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = x * 3.0
            return y

        for v in _VALUES:
            assert repr(nested(v)) == repr(alone(v)), f'disagree at {v}'

    def test_the_parent_contributes_nothing(self):
        """Same inner block under two different parents.  If any part of the
        parent leaked in, FP32 and FP64 would not agree."""
        @fp.fpy
        def under_fp32(x: fp.Real) -> fp.Real:
            with fp.FP32:
                with fp.FP16:
                    y = x * 3.0
            return y

        @fp.fpy
        def under_fp64(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with fp.FP16:
                    y = x * 3.0
            return y

        for v in _VALUES:
            assert repr(under_fp32(v)) == repr(under_fp64(v)), f'disagree at {v}'

    def test_the_parent_resumes_after_the_inner_block(self):
        """The replacement is scoped: leaving the inner block restores the
        parent's context rather than keeping the inner one."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                with fp.FP64:
                    a = x * 3.0
                b = x * 3.0
            return b

        @fp.fpy
        def baseline(x: fp.Real) -> fp.Real:
            with fp.FP16:
                b = x * 3.0
            return b

        for v in _VALUES:
            assert repr(f(v)) == repr(baseline(v)), f'disagree at {v}'


class TestContextExpressionScope:
    """The context expression is evaluated under ``REAL``, whatever it is
    nested in, so moving a ``with`` to a different depth cannot change how its
    own context expression evaluates.

    Only a **computed** argument can show this.  A literal one -- the
    ``fp.FP16`` or ``fp.IEEEContext(5, 11, ...)`` that ordinary code writes --
    is not rounded by the active context at all, so a test built on one passes
    whether or not the ``REAL`` reset is there.  The shape that does discriminate
    is the ``MPFixedContext(e - 12, rm)`` of the `PartialContext` docstring,
    where the argument is an FPy expression the active context would round.
    """

    def test_a_computed_argument_is_not_rounded_by_the_parent(self):
        # `MPFixedContext(0, RNE)` rounds 11 to 12, and an IEEE format of 11
        # bits differs from one of 12 at every value below.  So if the parent
        # reached the inner context expression, `nested` would run at 12 bits.
        @fp.fpy
        def nested(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.MPFixedContext(0, fp.RM.RNE):
                with fp.IEEEContext(5, n + 1.0, fp.RM.RNE):
                    y = x * 3.0
            return y

        @fp.fpy
        def alone(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.IEEEContext(5, n + 1.0, fp.RM.RNE):
                y = x * 3.0
            return y

        for v in (1.1, 3.7, 1e-5):
            assert repr(nested(v, 10.0)) == repr(alone(v, 10.0)), f'disagree at {v}'

    def test_the_two_widths_it_discriminates_really_differ(self):
        """Guards the test above: if 11 and 12 bits agreed, it would pass for
        the wrong reason."""
        @fp.fpy
        def at11(x: fp.Real) -> fp.Real:
            with fp.IEEEContext(5, 11, fp.RM.RNE):
                return x * 3.0

        @fp.fpy
        def at12(x: fp.Real) -> fp.Real:
            with fp.IEEEContext(5, 12, fp.RM.RNE):
                return x * 3.0

        for v in (1.1, 3.7, 1e-5):
            assert repr(at11(v)) != repr(at12(v)), f'indistinguishable at {v}'


class TestBindingsEscapeTheBlock:
    """A ``with`` is not a variable scope: what it binds outlives it.  A
    transform may therefore split a block without stranding a definition."""

    def test_a_name_bound_inside_is_readable_after(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = x + 1.0
            return y

        assert f(1.0) == 2

    def test_a_later_block_reads_an_earlier_one(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = x + 1.0
            with fp.FP16:
                b = a + 1.0
            return b

        assert f(1.0) == 3


class TestOuterTargetVisibility:
    """``with e as c:`` binds ``c`` for the whole body, nested blocks
    included.  This is why hoisting a nested block *out* of its parent has to
    check that the block does not read the parent's target."""

    def test_the_outer_target_is_usable_in_a_nested_block(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    with c:
                        y = x + 1.0
            return y

        @fp.fpy
        def baseline(x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = x + 1.0
            return y

        for v in _VALUES:
            assert repr(f(v)) == repr(baseline(v)), f'disagree at {v}'

    def test_the_outer_target_is_usable_after_a_nested_block(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32 as c:
                with fp.FP16:
                    a = x + 1.0
                with c:
                    y = x + 1.0
            return y

        @fp.fpy
        def baseline(x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = x + 1.0
            return y

        for v in _VALUES:
            assert repr(f(v)) == repr(baseline(v)), f'disagree at {v}'
