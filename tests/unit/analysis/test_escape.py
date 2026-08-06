"""Unit tests for :class:`fpy2.analysis.Escape`.

The distinction the whole thing turns on is *retention is not mutation*: a callee
that writes its argument keeps nothing afterwards, and a summary that confused
the two would give up exactly the kernels worth unboxing.
"""

import fpy2 as fp

from fpy2.analysis import Escape


def _retained(f: fp.Function, callees=None) -> set[int]:
    """Indices of *f*'s parameters that outlive a call to it."""
    summaries = {g.ast: s for g, s in (callees or {}).items()}
    return set(Escape.analyze(f.ast, summaries).retained)


def _summary(f: fp.Function, callees=None):
    summaries = {g.ast: s for g, s in (callees or {}).items()}
    return Escape.analyze(f.ast, summaries)


class TestLocal:
    """Routes visible without looking at any callee."""

    def test_read_only_parameter_is_not_retained(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        assert _retained(f) == set()

    def test_written_parameter_is_not_retained(self):
        """The case most worth getting right: ``xs[0] = 99`` mutates the
        caller's list *during* the call and keeps nothing after it."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                xs[0] = 99
                return xs[0]

        assert _retained(f) == set()

    def test_returned_parameter_is_retained(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return xs

        assert _retained(f) == {0}

    def test_retained_through_an_alias(self):
        """Read off alias regions, not syntax, so this needs no extra rule."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                ys = xs
                return ys

        assert _retained(f) == {0}

    def test_stored_into_another_parameter_is_not_retained(self):
        """A store *copies*, so the caller reaches xs's values through ``yss``
        afterwards but not xs itself — and retention is about identity."""
        @fp.fpy
        def f(yss: list[list[fp.Real]], xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                yss[0] = xs
                return xs[0]

        assert 1 not in _retained(f)

    def test_a_fresh_local_does_not_retain_the_parameter(self):
        @fp.fpy
        def f(xs: list[fp.Real], x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                ys = [x, x]
                ys[0] = xs[0]
                return ys

        assert _retained(f) == set()

    def test_a_scalar_parameter_is_never_retained(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x

        assert _retained(f) == set()


class TestThroughCalls:
    """A callee's summary is what makes the caller's precise."""

    @staticmethod
    def _leaf_reads():
        @fp.fpy
        def g(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return zs[0]
        return g

    @staticmethod
    def _leaf_retains():
        @fp.fpy
        def g(zs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return zs
        return g

    def test_argument_to_a_non_retaining_callee(self):
        g = self._leaf_reads()

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return g(xs)

        assert _retained(g) == set()
        assert _retained(f, {g: _summary(g)}) == set()

    def test_argument_to_a_retaining_callee(self):
        g = self._leaf_retains()

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = g(xs)
                return ys[0]

        assert _retained(g) == {0}
        assert _retained(f, {g: _summary(g)}) == {0}

    def test_a_callee_with_no_summary_retains_everything(self):
        """A foreign function, an unresolved target, or one in a cycle: the
        absence of a summary is the conservative answer."""
        g = self._leaf_reads()

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return g(xs)

        assert _retained(f, callees=None) == {0}

    def test_only_the_retained_position_matters(self):
        """A callee that retains its *second* argument says nothing about the
        first."""
        @fp.fpy
        def g(a: list[fp.Real], b: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                n = a[0]
                return b

        @fp.fpy
        def f(p: list[fp.Real], q: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                r = g(p, q)
                return r[0]

        assert _retained(g) == {1}
        assert _retained(f, {g: _summary(g)}) == {1}
