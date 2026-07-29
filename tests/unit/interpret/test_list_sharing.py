"""
Interpreter behaviour of list sharing across an FPy call boundary.

FPy lists are **shared**: assignment aliases, ``xs[i] = e`` mutates the list
object, and passing or returning a list carries the identity with it.

``Interpreter.eval`` broke that by applying the Python-boundary conversions to
*every* call, FPy-to-FPy included.  Those conversions rebuild lists and tuples,
which severs the alias.  They fire only when an element isn't already a
``Float``/``bool``/``Context``, so an unrounded literal (``xs[0] = 77`` stores a
``Fraction``) was enough to trigger them — making sharing depend on whether the
last store was a literal or a computed value.  That is why the tests below cover
*both* spellings of the same program.

The last class pins what must survive: the conversions still apply at the Python
edge, so a Python caller keeps getting canonical values.  Its final test records a
*separate*, pre-existing gap this work surfaced but did not change — whether that
edge isolates a caller's list at all is representation-dependent.
"""

from fractions import Fraction

import fpy2 as fp

FP64 = fp.FP64


class TestParameterSharing:
    """A callee writing a list parameter writes the caller's list."""

    def test_callee_write_is_visible_to_caller(self):
        @fp.fpy
        def callee(zs: list[fp.Real]) -> fp.Real:
            with FP64:
                zs[0] = 77
                return 0

        @fp.fpy
        def caller(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                _ = callee(xs)
                return xs[0]

        assert float(caller([1.0, 2.0], ctx=FP64)) == 77.0

    def test_sharing_holds_when_the_list_already_holds_a_fraction(self):
        """``xs[0] = 77`` stores an unrounded ``Fraction``, which is what used
        to trip the conversion; the list must still be passed by identity."""
        @fp.fpy
        def touch(zs: list[fp.Real]) -> fp.Real:
            with FP64:
                zs[1] = zs[1] + 1        # computed: stays a Float
                return 0

        @fp.fpy
        def caller(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                xs[0] = 77               # unrounded literal -> Fraction
                _ = touch(xs)
                return xs[1]             # 3 if shared, 2 if the list was copied

        assert float(caller([1.0, 2.0], ctx=FP64)) == 3.0


class TestReturnSharing:
    """A returned list keeps its identity, so the caller's binding aliases it."""

    def test_returned_parameter_aliases(self):
        @fp.fpy
        def ident(zs: list[fp.Real]) -> list[fp.Real]:
            return zs

        @fp.fpy
        def caller(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                ys = ident(xs)
                ys[0] = 99
                return xs[0]

        assert float(caller([1.0, 2.0], ctx=FP64)) == 99.0

    def test_mutate_then_return_aliases(self):
        """The case that revealed the bug: the callee's write was visible to the
        caller *and* the returned list was a copy — an incoherent pair.  The
        write turned an element into a ``Fraction``, which tripped the return
        conversion into rebuilding the list."""
        @fp.fpy
        def bump(zs: list[fp.Real]) -> list[fp.Real]:
            with FP64:
                zs[0] = 77
                return zs

        @fp.fpy
        def caller(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                ys = bump(xs)
                ys[1] = 55           # through the returned alias
                return xs[1]         # 55 if shared, 2 if copied

        assert float(caller([1.0, 2.0], ctx=FP64)) == 55.0

    def test_literal_and_computed_stores_behave_alike(self):
        """Sharing must not depend on element representation.  These two callees
        differ only in whether the stored value is a literal (a ``Fraction``) or
        computed (a ``Float``); before the fix they disagreed."""
        @fp.fpy
        def store_literal(zs: list[fp.Real]) -> list[fp.Real]:
            with FP64:
                zs[0] = 77
                return zs

        @fp.fpy
        def store_computed(zs: list[fp.Real]) -> list[fp.Real]:
            with FP64:
                zs[0] = zs[0] + 76
                return zs

        @fp.fpy
        def via_literal(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                ys = store_literal(xs)
                ys[1] = ys[1] + 53
                return xs[1]

        @fp.fpy
        def via_computed(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                ys = store_computed(xs)
                ys[1] = ys[1] + 53
                return xs[1]

        lit = float(via_literal([1.0, 2.0], ctx=FP64))
        calc = float(via_computed([1.0, 2.0], ctx=FP64))
        assert lit == calc == 55.0


class TestNestedSharing:
    """An inner list handed to a callee is shared too — the identity travels
    through a projection, not just a bare name."""

    def test_callee_writes_an_inner_list(self):
        @fp.fpy
        def callee(row: list[fp.Real]) -> fp.Real:
            with FP64:
                row[0] = 42
                return 0

        @fp.fpy
        def caller(xss: list[list[fp.Real]]) -> fp.Real:
            with FP64:
                _ = callee(xss[0])
                return xss[0][0]

        assert float(caller([[1.0, 2.0], [3.0, 4.0]], ctx=FP64)) == 42.0


class TestPythonBoundary:
    """The fix narrows the boundary conversions to the Python edge rather than
    removing them, so a Python caller still gets canonical values.  Over-applying
    it would have started handing raw ``Fraction``\ s back to Python code.

    The last test pins a separate, pre-existing gap that this work surfaced but
    did not change.
    """

    def test_returned_literal_is_a_float_not_a_fraction(self):
        @fp.fpy
        def f() -> fp.Real:
            with FP64:
                xs = [0.0, 0.0]
                xs[0] = 77          # unrounded literal -> Fraction internally
                return xs[0]

        out = f(ctx=FP64)
        assert isinstance(out, fp.Float), f'expected Float at the boundary, got {type(out)}'
        assert float(out) == 77.0

    def test_returned_list_elements_are_converted(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with FP64:
                xs[0] = 77
                return xs

        out = f([0.0, 0.0], ctx=FP64)
        assert all(isinstance(v, fp.Float) for v in out), (
            f'expected all Float at the boundary, got {[type(v) for v in out]}'
        )

    def test_non_dyadic_rational_stays_exact(self):
        """``_cvt_return`` converts only *dyadic* rationals; a non-dyadic one is
        left as a ``Fraction`` so no precision is invented."""
        @fp.fpy(ctx=fp.REAL)
        def f() -> fp.Real:
            return fp.rational(1, 3)

        assert f() == Fraction(1, 3)

    def test_python_list_isolation_is_representation_dependent(self):
        """**Known gap, pinned deliberately.**

        Whether a Python caller's list is isolated from an FPy callee's writes
        depends on the element representation, because ``_arg_to_value`` uses
        "is every element already an FPy value?" to decide whether to rebuild the
        container.  Those are different questions:

        - a list of Python ``float``\ s needs conversion, so it is rebuilt and
          the caller is isolated;
        - a list of ``Float``\ s (or ``Fraction``\ s) needs none, so it is
          passed by identity and the callee's write reaches the caller.

        Both are asserted below so the inconsistency is visible rather than
        latent.  Deciding it needs a call on whether the Python boundary should
        isolate at all; if it should, the boundary must copy unconditionally
        instead of only when a conversion happens to be required.
        """
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with FP64:
                xs[0] = 99
                return xs[0]

        # needs conversion -> rebuilt -> caller isolated
        py_floats = [1.0, 2.0]
        assert float(f(py_floats, ctx=FP64)) == 99.0
        assert py_floats == [1.0, 2.0]

        # needs no conversion -> passed by identity -> caller is written
        fpy_floats = [fp.Float.from_float(1.0), fp.Float.from_float(2.0)]
        assert float(f(fpy_floats, ctx=FP64)) == 99.0
        assert float(fpy_floats[0]) == 99.0
