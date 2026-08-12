"""
Structural equality in the interpreter.

FPy's ``==`` / ``!=`` are ``a -> a -> bool``, so a tuple or list can reach
them.  Python's own sequence equality compares elements with ``is`` before
``==``, which would let a NaN inside a container compare equal to itself --
so containers are walked elementwise and only scalars reach ``==``.
"""

import fpy2 as fp
import pytest


class TestNaNIsNeverEqual:
    """A NaN is unequal to everything, including itself, at any depth."""

    def test_a_scalar_nan(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            return fp.nan() == fp.nan()

        assert f() is False

    def test_a_tuple_compared_with_itself(self):
        """The case Python's identity shortcut gets wrong: the operands are
        the *same object*, so ``is`` would answer before ``__eq__`` runs."""
        @fp.fpy(ctx=fp.FP64)
        def f():
            t = (fp.nan(), 1.0)
            return t == t

        assert f() is False

    def test_a_list_compared_with_itself(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            xs = [fp.nan()]
            return xs == xs

        assert f() is False

    def test_two_separately_built_tuples(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = (fp.nan(), 1.0)
            b = (fp.nan(), 1.0)
            return a == b

        assert f() is False

    def test_a_nan_nested_two_deep(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = [(1.0, fp.nan())]
            return a == a

        assert f() is False


class TestStructuralEquality:
    """Containers compare elementwise, not by identity."""

    def test_equal_tuples(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = (1.0, 2.0)
            b = (1.0, 2.0)
            return a == b

        assert f() is True

    def test_unequal_tuples(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = (1.0, 2.0)
            b = (1.0, 3.0)
            return a == b

        assert f() is False

    def test_equal_lists(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            return [1.0, 2.0] == [1.0, 2.0]

        assert f() is True

    def test_nested_containers(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = [(1.0, 2.0), (3.0, 4.0)]
            b = [(1.0, 2.0), (3.0, 4.0)]
            return a == b

        assert f() is True

    def test_lists_of_different_length(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            return [1.0, 2.0] == [1.0]

        assert f() is False

    def test_inequality_negates(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = (1.0, 2.0)
            b = (1.0, 3.0)
            return a != b

        assert f() is True

    def test_signed_zeros_are_equal(self):
        """``-0.0 == 0.0`` is IEEE-true, and stays so inside a container."""
        @fp.fpy(ctx=fp.FP64)
        def f():
            return (-0.0, 1.0) == (0.0, 1.0)

        assert f() is True


class TestScalarsAndChains:
    """The common cases keep working."""

    @pytest.mark.parametrize('a, b, eq', [(1.0, 1.0, True), (1.0, 2.0, False)])
    def test_reals(self, a, b, eq):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real):
            return x == y

        @fp.fpy(ctx=fp.FP64)
        def g(x: fp.Real, y: fp.Real):
            return x != y

        assert f(a, b) is eq
        assert g(a, b) is (not eq)

    def test_booleans(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return fp.signbit(x) != fp.isnan(x)

        assert f(-1.0) is True
        assert f(1.0) is False

    @pytest.mark.parametrize('args, expect', [
        ((1.0, 1.0, 1.0), True), ((1.0, 1.0, 2.0), False),
    ])
    def test_an_equality_chain(self, args, expect):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real):
            return x == y == z

        assert f(*args) is expect

    @pytest.mark.parametrize('args, expect', [
        ((1.0, 2.0, 2.0), True), ((1.0, 2.0, 3.0), False),
    ])
    def test_a_chain_mixing_ordering_and_equality(self, args, expect):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real):
            return x < y == z

        assert f(*args) is expect

    def test_a_chain_evaluates_each_operand_once(self):
        """A middle operand is shared by two pairs, so it is bound rather than
        re-evaluated -- an FPy call may mutate a list it was handed."""
        import ast as pyast

        from fpy2.interpret.byte import BytecodeCompiler

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real):
            return x == y == z

        tree = BytecodeCompiler(f.ast, f.env)._visit_function(f.ast, None)
        src = pyast.unparse(pyast.Module(body=[tree], type_ignores=[]))
        assert src.count(':= y') == 1, src


class TestTheRuntimeRejectsWhatTypesReject:
    """Type inference is an optional analysis, so every program it rejects has
    to be rejected here too -- otherwise the interpreter answers a question the
    language does not define."""

    def test_ordering_rejects_tuples(self):
        """Python orders tuples lexicographically; FPy does not."""
        @fp.fpy(ctx=fp.FP64)
        def f():
            a = (1.0, 2.0)
            b = (1.0, 3.0)
            return a >= b

        with pytest.raises(TypeError, match='expects real operands'):
            f()

    def test_ordering_rejects_lists(self):
        @fp.fpy(ctx=fp.FP64)
        def f():
            return [1.0] < [2.0]

        with pytest.raises(TypeError, match='expects real operands'):
            f()

    def test_ordering_rejects_booleans(self):
        """Python orders a ``bool`` as an integer; FPy does not."""
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return fp.signbit(x) < fp.isnan(x)

        with pytest.raises(TypeError, match='expects real operands'):
            f(-1.0)

    def test_equality_rejects_mismatched_scalars(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return fp.signbit(x) == x

        with pytest.raises(TypeError, match='same type'):
            f(1.0)

    def test_equality_rejects_a_tuple_against_a_list(self):
        """Different FPy types even when the elements match."""
        @fp.fpy(ctx=fp.FP64)
        def f():
            return (1.0, 2.0) == [1.0, 2.0]

        with pytest.raises(TypeError, match='same type'):
            f()

    def test_a_guarded_chain_still_orders_reals(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real, z: fp.Real):
            return x < y < z

        assert f(1.0, 2.0, 3.0) is True
        assert f(1.0, 3.0, 2.0) is False


class TestGeneratedNamesDoNotShadow:
    """The bytecode compiler mints temporaries for comparison chains and
    ``with`` scopes.  ``Gensym`` only avoids names it was told about, so the
    program's own names have to be reserved."""

    def test_a_chain_temp_does_not_shadow(self):
        @fp.fpy(ctx=fp.FP64)
        def f(a: fp.Real, b: fp.Real, c: fp.Real):
            __fpy_cmp = 5.0
            r = a == b == c
            return __fpy_cmp

        assert f(1.0, 2.0, 3.0) == 5.0

    def test_a_context_temp_does_not_shadow(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            __fpy_ctx_tmp = 5.0
            with fp.FP32:
                y = fp.round(x)
            return __fpy_ctx_tmp

        assert f(1.0) == 5.0
