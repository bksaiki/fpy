"""
Foreign values: the ``Foreign`` arm of the interpreter's value ADT
(``fpy2/interpret/value.py``).

A Python object with no FPy form is wrapped as ``Foreign`` when it
enters a program (argument, free variable, attribute read) and unwrapped
when it leaves (return, context-constructor argument, foreign call).
FPy cannot operate on one — only pass it along.
"""

import random
import types
from fractions import Fraction

import pytest

import fpy2 as fp
from fpy2 import FP64, REAL, Foreign
from fpy2.interpret.value import from_value, to_value

_OBJ = object()
_ATTR = types.SimpleNamespace(x=5.0, inner=types.SimpleNamespace(y=2))
_FRAC = Fraction(1, 3)
_STR = 'hello'


class TestForeignClass:

    def test_identity_eq(self):
        # payload identity, not `==`: two equal-but-distinct payloads differ
        assert Foreign(_OBJ) == Foreign(_OBJ)
        assert Foreign([1]) != Foreign([1])
        assert hash(Foreign(_OBJ)) == id(_OBJ)

    def test_to_value_wraps_opaque(self):
        v = to_value(_OBJ)
        assert isinstance(v, Foreign) and v.val is _OBJ
        # idempotent
        assert to_value(v) is v

    def test_from_value_unwraps(self):
        assert from_value(Foreign(_OBJ)) is _OBJ
        assert from_value((Foreign(_OBJ),))[0] is _OBJ

    def test_value_kinds_never_wrap(self):
        for x in [True, fp.Float.from_float(1.0), Fraction(1, 3), FP64]:
            assert not isinstance(to_value(x), Foreign)


class TestBoundary:

    def test_free_var_round_trip(self):
        @fp.fpy
        def f():
            return _OBJ
        assert f(ctx=FP64) is _OBJ

    def test_argument_round_trip(self):
        @fp.fpy
        def f(x):
            return x
        assert f(_OBJ, ctx=FP64) is _OBJ

    def test_container_round_trip(self):
        @fp.fpy
        def f(x):
            return x
        out = f((_OBJ, _STR), ctx=FP64)
        assert out[0] is _OBJ and out[1] is _STR

    def test_fraction_free_var_is_not_foreign(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return _FRAC + 1
        assert f() == Fraction(4, 3)


class TestAttribute:

    def test_result_is_classified(self):
        # `e.name` re-classifies: a native `float` attribute becomes `Float`
        @fp.fpy
        def f() -> fp.Real:
            return _ATTR.x
        assert isinstance(f(ctx=FP64), fp.Float)

    def test_nested_attribute(self):
        @fp.fpy
        def f() -> fp.Real:
            return _ATTR.inner.y + 1
        assert f(ctx=FP64) == 3


class TestExits:

    def test_context_constructor_arg(self):
        # opaque payload passes into a constructor's pass-through parameter
        r = random.Random(0)

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.IEEEContext(8, 32, rng=r):
                y = x + 0
            return y
        assert float(f(1.5, ctx=FP64)) == 1.5

    def test_callee_unwraps(self):
        g = _helper

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x)
        assert float(f(1.0, ctx=FP64)) == 2.0

    def test_print_returns_none(self, capsys):
        @fp.fpy
        def f(x: fp.Real):
            y = print(_STR, x)
            return y
        assert f(1.0, ctx=FP64) is None
        assert _STR in capsys.readouterr().out


class TestOpaque:

    def test_arithmetic_rejected(self):
        @fp.fpy
        def f() -> fp.Real:
            return _OBJ + 1  # type: ignore[operator]
        with pytest.raises(TypeError):
            f(ctx=FP64)

    def test_comparison_rejected(self):
        @fp.fpy
        def f() -> bool:
            return _OBJ < 1  # type: ignore[operator]
        with pytest.raises(TypeError):
            f(ctx=FP64)

    def test_iteration_rejected(self):
        @fp.fpy
        def f() -> fp.Real:
            t = 0
            for x in xs2:
                t = t + x
            return t
        with pytest.raises(TypeError):
            f(ctx=FP64)


# iterable, but not a list — foreign to FPy
class _NotAList:
    def __iter__(self):
        return iter([1.0])

xs2 = _NotAList()


@fp.fpy
def _helper(x: fp.Real) -> fp.Real:
    return x + 1
