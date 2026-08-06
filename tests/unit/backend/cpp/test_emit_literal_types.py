"""A literal argument to a C++ call must state its own type.

A literal's storage comes from its *value*, so ``1.5`` under FP32 matches the
``float`` signature while the token it prints is a ``double`` — and nothing
inserts a cast, because storage and signature agree.  Harmless wherever a
declaration supplies the type, and exact for ``+ - * /`` since double rounding
equals single rounding when ``2p + 2 <= 53``.  Not harmless where the callee
takes its type *from* the argument: ``fpy::min`` / ``fpy::max`` are templates, so
mixed types fail to deduce and the C++ does not compile, and the ``<cmath>``
overload sets silently pick the wider overload — which for ``fma``, outside
``2p + 2``, rounds twice where FPy rounds once.

**Paths, not ops.**  The op table's 40 call-form signatures share four dispatch
paths, so testing each op would exercise the same four routes forty times.  The
risk worth guarding is a *new* path added without routing literals through
``_call_arg``, so the last test asserts the call-form arity set is closed.

``optimize=False`` throughout, or ``RoundElim`` and const-folding rewrite these
before the emitter sees them.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.target import make_op_table
from fpy2.types import RealType

_R32 = RealType(fp.FP32)


def _emit(func, nargs: int) -> str:
    return CppCompiler(optimize=False).compile(
        func, ctx=fp.FP32, arg_types=[_R32] * nargs,
    )


class TestLiteralArgumentsStateTheirType:
    """One test per dispatch path that emits a C++ call."""

    def test_unary_call(self):
        """The unary op-table path: ``std::sqrt`` and 31 others."""

        @fp.fpy(ctx=fp.FP32)
        def f(y: fp.Real) -> fp.Real:
            return y + fp.sqrt(0.25)

        out = _emit(f, 1)
        assert 'std::sqrt(static_cast<float>(0.25))' in out, out

    def test_binary_call(self):
        """The binary op-table path: ``std::copysign`` and 6 others."""

        @fp.fpy(ctx=fp.FP32)
        def f(y: fp.Real) -> fp.Real:
            return fp.copysign(y, 1.5)

        out = _emit(f, 1)
        assert 'static_cast<float>(1.5)' in out, out

    def test_ternary_call_is_fma(self):
        """The ternary path, which is ``std::fma`` alone.

        The one operation ``2p + 2`` does not cover, so this is where a wide
        overload changes the answer rather than just the spelling.  The
        documented case: ``y * z`` exactly a binary32 midpoint gives the
        interpreter ``0x1.000002p+84`` and the ``double`` overload
        ``0x1p+84``.
        """

        @fp.fpy(ctx=fp.FP32)
        def f(y: fp.Real, z: fp.Real) -> fp.Real:
            return fp.fma(y, z, 0.25)

        out = _emit(f, 2)
        assert 'std::fma(y, z, static_cast<float>(0.25))' in out, out

    def test_min_max_template(self):
        """``_emit_min_max``, which emits our own ``fpy::max`` template.

        The only path where the failure is a *compile error* rather than a
        silent choice: template deduction sees ``float`` and ``double`` and
        reports conflicting types for ``T``.
        """

        @fp.fpy(ctx=fp.FP32)
        def f(y: fp.Real) -> fp.Real:
            return fp.fmax(y, 1.5)

        out = _emit(f, 1)
        assert 'fpy::max(y, static_cast<float>(1.5))' in out, out

    def test_a_literal_already_of_the_target_type_is_left_alone(self):
        """The rule that stops this becoming the uniform-cast alternative.

        Pinned on ``_literal_cpp_type`` rather than on emitted text, because an
        FP64 program cannot exhibit it: a small dyadic literal's *storage* is
        ``float`` even under FP64 (its value fits binary32), so the pre-existing
        ``_maybe_cast`` inserts a widening cast before ``_call_arg`` is asked.
        A literal that is exactly a ``double`` and too fine for a ``float``
        would show it, and is not expressible -- ``Decnum`` comes from Python's
        float ``repr``, so anything finer than a small dyadic is not exactly a
        double either and is refused outright.
        """
        from fpy2.ast.fpyast import Decnum, Integer
        from fpy2.backend.cpp.emitter import CppEmitter
        from fpy2.backend.cpp.types import CppScalar

        lit = CppEmitter._literal_cpp_type
        # a fraction goes out through `repr(float)`, so the token is a double
        assert lit(Decnum('1.5', None)) == CppScalar.F64
        # a decimal integer literal is `int` while it fits
        assert lit(Integer(1, None)) == CppScalar.S32
        assert lit(Integer(2 ** 40, None)) == CppScalar.S64
        # ...and past `long long` no integer literal holds it at all
        assert lit(Integer(2 ** 100, None)) is None

    def test_the_bound_is_on_magnitude_not_the_signed_range(self):
        """C++ has no negative literal, so the bound is two-sided.

        ``-2147483648`` is unary minus applied to ``2147483648``, which does not
        fit an ``int`` -- so the expression is a ``long``, not the ``int`` its
        value would suggest.  Getting this wrong would skip a cast in a deduced
        position for exactly that value.
        """
        from fpy2.ast.fpyast import Integer
        from fpy2.backend.cpp.emitter import CppEmitter
        from fpy2.backend.cpp.types import CppScalar

        lit = CppEmitter._literal_cpp_type
        assert lit(Integer(2 ** 31 - 1, None)) == CppScalar.S32
        assert lit(Integer(-(2 ** 31 - 1), None)) == CppScalar.S32
        assert lit(Integer(-(2 ** 31), None)) == CppScalar.S64
        assert lit(Integer(2 ** 31, None)) == CppScalar.S64

    def test_infix_operators_are_left_alone(self):
        """An infix operator deduces nothing, so the promotion is C++'s own.

        Sound by the ``2p + 2`` argument: the exact result of a binary32
        ``+ - * /`` fits binary64, so computing there and rounding to binary32
        gives the same answer as rounding once.  Pinned because casting here
        instead would be the uniform-but-noisy alternative.
        """

        @fp.fpy(ctx=fp.FP32)
        def f(y: fp.Real) -> fp.Real:
            return y * 1.5

        out = _emit(f, 1)
        assert '(y * 1.5)' in out, out


class TestThePathSetIsClosed:
    """The enumeration: no call-form arity may exist without a test above."""

    def test_no_unhandled_call_form_arity(self):
        """Fails when the op table grows a call-form arity with no case here.

        This is the failure mode option A carries: the fix is per-path, so a new
        path is a new way to reintroduce the bug.  Checking it is cheaper than
        remembering it.
        """
        table = make_op_table()
        arities_with_calls = set()
        for name, sigs_by_op in (
            ('unary', table.unary),
            ('binary', table.binary),
            ('ternary', table.ternary),
        ):
            for sigs in sigs_by_op.values():
                for s in sigs:
                    if s.is_call:
                        arities_with_calls.add(name)
                        break
        assert arities_with_calls == {'unary', 'binary', 'ternary'}, (
            f'call-form arities are {sorted(arities_with_calls)}; every one '
            f'needs a literal-argument test in this file, and its dispatch '
            f'path needs to route literals through `_call_arg`'
        )

    def test_every_call_form_op_shares_a_tested_path(self):
        """The count, so a *large* table change is visible.

        Not a correctness property -- the paths are what matter -- but a jump
        here means ops were added, which is the moment to check they dispatch
        the same way.
        """
        table = make_op_table()
        n = 0
        for name, sigs_by_op in (
            ('unary', table.unary),
            ('binary', table.binary),
            ('ternary', table.ternary),
        ):
            for sigs in sigs_by_op.values():
                if any(s.is_call for s in sigs):
                    n += 1
        assert n == 40, (
            f'{n} ops have a call-form signature, expected 40 -- update this '
            f'count deliberately, having checked the new ops dispatch through '
            f'the paths tested above'
        )
