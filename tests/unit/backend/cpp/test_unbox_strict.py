"""``UnboxMode.STRICT``: unbox everywhere or refuse to compile.

``ALLOW``'s contract is per-level best effort; ``STRICT``'s is a guarantee --
the emitted unit contains no ``fpy::list`` (``std::shared_ptr``).  A program
whose semantics need a shared handle is a compile error naming the list and
the reason it kept its handle, not a handle in the output.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)

STRICT = CppCompiler(unbox=UnboxMode.STRICT)


@fp.fpy
def _scale(xs: list[fp.Real], a: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        return [a * x for x in xs]


@fp.fpy
def _shared(xs: list[fp.Real], c: bool, x: fp.Real) -> fp.Real:
    """`ys` may be `xs` or a fresh list, and `ys[0] = 99` must reach `xs`
    on one path -- the canonical program only a handle can compile."""
    with fp.FP64:
        if c:
            ys = [x, x]
        else:
            ys = xs
        ys[0] = 99
        return xs[0]


@fp.fpy
def _identity(xs: list[fp.Real]) -> list[fp.Real]:
    """Returning a parameter hands the caller a second name for its own
    storage -- unboxed, the return would be a copy and lose writes."""
    with fp.FP64:
        return xs


class TestStrictAccepts:
    """Where every level unboxes, STRICT is exactly ALLOW."""

    def test_fully_unboxable_program_compiles_clean(self):
        out = STRICT.compile(_scale, ctx=fp.FP64, arg_types=[L, R])
        assert 'std::vector<double>' in out
        assert 'fpy::list' not in out

    def test_signature_reports_native_types(self):
        params, ret = STRICT.signature(_scale, ctx=fp.FP64, arg_types=[L, R])
        assert params[0].format() == 'std::vector<double>'
        assert ret.format() == 'std::vector<double>'


class TestStrictRefuses:
    """Where a list must keep its handle, compilation fails -- naming the
    list, the reason, and the way out."""

    def test_shared_list_is_a_compile_error(self):
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            STRICT.compile(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )

    def test_the_error_names_the_list_reason_and_escape_hatch(self):
        with pytest.raises(
            CppCompileError,
            match=r'(?s)`ys` \(depth 0\): shared.*UnboxMode\.ALLOW',
        ):
            STRICT.compile(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )

    def test_returned_parameter_is_a_compile_error(self):
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            STRICT.compile(_identity, ctx=fp.FP64, arg_types=[L])

    def test_a_retaining_callee_fails_strict(self):
        """A callee that keeps its argument forces handles on both ends of
        the call -- leaves-first, the callee is refused before the caller
        is ever emitted."""
        @fp.fpy
        def keep(zs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return zs

        @fp.fpy
        def hand_over(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = keep(xs)
                return ys[0]

        m = Module()
        m.add(hand_over, ctx=fp.FP64, arg_types=[L])
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            STRICT.compile_module(m)

    def test_signature_is_strict_too(self):
        """`signature` goes through the same `analyze`, so an embedding
        program cannot be told types that `compile_module` would refuse."""
        with pytest.raises(CppCompileError, match='strict unboxing failed'):
            STRICT.signature(
                _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
            )


class TestOtherModesUnchanged:
    """STRICT is a third mode, not a change to the other two."""

    @pytest.mark.parametrize(
        'mode', [UnboxMode.ALLOW, UnboxMode.NEVER], ids=['allow', 'never'],
    )
    def test_other_modes_still_compile_shared_programs(self, mode):
        out = CppCompiler(unbox=mode).compile(
            _shared, ctx=fp.FP64, arg_types=[L, BoolType(), R],
        )
        assert 'fpy::list<double>' in out
