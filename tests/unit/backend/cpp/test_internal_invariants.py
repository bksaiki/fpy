"""Invariants the emitter assumes of the analyses ahead of it, tested directly.

A broken one raises :class:`CppInternalError`; the corpus-wide gate that no
program reaches one is in `test_corpus_profile`.
"""

import pytest

import fpy2 as fp
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.types import RealType


class TestBothGuardsAskOneQuestion:
    """"Can this value live in that type?" has one implementation.

    The slot-store guard and the operand guard must *route* through
    `_value_fits` rather than each spelling their own test, or a fix to one
    leaves the other behind.  Wiring, not behaviour.
    """

    @staticmethod
    def _slot_store():
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            xs = [fp.round(x)]
            xs[0] = fp.round(x)
            return xs[0]
        return f, [RealType(fp.FP32)], fp.FP32

    @staticmethod
    def _operand():
        """`FP32` operands under an `FP64` context: no signature matches on
        storage, so dispatch reaches cast-to-active and has to convert."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return a + b
        return f, [RealType(fp.FP32), RealType(fp.FP32)], fp.FP64

    @pytest.mark.parametrize('shape', ['_slot_store', '_operand'])
    def test_refusing_the_predicate_refuses_the_shape(self, shape, monkeypatch):
        """With the predicate saying no, every guard that asks it says no.

        A guard keeping its own copy of the test would go on accepting.
        """
        from fpy2.backend.cpp.emitter import CppEmitter

        f, args, ctx = getattr(self, shape)()
        assert CppCompiler().compile(f, ctx=ctx, arg_types=args)

        monkeypatch.setattr(
            CppEmitter, '_value_fits', lambda self, bound, src, want: False)
        with pytest.raises(Exception, match='lossy|narrow'):
            CppCompiler().compile(f, ctx=ctx, arg_types=args)
