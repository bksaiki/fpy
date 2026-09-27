"""Invariants the emitter assumes of the analyses ahead of it, tested directly.

A broken one raises :class:`CppInternalError`; the corpus-wide gate that no
program reaches one is in `test_corpus_profile`.
"""

import pytest

import fpy2 as fp
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.types import BoolType, RealType


class TestReferenceBindingStorage:
    """A reference binding and a shared storage class are the same claim.

    Where they disagree, the name has the type C++ deduced from its initializer
    rather than the one storage inference chose, and every consumer of
    `storage_of` must remember to compensate — which is how a boxed `uint8_t`
    list once reached a vector of boxed `float` lists.  `binds_by_reference` now
    requires the two to agree, so the divergence cannot arise.
    """

    def test_a_reference_binding_keeps_its_source_storage(self):
        @fp.fpy
        def f(n: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs = [1.0, 2.0]
                ys = xs
                ys[0] = n
                return ys

        out = CppCompiler(unbox=UnboxMode.ALLOW).compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        # the binding is a reference, and both names spell the same type
        assert 'const auto& ys = xs;' in out, out
        decl = next(ln for ln in out.splitlines() if ln.strip().startswith(
            'std::shared_ptr') and ' xs =' in ln)
        assert 'std::shared_ptr<std::vector<double>>' in decl, decl

    def test_a_rebound_name_is_not_a_reference(self):
        """A `const` reference cannot be reassigned, so a rebind copies."""

        @fp.fpy
        def f(c: bool, n: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs = [n, n]
                ys = xs
                if c:
                    ys = [n]
                return ys

        out = CppCompiler(unbox=UnboxMode.ALLOW).compile(
            f, ctx=fp.FP64, arg_types=[BoolType(), RealType(fp.FP64)])
        assert 'const auto& ys' not in out, out


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
