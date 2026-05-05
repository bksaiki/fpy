"""
Phase 4e tests for the cpp2 emitter — ``IndexedAssign`` (``xs[i] = e``).

The pipeline runs ``FuncUpdate`` first, rewriting every
``IndexedAssign`` into ``Assign(xs, ListSet(xs, [i], e))``.  Per the
FPy interpreter (``interpret/byte.py``), ``xs[i] = e`` is *in-place*
mutation — no fresh vector.  ``StorageInfer`` recovers that by
detecting the canonical FuncUpdate shape and unioning the new SSA
def with its ``prev``: they share a storage class and a C++ name,
so the emitter produces a direct subscript-store.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import ListType, RealType


class TestIndexedAssign:

    def test_loop_mutation_is_in_place(self):
        """``xs[i] = e`` inside a loop where ``xs`` keeps the same
        storage class throughout collapses to a direct subscript-store
        — no per-iteration copy of the vector."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                for i in range(len(xs)):
                    xs[i] = xs[i] * 2
                return xs

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'xs[i] = (xs[i] * 2);' in out
        # No copy temp inside the loop body.
        assert '__cpp2_tmp' not in out

    def test_alias_rebind_is_still_in_place(self):
        """``ys = xs; ys[0] = 99`` mutates ``ys`` in place after the
        copy — ``ys[0] = 99`` is a direct subscript-store on ``ys``.
        (C++ value semantics means the prior ``ys = xs;`` already
        copies, so ``xs`` is unaffected — no second copy is
        introduced for the mutation itself.)"""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                ys = xs
                ys[0] = 99
                return ys

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'std::vector<double> ys = xs;' in out
        assert 'ys[0] = 99;' in out
        # No copy temp for the mutation.
        assert '__cpp2_tmp' not in out

    def test_sequential_mutations_in_place(self):
        """Sequential mutations of a freshly-built list reuse the
        same C++ variable — each ``xs[i] = e`` is in-place."""

        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.0, 2.0, 3.0]
                xs[0] = 5.0
                xs[1] = 10.0
                return xs[1]

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        # Single ``xs`` declaration; both mutations are direct stores.
        assert 'xs[0] = 5;' in out
        assert 'xs[1] = 10;' in out
        # No suffixed copy variables.
        assert 'xs_1' not in out
        assert 'xs_2' not in out
        assert '__cpp2_tmp' not in out

    def test_indexed_assign_arg(self):
        """A function-arg list mutated directly compiles to a direct
        subscript-store; the arg's class absorbs the post-mutation
        SSA def via the in-place coalescing edge."""

        @fp.fpy
        def f(xs: list[fp.Real], i: fp.Real, v: fp.Real) -> fp.Real:
            with fp.FP64:
                xs[i] = v
                return xs[0]

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[
                ListType(RealType(fp.FP64)),
                RealType(fp.FP64),
                RealType(fp.FP64),
            ],
        )
        assert 'xs[i] = v;' in out
        assert 'return xs[0];' in out
        # No SSA-suffix variable, no copy temp.
        assert 'xs_1' not in out
        assert '__cpp2_tmp' not in out
