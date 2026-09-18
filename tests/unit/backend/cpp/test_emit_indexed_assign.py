"""
Phase 4e tests for the cpp emitter — ``IndexedAssign`` (``xs[i] = e``).

Per the FPy interpreter (``interpret/byte.py``), ``xs[i] = e`` is
*in-place* mutation — no fresh vector.  ``StorageInfer`` recovers
that by unioning the new SSA def the ``IndexedAssign`` introduces
with its ``prev``: they share a storage class and a C++ name, so
the emitter produces a direct subscript-store.
"""

import pytest

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.backend.cpp.unbox import UnboxMode
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

        # `xs` is a parameter *and* the result, so the caller ends up with
        # two handles to one list: shared, and it keeps its handle -- so
        # ALLOW, not the strict default.
        out = CppCompiler(unbox=UnboxMode.ALLOW).compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # the store is through the handle, in place -- and with no copy temp
        assert (
            '(*xs)[static_cast<size_t>(i)] = '
            '((*xs)[static_cast<size_t>(i)] * static_cast<double>(2));'
        ) in out
        assert '_tmp' not in out

    def test_alias_then_mutate_is_observable_through_the_original(self):
        """``ys = xs`` aliases in FPy, so ``ys[0] = 99`` is observable through
        ``xs`` — and now through the generated C++ too.

        ``ys`` binds a ``const`` reference to the *handle*: it cannot be
        rebound, but the elements it points at are mutable, so the
        subscript-store lands on the list ``xs`` also names.  The parameter
        stays ``const&`` for the same reason — ``const`` applies to the handle,
        not the elements.

        Executed end to end by ``_regression_alias_then_mutate`` in the infra
        corpus; this only pins the shape.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                ys = xs
                ys[0] = 99
                return ys

        out = CppCompiler(unbox=UnboxMode.ALLOW).compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'const std::shared_ptr<std::vector<double>>& xs' in out
        assert 'const auto& ys = xs;' in out
        assert '(*ys)[static_cast<size_t>(0)] = 99;' in out
        # Nothing is copied: no temp, and no fresh list.
        assert '_tmp' not in out
        assert 'make_shared' not in out

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

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        # Single ``xs`` declaration; both mutations are direct stores.
        assert 'xs[static_cast<size_t>(0)] = 5;' in out
        assert 'xs[static_cast<size_t>(1)] = 10;' in out
        # No suffixed copy variables.
        assert 'xs_1' not in out
        assert 'xs_2' not in out
        assert '_tmp' not in out

    def test_indexed_assign_arg(self):
        """A function-arg list mutated directly compiles to a direct
        subscript-store; the arg's class absorbs the post-mutation
        SSA def via the in-place coalescing edge."""

        @fp.fpy
        def f(xs: list[fp.Real], i: fp.Real, v: fp.Real) -> fp.Real:
            with fp.FP64:
                xs[i] = v
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[
                ListType(RealType(fp.FP64)),
                RealType(fp.FP64),
                RealType(fp.FP64),
            ],
        )
        assert 'xs[static_cast<size_t>(i)] = v;' in out
        assert 'return xs[static_cast<size_t>(0)];' in out
        # No SSA-suffix variable, no copy temp.
        assert 'xs_1' not in out
        assert '_tmp' not in out

    def test_projection_widened_by_its_own_writes_is_refused(self):
        """``row = xss[i]`` binds a reference, so ``row`` and ``xss``'s element
        are one object -- but ``row``'s class widens on its own writes while the
        container's does not.  Two storages for one object bridge by no
        conversion, so the emitter refuses rather than aliasing at the wrong
        type."""

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            with fp.INTEGER:
                xss = [[1, 2], [3, 4]]
                row = xss[0]
                row[0] = 1000
                return xss

        with pytest.raises(CppCompileError, match='aliases storage of type'):
            CppCompiler(unbox=UnboxMode.ALLOW).compile(f, ctx=fp.INTEGER)


class TestSlotStoreTypes:
    """A slot takes its type from the container, so the value must already fit
    -- unless the emitter is *constructing* it, in which case it is built at the
    slot's type and its own storage is not the question."""

    def test_a_value_that_fits_stores_though_its_type_does_not(self):
        """"Does it fit" is about the *values*, not whether the types nest.
        A `Round` reports its context's type where the value is bounded by the
        operand: ``round_SINT64`` of an ``FP32`` is 24 significand bits, which
        a ``float`` holds exactly.  A store and a fresh list are the same
        conversion and must agree.
        """
        @fp.fpy
        def slot(x: fp.Real):
            zs = [1.5, 2.5]
            with fp.SINT64:
                zs[0] = fp.round(x)
            return zs[0]

        @fp.fpy
        def fresh(x: fp.Real):
            with fp.SINT64:
                zs = [fp.round(x)]
            return zs[0]

        for f in (slot, fresh):
            out = CppCompiler().compile(f, arg_types=[RealType(fp.FP32)])
            assert 'static_cast<float>' in out, f.name

    def test_a_container_widens_rather_than_refusing(self):
        """Why the refusal is hard to reach: a slot's type comes from the
        container's *class*, which the store joins into -- so a value needing
        53 bits makes the container ``double`` rather than being refused."""
        @fp.fpy
        def f(zs: list[fp.Real], x: fp.Real):
            with fp.SINT64:
                zs[0] = fp.round(x)
            return zs[0]

        out = CppCompiler().compile(
            f, arg_types=[ListType(RealType(fp.FP32)), RealType(fp.FP64)])
        assert 'std::vector<double>& zs' in out

    def test_an_allocation_into_a_slot_is_built_at_the_slot(self):
        """``fp.empty``'s bound is the lattice bottom, so its own storage is the
        ladder's first rung -- ``std::vector<uint8_t>`` -- whatever the row it is
        allocating actually holds."""

        @fp.fpy(ctx=fp.FP64)
        def f(xss: list[list[fp.Real]]) -> list[list[fp.Real]]:
            out = fp.empty(len(xss))
            for i in range(len(xss)):
                row = xss[i]
                out[i] = fp.empty(len(row))
                for j in range(len(row)):
                    out[i][j] = row[j]
            return out

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(ListType(RealType(fp.FP64)))],
        )
        assert 'std::vector<std::vector<double>> f(' in out
        assert 'uint8_t' not in out
