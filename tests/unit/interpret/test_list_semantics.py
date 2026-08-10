"""
Interpreter conformance with the *Lists* section of the derived semantics.

An FPy list is a core list of *references* — one cell per element — so the three
consequences the documentation draws from that encoding are what these tests
pin:

* elements are mutable, the length is not;
* binding shares cells, including through a tuple field;
* ownership stops at the cell — construction gives every element a fresh cell,
  but that cell holds the element's *value*, so a nested list's rows stay shared.

The third is the one worth having tests for: it is the boundary that makes
``[xs]`` cheap and ``[xs][0][0] = e`` visible through ``xs``, and nothing else in
the suite states it.  ``test_derived_semantics`` covers direct aliasing
(``test_indexed_assign_is_inplace``) and ``test_list_sharing`` covers the call
boundary, so neither is repeated here.

Every program observes sharing *inside* a single FPy function and returns a
scalar, because the Python boundary rebuilds containers unconditionally (see
``test_list_sharing.TestPythonBoundary``) — an alias checked from Python would
be testing the boundary instead.
"""

import fpy2 as fp

FP64 = fp.FP64


class TestElementsMutableLengthFixed:
    """``E-Update`` replaces a cell's contents; only ``E-List`` builds a list
    value, and it fixes the length."""

    def test_store_leaves_the_length_alone(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            xs[0] = 9
            return len(xs)

        assert float(f([1.0, 2.0], ctx=FP64)) == 2.0


class TestBindingSharesCells:
    """``E-Assign`` copies nothing, so a second name — or a tuple field — reaches
    the same cells."""

    def test_a_tuple_field_shares_the_list(self):
        """A tuple holds its fields' values, and a list's value is its structure
        of cells, so a list reached through a tuple is shared, not copied."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            t = (xs, 1.0)
            a, b = t
            a[0] = 9
            return xs[0]

        assert float(f([1.0, 2.0], ctx=FP64)) == 9.0


class TestOwnershipStopsAtTheCell:
    """Construction allocates a fresh cell per element, but the cell holds the
    element's *value* — so a list owns its own cells and nothing deeper."""

    def test_construction_shares_the_rows_it_is_built_from(self):
        """``[xs]`` allocates one fresh cell holding ``xs``'s structure, so
        ``xs``'s own cells are reachable through it."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            yss = [xs]
            yss[0][0] = 9
            return xs[0]

        assert float(f([1.0, 2.0], ctx=FP64)) == 9.0

    def test_a_comprehension_shares_them_too(self):
        """A comprehension is construction, so it copies no deeper either."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            yss = [row for row in xss]
            yss[0][0] = 9
            return xss[0][0]

        assert float(f([[1.0, 2.0]], ctx=FP64)) == 9.0

    def test_the_new_cell_is_fresh(self):
        """The other half of the boundary: the cell ``[xs]`` allocates is its
        own, so overwriting it does not reach ``xs``."""
        @fp.fpy
        def f(xs: list[fp.Real], zs: list[fp.Real]) -> fp.Real:
            yss = [xs]
            yss[0] = zs
            return xs[0]

        assert float(f([1.0, 2.0], [7.0, 8.0], ctx=FP64)) == 1.0

    def test_a_projection_reads_through_its_cell(self):
        """``xss[i]`` in value position is **E-Index** then **E-Deref**, so it
        binds the row's *contents*.  Overwriting the slot afterwards replaces
        those contents in the slot's cell, which the earlier binding no longer
        refers to — as in Python."""
        @fp.fpy
        def f(xss: list[list[fp.Real]], zs: list[fp.Real]) -> fp.Real:
            row = xss[0]
            xss[0] = zs
            return row[0]

        assert float(f([[1.0, 2.0], [3.0, 4.0]], [7.0, 8.0], ctx=FP64)) == 1.0


class TestSliceCopiesTheSpine:
    """``xs[start:stop]`` reads each element and rebuilds, so it is construction:
    a fresh cell per element, no deeper."""

    def test_a_write_to_the_slice_does_not_reach_the_original(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            ys = xs[:]
            ys[0] = 9
            return xs[0]

        assert float(f([1.0, 2.0], ctx=FP64)) == 1.0

    def test_but_the_rows_are_shared(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            yss = xss[:]
            yss[0][0] = 9
            return xss[0][0]

        assert float(f([[1.0, 2.0], [3.0, 4.0]], ctx=FP64)) == 9.0


class TestListProducingBuiltinsHoldTuples:
    """``zip`` and ``enumerate`` build lists whose elements are tuples, so their
    cells hold tuples whose fields hold what the projection dereferenced to — a
    copy for a scalar, the shared row for a list."""

    def test_zip_pairs_share_their_rows(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]], yss: list[list[fp.Real]]) -> fp.Real:
            ps = zip(xss, yss)
            a, b = ps[0]
            a[0] = 9
            return xss[0][0]

        assert float(f([[1.0, 2.0]], [[3.0, 4.0]], ctx=FP64)) == 9.0

    def test_enumerate_pairs_share_their_rows(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            ps = enumerate(xss)
            i, row = ps[0]
            row[0] = 9
            return xss[0][0]

        assert float(f([[1.0, 2.0]], ctx=FP64)) == 9.0
