"""Which list operations share and which copy.

Construction owns its elements; projection and assignment hand out references.
The full rule table is in ``docs/todos/list-value-semantics.md``; this file is
that table as executable assertions, since nothing else pins it.

The two-value convention throughout: a test mutates one name and reads through
the other, so ``99`` means *shared* and the original value means *copied*.
"""

import fpy2 as fp


def _r(f, *args) -> float:
    return float(f(*args))


class TestConstructionCopies:
    """A list owns its elements, so building one materialises them."""

    def test_list_literal_copies_a_list_element(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            y = [a]
            a[0] = 99.0
            return y[0][0]

        assert _r(f, 1.0) == 1.0

    def test_the_same_list_twice_gives_two_allocations(self):
        """``[a, a]`` -- the case that forces a handle inside a C++ vector while
        construction shares."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            x = [a, a]
            x[0][0] = 99.0
            return x[1][0]

        assert _r(f, 1.0) == 1.0

    def test_comprehension_copies_its_elements(self):
        """``[row for row in xss]`` is construction, so the rows are copied.
        This is the idiom whose cost changes most."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            ys = [row for row in xss]
            ys[0][0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 1.0

    def test_deep_through_nested_lists(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            inner = [n, n]
            mid = [inner]
            outer = [mid]
            inner[0] = 99.0
            return outer[0][0][0]

        assert _r(f, 1.0) == 1.0


class TestSliceCopies:
    """A slice is construction, not a view."""

    def test_slice_copies_rows(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            s = xss[0:2]
            s[0][0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 1.0

    def test_full_slice_is_a_deep_copy(self):
        """``xs[:]`` was a *shallow* copy before this change, and is the idiom
        the libraries used to defend against sharing."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            s = xss[:]
            s[0][0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 1.0


class TestStoreCopies:
    """``xs[i] = v`` writes *v*'s values into storage ``xs`` owns, mirroring
    ``arr[0] = row``."""

    def test_storing_a_list_copies_it(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            xss = [[n, n], [n, n]]
            xss[0] = a
            a[0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 1.0

    def test_and_the_slot_is_not_the_source(self):
        """The other direction: writing through the slot must not reach *a*."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            xss = [[n, n], [n, n]]
            xss[0] = a
            xss[0][0] = 99.0
            return a[0]

        assert _r(f, 1.0) == 1.0


class TestStoreOverwritesTheSlot:
    """The other half of the store rule, and the one that is not Python's: the
    values go *into* the list already in the slot, rather than the slot coming to
    hold a different list.  ``arr[0] = row`` in numpy, ``operator=`` in
    ``std::vector`` -- and the reason the C++ backend can hold a nested list by
    value."""

    def test_a_projection_taken_earlier_sees_the_store(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            ys = [99.0, 99.0]
            row = xss[0]
            xss[0] = ys
            return row[0]

        assert _r(f, 1.0) == 99.0

    def test_and_writing_through_that_projection_reaches_the_container(self):
        """The store does not detach the slot, so the projection is still a
        reference to it afterwards."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            ys = [2.0, 2.0]
            row = xss[0]
            xss[0] = ys
            row[0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 99.0

    def test_overwriting_is_still_a_copy_of_the_source(self):
        """Overwrite says where the values land, not whether they are shared."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n], [n, n]]
            ys = [2.0, 2.0]
            xss[0] = ys
            ys[0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 2.0

    def test_a_scalar_store_is_unaffected(self):
        """Rebinding a slot that holds a scalar cannot be told from overwriting
        it, so nothing about the flat case changes."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xs = [n, n]
            xs[1] = 42.0
            return xs[1]

        assert _r(f, 1.0) == 42.0

    def test_a_store_through_two_indices(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xsss = [[[n, n]]]
            ys = [7.0, 7.0]
            row = xsss[0][0]
            xsss[0][0] = ys
            return row[0]

        assert _r(f, 1.0) == 7.0


class TestReferencesAreKept:
    """The counterweight: not everything copies, or in-place mutation would be
    inexpressible."""

    def test_projection_is_a_reference(self):
        """(a): mutating a projected row mutates the container."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            x = [[n, n], [n, n]]
            r = x[0]
            r[0] = 99.0
            return x[0][0]

        assert _r(f, 1.0) == 99.0

    def test_assignment_is_a_reference(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            b = a
            b[0] = 99.0
            return a[0]

        assert _r(f, 1.0) == 99.0


class TestTuplesGroupRatherThanOwn:
    """A tuple is a transparent grouping: never copied, and copying a list does
    not descend into one."""

    def test_a_tuple_field_shares_its_list(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            a = [n, n]
            t = a, 1.0
            a[0] = 99.0
            u, _v = t
            return u[0]

        assert _r(f, 1.0) == 99.0

    def test_no_provenance_discrepancy(self):
        """The reason the copy stops at tuples: a tuple built directly and the
        same tuple copied into a list must behave alike.  Descending would make
        these two disagree."""

        @fp.fpy(ctx=fp.FP64)
        def direct(n: fp.Real) -> fp.Real:
            a = [n, n]
            t = a, 1.0
            a[0] = 99.0
            u, _v = t
            return u[0]

        @fp.fpy(ctx=fp.FP64)
        def through_a_list(n: fp.Real) -> fp.Real:
            a = [n, n]
            t = a, 1.0
            xs = [t]
            a[0] = 99.0
            u, _v = xs[0]
            return u[0]

        assert _r(direct, 1.0) == _r(through_a_list, 1.0) == 99.0


class TestBuiltinsStayCheap:
    """Every list-producing builtin has tuple or scalar elements, never list
    ones -- so none of them copies anything expensive, with no exemption."""

    def test_zip_shares_the_rows_it_pairs(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n]]
            yss = [[n, n]]
            for a, _b in zip(xss, yss):
                a[0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 99.0

    def test_enumerate_shares_the_rows_it_indexes(self):
        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real) -> fp.Real:
            xss = [[n, n]]
            for _i, row in enumerate(xss):
                row[0] = 99.0
            return xss[0][0]

        assert _r(f, 1.0) == 99.0
