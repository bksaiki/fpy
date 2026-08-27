"""Unit tests for :class:`fpy2.analysis.Alias`.

One test per aliasing route the analysis models, plus the properties that make it
usable: element cells terminate on nested types, and a uniquely-owned list is
distinguished from a shared one.
"""

import fpy2 as fp

from fpy2.analysis import Alias, DefineUse


def _sites(func: fp.Function):
    """``(owned, shared)`` allocation sites of *func*."""
    a = Alias.analyze(func.ast)
    owned = [s for s in a.sites if a.is_uniquely_owned(s)]
    shared = [s for s in a.sites if not a.is_uniquely_owned(s)]
    return a, owned, shared


class TestUniqueOwnership:
    def test_fresh_local_is_uniquely_owned(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                xs = [x, x]
                xs[0] = 1
                return xs[0]

        _, owned, shared = _sites(f)
        assert len(owned) == 1 and not shared

    def test_read_only_parameter_is_uniquely_owned(self):
        """A parameter is owned by the caller, but nothing here shares it."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        _, owned, shared = _sites(f)
        assert len(owned) == 1 and not shared

    def test_element_write_does_not_share(self):
        """``xs[i] = e`` mutates the list; it does not create a referrer."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                xs[0] = 99
                return xs[0]

        _, owned, shared = _sites(f)
        assert len(owned) == 1 and not shared


class TestAliasingRoutes:
    """Each route that makes a list reachable from a second place."""

    def test_assignment_aliases(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs
                ys[0] = 99
                return xs[0]

        _, _, shared = _sites(f)
        assert shared, 'ys = xs must make xs shared'

    def test_projection_shares_the_element(self):
        """``row = xss[i]`` refers to the element, not a copy of it."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                row = xss[0]
                row[0] = 99
                return xss[0][0]

        a, owned, shared = _sites(f)
        # the *outer* list is still unshared -- only its elements are
        assert [s.depth for s in shared] == [1]
        assert [s.depth for s in owned] == [0]

    def test_loop_variable_shares_the_element(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                for row in xss:
                    row[0] = 99
                return xss[0][0]

        _, _, shared = _sites(f)
        assert [s.depth for s in shared] == [1]

    def test_construction_shares_what_it_holds(self):
        """``[xs]`` is a new list holding the *same* xs."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zss = [xs]
                zss[0][0] = 99
                return xs[0]

        _, _, shared = _sites(f)
        assert any(s.kind == 'param' for s in shared)

    def test_one_list_at_two_indices(self):
        """Only expressible because a slot holds a reference."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = [x, x]
                m = [a, a]
                m[0][0] = 99
                return m[1][0]

        a_, _, shared = _sites(f)
        assert any(s.kind == 'literal' for s in shared)

    def test_slice_is_shallow(self):
        """A slice makes a fresh outer list over the same elements, so a nested
        slice shares them while a flat one does not."""
        @fp.fpy
        def nested(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                yss = xss[0:1]
                yss[0][0] = 99
                return xss[0][0]

        @fp.fpy
        def flat(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs[0:1]
                ys[0] = 99
                return xs[0]

        _, _, nested_shared = _sites(nested)
        assert nested_shared, 'a nested slice shares its elements'

        _, flat_owned, _ = _sites(flat)
        assert len(flat_owned) >= 1, 'a flat slice copies; nothing is shared'

    def test_slot_store_shares(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                xss[0] = ys
                return xss[0][0]

        a, _, shared = _sites(f)
        assert any(
            s.kind == 'param' and getattr(s.node.name, 'base', str(s.node.name))
            for s in shared
        )


class TestEscape:
    """The two routes out of a function: a ``return`` transfers ownership, a
    call argument shares it."""

    def test_returned_fresh_list_transfers_ownership(self):
        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs = [x, x]
                return xs

        a, owned, _ = _sites(f)
        assert [s.kind for s in owned] == ['literal']
        site, = a.sites
        assert a.escapes(site), 'it does leave the function'
        assert a.is_returned(site) and a.transfers_ownership(site)

    def test_returning_a_parameter_is_sharing_not_transfer(self):
        """``return xs`` leaves the caller holding two handles to one list, so the
        ``param`` site in the class blocks the transfer."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return xs

        a, owned, _ = _sites(f)
        assert not owned
        site, = a.sites
        assert a.is_returned(site) and not a.transfers_ownership(site)

    def test_returning_a_row_of_a_parameter_is_sharing(self):
        """The indirect route: the row's class carries the depth-1 ``param``
        site, so the guard catches it without a special case.

        Only the row, though.  The outer list is not what left, and a shallow copy
        of it would hold the same rows — so it stays owned, and a consumer may
        represent it by value while keeping the rows boxed.
        """
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            with fp.FP64:
                return xss[0]

        a, owned, _ = _sites(f)
        row, = (s for s in a.sites if s.depth == 1)
        assert a.is_returned(row) and not a.transfers_ownership(row)
        assert [s.depth for s in owned] == [0]

    def test_returning_a_local_stored_into_a_parameter_is_sharing(self):
        """The other indirect route: fresh, but the caller can reach it through
        ``xss`` after the store."""
        @fp.fpy
        def f(xss: list[list[fp.Real]], x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                ys = [x, x]
                xss[0] = ys
                return ys

        a, _, _ = _sites(f)
        assert not any(a.transfers_ownership(s) for s in a.sites)

    def test_returned_and_also_passed_to_a_call_is_not_owned(self):
        """Shared outward beats a transfer: the callee may still hold it."""
        @fp.fpy
        def g(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return zs[0]

        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                ys = [x, x]
                n = g(ys)
                return ys

        a, owned, _ = _sites(f)
        assert not owned
        assert not any(a.transfers_ownership(s) for s in a.sites)

    def test_nested_transfer_carries_its_rows(self):
        """A returned fresh nested list transfers at every level: the rows go with
        the outer list, so neither has to stay boxed."""
        @fp.fpy
        def f(x: fp.Real) -> list[list[fp.Real]]:
            with fp.FP64:
                yss = [[x, x], [x, x]]
                return yss

        a, owned, _ = _sites(f)
        assert owned, 'a fresh nested list transfers whole'

    def test_call_argument_escapes(self):
        @fp.fpy
        def g(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return zs[0]

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return g(xs)

        a, owned, _ = _sites(f)
        assert not owned, 'conservative: a callee may retain its argument'


class TestNesting:
    """Element cells are what make nesting work, and they terminate."""

    def test_one_site_per_level(self):
        @fp.fpy
        def f(xsss: list[list[list[fp.Real]]]) -> fp.Real:
            with fp.FP64:
                return xsss[0][0][0]

        a, _, _ = _sites(f)
        depths = sorted(s.depth for s in a.sites if s.kind == 'param')
        assert depths == [0, 1, 2], 'one site per level of the list type'

    def test_deep_projection_shares_only_its_level(self):
        @fp.fpy
        def f(xsss: list[list[list[fp.Real]]]) -> fp.Real:
            with fp.FP64:
                mid = xsss[0]
                row = mid[0]
                row[0] = 99
                return xsss[0][0][0]

        a, owned, shared = _sites(f)
        assert sorted(s.depth for s in shared) == [1, 2]
        assert [s.depth for s in owned] == [0]


class TestMayAlias:
    def test_aliased_names_may_alias(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs
                return ys[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        defs = {str(d.name): d for d in du.defs}
        assert a.may_alias(defs['xs'], defs['ys'])

    def test_unrelated_lists_do_not_alias(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = [x, x]
                b = [x, x]
                return a[0] + b[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        defs = {str(d.name): d for d in du.defs}
        assert not al.may_alias(defs['a'], defs['b'])


class TestCapturedByNonLists:
    """A list can be captured by an expression that is not itself a list."""

    def test_comprehension_variable_shares_the_element(self):
        """``[row for row in xss]`` is a new outer list over the *same* rows, so
        the comprehension's loop variable has to be bound to xss's elements."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                yss = [row for row in xss]
                yss[0][0] = 99
                return xss[0][0]

        _, _, shared = _sites(f)
        assert shared, 'the comprehension shares xss rows'

    def test_list_in_a_tuple_is_shared(self):
        """A tuple is not list-typed, so the list inside it is only reachable if
        tuple-valued expressions are given cells too."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, 1.0)
                ys = fp.fst(t)
                ys[0] = 99
                return xs[0]

        a, _, shared = _sites(f)
        param = [s for s in a.sites if s.kind == 'param']
        assert param and all(s in shared for s in param)
        # a *local* tuple is not a way out of the function
        assert not any(a.escapes(s) for s in param)


class TestTupleFields:
    """Tuple fields are kept apart: arity is static, so an index is known."""

    def test_fields_do_not_alias_each_other(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, ys)
                a = fp.fst(t)
                b = fp.snd(t)
                return a[0] + b[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        d = {str(x.name): x for x in du.defs}
        assert al.may_alias(d['a'], d['xs'])
        assert al.may_alias(d['b'], d['ys'])
        assert not al.may_alias(d['a'], d['ys']), 'fields must stay apart'

    def test_destructuring_binds_field_by_field(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, ys)
                a, b = t
                return a[0] + b[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        d = {str(x.name): x for x in du.defs}
        assert al.may_alias(d['a'], d['xs'])
        assert not al.may_alias(d['a'], d['ys'])


class TestConservativeRoutes:
    """Routes that would authorise an unobservable-copy claim if unmodelled."""

    def test_escape_reaches_nested_elements(self):
        """Returning a ``list[list[Real]]`` hands out its rows as well, so a copy
        of a row would be observable through the caller."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[list[fp.Real]]:
            with fp.FP64:
                return xss

        a, owned, _ = _sites(f)
        assert not owned, 'every level of a returned nested list escapes'
        assert all(a.escapes(s) for s in a.sites)

    def test_conditional_expression_aliases_both_branches(self):
        """``xs if c else ys`` *is* one of them, so writing through the result
        writes through both."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real], c: bool) -> fp.Real:
            with fp.FP64:
                zs = xs if c else ys
                zs[0] = 99
                return xs[0]

        a, owned, _ = _sites(f)
        assert not [s for s in owned if s.kind == 'param']
        assert f([1.0, 2.0], [7.0, 8.0], True, ctx=fp.FP64) == 99

    def test_enumerate_element_is_a_tuple_over_the_rows(self):
        """The loop variable of ``for i, row in enumerate(xss)`` is a row of xss,
        one level up from where a blanket element merge would put it."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                other = xss[0]
                acc = 0.0
                for i, row in enumerate(xss):
                    acc = acc + row[0]
                return acc + other[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        d = {str(x.name): x for x in du.defs}
        assert al.may_alias(d['row'], d['other'])

    def test_zip_takes_field_i_from_argument_i(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]], yss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                a = xss[0]
                acc = 0.0
                for p, q in zip(xss, yss):
                    acc = acc + p[0] + q[0]
                return acc + a[0]

        from fpy2.analysis import DefineUse

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        d = {str(x.name): x for x in du.defs}
        assert al.may_alias(d['p'], d['a'])
        assert not al.may_alias(d['q'], d['a'])


class TestRegionFacts:
    """The syntactic facts a representation decision reads off the region graph.

    Each is stated against the region a *name* denotes, which is the form a
    consumer asks in.
    """

    def test_element_store_marks_the_region_written(self):
        @fp.fpy
        def f(xs: list[fp.Real], v: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs[0] = v
                return xs

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        region = a.region_of(next(d for d in du.defs if str(d.name) == 'xs'))
        assert region in a.written_regions

    def test_a_read_only_list_is_not_written(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        region = a.region_of(next(d for d in du.defs if str(d.name) == 'xs'))
        assert region not in a.written_regions

    def test_replacing_a_slot_marks_the_element_region(self):
        """``xss[i] = <list>`` puts a different list in the cell, so a reference
        bound from that slot would re-read it."""

        @fp.fpy
        def f(xss: list[list[fp.Real]], row: list[fp.Real]) -> list[list[fp.Real]]:
            with fp.FP64:
                xss[0] = row
                return xss

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        d = next(d for d in du.defs if str(d.name) == 'xss')
        assert a.region_of(d, 1) in a.slot_replaced
        # the outer spine is written, not replaced
        assert a.region_of(d, 0) not in a.slot_replaced

    def test_a_scalar_store_replaces_no_slot(self):
        @fp.fpy
        def f(xs: list[fp.Real], v: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs[0] = v
                return xs

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        assert not a.slot_replaced

    def test_two_returns_put_their_regions_in_one_level(self):
        """A function has one return type but several ``return``s, and nothing
        unifies their regions -- so the grouping is the fact."""

        @fp.fpy
        def f(c: bool, xs: list[fp.Real], y: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                if c:
                    return xs
                else:
                    return [y, y]

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        assert len(a.returned_levels) == 1
        assert len(a.returned_levels[0]) == 2

    def test_returned_levels_are_indexed_by_depth(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[list[fp.Real]]:
            with fp.FP64:
                return xss

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        d = next(d for d in du.defs if str(d.name) == 'xss')
        assert len(a.returned_levels) == 2
        assert a.returned_levels[0] == {a.region_of(d, 0)}
        assert a.returned_levels[1] == {a.region_of(d, 1)}

    def test_a_function_returning_a_scalar_has_no_returned_levels(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        du = DefineUse.analyze(f.ast)
        a = Alias.analyze(f.ast, def_use=du)
        assert a.returned_levels == []


class TestConsumedNames:
    """A name handed to a container and never read again is not a place beside
    the container's slot -- the value moves in.  The fact only."""

    @staticmethod
    def _consumed(func: fp.Function) -> set[str]:
        du = DefineUse.analyze(func.ast)
        a = Alias.analyze(func.ast, def_use=du)
        return {str(d.name) for ds in a.consumed_defs.values() for d in ds}

    def test_sole_use_in_a_tuple_is_consumed(self):
        @fp.fpy
        def f(n: fp.Real) -> tuple[list[fp.Real], fp.Real]:
            with fp.FP64:
                xs = [n, n]
                return (xs, 1.0)

        assert self._consumed(f) == {'xs'}

    def test_sole_use_in_a_list_is_consumed(self):
        @fp.fpy
        def f(n: fp.Real) -> list[list[fp.Real]]:
            with fp.FP64:
                xs = [n, n]
                return [xs]

        assert self._consumed(f) == {'xs'}

    def test_a_second_use_disqualifies(self):
        @fp.fpy
        def f(n: fp.Real) -> tuple[list[fp.Real], fp.Real]:
            with fp.FP64:
                xs = [n, n]
                y = xs[0]
                return (xs, y)

        assert self._consumed(f) == set()

    def test_a_use_one_loop_level_in_is_refused(self):
        """The sole *syntactic* use runs once per iteration, so the value must
        survive the first; a block is the loop boundary."""

        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                xs = [n, n]
                acc = 0.0
                for i in range(3):
                    yss = [xs]
                    acc = acc + yss[0][0]
                return acc

        assert self._consumed(f) == set()

    def test_a_definition_inside_the_loop_is_consumed(self):
        """Redefined each iteration, so there is nothing to preserve."""

        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for i in range(3):
                    xs = [n, n]
                    yss = [xs]
                    acc = acc + yss[0][0]
                return acc

        assert self._consumed(f) == {'xs'}

    def test_a_parameter_is_never_consumed(self):
        """It names the caller's storage."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> tuple[list[fp.Real], fp.Real]:
            with fp.FP64:
                return (xs, 1.0)

        assert self._consumed(f) == set()

    def test_a_use_that_is_not_a_construction_is_not_a_move(self):
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                xs = [n, n]
                return xs[0]

        assert self._consumed(f) == set()

    def test_a_construction_in_a_comprehension_is_refused(self):
        """The body re-runs per item, but it is an expression rather than a
        block, so block identity alone does not see the repetition."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = [x * 2.0 for x in xs]
                zss = [[ys] for i in range(3)]
                return zss[0][0][0]

        assert self._consumed(f) == set()

    def test_a_construction_in_a_while_cond_is_refused(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = [x * 2.0 for x in xs]
                i = 0
                # the cond is `ys`'s only use, so nothing but the repetition
                # guard refuses it
                while [ys][0][0] > 0.0 and i < 2:
                    i = i + 1
                return i

        assert self._consumed(f) == set()

    def test_a_value_leaving_a_branch_through_a_phi_is_refused(self):
        """A phi is not a use site, so sole-use holds inside the branch while
        the merged value is still read after it."""

        @fp.fpy
        def f(xs: list[fp.Real], c: fp.Real) -> fp.Real:
            with fp.FP64:
                ys = [x * 1.0 for x in xs]
                acc = 0.0
                if c > 0:
                    ys = [x * 2.0 for x in xs]
                    zss = [ys]
                    acc = zss[0][0]
                return acc + ys[0]

        assert self._consumed(f) == set()

    def test_a_sibling_definition_of_the_same_name_is_not_discounted(self):
        """``referrers`` counts names, and the move is per definition: one
        definition read twice must not inherit a sibling's discount."""

        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                xs = [a, b]
                ys = [xs]
                xs = ys[0]
                zs = [xs]
                return zs[0][0] + xs[1]

        du = DefineUse.analyze(f.ast)
        al = Alias.analyze(f.ast, def_use=du)
        consumed = {d for ds in al.consumed_defs.values() for d in ds}
        assert consumed, 'the single-use definition should be consumed'
        assert all(len(du.uses.get(d, ())) == 1 for d in consumed)
