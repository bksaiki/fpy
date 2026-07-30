"""Unit tests for :class:`fpy2.analysis.Alias`.

One test per aliasing route the analysis models, plus the properties that make it
usable: element cells terminate on nested types, and a uniquely-owned list is
distinguished from a shared one.

The predicate that matters to a consumer is ``is_uniquely_owned``: it holds when
exactly one cell refers to the allocation and it does not escape, which is
precisely when copying the list would be unobservable.
"""

import fpy2 as fp

from fpy2.analysis import Alias


def _sites(func: fp.Function):
    """``(owned, shared)`` allocation sites of *func*."""
    a = Alias.analyze(func.ast)
    owned = [s for s in a.sites if a.is_uniquely_owned(s)]
    shared = [s for s in a.sites if not a.is_uniquely_owned(s)]
    return a, owned, shared


class TestUniqueOwnership:
    """A list nothing else refers to is uniquely owned."""

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
    """A list handed outside the function is not decidable here."""

    def test_returned_list_escapes(self):
        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                xs = [x, x]
                return xs

        a, owned, _ = _sites(f)
        assert not owned
        assert all(a.escapes(s) for s in a.sites)

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
    """The pairwise query, for consumers that want it."""

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
    """A list can be captured by an expression that is not itself a list.

    Both of these looked uniquely owned until ground truth said otherwise, so
    they are pinned here rather than left to the integration check.
    """

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
    """Routes that reported a list uniquely owned until they were modelled.

    Each of these silently authorised an unobservable-copy claim that a consumer
    would have acted on, so they are pinned individually.
    """

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
