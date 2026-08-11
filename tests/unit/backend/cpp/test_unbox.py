"""Where allocation-site ownership meets C++ storage classes.

Neither partition refines the other, so this is where an optimistic answer
becomes a miscompilation rather than a missed optimization.  See
:mod:`fpy2.backend.cpp.unbox`.
"""

import re

import fpy2 as fp

from fpy2.analysis import Alias
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.types import CppList
from fpy2.backend.cpp.unbox import Unbox, UnboxMode
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

R = RealType(fp.FP64)

# This file is about the opportunistic analysis, so every compile is ALLOW;
# the strict default's refusals live in `test_unbox_strict.py`.
ALLOW = CppCompiler(unbox=UnboxMode.ALLOW)


def _decide(f: fp.Function, arg_types):
    """``(names -> chosen storage, reasons)`` for *f*.

    Goes through the compiler's own analyses: the decision has to be made on the
    *specialized* AST, whose types are concrete.
    """
    cc = ALLOW
    m = Module()
    m.add(f, ctx=fp.FP64, arg_types=list(arg_types))
    a = cc.analyze(cc.specialize(m)[-1])
    alias = Alias.analyze(a.ast, def_use=a.def_use)
    ub = Unbox.decide(a.ast, a.storage, alias, a.def_use)
    by_name = {
        a.storage.def_to_name[cls]: ty for cls, ty in ub.storage.items()
    }
    reasons = {
        (a.storage.def_to_name[cls], depth): why
        for (cls, depth), why in ub.boxed_because.items()
    }
    return by_name, reasons


def _levels(ty) -> list[bool]:
    """``boxed`` for each list level, outermost first."""
    out = []
    while isinstance(ty, CppList):
        out.append(ty.boxed)
        ty = ty.elt
    return out


class TestTheJoin:
    """One C++ variable admits one representation.

    ``alias`` mirrors both edges ``storage_infer`` unions on, so a class
    normally maps to a single region.  The conjunction in
    :mod:`~fpy2.backend.cpp.unbox` is a guard: a ``sites disagree`` verdict
    means the backend has grown a third kind of edge.
    """

    def test_disagreeing_class_stays_boxed(self):
        """One C++ variable, two values: a fresh literal and a parameter.
        Unboxing ``ys`` while ``xs`` stays boxed would make ``ys = xs`` a silent
        conversion and lose the write."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                if c:
                    ys = [x, x]
                else:
                    ys = xs
                ys[0] = 99
                return xs[0]

        storage, reasons = _decide(f, [ListType(R), BoolType(), R])
        assert _levels(storage['ys']) == [True]
        assert _levels(storage['xs']) == [True]

    def test_no_class_is_unboxed_against_one_it_reads(self):
        """The property that makes the conjunction sufficient: a class cannot
        come out unboxed while a class it is assigned from stays boxed, because
        the assignment merged their alias classes and hence their site sets."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                if c:
                    ys = [x, x]
                else:
                    ys = xs
                ys[0] = 99
                return xs[0]

        storage, _ = _decide(f, [ListType(R), BoolType(), R])
        boxed = {n for n, ty in storage.items() if _levels(ty)[0]}
        assert {'xs', 'ys'} <= boxed


class TestUnboxed:
    """Where the decision should be positive — rejecting everything would be
    sound and useless."""

    def test_fresh_local(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                ys = [x, x]
                ys[0] = 99
                return ys[0]

        storage, _ = _decide(f, [R])
        assert _levels(storage['ys']) == [False]

    def test_nested_read_only_parameter(self):
        """The case worth the whole exercise: a nested parameter costs 8.5-9.3x
        to convert at a native boundary, and unboxed it costs nothing."""
        import fpy2.libraries.matrix as M

        storage, _ = _decide(M.matvec, [ListType(ListType(R)), ListType(R)])
        assert _levels(storage['A']) == [False, False]
        assert _levels(storage['x']) == [False]


class TestStaysBoxed:
    """The conservative direction, one reason at a time."""

    def test_parameter_handed_to_a_retaining_call(self):
        """A callee that *keeps* its argument shares it.

        Writing does not: ``zs[0] = 99`` holds nothing once the call returns,
        which is why the callee here returns the list instead.  See
        :mod:`fpy2.analysis.escape`.
        """
        @fp.fpy
        def g(zs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return zs

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = g(xs)
                return ys[0]

        storage, reasons = _decide(f, [ListType(R)])
        assert _levels(storage['xs']) == [True]
        assert reasons[('xs', 0)] == 'reached across a boundary'

    def test_levels_are_decided_independently(self):
        """Returning a row hands out the row, not the outer list — so the outer
        may be a value while its elements keep their handles."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            with fp.FP64:
                return xss[0]

        storage, reasons = _decide(f, [ListType(ListType(R))])
        assert _levels(storage['xss']) == [False, True]
        assert reasons[('xss', 1)] == 'shared'


class TestReferenceBoundNames:
    """A name the emitter binds by reference is not a second place.

    See ``unbox._shares_storage``.
    """

    def test_named_loop_variable_does_not_box_the_rows(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for xs in xss:
                    for x in xs:
                        acc = acc + x
                return acc

        storage, _ = _decide(f, [ListType(ListType(R))])
        assert _levels(storage['xss']) == [False, False]

    def test_zip_elim_does_not_box_its_operands(self):
        """An *optimization* must not cost a representation.  ``ZipElim``
        rewrites ``zip(xs, ys)`` into ``_src = xs`` aliases plus indexed
        access; those are reference bindings, so nothing is copied."""
        @fp.fpy
        def g(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for x, y in zip(xs, ys):
                    acc = acc + x * y
                return acc

        args = [ListType(R), ListType(R)]
        opt, _ = _decide(g, args)
        assert _levels(opt['xs']) == [False]
        assert _levels(opt['ys']) == [False]

        # ...and it agrees with the unoptimized pipeline, which is the property
        # that actually matters: the two must not disagree about storage.
        unopt = CppCompiler(unbox=UnboxMode.ALLOW, optimize=False).compile(
            g, ctx=fp.FP64, arg_types=args,
        )
        assert 'const std::vector<double>& xs' in unopt


class TestDiscountHasLimits:
    """Where discounting a reference-bound name would be *wrong*."""

    def test_a_parameter_is_never_discounted(self):
        """``zss = [xs]`` puts the caller's list in a container; the parameter's
        reference points at the caller's storage, which is a place of its
        own."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zss = [xs]
                zss[0][0] = 99
                return xs[0]

        storage, _ = _decide(f, [ListType(R)])
        assert _levels(storage['xs']) == [True]

    def test_alias_that_writes_stays_writable(self):
        """``ys = xs; ys[0] = 99`` may unbox — both names reference one vector
        — but then neither may be ``const``, and the write is on a *different*
        storage class than the parameter, so const-ness is a question about the
        whole alias region."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs
                ys[0] = 99
                return xs[0]

        out = ALLOW.compile(f, ctx=fp.FP64, arg_types=[ListType(R)])
        assert 'std::vector<double>& xs' in out
        assert 'const std::vector<double>& xs' not in out
        assert 'auto& ys = xs;' in out
        assert 'const auto& ys' not in out
        assert f([1.0, 2.0], ctx=fp.FP64) == 99

    def test_const_propagates_out_through_unboxed_levels(self):
        """A write to a *row* makes the whole nested parameter non-const:
        ``const`` reaches through a value container, unlike a handle."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                for row in xss:
                    row[0] = 99
                return xss[0][0]

        out = ALLOW.compile(
            f, ctx=fp.FP64, arg_types=[ListType(ListType(R))],
        )
        assert 'std::vector<std::vector<double>>& xss' in out
        assert 'const std::vector<std::vector<double>>&' not in out
        assert 'std::vector<double>& row' in out
        assert f([[1.0, 2.0]], ctx=fp.FP64) == 99


class TestInvisibleToTheHarness:
    """Wrong answers the differential harness cannot detect."""

    def test_a_rebound_parameter_keeps_its_handle(self):
        """Silent if got wrong: the return value is right either way, and the
        interpreter cannot see it because Python-level calls copy list
        arguments.  See ``unbox._shares_storage``."""
        @fp.fpy
        def k(xs: list[fp.Real], c: fp.Real) -> fp.Real:
            with fp.FP64:
                xs[0] = 99.0
                if c > 0:
                    xs = [7.0]
                return xs[0]

        storage, _ = _decide(k, [ListType(R), R])
        assert _levels(storage['xs']) == [True]

    def test_signature_agrees_with_what_the_module_emits(self):
        """A function another compiled function calls keeps its handles, so a
        signature computed for it *alone* is not the one it gets in company —
        and ``signature`` is what an embedding program builds arguments from."""
        @fp.fpy
        def callee(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zs[0] = 1
                return zs[0]

        @fp.fpy
        def caller(ws: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return callee(ws)

        cc = ALLOW
        m = Module()
        m.add(caller, ctx=fp.FP64, arg_types=[ListType(R)])
        m.add(callee, ctx=fp.FP64, arg_types=[ListType(R)])
        params, _ = cc.signature(
            callee, ctx=fp.FP64, arg_types=[ListType(R)], module=m,
        )
        # const-ness is a separate question; what must agree is the type
        assert f'{params[0].format()}& zs' in cc.compile_module(m)


class TestAcrossACall:
    """What a summary buys, and what it must not.

    A compiled-to-compiled boundary used to keep its handles unconditionally.
    With :mod:`fpy2.analysis.escape` it keeps them only where the callee really
    holds on to the list.
    """

    def test_both_ends_unbox_when_the_callee_keeps_nothing(self):
        @fp.fpy
        def inner(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for i in range(len(xs)):
                    acc = acc + xs[i]
                return acc

        @fp.fpy
        def outer(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for i in range(len(xss)):
                    acc = acc + inner(xss[i])
                return acc

        m = Module()
        m.add(outer, ctx=fp.FP64, arg_types=[ListType(ListType(R))])
        out = ALLOW.compile_module(m)
        assert 'const std::vector<double>& xs' in out
        assert 'const std::vector<std::vector<double>>& xss' in out
        assert 'std::shared_ptr' not in out

    def test_a_retaining_callee_keeps_both_ends_boxed(self):
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
        m.add(hand_over, ctx=fp.FP64, arg_types=[ListType(R)])
        out = ALLOW.compile_module(m)
        assert 'std::vector<double>& xs' not in out
        assert 'std::shared_ptr<std::vector<double>>' in out

    def test_a_writing_callee_makes_both_ends_non_const(self):
        """Writing is not retaining, so both unbox — but ``const`` has to cross
        the call edge or the argument will not bind."""
        @fp.fpy
        def bump(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zs[0] = 99
                return zs[0]

        @fp.fpy
        def call_it(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return bump(xs)

        m = Module()
        m.add(call_it, ctx=fp.FP64, arg_types=[ListType(R)])
        out = ALLOW.compile_module(m)
        assert 'std::vector<double>& zs' in out
        assert 'std::vector<double>& xs' in out
        assert 'const std::vector<double>&' not in out

    def test_a_boxed_caller_hands_over_the_pointee(self):
        """The callee's signature is fixed by its own body, so a caller that
        keeps a handle for a local reason passes ``*handle`` — same elements,
        no copy."""
        @fp.fpy
        def reads(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return zs[0]

        @fp.fpy
        def shares(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, 1.0)          # a tuple keeps `xs` boxed
                return reads(xs) + fp.fst(t)[0]

        m = Module()
        m.add(shares, ctx=fp.FP64, arg_types=[ListType(R)])
        out = ALLOW.compile_module(m)
        assert 'const std::vector<double>& zs' in out
        assert 'const std::shared_ptr<std::vector<double>>& xs' in out
        assert re.search(r'reads__\w+\(\*xs\)', out), out


class TestFreshNestedAllocations:
    """A fresh ``list[list[T]]`` allocates its rows too.

    Without a site at each level, the inner one has no allocation recorded
    against it, and a consumer cannot tell "nothing owns this" from "nothing is
    known about it" — so it kept its handle.  Every nested-returning matrix
    kernel was affected.
    """

    def test_a_returned_nested_result_unboxes_at_every_level(self):
        import fpy2.libraries.matrix as M

        N = ListType(ListType(R))
        _params, ret = ALLOW.signature(
            M.add, ctx=fp.FP64, arg_types=[N, N],
        )
        assert _levels(ret) == [False, False], ret.format()

    def test_rows_of_a_fresh_nested_list_are_owned(self):
        @fp.fpy
        def f(n: fp.Real) -> list[list[fp.Real]]:
            with fp.FP64:
                m = [[n, n], [n, n]]
                m[0][0] = n + 1
                return m

        storage, _ = _decide(f, [R])
        assert _levels(storage['m']) == [False, False]

    def test_rows_that_are_shared_still_keep_their_handles(self):
        """The seeding must not paper over real sharing: here the rows of the
        fresh outer list *are* the caller's."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                m = [xs, xs]
                m[0][0] = 99
                return xs[0]

        storage, _ = _decide(f, [ListType(R)])
        assert _levels(storage['m']) == [False, True]


class TestCalleeReturn:
    """The return half of a compiled-to-compiled boundary.

    A callee used to keep a handle on its result purely because it was called.
    Now its callers read the representation off its signature, the way they
    already do for its parameters.
    """

    def test_a_fresh_result_crosses_the_boundary_unboxed(self):
        @fp.fpy
        def make(n: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                return [n, n]

        @fp.fpy
        def use(n: fp.Real) -> fp.Real:
            with fp.FP64:
                v = make(n)
                return v[0] + v[1]

        m = Module()
        m.add(use, ctx=fp.FP64, arg_types=[R])
        out = ALLOW.compile_module(m)
        assert 'std::shared_ptr' not in out, out

    def test_a_caller_that_needs_a_handle_makes_one(self):
        """The callee's signature is fixed by its own body, so a caller with a
        local reason to hold a handle wraps the result.  Here the callee returns
        a *sized* value, so the wrap goes through a vector rebuild."""
        @fp.fpy
        def make(n: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                return [n, n]

        @fp.fpy
        def boxes_it(n: fp.Real) -> fp.Real:
            with fp.FP64:
                v = make(n)
                t = (v, 1.0)       # a tuple keeps `v` boxed
                w = fp.fst(t)
                w[0] = 9
                return v[0]

        m = Module()
        m.add(boxes_it, ctx=fp.FP64, arg_types=[R])
        out = ALLOW.compile_module(m)
        # the callee hands back a fixed-size value; boxing it spells the
        # vector copy first -- `make_shared<vector>(array)` does not exist
        assert 'std::make_shared<std::vector<double>>(std::vector<double>(' in out, out
        assert boxes_it(3.0, ctx=fp.FP64) == 9


class TestProjectionByReference:
    """``row = xss[i]`` binds a reference where it safely can.

    It used to always copy, which made the row a second place and boxed it.
    ``ZipElim`` manufactures exactly this shape, so `for a, b in zip(...)` over
    nested lists was paying for it.
    """

    def test_a_projection_binds_a_reference(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                row = xss[0]
                return row[0]

        out = ALLOW.compile(
            f, ctx=fp.FP64, arg_types=[ListType(ListType(R))],
        )
        assert 'const auto& row =' in out, out
        assert 'std::shared_ptr' not in out, out

    def test_zip_over_nested_lists_unboxes(self):
        """The shape from the report: `ZipElim` lowers the loop variable to a
        projection, and the rows used to keep their handles because of it."""
        @fp.fpy
        def inner(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for x, y in zip(xs, ys):
                    acc = acc + x * y
                return acc

        @fp.fpy
        def outer(
            xss: list[list[fp.Real]], yss: list[list[fp.Real]],
        ) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                for xs, ys in zip(xss, yss):
                    acc = acc + inner(xs, ys)
                return acc

        N = ListType(ListType(R))
        m = Module()
        m.add(outer, ctx=fp.FP64, arg_types=[N, N])
        out = ALLOW.compile_module(m)
        assert 'std::shared_ptr' not in out, out

    def test_a_replaced_slot_still_copies(self):
        """The guard.  A C++ reference follows the slot; FPy keeps referring to
        the list that was in it, so a store of a *different* list anywhere in
        the function rules the reference out.
        """
        import tests.infra.backend.cpp as corpus

        out = ALLOW.compile(
            corpus._regression_replaced_slot, ctx=fp.FP64,
            arg_types=[ListType(ListType(R)), ListType(R)],
        )
        assert 'std::shared_ptr<std::vector<double>> row = ' in out, out
        assert 'auto& row' not in out, out

    def test_the_guard_is_function_wide(self):
        """Conservative on purpose: the store is *after* the last read here, so
        a flow-sensitive guard would allow the reference.  Deliberately not —
        nothing else in the analysis is flow-sensitive."""
        @fp.fpy
        def f(xss: list[list[fp.Real]], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                row = xss[0]
                n = row[0]
                xss[0] = ys
                return n

        out = ALLOW.compile(
            f, ctx=fp.FP64, arg_types=[ListType(ListType(R)), ListType(R)],
        )
        assert 'auto& row' not in out, out


    def test_a_literal_nested_three_deep_unboxes_at_every_level(self):
        """Seeding a site per level is only right where nothing else describes
        the elements.

        A literal's elements are expressions that allocate cells of their own,
        so seeding on top of them gave one place two parts — which merge into
        two referrers and read as shared.  It showed only at depth three,
        because at depth two the inner literal has no parts to collide with.
        """
        @fp.fpy
        def f() -> list[list[list[fp.Real]]]:
            with fp.FP64:
                x = [[[1.0, 2.0]]]
                x[0][0][0] = 0
                return x

        out = ALLOW.compile(f, ctx=fp.FP64, arg_types=[])
        assert 'std::shared_ptr' not in out, out

    def test_a_call_result_still_seeds_its_levels(self):
        """The other half of the rule: nothing local describes the elements of
        a value a callee hands back, so those levels do need seeding."""
        import fpy2.libraries.matrix as M

        _p, ret = ALLOW.signature(
            M.identity, ctx=fp.FP64, arg_types=[R],
        )
        assert _levels(ret) == [False, False], ret.format()


class TestListsInsideTuples:
    """A list held in a tuple is decided like any other.

    It used to be forced boxed, because ``Unbox`` walked only the ``CppList``
    spine and stopping there is not the same as deciding.
    """

    def test_a_destructured_component_is_not_a_reference(self):
        """The precondition, and the reason this could not be done earlier.

        ``a, b = t`` reads ``a`` with ``std::get``, which *copies*.  Discounting
        it as a reference binding is harmless while the component is a handle —
        the copy still shares — but the moment tuples unbox it is a copy of a
        value, and the write below would be lost.
        """
        from fpy2.backend.cpp.storage_infer import binds_by_reference

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, 1.0)
                a, b = t
                a[0] = 99
                return xs[0]

        cc = ALLOW
        m = Module()
        m.add(f, ctx=fp.FP64, arg_types=[ListType(R)])
        a = cc.analyze(cc.specialize(m)[-1])
        component = next(
            d for d in a.def_use.defs
            if str(d.name) == 'a' and type(d.site).__name__ == 'Assign'
        )
        assert not binds_by_reference(a.storage, a.def_use, component)
        assert f([1.0, 2.0], ctx=fp.FP64) == 99

    def test_a_fresh_list_in_a_returned_tuple_unboxes(self):
        @fp.fpy
        def f(n: fp.Real) -> tuple[list[fp.Real], fp.Real]:
            with fp.FP64:
                return ([n, n], 1.0)

        cc = ALLOW
        m = Module()
        m.add(f, ctx=fp.FP64, arg_types=[R])
        _p, ret = cc.signature(f, ctx=fp.FP64, arg_types=[R], module=m)
        assert ret.format() == 'std::tuple<std::array<double, 2>, uint8_t>', (
            ret.format()
        )

    def test_a_shared_list_in_a_tuple_still_keeps_its_handle(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = (xs, 1.0)
                w = fp.fst(t)
                w[0] = 55
                return xs[0]

        out = ALLOW.compile(f, ctx=fp.FP64, arg_types=[ListType(R)])
        assert 'std::tuple<std::shared_ptr<std::vector<double>>, uint8_t>' in out, out
        assert f([1.0, 2.0], ctx=fp.FP64) == 55

    def test_a_read_only_tuple_parameter_unboxes(self):
        """A tuple *parameter* holding a list, only read.

        The caller owns the vector and we take the tuple by ``const`` reference,
        so there is nothing to share and no allocation to make.  This case was
        pinned the other way until the declaration stopped being decided by a
        traversal that skipped tuples -- and the two tests contradicted each
        other, which is how the skip survived.
        """
        from fpy2.types import TupleType

        @fp.fpy
        def f(t: tuple[list[fp.Real], fp.Real]) -> fp.Real:
            with fp.FP64:
                return fp.fst(t)[0]

        out = ALLOW.compile(
            f, ctx=fp.FP64, arg_types=[TupleType(ListType(R), R)],
        )
        assert 'const std::tuple<std::vector<double>, double>&' in out, out
        assert 'std::shared_ptr' not in out, out
