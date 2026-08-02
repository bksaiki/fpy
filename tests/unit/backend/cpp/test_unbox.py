"""Where allocation-site ownership meets C++ storage classes.

Neither partition refines the other, so this is where an optimistic answer
becomes a miscompilation rather than a missed optimization.  See
:mod:`fpy2.backend.cpp.unbox`.
"""

import fpy2 as fp

from fpy2.analysis import Alias
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.types import CppList
from fpy2.backend.cpp.unbox import Unbox
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

R = RealType(fp.FP64)


def _decide(f: fp.Function, arg_types):
    """``(names -> chosen storage, reasons)`` for *f*.

    Goes through the compiler's own analyses: the decision has to be made on the
    *specialized* AST, whose types are concrete.
    """
    cc = CppCompiler()
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

    def test_parameter_handed_to_a_call(self):
        """Conservative: a callee may retain its argument, so the caller keeps
        a handle — and the callee's parameter has to match it."""
        @fp.fpy
        def g(zs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zs[0] = 99
                return zs[0]

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return g(xs)

        storage, reasons = _decide(f, [ListType(R)])
        assert _levels(storage['xs']) == [True]
        assert reasons[('xs', 0)] == 'shared'

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
        unopt = CppCompiler(optimize=False).compile(
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

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[ListType(R)])
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

        out = CppCompiler().compile(
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

    def test_a_list_inside_a_tuple_keeps_its_handle(self):
        """``Unbox`` walks the ``CppList`` spine and stops at a tuple.
        Undecided is not the same as boxed — the two representations then do
        not compile together."""
        from fpy2.types import TupleType

        @fp.fpy
        def f(t: tuple[list[fp.Real], fp.Real]) -> fp.Real:
            with fp.FP64:
                return fp.fst(t)[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[TupleType(ListType(R), R)],
        )
        assert 'std::tuple<fpy::list<double>, double>' in out
        assert 'std::vector' not in out

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

        cc = CppCompiler()
        m = Module()
        m.add(caller, ctx=fp.FP64, arg_types=[ListType(R)])
        m.add(callee, ctx=fp.FP64, arg_types=[ListType(R)])
        params, _ = cc.signature(
            callee, ctx=fp.FP64, arg_types=[ListType(R)], module=m,
        )
        assert f'const {params[0].format()}& zs' in cc.compile_module(m)
