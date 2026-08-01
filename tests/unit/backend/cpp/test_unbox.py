"""The join between allocation-site ownership and C++ storage classes.

:mod:`fpy2.analysis.alias` decides per *allocation site*; the emitter declares one
variable per *storage class*.  Neither partition refines the other, so the join is
where an optimistic answer would become a miscompilation rather than a missed
optimization.  These tests pin the conservative direction at each shape.

Nothing consumes the decision yet — that is the point of testing it first.
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
    ub = Unbox.decide(a.storage, alias)
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
    """A storage class can span sites the alias analysis kept apart."""

    def test_disagreeing_class_stays_boxed(self):
        """One C++ variable, two sites: a fresh literal (owned) and a parameter
        (shared).  Unboxing ``ys`` while ``xs`` stays boxed would make ``ys = xs``
        a silent conversion and lose the write."""
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
        assert reasons[('ys', 0)] == 'sites disagree'
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

    def test_shared_parameter(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs
                ys[0] = 99
                return xs[0]

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
