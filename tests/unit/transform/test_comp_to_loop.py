"""
Unit tests for the :class:`fpy2.transform.CompToLoop` transform.

The rewrite mints fresh ``t`` / ``acc`` / ``i`` / ``j`` names via ``Gensym``, so
comparing against a hand-written golden AST is brittle.  These tests assert

1. **Structural shape** — the comprehension is gone, replaced by an `fp.empty`
   allocation and a loop that writes every slot.
2. **Semantic equivalence** via the interpreter: one clause, several
   independent clauses, a `TupleBinding` target, and `[row for row in xss]`,
   whose outer list is fresh over the *same* inner lists.
3. **Refusals**, by the reason each gives -- a dependent clause list, whose
   length is a sum rather than a product, and the positions with no statement
   slot.
4. **The `where` contract** — a listing reports exactly what `where=None`
   rewrites.
"""

import re

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import ArraySizeInfer, ContextUse, DefineUse
from fpy2.analysis.array_size import ListSize, concrete_size
from fpy2.analysis.format_infer import FormatInfer
from fpy2.ast.fpyast import Empty, ForStmt, IndexedAssign, ListComp
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import (
    CompToLoop,
    ExprCursor,
    TransformDeclined,
    TransformReferenceError,
)
from fpy2.utils import NamedId

# ----------------------------------------------------------------------
# Helpers


def _count(ast, kind) -> int:
    """How many *kind* nodes are in *ast*."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_list_comp(self, e, ctx):
            nonlocal n
            if kind is ListComp:
                n += 1
            super()._visit_list_comp(e, ctx)

        def _visit_for(self, stmt, ctx):
            nonlocal n
            if kind is ForStmt:
                n += 1
            super()._visit_for(stmt, ctx)

        def _visit_indexed_assign(self, stmt, ctx):
            nonlocal n
            if kind is IndexedAssign:
                n += 1
            super()._visit_indexed_assign(stmt, ctx)

        def _visit_naryop(self, e, ctx):
            nonlocal n
            if kind is Empty and isinstance(e, Empty):
                n += 1
            super()._visit_naryop(e, ctx)

    _C()._visit_function(ast, None)
    return n


def _lower(f, **kw) -> Function:
    return Function(CompToLoop.apply(f.ast, **kw), runtime=f.runtime)


def _vals(xs):
    """A nested list of floats, so results compare by value."""
    return [_vals(x) for x in xs] if isinstance(xs, list) else float(xs)


def _agree(f, *args, **kw) -> bool:
    return _vals(_lower(f, **kw)(*args)) == _vals(f(*args))


# ----------------------------------------------------------------------
# Programs


@fp.fpy(ctx=fp.FP64)
def _one(xs: list[fp.Real]) -> list[fp.Real]:
    return [x * x for x in xs]


@fp.fpy(ctx=fp.FP64)
def _product(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
    return [a + b for a in xs for b in ys]


@fp.fpy(ctx=fp.FP64)
def _ragged(xss: list[list[fp.Real]]) -> list[fp.Real]:
    return [b for a in xss for b in a]


@fp.fpy(ctx=fp.FP64)
def _ragged3(xsss: list[list[list[fp.Real]]]) -> list[fp.Real]:
    return [c for a in xsss for b in a for c in b]


@fp.fpy(ctx=fp.FP64)
def _pairs(ps: list[tuple[fp.Real, fp.Real]]) -> list[fp.Real]:
    return [x + y for x, y in ps]


@fp.fpy(ctx=fp.FP64)
def _rows(xss: list[list[fp.Real]]) -> list[list[fp.Real]]:
    return [row for row in xss]


# ----------------------------------------------------------------------
# The rewrite


class TestCompToLoop:
    def test_one_clause_becomes_an_allocation_and_a_loop(self):
        out = CompToLoop.apply(_one.ast)
        assert _count(out, ListComp) == 0
        assert _count(out, Empty) == 1
        assert _count(out, ForStmt) == 1
        assert _count(out, IndexedAssign) == 1
        assert _agree(_one, [1.5, 2.0, -3.0])

    def test_several_independent_clauses_nest(self):
        out = CompToLoop.apply(_product.ast)
        assert _count(out, ListComp) == 0
        assert _count(out, ForStmt) == 2      # one per clause, nested
        assert _agree(_product, [1.0, 2.0, 3.0], [10.0, 20.0])

    def test_a_tuple_binding_target_destructures(self):
        out = CompToLoop.apply(_pairs.ast)
        assert _count(out, ListComp) == 0
        assert _agree(_pairs, [(1.0, 2.0), (3.0, 4.0)])

    def test_a_row_comprehension_keeps_sharing_its_rows(self):
        """``[row for row in xss]`` is a fresh *outer* list over the *same* inner
        lists; the lowering must not deep-copy them."""
        rows = [[1.0, 2.0], [3.0]]
        assert _agree(_rows, rows)
        out = _lower(_rows)(rows)
        assert _vals(out[0]) == [1.0, 2.0]

    def test_an_assignment_target_is_filled_directly(self):
        """``ys = [...]`` writes into ``ys``.  An ``acc`` copied into ``ys``
        would leave two names on one list, and a second name is a second
        *place*: the cpp backend's ``UnboxMode.STRICT`` refuses it."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = [x * 2.0 for x in xs]
            return ys

        out = CompToLoop.apply(f.ast)
        assert 'acc' not in out.format()
        assert 'ys = fp.empty' in out.format()
        assert _agree(f, [1.0, 2.0])

    def test_a_comprehension_with_no_target_still_mints_one(self):
        """A ``return`` has no name to fill, so the accumulator stays."""
        out = CompToLoop.apply(_one.ast)
        assert 'acc = fp.empty' in out.format()

    def test_an_element_reading_the_target_keeps_its_accumulator(self):
        """The loops overwrite ``ys`` before the element runs, so an element
        that reads ``ys`` must read a list the fill would have destroyed."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = [1.0, 2.0]
            ys = [ys[0] + x for x in xs]
            return ys

        out = CompToLoop.apply(f.ast)
        assert 'acc = fp.empty' in out.format()
        assert _agree(f, [1.0, 2.0])

    def test_a_nested_comprehension_allocates_into_its_slot(self):
        """``[[...] for ...]`` lowers in two passes: the outer one turns the
        inner into ``acc[i] = [...]``, and that slot is the place the inner's
        loops write into.  An accumulator here would be a second name on the
        list ``acc[i]`` holds."""
        @fp.fpy(ctx=fp.FP64)
        def f(rows: list[fp.Real], cols: list[fp.Real]) -> list[list[fp.Real]]:
            return [[c for c in cols] for _ in rows]

        out = f.ast
        for _ in range(3):
            if not CompToLoop.sites(out):
                break
            out = CompToLoop.apply(out)
        assert _count(out, ListComp) == 0
        src = out.format()
        # one accumulator only -- the outer, which is in a `return`; the inner
        # allocates into the slot the outer's loop stores to
        assert len(re.findall(r'\bacc\w* = fp\.empty', src)) == 1
        assert re.search(r'\bacc\[\w+\] = fp\.empty', src)
        assert _agree(f, [1.0, 2.0], [3.0, 4.0])

    def test_an_element_reading_the_slots_base_keeps_its_accumulator(self):
        """The inner loops overwrite ``out[i]`` before the element runs, so an
        element reading ``out`` must read a list the fill would have
        destroyed."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> list[list[fp.Real]]:
            out = fp.empty(len(xs), 1)
            out[0][0] = 5.0
            for i in range(len(xs)):
                out[i] = [out[0][0] + x for x in xs]
            return out

        lowered = CompToLoop.apply(f.ast)
        assert 'acc = fp.empty' in lowered.format()
        assert _agree(f, [1.0, 2.0])

    def test_a_comprehension_in_an_iterable_does_not_claim_the_target(self):
        """Only the assignment's own right-hand side may fill ``zs``; the
        comprehension inside its iterable is a different list."""
        @fp.fpy(ctx=fp.FP64)
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            zs = [r[0] for r in [q for q in xss]]
            return zs

        out = CompToLoop.apply(f.ast)
        assert _count(out, ListComp) == 0
        assert _count(out, Empty) == 2      # one fills `zs`, one the inner list
        assert _agree(f, [[1.0, 2.0], [3.0]])

    def test_does_not_mutate_the_input(self):
        CompToLoop.apply(_one.ast)
        assert _count(_one.ast, ListComp) == 1

    def test_temp_id_names_the_iterable_binding(self):
        out = CompToLoop.apply(_one.ast, temp_id=NamedId('src'))
        assert 'src = xs' in out.format()


# ----------------------------------------------------------------------
# A dependent clause list is left alone


class TestDependentClauses:
    @pytest.mark.parametrize('f', ['_ragged', '_ragged3'])
    def test_a_dependent_clause_list_is_left_alone(self, f):
        """`[b for a in xss for b in a]` has length `sum(len(a) for a in xss)`,
        not a product -- and `fp.empty` needs its length up front with no
        `append` to fall back on.  So the pass leaves it exactly as it was."""
        func = {'_ragged': _ragged, '_ragged3': _ragged3}[f]
        assert CompToLoop.sites(func.ast) == []
        why = CompToLoop.refusals(func.ast)
        assert len(why) == 1 and 'mentions an earlier' in why[0][1]
        assert CompToLoop.apply(func.ast).is_equiv(func.ast)

    def test_left_alone_is_not_an_error(self):
        """The pass never raises over a comprehension it cannot lower; a caller
        that needs none left checks `refusals` itself."""
        out = CompToLoop.apply(_ragged.ast)     # no exception
        assert _count(out, ListComp) == 1

    def test_dependence_is_a_free_variable_check_not_a_syntactic_one(self):
        """`range(len(a))` mentions `a`, so it is dependent even though it is
        not the bare target."""

        @fp.fpy(ctx=fp.FP64)
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            return [a[i] for a in xss for i in range(len(a))]

        assert CompToLoop.sites(f.ast) == []

    def test_an_independent_clause_that_merely_looks_dependent_lowers(self):
        """`range(k)` reads an outer name, not an earlier target."""

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
            with fp.INTEGER:
                n = fp.round(k)
            return [x for x in xs for _ in range(n)]

        assert len(CompToLoop.sites(f.ast)) == 1
        assert _agree(f, [1.0, 2.0], 2.0)


# ----------------------------------------------------------------------
# Reaching a comprehension-free program


class TestFixpoint:
    def test_a_nested_comprehension_clears_after_a_second_pass(self):
        """The inner one has no statement slot until the outer is lowered."""

        @fp.fpy(ctx=fp.FP64)
        def f(xss: list[list[fp.Real]]) -> list[list[fp.Real]]:
            return [[x * 2.0 for x in a] for a in xss]

        once = CompToLoop.apply(f.ast)
        assert _count(once, ListComp) == 1      # the inner one, now in a slot
        assert _count(CompToLoop.apply(once), ListComp) == 0
        assert _agree(f, [[1.0, 2.0], [3.0]])


# ----------------------------------------------------------------------
# Refusals


class TestRefusals:
    def test_a_comprehension_with_no_statement_slot(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], p: bool) -> list[fp.Real]:
            return [x * x for x in xs] if p else xs

        assert CompToLoop.sites(f.ast) == []
        why = CompToLoop.refusals(f.ast)
        assert len(why) == 1 and 'no statement-level position' in why[0][1]
        assert CompToLoop.apply(f.ast).is_equiv(f.ast)

    def test_naming_a_refused_comprehension_by_cursor_says_why(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], p: bool) -> list[fp.Real]:
            return [x * x for x in xs] if p else xs

        cursor, _ = CompToLoop.refusals(f.ast)[0]
        with pytest.raises(TransformDeclined, match='no statement-level position'):
            CompToLoop.apply(f.ast, where=cursor)

    def test_rejects_non_funcdef(self):
        with pytest.raises(TypeError):
            CompToLoop.apply(_one)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# The `where` contract


class TestWhere:
    def test_sites_are_comprehensions(self):
        found = CompToLoop.sites(_one.ast)
        assert all(isinstance(c, ExprCursor) for c in found)
        assert [type(c.resolve()).__name__ for c in found] == ['ListComp']

    @pytest.mark.parametrize('f', [_one, _product, _pairs])
    def test_every_index_rewrites_and_none_does_them_all(self, f):
        listed = CompToLoop.sites(f.ast)
        assert listed
        for j in range(len(listed)):
            assert not CompToLoop.apply(f.ast, where=j).is_equiv(f.ast)
        assert not CompToLoop.apply(f.ast).is_equiv(f.ast)

    def test_a_cursor_aims_the_same_as_its_index(self):
        for j, cursor in enumerate(CompToLoop.sites(_product.ast)):
            by_index = CompToLoop.apply(_product.ast, where=j)
            assert CompToLoop.apply(_product.ast, where=cursor).is_equiv(by_index)

    def test_an_index_past_the_end_is_an_error(self):
        with pytest.raises(TransformReferenceError):
            CompToLoop.apply(_one.ast, where=7)


# ----------------------------------------------------------------------
# Precision: the point of the exercise is to lose nothing


class TestPrecision:
    @staticmethod
    def _probe(ast):
        from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn

        ast.args[0].type = ListTypeAnn(RealTypeAnn(None, None), 3, None)
        du = DefineUse.analyze(ast)
        cu = ContextUse.analyze(ast, def_use=du)
        size = ArraySizeInfer.analyze(ast).ret_size
        fmt = FormatInfer.analyze(ast, def_use=du, ctx_use=cu)
        ret = ast.body.stmts[-1].expr
        return (
            concrete_size(size.size) if isinstance(size, ListSize) else None,
            str(fmt.by_expr.get(ret)),
        )

    def test_the_lowered_form_matches_the_comprehension(self):
        """Element format *and* length survive, which is what makes lowering
        free: `Empty` starts at the bottom of the type and each `IndexedAssign`
        joins the stored value back in."""

        @fp.fpy(ctx=fp.FP64)
        def comp(xs: list[fp.Real]) -> list[fp.Real]:
            return [x * x for x in xs]

        assert self._probe(CompToLoop.apply(comp.ast)) == self._probe(comp.ast)


# ----------------------------------------------------------------------
# Compound statements: a sub-expression is not always a place to hoist to
#
# Regressions.  `SiteRewriter._visit_block` clears `_replaced` for every
# statement of a nested block, so a rewrite in the *enclosing* statement's own
# sub-expression was recorded and then lost -- and for a `while` condition the
# hoist is not even sound.


class TestCompoundStatements:
    def test_a_while_condition_is_refused(self):
        """A `while` condition is re-evaluated every iteration, so a loop
        hoisted before the `while` freezes it at its first value.  Measured: it
        turned a terminating loop into an out-of-bounds slice."""

        @fp.fpy(ctx=fp.FP64)
        def shrink(xs: list[fp.Real]) -> fp.Real:
            n = 0.0
            while len([x for x in xs]) > 1:
                xs = xs[1:]
                n = n + 1
            return n

        assert CompToLoop.sites(shrink.ast) == []
        why = CompToLoop.refusals(shrink.ast)
        assert len(why) == 1 and '`while` condition' in why[0][1]
        assert CompToLoop.apply(shrink.ast).is_equiv(shrink.ast)
        assert _agree(shrink, [1.0, 2.0, 3.0])

    @pytest.mark.parametrize('build', ['for_iterable', 'if_cond'])
    def test_an_evaluated_once_position_lowers_and_records_its_edit(self, build):
        """A `for` iterable and an `if` condition each run exactly once, so
        hoisting out of them is sound -- but the edit has to survive the nested
        block walk, or every later statement mis-forwards."""

        if build == 'for_iterable':
            @fp.fpy(ctx=fp.FP64)
            def f(xs: list[fp.Real]) -> fp.Real:
                a = 0.0
                for y in [x * 2.0 for x in xs]:
                    a = a + y
                return a
        else:
            @fp.fpy(ctx=fp.FP64)
            def f(xs: list[fp.Real]) -> fp.Real:
                a = 0.0
                if sum([x for x in xs]) > 0.0:
                    a = 1.0
                return a

        assert len(CompToLoop.sites(f.ast)) == 1
        log = CompToLoop.apply_with_edits(f.ast)
        assert len(log.edits) == 1, 'the rewrite fired but recorded nothing'
        assert _count(log.result, ListComp) == 0
        assert _agree(f, [1.0, 2.0, 3.0])

    def test_a_statement_after_the_rewrite_forwards(self):
        """What the edit log is for: a cursor taken before the pass still names
        the same statement afterwards."""
        from fpy2.transform import StmtCursor

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            a = 0.0
            if sum([x for x in xs]) > 0.0:
                a = 1.0
            b = a + 1.0
            return b

        log = CompToLoop.apply_with_edits(f.ast)
        before = StmtCursor(f.ast, log.edits[0].block_path.stmt(2))
        assert log.forward(before).resolve().is_equiv(before.resolve())
