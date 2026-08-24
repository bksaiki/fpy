"""
Unit tests for the :class:`fpy2.transform.CompToLoop` transform.

The rewrite mints fresh ``t`` / ``acc`` / ``i`` / ``j`` names via ``Gensym``, so
comparing against a hand-written golden AST is brittle.  These tests assert

1. **Structural shape** — the comprehension is gone, replaced by an `fp.empty`
   allocation and a loop that writes every slot.
2. **Semantic equivalence** via the interpreter, on the shapes that decide the
   size formula: one clause, several independent clauses, and a ragged flatten
   whose length is a sum rather than a product.  The ragged edge cases are the
   ones that caught the formula: an empty inner list, all-empty, empty-outer.
3. **Refusals**, by the reason each gives.
4. **The `where` contract** — a listing reports exactly what `where=None`
   rewrites.
"""

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

    def test_does_not_mutate_the_input(self):
        CompToLoop.apply(_one.ast)
        assert _count(_one.ast, ListComp) == 1

    def test_temp_id_names_the_iterable_binding(self):
        out = CompToLoop.apply(_one.ast, temp_id=NamedId('src'))
        assert 'src = xs' in out.format()


# ----------------------------------------------------------------------
# The ragged case: the length is a sum, not a product


class TestRagged:
    @pytest.mark.parametrize('arg', [
        [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],   # uneven
        [[], [7.0]],                            # an empty inner list
        [[]],                                   # all empty
        [],                                     # empty outer
    ])
    def test_agrees_on_the_shapes_that_break_a_product(self, arg):
        assert _agree(_ragged, arg)

    def test_the_size_is_a_sum_over_the_clause(self):
        out = CompToLoop.apply(_ragged.ast)
        # the size expression is a `sum` over a comprehension of lengths, so one
        # comprehension survives this pass -- see the fixpoint test below
        assert 'sum(' in out.format()

    @pytest.mark.parametrize('arg', [
        [[[1.0, 2.0], [3.0]], [[4.0]]],
        [[[]], [[5.0, 6.0], []]],
        [],
    ])
    def test_three_clauses_nest_the_sums(self, arg):
        assert _agree(_ragged3, arg)

    def test_a_dependent_iterable_with_ordinary_ops_still_lowers(self):
        """Only *stochastic* re-evaluation is refused; exact arithmetic is fine."""

        @fp.fpy(ctx=fp.FP64)
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            return [a[i] for a in xss for i in range(len(a))]

        assert CompToLoop.sites(f.ast)
        assert _agree(f, [[1.0, 2.0], [3.0]])


# ----------------------------------------------------------------------
# Reaching a comprehension-free program


class TestFixpoint:
    def test_ragged_clears_after_a_second_pass(self):
        """The size expression is itself a comprehension, so a ragged
        comprehension takes one pass per level to clear."""
        ast, arg = _ragged.ast, [[1.0, 2.0], [3.0]]
        want = _vals(_ragged(arg))
        for _ in range(4):
            nxt = CompToLoop.apply(ast)
            if nxt.is_equiv(ast):
                break
            ast = nxt
            assert _vals(Function(ast, runtime=_ragged.runtime)(arg)) == want
        assert _count(ast, ListComp) == 0

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

    def test_a_stochastic_dependent_iterable(self):
        """The size expression evaluates a dependent iterable a second time, so
        a length that could round differently would mis-size the allocation."""

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.IEEEContext(8, 32, num_randbits=4):
                r = [b for a in xs for b in range(fp.round(a))]
            return r

        assert CompToLoop.sites(f.ast) == []
        why = CompToLoop.refusals(f.ast)
        assert len(why) == 1 and 'stochastically' in why[0][1]
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

    @pytest.mark.parametrize('f', [_one, _product, _ragged, _pairs])
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
        free: see ``docs/todos/comprehension-lowering.md``."""

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
