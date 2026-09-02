"""
Unit tests for dead code elimination.
"""

import fpy2 as fp

@fp.fpy
def _example_simple_1():
    x = 1
    y = 2
    return x

@fp.fpy
def _example_simple_1_expect():
    x = 1
    return x


@fp.fpy
def _example_simple_2():
    z = 3 * 4
    x = 1
    return x

@fp.fpy
def _example_simple_2_expect():
    x = 1
    return x

@fp.fpy
def _example_dead_if1_1():
    if False:
        x = 1
    return 2

@fp.fpy
def _example_dead_if1_1_expect():
    return 2

@fp.fpy
def _example_dead_if1_2(t: fp.Real):
    if t < 0:
        pass
    return t

@fp.fpy
def _example_dead_if1_2_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if1_1(t: fp.Real):
    if t < 0:
        x = 1
    return t

@fp.fpy
def _example_if1_1_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if1_2(t: fp.Real):
    if t < 0:
        x = 1
        t = 2
    return t

@fp.fpy
def _example_if1_2_expect(t: fp.Real):
    if t < 0:
        t = 2
    return t

@fp.fpy
def _example_dead_if_1(t: fp.Real):
    if False:
        t = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_1_expect(t: fp.Real):
    t = 2
    return t

@fp.fpy
def _example_dead_if_2(t: fp.Real):
    if True:
        t = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_2_expect(t: fp.Real):
    t = 1
    return t

@fp.fpy
def _example_dead_if_3(t: fp.Real):
    if t < 0:
        pass
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_3_expect(t: fp.Real):
    if not t < 0:
        t = 2
    return t

@fp.fpy
def _example_dead_if_4(t: fp.Real):
    if t < 0:
        t = 1
    else:
        pass
    return t

@fp.fpy
def _example_dead_if_4_expect(t: fp.Real):
    if t < 0:
        t = 1
    return t

@fp.fpy
def _example_dead_if_5(t: fp.Real):
    if t < 0:
        pass
    else:
        pass
    return t

@fp.fpy
def _example_dead_if_5_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if_1(t: fp.Real):
    if t < 0:
        x = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_if_1_expect(t: fp.Real):
    if not t < 0:
        t = 2
    return t

@fp.fpy
def _example_if_2(t: fp.Real):
    if t < 0:
        t = 2
    else:
        x = 1
    return t

@fp.fpy
def _example_if_2_expect(t: fp.Real):
    if t < 0:
        t = 2
    return t

@fp.fpy
def _example_if_3(t: fp.Real):
    if t < 0:
        x = 1
        t = 2
    else:
        t = 1
    return t

@fp.fpy
def _example_if_3_expect(t: fp.Real):
    if t < 0:
        t = 2
    else:
        t = 1
    return t

@fp.fpy
def _example_if_4(t: fp.Real):
    if t < 0:
        x = 1
        y = 0
    else:
        x = 2
        z = 0
    return t

@fp.fpy
def _example_if_4_expect(t: fp.Real):
    return t

@fp.fpy
def _example_dead_while_1():
    x = 0
    while False:
        x = x + 1
    return x

@fp.fpy
def _example_dead_while_1_expect():
    x = 0
    return x

@fp.fpy
def _example_dead_while_2():
    # ``while <cond>: pass`` with a non-trivially-False cond is
    # preserved — eliminating it would convert possibly-divergent
    # behaviour into termination, which is unsound.
    x = 0
    while x < 10:
        pass
    return x

@fp.fpy
def _example_dead_while_2_expect():
    x = 0
    while x < 10:
        pass
    return x

@fp.fpy
def _example_while_1():
    x = 0
    while x < 10:
        y = 5
        x = x + 1
    return x

@fp.fpy
def _example_while_1_expect():
    x = 0
    while x < 10:
        x = x + 1
    return x

@fp.fpy
def example_simple_3():
    x = 1
    y = 2
    x = 3
    return x

@fp.fpy
def example_simple_3_expect():
    x = 3
    return x

@fp.fpy
def example_simple_4():
    x = 1
    x = 2
    x = 3
    return x

@fp.fpy
def example_simple_4_expect():
    x = 3
    return x


@fp.fpy
def _example_assert_true():
    assert True
    return 1

@fp.fpy
def _example_assert_true_expect():
    return 1


@fp.fpy
def _example_assert_true_with_msg():
    assert True, 'should never fire'
    return 1

@fp.fpy
def _example_assert_true_with_msg_expect():
    return 1


@fp.fpy
def _example_assert_false_preserved():
    assert False
    return 1

@fp.fpy
def _example_assert_false_preserved_expect():
    assert False
    return 1


@fp.fpy
def _example_pure_effect():
    1 + 2
    return 0

@fp.fpy
def _example_pure_effect_expect():
    return 0


# Rule (i): unused TupleBinding leaf → UnderscoreId.
@fp.fpy
def _example_tuple_scrub_one(x: fp.Real) -> fp.Real:
    a, b = (1.0, 2.0)
    return a + x

@fp.fpy
def _example_tuple_scrub_one_expect(x: fp.Real) -> fp.Real:
    a, _ = (1.0, 2.0)
    return a + x


# Rule (ii): all-underscore + pure RHS → drop.
@fp.fpy
def _example_tuple_drop_all_unused(x: fp.Real) -> fp.Real:
    a, b = (1.0, 2.0)
    return x

@fp.fpy
def _example_tuple_drop_all_unused_expect(x: fp.Real) -> fp.Real:
    return x


# Rule (ii) composed with already-present underscore.
@fp.fpy
def _example_tuple_drop_with_underscore(x: fp.Real) -> fp.Real:
    _, b = (1.0, 2.0)
    return x

@fp.fpy
def _example_tuple_drop_with_underscore_expect(x: fp.Real) -> fp.Real:
    return x


# Nested binding: outer leaf still used, inner all-unused → scrubbed.
@fp.fpy
def _example_tuple_nested_scrub(x: fp.Real) -> fp.Real:
    (a, b), c = ((1.0, 2.0), 3.0)
    return c + x

@fp.fpy
def _example_tuple_nested_scrub_expect(x: fp.Real) -> fp.Real:
    (_, _), c = ((1.0, 2.0), 3.0)
    return c + x


# ----------------------------------------------------------------------
# A `with` block whose installed context nothing observes: the body is
# spliced into the enclosing block.


@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_pass(y: fp.Real) -> fp.Real:
    with fp.FP16:
        pass
    return y

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_pass_expect(y: fp.Real) -> fp.Real:
    return y


@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_literal(y: fp.Real) -> fp.Real:
    with fp.FP16:
        x = 1.0
    return x + y

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_literal_expect(y: fp.Real) -> fp.Real:
    x = 1.0
    return x + y


# The inner block owns the operation, so the outer one has no use of its own.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_nested(y: fp.Real) -> fp.Real:
    with fp.FP16:
        with fp.FP32:
            z = y + y
    return z

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_nested_expect(y: fp.Real) -> fp.Real:
    with fp.FP32:
        z = y + y
    return z


# Several statements: the whole body lifts, in order.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_many_stmts(xs: list[fp.Real]) -> fp.Real:
    with fp.FP16:
        a = xs[0]
        b = xs[1]
        c = a
    return b + c

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_many_stmts_expect(xs: list[fp.Real]) -> fp.Real:
    a = xs[0]
    b = xs[1]
    c = a
    return b + c


# A comparison is exact, so a block holding only one is dead.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_compare(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        if x > y:
            return x
        else:
            return y

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_compare_expect(x: fp.Real, y: fp.Real) -> fp.Real:
    if x > y:
        return x
    else:
        return y


# Nested in a loop body: the splice lands in the `for` block, not the function.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_in_loop(xs: list[fp.Real]) -> fp.Real:
    a = 0.0
    for x in xs:
        with fp.FP16:
            y = x
        a = a + y
    return a

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_in_loop_expect(xs: list[fp.Real]) -> fp.Real:
    a = 0.0
    for x in xs:
        y = x
        a = a + y
    return a


# An operation under the block is a use, so the block stays.
@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_op(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = x * x
    return y

@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_op_expect(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = x * x
    return y


@fp.fpy(ctx=fp.FP64)
def _example_ctx_callee(x: fp.Real) -> fp.Real:
    return x * x


# A call inherits the ambient context, so it counts as a use.
@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_call(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = _example_ctx_callee(x)
    return y

@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_call_expect(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = _example_ctx_callee(x)
    return y


# The block binds a name that escapes, so it stays.
@fp.fpy
def _example_live_ctx_escapes():
    with fp.MPFixedContext(-4) as c:
        a = 1.0
    return a, c

@fp.fpy
def _example_live_ctx_escapes_expect():
    with fp.MPFixedContext(-4) as c:
        a = 1.0
    return a, c


# ----------------------------------------------------------------------
# A `with` block that installs the context already in force -- droppable even
# though operations under it do read a context.


@fp.fpy(ctx=fp.REAL)
def _example_same_ctx_as_function(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = x * x
    return y

@fp.fpy(ctx=fp.REAL)
def _example_same_ctx_as_function_expect(x: fp.Real) -> fp.Real:
    y = x * x
    return y


# Two contexts that differ: only the outer is dead, and only after the
# elimination loop revisits it.
@fp.fpy(ctx=fp.FP64)
def _example_diff_ctx_nested(x: fp.Real) -> fp.Real:
    with fp.FP16:
        with fp.FP32:
            y = x * x
    return y

@fp.fpy(ctx=fp.FP64)
def _example_diff_ctx_nested_expect(x: fp.Real) -> fp.Real:
    with fp.FP32:
        y = x * x
    return y


# A symbolic context is *unknown*, not equal to anything -- the function has
# no annotation, so the block has to stay.
@fp.fpy
def _example_symbolic_ctx(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = x * x
    return y

@fp.fpy
def _example_symbolic_ctx_expect(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = x * x
    return y


# Two blocks at the same level: neither is the other's ancestor, so the
# innermost-only filter has to let both go in one round.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_siblings(x: fp.Real) -> fp.Real:
    with fp.FP16:
        p = 1.0
    with fp.FP32:
        q = 2.0
    return (p + q) + x

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_siblings_expect(x: fp.Real) -> fp.Real:
    p = 1.0
    q = 2.0
    return (p + q) + x


# A context held in a variable: the block reads it, so `Purity` sees a `Var`
# rather than a constructor call.
@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_var(x: fp.Real) -> fp.Real:
    ctx = fp.FP16
    with ctx:
        y = x * x
    return y

@fp.fpy(ctx=fp.FP64)
def _example_live_ctx_var_expect(x: fp.Real) -> fp.Real:
    ctx = fp.FP16
    with ctx:
        y = x * x
    return y


# The same, with nothing under the block that reads it: the block goes, and
# `ctx` goes with it as an unused assign.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_var(x: fp.Real) -> fp.Real:
    ctx = fp.FP16
    with ctx:
        y = 1.0
    return y + x

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_var_expect(x: fp.Real) -> fp.Real:
    y = 1.0
    return y + x


# A list literal builds exact values, so it reads no context.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_list_literal(x: fp.Real) -> fp.Real:
    with fp.FP16:
        xs = [1.0, 2.0]
    return xs[0] + x

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_list_literal_expect(x: fp.Real) -> fp.Real:
    xs = [1.0, 2.0]
    return xs[0] + x


# An indexed store rounds nothing either -- it stores the value it is given.
@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_indexed_store(xs: list[fp.Real], x: fp.Real) -> fp.Real:
    with fp.FP16:
        xs[0] = 1.0
    return xs[0] + x

@fp.fpy(ctx=fp.FP64)
def _example_dead_ctx_indexed_store_expect(xs: list[fp.Real], x: fp.Real) -> fp.Real:
    xs[0] = 1.0
    return xs[0] + x


_examples: list[tuple[fp.Function, fp.Function]] = [
    (_example_simple_1, _example_simple_1_expect),
    (_example_simple_2, _example_simple_2_expect),
    (_example_dead_if1_1, _example_dead_if1_1_expect),
    (_example_dead_if1_2, _example_dead_if1_2_expect),
    (_example_if1_1, _example_if1_1_expect),
    (_example_if1_2, _example_if1_2_expect),
    (_example_dead_if_1, _example_dead_if_1_expect),
    (_example_dead_if_2, _example_dead_if_2_expect),
    (_example_dead_if_3, _example_dead_if_3_expect),
    (_example_dead_if_4, _example_dead_if_4_expect),
    (_example_dead_if_5, _example_dead_if_5_expect),
    (_example_if_1, _example_if_1_expect),
    (_example_if_2, _example_if_2_expect),
    (_example_if_3, _example_if_3_expect),
    (_example_if_4, _example_if_4_expect),
    (_example_dead_while_1, _example_dead_while_1_expect),
    (_example_dead_while_2, _example_dead_while_2_expect),
    (_example_while_1, _example_while_1_expect),
    (example_simple_3, example_simple_3_expect),
    (example_simple_4, example_simple_4_expect),
    (_example_assert_true, _example_assert_true_expect),
    (_example_assert_true_with_msg, _example_assert_true_with_msg_expect),
    (_example_assert_false_preserved, _example_assert_false_preserved_expect),
    (_example_pure_effect, _example_pure_effect_expect),
    (_example_tuple_scrub_one, _example_tuple_scrub_one_expect),
    (_example_tuple_drop_all_unused, _example_tuple_drop_all_unused_expect),
    (_example_tuple_drop_with_underscore, _example_tuple_drop_with_underscore_expect),
    (_example_tuple_nested_scrub, _example_tuple_nested_scrub_expect),
    (_example_dead_ctx_pass, _example_dead_ctx_pass_expect),
    (_example_dead_ctx_literal, _example_dead_ctx_literal_expect),
    (_example_dead_ctx_nested, _example_dead_ctx_nested_expect),
    (_example_dead_ctx_many_stmts, _example_dead_ctx_many_stmts_expect),
    (_example_dead_ctx_compare, _example_dead_ctx_compare_expect),
    (_example_dead_ctx_in_loop, _example_dead_ctx_in_loop_expect),
    (_example_live_ctx_op, _example_live_ctx_op_expect),
    (_example_live_ctx_call, _example_live_ctx_call_expect),
    (_example_live_ctx_escapes, _example_live_ctx_escapes_expect),
    (_example_same_ctx_as_function, _example_same_ctx_as_function_expect),
    (_example_diff_ctx_nested, _example_diff_ctx_nested_expect),
    (_example_symbolic_ctx, _example_symbolic_ctx_expect),
    (_example_dead_ctx_siblings, _example_dead_ctx_siblings_expect),
    (_example_live_ctx_var, _example_live_ctx_var_expect),
    (_example_dead_ctx_var, _example_dead_ctx_var_expect),
    (_example_dead_ctx_list_literal, _example_dead_ctx_list_literal_expect),
    (_example_dead_ctx_indexed_store, _example_dead_ctx_indexed_store_expect),
]


def _with_count(ast) -> int:
    return str(fp.Function(ast, runtime=None).format()).count('with ')


class TestDeadCode():

    def test_examples(self):
        for f, f_expect in _examples:
            f_opt = fp.transform.DeadCodeEliminate.apply(f.ast)
            f_opt.name = f_expect.name
            assert f_opt.is_equiv(f_expect.ast), f'expect:\n{f_expect.format()}\nactual:\n{f_opt.format()}'


class TestDeadContext:
    """What a shape comparison cannot check: that the elimination loop reaches
    the blocks other rules strand, and that the survivor is the right one."""

    def test_values_survive_the_splice(self):
        f_opt = fp.Function(
            fp.transform.DeadCodeEliminate.apply(_example_dead_ctx_literal.ast),
            runtime=_example_dead_ctx_literal.runtime,
        )
        for y in (0.0, 1.5, -3.25):
            assert repr(f_opt(y)) == repr(_example_dead_ctx_literal(y))

    def test_an_unused_assign_strands_its_block(self):
        """The elimination loop is what makes this work: removing `y` leaves the
        block with nothing that reads the context, so the next round drops it."""

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = x * x
            return x

        out = fp.transform.DeadCodeEliminate.apply(f.ast)
        assert _with_count(out) == 0
        assert repr(fp.Function(out, runtime=f.runtime)(1.5)) == repr(f(1.5))

    def test_an_eliminated_rounding_strands_its_block(self):
        """`RoundElim` is the main producer: dropping the only rounding in a
        block leaves the context with no use."""

        @fp.fpy
        def f():
            with fp.IEEEContext(8, 32, fp.RM.RNE):
                return fp.round(0)

        hoisted = fp.transform.RoundElim.apply(f.ast)
        assert _with_count(hoisted) == 1
        assert _with_count(fp.transform.DeadCodeEliminate.apply(hoisted)) == 0

    def test_the_same_context_twice_keeps_one(self):
        """The regression this rule needs care about: the inner block is
        redundant with the outer, and the outer looks unused only because the
        inner one owns the multiply.  Dropping both would leave the multiply
        under the function's own FP64."""

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                with fp.FP16:
                    y = x * x
            return y

        out = fp.transform.DeadCodeEliminate.apply(f.ast)
        assert _with_count(out) == 1
        assert 'with fp.FP16:' in fp.Function(out, runtime=f.runtime).format()
        assert repr(fp.Function(out, runtime=f.runtime)(1.1)) == repr(f(1.1))

    def test_two_structurally_equal_contexts_are_the_same_context(self):
        """`_same_context` compares resolved values, not identity: the block
        installs a distinct but equal `Context` object."""

        @fp.fpy(ctx=fp.IEEEContext(5, 16, fp.RM.RNE))
        def f(x: fp.Real) -> fp.Real:
            with fp.IEEEContext(5, 16, fp.RM.RNE):
                y = x * x
            return y

        out = fp.transform.DeadCodeEliminate.apply(f.ast)
        assert _with_count(out) == 0
        assert repr(fp.Function(out, runtime=f.runtime)(1.1)) == repr(f(1.1))

    def test_alternating_rounding_rewrites_stop_diverging(self):
        """`elim_round` and `insert_round` each hoist into a fresh block, so
        alternating them nests one deeper every round.  Every block but the
        innermost is dead, and the innermost then matches the function's own
        context."""
        import fpy2.strategies as st
        from fpy2.types import RealType

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        g = st.monomorphize(f, fp.FP64, [RealType(fp.FP32), RealType(fp.FP32)])
        for _ in range(4):
            g = st.insert_round(st.elim_round(g), fp.FP64)
        assert _with_count(g.ast) == 8

        out = fp.transform.DeadCodeEliminate.apply(g.ast)
        assert _with_count(out) == 0
        assert repr(fp.Function(out, runtime=g.runtime)(1.5, 2.5)) == repr(g(1.5, 2.5))

    def test_the_unfold_pipeline_loses_its_redundant_blocks(self):
        """The motivating case: `unfold_*` / `float_to_fixed` / `rescale_fixed`
        emit `with fp.REAL:` inside scopes that are already REAL."""
        import fpy2.strategies as st

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.BF16:
                y = fp.round(x)
            return y

        g = st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(st.unfold_special(f), early_check=True)
        ))
        reals = g.format().count('with fp.REAL')

        out = fp.Function(fp.transform.DeadCodeEliminate.apply(g.ast), runtime=g.runtime)
        # the three that went are REAL blocks nested in REAL scopes
        assert _with_count(out.ast) == _with_count(g.ast) - 3
        assert out.format().count('with fp.REAL') == reals - 3
        for v in (float('nan'), float('inf'), float('-inf'), 0.0, -0.0,
                  1.5, -1.5, 1e40, -1e-42):
            assert repr(out(v)) == repr(g(v)), f'disagree at {v}'

    def test_a_target_read_after_a_loop_is_a_use(self):
        """A name read after the loop has no *direct* use -- the read goes
        through a phi -- so the successors have to decide it.  Missing that
        dropped the binding and moved the multiply to the outer context."""

        @fp.fpy(ctx=fp.FP64)
        def f(n: fp.Real, x: fp.Real) -> fp.Real:
            c = fp.FP64
            for _i in range(n):
                with fp.FP16 as c:
                    y = 1.0
            with c:
                z = x * x
            return z

        out = fp.transform.DeadCodeEliminate.apply(f.ast)
        assert 'fp.FP16 as c' in fp.Function(out, runtime=f.runtime).format()
        # 1.1 * 1.1 rounds differently in FP16 than in FP64
        assert repr(fp.Function(out, runtime=f.runtime)(1, 1.1)) == repr(f(1, 1.1))

    def test_a_target_read_after_a_branch_is_a_use(self):
        """The same through an `if` phi, where dropping the binding left the
        phi pointing at a deleted definition -- a `KeyError`, not a wrong
        answer, and `SyntaxCheck` did not catch it."""

        @fp.fpy(ctx=fp.FP64)
        def f(b: fp.Real, x: fp.Real) -> fp.Real:
            if b > 0.0:
                with fp.FP16 as c:
                    y = 1.0
            else:
                c = fp.FP32
            with c:
                z = x * x
            return z

        out = fp.transform.DeadCodeEliminate.apply(f.ast)      # no KeyError
        assert 'fp.FP16 as c' in fp.Function(out, runtime=f.runtime).format()
        for b in (1.0, -1.0):
            assert repr(fp.Function(out, runtime=f.runtime)(b, 1.1)) == repr(f(b, 1.1))

    def test_a_stochastic_context_is_never_redundant(self):
        """`Context` equality ignores `rng`, so two seeded streams compare
        equal; dropping the block would draw from the outer generator."""
        import random

        outer = fp.IEEEContext(5, 11, num_randbits=2, rng=random.Random(1))
        inner = fp.IEEEContext(5, 11, num_randbits=2, rng=random.Random(999))
        assert outer == inner and outer is not inner

        @fp.fpy(ctx=outer)
        def f(x: fp.Real) -> fp.Real:
            with inner:
                y = x * x
            return y

        assert _with_count(fp.transform.DeadCodeEliminate.apply(f.ast)) == 1

    def test_an_impure_context_expression_is_kept(self):
        """Dropping the block would skip evaluating the expression."""

        def make_ctx():
            return fp.FP16

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with make_ctx():
                y = 1.0
            return y + x

        assert _with_count(fp.transform.DeadCodeEliminate.apply(f.ast)) == 1


class TestUnusedPhiArguments:
    """An unused phi does not make its arguments unused."""

    def test_an_argument_read_in_its_own_branch_survives(self):
        """``z``'s phi at the merge has no uses, but the ``else`` branch's
        definition is read by ``y = z * 3`` -- and a phi is not a use site, so
        the merge cannot see that.  Removing it left a read with no definition,
        which the pass then crashed re-analyzing."""

        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            if x > 0:
                y = 1.0
            elif x > -1:
                t = x * 2.0
                z = t
                y = t * 2.0
            else:
                z = x * 3.0
                y = z * 3.0
            return y

        out = fp.transform.DeadCodeEliminate.apply(f.ast)
        fp.analysis.DefineUse.analyze(out)      # well-formed
        src = out.format()
        assert 'z = (x * 3)' in src             # read in its branch: kept
        assert 'z = t' not in src               # read only by the phi: dropped
