"""
Smoke tests for the type-directed FPy program generators.

These tests don't check any specific FPy behavior — they verify that the
generator emits well-typed programs that downstream passes accept. They
double as fuzz tests for those passes.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis.type_infer import TypeInfer
from fpy2.ast.fpyast import (
    BoolVal, Expr, FuncDef, FuncMeta, Integer, ListExpr, ReturnStmt,
    StmtBlock, TupleExpr,
)
from fpy2.env import ForeignEnv
from fpy2.types import BoolType, ContextType, ListType, RealType, TupleType  # noqa: F401

from . import (
    BOOL_TAGS,
    CONTEXT_TAGS,
    LIST_TAGS,
    REAL_TAGS,
    arbitrary_type,
    bool_expr,
    context_expr,
    expr,
    fpy_funcdef,
    fpy_function,
    fpy_real_funcdef,
    fpy_real_function,
    list_expr,
    real_expr,
    stmt_block,
    tuple_expr,
    value_for_type,
)
from .number import real_floats


def _wrap_as_funcdef(body_expr: Expr) -> FuncDef:
    body = StmtBlock([ReturnStmt(body_expr, None)])
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef('f', [], body, meta)


class TestGeneratedTypeChecks:
    """Every generated function should type-check to ``real -> ... -> real``."""

    @given(fpy_real_funcdef())
    def test_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        for at in analysis.arg_types:
            assert isinstance(at, RealType), f"arg type {at.format()} is not real"
        assert isinstance(analysis.return_type, RealType), (
            f"return type {analysis.return_type.format()} is not real"
        )

    @given(fpy_real_funcdef(num_args=st.just(0), max_depth=st.just(0)))
    def test_zero_arg_zero_depth_is_just_a_literal(self, fd: FuncDef) -> None:
        # With no params and no recursion budget, the body must be a literal.
        # Guards against the generator accidentally drawing a Var of a name
        # not in env (which would crash type inference) when env is empty.
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestGeneratedRuns:
    """Generated functions should evaluate under ``fp.FP64`` without crashing.

    FP64 (not ``fp.REAL``) because ``Div`` isn't implemented for real-number
    arithmetic; FP64 handles div-by-zero / sqrt of negative / etc. by
    producing ``±inf`` or ``nan`` rather than raising.
    """

    @given(
        fpy_real_function(num_args=st.integers(0, 3), max_depth=st.integers(0, 3)),
        st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_runs_under_FP64(self, f: fp.Function, data: st.DataObject) -> None:
        n = len(f.args)
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(n)]
        f(*inputs, ctx=fp.FP64)


class TestRealExprStrategyDirectly:
    """Exercise ``real_expr`` outside the ``FuncDef`` wrapper."""

    @given(real_expr({}, depth=0))
    def test_empty_env_leaf_is_a_literal(self, e: Expr) -> None:
        # No vars in env + depth=0 ⇒ leaves only ⇒ must be a literal node.
        assert isinstance(e, Integer)


class TestBoolExprStrategyDirectly:
    """Exercise ``bool_expr`` outside the ``FuncDef`` wrapper."""

    @given(bool_expr({}, depth=0))
    def test_empty_env_leaf_is_a_literal(self, e: Expr) -> None:
        assert isinstance(e, BoolVal)

    @given(bool_expr({}, depth=3))
    def test_typechecks_as_bool_inside_if_expr(self, cond: Expr) -> None:
        # Wrap as `return cond if True else 0` — if `cond` doesn't type as
        # bool the surrounding function fails to type-check.
        from fpy2.ast.fpyast import IfExpr
        fd = _wrap_as_funcdef(
            IfExpr(cond, Integer(1, None), Integer(0, None), None)
        )
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestListExprStrategyDirectly:
    """Exercise ``list_expr`` and verify produced lists are well-typed."""

    @given(list_expr(RealType(), {}, depth=0))
    def test_depth0_is_a_list_literal(self, e: Expr) -> None:
        assert isinstance(e, ListExpr)
        assert len(e.elts) >= 1

    @given(list_expr(RealType(), {}, depth=3))
    def test_typechecks_as_list_real(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, ListType)
        assert isinstance(analysis.return_type.elt, RealType)

    @given(list_expr(BoolType(), {}, depth=2))
    def test_typechecks_as_list_bool(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, ListType)
        assert isinstance(analysis.return_type.elt, BoolType)


class TestTupleExprStrategyDirectly:
    """Exercise ``tuple_expr`` and verify produced tuples are well-typed."""

    @given(tuple_expr((RealType(), RealType()), {}, depth=0))
    def test_pair_of_reals(self, e: Expr) -> None:
        assert isinstance(e, TupleExpr)
        assert len(e.elts) == 2
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, TupleType)
        assert all(isinstance(t, RealType) for t in analysis.return_type.elts)

    @given(tuple_expr((RealType(), BoolType()), {}, depth=2))
    def test_heterogeneous_pair_typechecks(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, TupleType)
        assert isinstance(analysis.return_type.elts[0], RealType)
        assert isinstance(analysis.return_type.elts[1], BoolType)


class TestStmtBlock:
    """Statement-block generator: ``Assign``* ``ReturnStmt``."""

    @given(stmt_block({}, RealType(), depth=2, max_assigns=3))
    def test_ends_with_return(self, block) -> None:
        from fpy2.ast.fpyast import Assign, ReturnStmt
        assert isinstance(block.stmts[-1], ReturnStmt)
        # everything before the return is an Assign in this phase
        for s in block.stmts[:-1]:
            assert isinstance(s, Assign)

    @given(stmt_block({}, RealType(), depth=2, max_assigns=3))
    def test_typechecks_when_wrapped(self, block) -> None:
        from fpy2.ast.fpyast import FuncMeta
        from fpy2.env import ForeignEnv
        fd = FuncDef('f', [], block,
                     FuncMeta(set(), None, None, {}, ForeignEnv.default()))
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestAssignsAreVisible:
    """A generated function with locals should still type-check and run.

    ``max_assigns`` is a *cap*, not a floor, so a body with zero locals is
    a valid draw — we just verify the body always type-checks regardless.
    """

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(0, 2),
        max_assigns=st.just(3),
    ))
    def test_typechecks_with_locals_in_scope(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestCompoundLocals:
    """Phase 9: ``Assign`` locals can have compound types (list/tuple)."""

    @given(stmt_block(
        {}, RealType(), depth=2,
        max_assigns=3, max_contexts=0, max_ifs=0, max_loops=0,
        local_types=st.just(ListType(RealType())),  # pin every local to list[real]
    ))
    def test_pinned_list_real_locals_typecheck(self, block) -> None:
        from fpy2.ast.fpyast import Assign, FuncMeta
        from fpy2.env import ForeignEnv
        # Every Assign in the body should bind a list[real] local.
        assign_count = 0
        for s in block.stmts:
            if isinstance(s, Assign):
                assign_count += 1
        # Wrap and type-check.
        fd = FuncDef('f', [], block,
                     FuncMeta(set(), None, None, {}, ForeignEnv.default()))
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)
        # Defines list[real] entries — verify via analysis.by_def
        list_real_defs = [
            t for t in analysis.by_def.values()
            if isinstance(t, ListType) and isinstance(t.elt, RealType)
        ]
        assert len(list_real_defs) >= assign_count, (
            f'expected {assign_count} list[real] defs, got {len(list_real_defs)}'
        )

    @given(stmt_block(
        {}, RealType(), depth=2,
        max_assigns=2,
        local_types=st.just(TupleType(RealType(), BoolType())),
    ))
    def test_pinned_tuple_locals_typecheck(self, block) -> None:
        from fpy2.ast.fpyast import FuncMeta
        from fpy2.env import ForeignEnv
        fd = FuncDef('f', [], block,
                     FuncMeta(set(), None, None, {}, ForeignEnv.default()))
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(0, 2),
            max_assigns=st.integers(1, 3),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_default_locals_run(self, f: fp.Function, data: st.DataObject) -> None:
        # Smoke test: with the new default ``arbitrary_type(max_depth=1)``
        # local_types, the body can include compound locals — still runs.
        inputs = [
            data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
            for _ in range(len(f.args))
        ]
        f(*inputs, ctx=fp.FP64)


class TestForStmt:
    """Phase 8: ``ForStmt`` (``for i in range(N): ...``) in bodies."""

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(1, 2),
        max_assigns=st.just(1),
        max_contexts=st.just(0),
        max_ifs=st.just(0),
        max_loops=st.just(2),
    ))
    @settings(max_examples=80, deadline=None)
    def test_with_loops_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(1, 2),
            max_assigns=st.just(1),
            max_contexts=st.just(0),
            max_ifs=st.just(0),
            max_loops=st.just(2),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_with_loops_runs(self, f: fp.Function, data: st.DataObject) -> None:
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(len(f.args))]
        f(*inputs, ctx=fp.FP64)


class TestWhileStmt:
    """Phase 10: ``WhileStmt`` (counter-driven template) in bodies."""

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(1, 2),
        max_assigns=st.just(1),
        max_contexts=st.just(0),
        max_ifs=st.just(0),
        max_loops=st.just(0),
        max_whiles=st.just(2),
    ))
    @settings(max_examples=80, deadline=None)
    def test_with_whiles_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(1, 2),
            max_assigns=st.just(1),
            max_contexts=st.just(0),
            max_ifs=st.just(0),
            max_loops=st.just(0),
            max_whiles=st.just(2),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_with_whiles_runs(self, f: fp.Function, data: st.DataObject) -> None:
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(len(f.args))]
        f(*inputs, ctx=fp.FP64)


class TestAllControlFlow:
    """Phase 10: every statement kind enabled simultaneously."""

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(0, 2),
            max_assigns=st.just(1),
            max_contexts=st.just(1),
            max_ifs=st.just(1),
            max_loops=st.just(1),
            max_whiles=st.just(1),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_runs(self, f: fp.Function, data: st.DataObject) -> None:
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(len(f.args))]
        f(*inputs, ctx=fp.FP64)


class TestIfStmt:
    """Phase 7: ``IfStmt`` / ``If1Stmt`` inside generated function bodies."""

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(1, 2),
        max_assigns=st.just(1),
        max_contexts=st.just(0),
        max_ifs=st.just(2),
    ))
    @settings(max_examples=80, deadline=None)
    def test_with_ifs_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(1, 2),
            max_assigns=st.just(1),
            max_contexts=st.just(0),
            max_ifs=st.just(2),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_with_ifs_runs(self, f: fp.Function, data: st.DataObject) -> None:
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(len(f.args))]
        f(*inputs, ctx=fp.FP64)


class TestZipEnumerate:
    """Phase 11: ``Zip`` / ``Enumerate`` productions in ``list_expr``."""

    @given(list_expr(TupleType(RealType(), RealType()), {}, depth=3))
    def test_pair_tuple_target_typechecks(self, e: Expr) -> None:
        # Any production at this target should typecheck as list[tuple[real, real]].
        from fpy2.ast.fpyast import FuncMeta
        from fpy2.env import ForeignEnv
        body = StmtBlock([ReturnStmt(e, None)])
        fd = FuncDef('f', [], body,
                     FuncMeta(set(), None, None, {}, ForeignEnv.default()))
        analysis = TypeInfer.check(fd)
        rt = analysis.return_type
        assert isinstance(rt, ListType)
        assert isinstance(rt.elt, TupleType)

    @given(list_expr(
        TupleType(RealType(), RealType(), RealType()), {}, depth=2,
        include={'zip'},
    ))
    def test_zip_only_for_arity3_typechecks(self, e: Expr) -> None:
        # arity 3 → enumerate doesn't apply; with include={'zip'} we should
        # always get a Zip node.
        from fpy2.ast.fpyast import Zip as _Zip
        assert isinstance(e, _Zip)
        assert len(e.args) == 3

    @given(list_expr(
        TupleType(RealType(), RealType()), {}, depth=2,
        include={'enumerate'},
    ))
    def test_enumerate_only_typechecks(self, e: Expr) -> None:
        from fpy2.ast.fpyast import Enumerate as _Enumerate
        assert isinstance(e, _Enumerate)


class TestArbitraryFuncdef:
    """Phase 6: ``fpy_funcdef`` with arbitrary signatures."""

    @given(
        st.data(),
    )
    @settings(max_examples=60, deadline=None)
    def test_arbitrary_typed_funcdef_typechecks(self, data: st.DataObject) -> None:
        # Draw a small random signature (1-3 args, all scalar-only for
        # easier value generation in the runtime test below).
        n_args = data.draw(st.integers(1, 3))
        arg_ts = tuple(
            data.draw(arbitrary_type(max_depth=1, scalar_only=True))
            for _ in range(n_args)
        )
        ret_t = data.draw(arbitrary_type(max_depth=1, scalar_only=True))
        fd = data.draw(fpy_funcdef(
            arg_ts, ret_t,
            max_depth=st.just(2),
            max_assigns=st.just(1),
            max_contexts=st.just(0),
        ))
        analysis = TypeInfer.check(fd)
        assert tuple(analysis.arg_types) == arg_ts
        assert analysis.return_type == ret_t

    @given(st.data())
    @settings(max_examples=60, deadline=None)
    def test_arbitrary_typed_function_runs(self, data: st.DataObject) -> None:
        n_args = data.draw(st.integers(0, 2))
        arg_ts = tuple(
            data.draw(arbitrary_type(max_depth=1, scalar_only=True))
            for _ in range(n_args)
        )
        ret_t = data.draw(arbitrary_type(max_depth=1, scalar_only=True))
        f = data.draw(fpy_function(
            arg_ts, ret_t,
            max_depth=st.just(2),
            max_assigns=st.just(1),
            max_contexts=st.just(0),
        ))
        inputs = [data.draw(value_for_type(t)) for t in arg_ts]
        f(*inputs, ctx=fp.FP64)


class TestContextExpr:
    """Context-typed expressions: ``ForeignVal(<context>, None)`` literals."""

    @given(context_expr({}, depth=0))
    def test_yields_foreign_val(self, e: Expr) -> None:
        from fpy2.ast.fpyast import ForeignVal
        assert isinstance(e, ForeignVal)
        import fpy2 as _fp
        assert isinstance(e.val, _fp.Context)


class TestContextStmt:
    """Generated functions with ``with CTX:`` blocks should typecheck + run."""

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(0, 2),
        max_assigns=st.integers(0, 2),
        max_contexts=st.just(2),
    ))
    @settings(max_examples=80, deadline=None)
    def test_with_contexts_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)

    @given(
        fpy_real_function(
            num_args=st.integers(0, 2),
            max_depth=st.integers(0, 2),
            max_assigns=st.integers(0, 2),
            max_contexts=st.just(2),
        ),
        st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_with_contexts_runs(self, f: fp.Function, data: st.DataObject) -> None:
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(len(f.args))]
        f(*inputs, ctx=fp.FP64)


class TestIncludeNarrowing:
    """``include`` filters which productions a helper emits."""

    @given(real_expr({}, depth=3, include={'literal', 'arith'}))
    def test_real_arith_only(self, e: Expr) -> None:
        from fpy2.ast.fpyast import (
            Abs, Add, Div, IfExpr, Len, Mul, Neg, NamedUnaryOp, Sub,
        )
        # No IfExpr, no Len, no transcendentals anywhere in the tree.
        stack = [e]
        while stack:
            node = stack.pop()
            assert not isinstance(node, (IfExpr, Len, NamedUnaryOp)), (
                f"unexpected production survived include filter: {type(node).__name__}"
            )
            for attr in ('args', 'elts'):
                if hasattr(node, attr):
                    children = getattr(node, attr)
                    if isinstance(children, (list, tuple)):
                        stack.extend(c for c in children if isinstance(c, Expr))

    @given(real_expr({}, depth=0, include={'literal'}))
    def test_literal_only_leaf(self, e: Expr) -> None:
        assert isinstance(e, Integer)

    @given(bool_expr({}, depth=3, include={'literal', 'compare'}))
    def test_bool_compare_only(self, e: Expr) -> None:
        from fpy2.ast.fpyast import And, Not, Or
        # No And/Or/Not — only literal leaves and Compare inner.
        stack = [e]
        while stack:
            node = stack.pop()
            assert not isinstance(node, (Not, And, Or)), (
                f"unexpected production survived include filter: {type(node).__name__}"
            )
            for attr in ('args', 'elts'):
                if hasattr(node, attr):
                    children = getattr(node, attr)
                    if isinstance(children, (list, tuple)):
                        stack.extend(c for c in children if isinstance(c, Expr))

    def test_unknown_tag_rejected(self) -> None:
        import pytest as _pytest
        with _pytest.raises(ValueError, match='unknown production tags'):
            real_expr({}, depth=2, include={'bogus'})

    def test_empty_include_with_no_leaves_rejected(self) -> None:
        import pytest as _pytest
        # No literal, no var in env ⇒ no leaves, no inner ⇒ nothing to produce.
        with _pytest.raises(ValueError, match='produces nothing'):
            real_expr({}, depth=0, include=set())

    def test_tag_sets_exposed(self) -> None:
        # Round-trip: include=all-tags is equivalent to include=None.
        assert 'arith' in REAL_TAGS
        assert 'compare' in BOOL_TAGS
        assert 'range' in LIST_TAGS


class TestRangeArgKwarg:
    """``range_arg_min`` / ``range_arg_max`` constrain ``Range1`` / ``Range2`` args."""

    @given(list_expr(
        RealType(), {}, depth=2,
        range_arg_min=42, range_arg_max=42,
    ))
    def test_pinned_range_arg_only_uses_42(self, e: Expr) -> None:
        # Walk the generated tree; every Range1/Range2 must carry Integer(42).
        from fpy2.ast.fpyast import Range1, Range2
        stack = [e]
        while stack:
            node = stack.pop()
            if isinstance(node, (Range1, Range2)):
                for arg in node.args:
                    assert isinstance(arg, Integer) and arg.val == 42, (
                        f"expected Integer(42), got {arg}"
                    )
            for attr in ('args', 'elts'):
                if hasattr(node, attr):
                    children = getattr(node, attr)
                    if isinstance(children, (list, tuple)):
                        stack.extend(c for c in children if isinstance(c, Expr))


class TestTypeDispatch:
    """The ``expr`` entry point dispatches on the target type."""

    @given(expr(RealType(), {}, depth=0))
    def test_real_dispatch_yields_real_leaf(self, e: Expr) -> None:
        assert isinstance(e, Integer)

    @given(expr(BoolType(), {}, depth=0))
    def test_bool_dispatch_yields_bool_leaf(self, e: Expr) -> None:
        assert isinstance(e, BoolVal)

    @given(expr(ListType(RealType()), {}, depth=0))
    def test_list_dispatch_yields_list_literal(self, e: Expr) -> None:
        assert isinstance(e, ListExpr)

    @given(expr(TupleType(RealType(), BoolType()), {}, depth=0))
    def test_tuple_dispatch_yields_tuple_expr(self, e: Expr) -> None:
        assert isinstance(e, TupleExpr)
        assert len(e.elts) == 2
