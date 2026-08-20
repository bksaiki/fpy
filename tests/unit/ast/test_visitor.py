"""
Unit tests for the AST visitors' traversal.

A keyword argument is an expression like any other, so every visitor has to
reach it -- `LiveVars` among them, which `Pattern.vars()` is computed from.
"""

import fpy2 as fp

from fpy2.analysis import LiveVars
from fpy2.ast import Call, Var
from fpy2.ast.visitor import DefaultTransformVisitor, DefaultVisitor


@fp.fpy(ctx=fp.REAL)
def kwarg_use(x: fp.Real) -> fp.Real:
    m = 8
    with fp.MPBFixedContext(-4, 8, maxval=m):
        y = fp.round(x)
    return y


def _vars_seen(visitor_cls, ast) -> set[str]:
    """Every variable the visitor reaches."""
    seen: set[str] = set()

    class _C(visitor_cls):
        def _visit_var(self, e, ctx):
            seen.add(str(e.name))
            return super()._visit_var(e, ctx)

    _C()._visit_function(ast, None)
    return seen


def test_both_visitors_reach_a_keyword_argument():
    for cls in (DefaultVisitor, DefaultTransformVisitor):
        assert 'm' in _vars_seen(cls, kwarg_use.ast), cls.__name__


def test_a_keyword_argument_is_visited_once():
    """A visitor that walks kwargs itself *and* calls `super()` counts them
    twice."""
    seen: list[Var] = []

    class _C(DefaultVisitor):
        def _visit_var(self, e, ctx):
            if str(e.name) == 'm':
                seen.append(e)
            return super()._visit_var(e, ctx)

    _C()._visit_function(kwarg_use.ast, None)
    assert len(seen) == 1


def test_live_vars_sees_a_keyword_argument():
    """`Pattern.vars()` is built on this: a pattern variable appearing only in a
    keyword argument has to be a pattern variable."""
    ctx_expr = kwarg_use.ast.body.stmts[1].ctx
    assert isinstance(ctx_expr, Call)
    assert {str(v) for v in LiveVars.analyze(ctx_expr)} == {'m'}


def test_a_pattern_variable_in_a_keyword_argument_is_a_pattern_variable():
    @fp.pattern
    def bounded(m):
        fp.MPBFixedContext(-4, 8, maxval=m)

    assert {str(v) for v in bounded.vars()} == {'m'}
