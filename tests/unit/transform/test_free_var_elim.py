"""
Unit tests for the :class:`fpy2.transform.FreeVarElim` transform.

The pass materializes each captured *data* free variable as a leading
``x = <value>`` assignment (closing the function over it) and leaves
free variables with no literal form — e.g. a called function — untouched.
"""

import fpy2 as fp

from fpy2.ast.fpyast import Assign, Integer, NamedId
from fpy2.transform import FreeVarElim


def _leading_assigns(ast) -> dict:
    """Map ``name -> Assign`` for the leading assignment statements."""
    out = {}
    for stmt in ast.body.stmts:
        if isinstance(stmt, Assign) and isinstance(stmt.target, NamedId):
            out[str(stmt.target)] = stmt
    return out


def test_data_free_var_inlined_and_closed():
    def make_f(x):
        @fp.fpy
        def f(y):
            return x + y
        return f

    f = make_f(1)
    assert NamedId('x') in {NamedId(str(v)) for v in f.ast.free_vars}

    out = FreeVarElim.apply(f.ast)
    # `x` is now bound by a leading assignment and no longer free.
    assert not out.free_vars
    assigns = _leading_assigns(out)
    assert 'x' in assigns
    assert isinstance(assigns['x'].expr, Integer)
    assert assigns['x'].expr.val == 1


def test_callable_free_var_left_untouched():
    @fp.fpy
    def g(z):
        return z * z

    def make_h():
        c = 3.0
        @fp.fpy
        def h(y):
            return g(y) + c
        return h

    h = make_h()
    out = FreeVarElim.apply(h.ast)
    # `c` (data) is inlined and closed; `g` (a callable) stays free.
    assert {str(v) for v in out.free_vars} == {'g'}
    assert 'c' in _leading_assigns(out)


def test_no_free_vars_is_noop():
    @fp.fpy
    def f(x, y):
        return x + y

    out = FreeVarElim.apply(f.ast)
    assert out is f.ast


def test_semantics_preserved():
    def make_f(x):
        @fp.fpy
        def f(y):
            return x + y
        return f

    f = make_f(5)
    out = FreeVarElim.apply(f.ast)
    closed: fp.Function = fp.Function(out)
    assert closed(3.0) == f(3.0)
