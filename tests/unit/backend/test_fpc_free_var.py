"""
FPCore lowering of captured free variables.

FPCore has no closure environment, so a captured *data* free variable must be
materialized as a binding (via ``FreeVarElim``) rather than left as a dangling
name.  Callables are referenced by name and left free.
"""

import fpy2 as fp

from fpy2 import FPCoreCompiler


def test_data_free_var_closed_via_let():
    def make_f(x):
        @fp.fpy
        def f(y):
            return x + y
        return f

    f = make_f(1)
    # `unsafe_int_cast` lets the integer constant compile; the point of the
    # test is that `x` is *bound* rather than dangling.
    core = str(FPCoreCompiler(unsafe_int_cast=True).compile(f))
    assert '(let ([x' in core       # `x` is bound by a let, not a free name
    assert '(+ x y)' in core
