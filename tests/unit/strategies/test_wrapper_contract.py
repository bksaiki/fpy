"""The guard every strategy shares.

A strategy takes a `Function`, not the `FuncDef` inside it, and each one checks
that before doing anything.  It was hand-written per strategy, which left half
of them untested; here it is one table, so a new strategy is covered by adding
its row rather than by remembering to copy a test.
"""

import pytest

import fpy2 as fp
import fpy2.strategies as st


@fp.fpy(ctx=fp.FP64)
def _prog(x: fp.Real) -> fp.Real:
    with fp.FP32:
        y = fp.round(x)
    return y


_EXTRA_ARGS = {
    'insert_round': (fp.FP64,),
    'split_round': (fp.FP32,),
    'split': (2,),
}
"""Positional arguments a strategy needs beyond the function itself.  The guard
fires before any of them is read, so the values only have to be well-typed."""

_STRATEGIES = sorted(
    n for n in st.__all__
    if not n[0].isupper() and n not in ('sites', 'refusals')
)


def test_every_strategy_is_covered():
    """The table is the list of strategies, so it cannot silently fall behind
    one that is added later."""
    public = {
        n for n in st.__all__
        if not n[0].isupper() and n not in ('sites', 'refusals')
    }
    assert set(_STRATEGIES) == public


@pytest.mark.parametrize('name', _STRATEGIES)
def test_a_bare_funcdef_is_refused(name):
    """`func.ast` is a `FuncDef`; passing one is the mistake this catches."""
    with pytest.raises(TypeError):
        getattr(st, name)(_prog.ast, *_EXTRA_ARGS.get(name, ()))
