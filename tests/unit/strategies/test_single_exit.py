"""Unit tests for :func:`fpy2.strategies.single_exit`.

The rewrite is covered in `tests/unit/transform/test_single_exit.py`; what is
asserted here is the strategy layer, and the composition the pass exists for --
`inline`, `simplify_if` and the FPCore backend all refuse a function with more
than one exit.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.visitor import DefaultVisitor
from fpy2.module import _RebindCalls
from fpy2.analysis import Reachability
from fpy2.strategies import TransformDeclined, inline, simplify_if, single_exit


@fp.fpy
def _early(x: fp.Real) -> fp.Real:
    if x > 0:
        return 1.0
    y = x * 2
    return y


@fp.fpy
def _in_a_loop(xs: list[fp.Real]) -> fp.Real:
    for x in xs:
        if x < 0:
            return x
    return 0.0


@fp.fpy
def _calls_early(x: fp.Real) -> fp.Real:
    return _early(x) + 1.0


def _calls(ast) -> int:
    """How many `Call`s to an FPy function remain."""
    n = 0

    class _V(DefaultVisitor):
        def _visit_call(self, e, ctx):
            nonlocal n
            if isinstance(e.fn, Function):
                n += 1
            super()._visit_call(e, ctx)

    _V()._visit_function(ast, None)
    return n


class TestTheStrategyLayer:
    def test_returns_a_function(self):
        assert isinstance(single_exit(_early), Function)

    @pytest.mark.parametrize('x', [1.0, -1.0, 0.0])
    def test_semantics_are_preserved(self, x):
        assert repr(single_exit(_early)(x)) == repr(_early(x))

    def test_one_return_remains(self):
        assert len(Reachability.analyze(single_exit(_early).ast).ret_stmts) == 1

    def test_a_loop_return_declines(self):
        with pytest.raises(TransformDeclined, match='inside a loop'):
            single_exit(_in_a_loop)


class TestItUnblocksTheConsumers:
    """The point of the pass: each of these refuses `_early` as written."""

    def test_inline_skips_a_multi_return_callee(self):
        """`inline` refuses a callee without exactly one trailing return, and
        with `where=None` a refusal is skipped rather than raised -- so the
        call simply survives."""
        assert _calls(inline(_calls_early).ast) == 1

    def test_inline_flattens_it_once_the_callee_has_one_exit(self):
        retargeted = _calls_early.with_ast(
            _RebindCalls({_early: single_exit(_early)}).apply(_calls_early.ast)
        )
        assert _calls(inline(retargeted).ast) == 0

    def test_simplify_if_accepts_it_afterwards(self):
        with pytest.raises(TransformDeclined):
            simplify_if(_early)
        simplify_if(single_exit(_early))
