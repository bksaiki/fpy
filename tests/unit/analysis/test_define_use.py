"""Unit tests for :class:`fpy2.analysis.DefineUse`.

Regression coverage for the loop-cond def-use bug: a ``while`` cond's
uses of loop-mutated variables must resolve through the loop-header
phi (not the pre-loop ``AssignDef``), otherwise downstream optimisers
fold the cond against a stale pre-loop value.
"""

import fpy2 as fp

from fpy2.analysis import DefineUse, PhiDef, AssignDef
from fpy2.ast import Compare, NamedId, Var, WhileStmt


def _find_while(block):
    for s in block.stmts:
        if isinstance(s, WhileStmt):
            return s
        for attr in ('body', 'ift', 'iff'):
            sub = getattr(s, attr, None)
            if sub is not None:
                r = _find_while(sub)
                if r is not None:
                    return r
    return None


class TestWhileCondResolution:
    """The ``while``-cond's uses of loop-mutated variables resolve
    through the loop-header phi."""

    def test_cond_var_resolves_to_phi(self):
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                i = 0
                while i < n:
                    i = i + 1
                return i

        du = DefineUse.analyze(f.ast)
        whl = _find_while(f.ast.body)
        assert whl is not None

        cond = whl.cond
        assert isinstance(cond, Compare)
        i_in_cond = cond.args[0]
        assert isinstance(i_in_cond, Var)

        d = du.find_def_from_use(i_in_cond)
        assert isinstance(d, PhiDef), (
            f'cond Var(i) should resolve to the loop-header phi; '
            f'got {type(d).__name__}'
        )
        assert d.is_loop, 'phi should be a loop phi'

    def test_cond_var_not_mutated_resolves_to_outer_def(self):
        """A while-cond reference to a variable that *isn't* mutated
        in the body still resolves to the pre-loop def — only mutated
        variables get a header phi."""
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                i = 0
                while i < n:
                    i = i + 1
                return i

        du = DefineUse.analyze(f.ast)
        whl = _find_while(f.ast.body)
        cond = whl.cond
        # ``n`` is not mutated in the body — should resolve to its
        # argument AssignDef directly.
        n_in_cond = cond.args[1]
        d = du.find_def_from_use(n_in_cond)
        assert isinstance(d, AssignDef)
        assert not isinstance(d, PhiDef)


class TestConstFoldOnLoopCond:
    """End-to-end: with the fix, :class:`ConstFold` no longer folds
    the cond against a stale pre-loop value."""

    def test_cond_does_not_fold_to_pre_loop_value(self):
        from fpy2.transform import ConstFold
        from fpy2.ast import BoolVal, Integer

        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                i = 0
                while i < n:
                    i = i + 1
                return i

        folded = ConstFold.apply(f.ast)
        whl = _find_while(folded.body)
        # The cond should still be a Compare (not folded to a BoolVal),
        # and its first arg should still be Var(i) (not Integer(0)).
        assert isinstance(whl.cond, Compare), \
            f'cond should stay Compare; got {type(whl.cond).__name__}'
        first = whl.cond.args[0]
        assert isinstance(first, Var), \
            f'cond Var(i) should stay Var; got {type(first).__name__}'

    def test_runtime_preserved(self):
        from fpy2.transform import ConstFold

        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                i = 0
                while i < n:
                    i = i + 1
                return i

        folded = ConstFold.apply(f.ast)
        # Wrap the folded AST back as a callable Function via the
        # original.  Re-execute and compare.
        g = f.with_ast(folded)
        for n in (0.0, 1.0, 3.0, 10.0):
            assert float(f(n, ctx=fp.FP64)) == float(g(n, ctx=fp.FP64)), \
                f'runtime mismatch at n={n}'
