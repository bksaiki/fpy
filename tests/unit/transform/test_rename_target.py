"""Unit tests for :class:`fpy2.transform.RenameTarget`."""

import fpy2 as fp
from fpy2.ast import NamedId
from fpy2.transform import RenameTarget


def test_a_context_target_is_renamed_with_its_uses():
    @fp.fpy
    def f(x: fp.Real):
        with fp.FP32 as c:
            y = x + 1
        with c:
            z = y * 2
        return z

    @fp.fpy
    def f_expect(x: fp.Real):
        with fp.FP32 as d:
            y = x + 1
        with d:
            z = y * 2
        return z

    c = f.ast.body.stmts[0].target
    g = RenameTarget.apply(f.ast, {c: NamedId('d')})
    g.name = f_expect.name
    assert g.is_equiv(f_expect.ast), g.format()
