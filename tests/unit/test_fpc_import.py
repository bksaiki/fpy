"""Unit tests for importing FPCore (:meth:`fpy2.Function.from_fpcore`)."""

import titanfp.fpbench.fpcparser as fpcparser

from fpy2 import Function
from fpy2.ast import ConstPi
from fpy2.transform.path import walk_exprs


def test_each_constant_is_its_own_node():
    """Analyses key on node identity, so a repeated constant is not shared."""
    f = Function.from_fpcore(fpcparser.compile1('(FPCore (x) (+ (* PI x) PI))'))
    pis = [e for _, e in walk_exprs(f.ast) if isinstance(e, ConstPi)]
    assert len(pis) == 2 and pis[0] is not pis[1]
