"""`StatementForm`: every comprehension and derived iterable lowered."""

import fpy2 as fp
from fpy2 import Function
from fpy2.transform import Simplify, StatementForm


@fp.fpy(ctx=fp.FP64)
def _f(xs: list[fp.Real]) -> list[fp.Real]:
    return [x * 2.0 for x in xs]


class TestSimplify:
    def test_off_by_default(self):
        """The lowering binds its iterable, `t = xs`, which `Simplify` clears."""
        out = StatementForm.apply(_f.ast)
        assert out.format() != Simplify.apply(out).format()

    def test_on_it_is_the_simplified_form(self):
        plain = StatementForm.apply(_f.ast)
        out = StatementForm.apply(_f.ast, simplify=True)
        assert out.format() == Simplify.apply(plain).format()
        xs = [1.5, -2.0]
        assert repr(Function(out, runtime=_f.runtime)(xs)) == repr(_f(xs))
