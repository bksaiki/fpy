import fpy2 as fp
from fpy2.ast import DefaultVisitor
from fpy2.module import Module
from fpy2.transform import BindElements, Specialize
from fpy2.types import ListType, RealType

_L2 = ListType(RealType(fp.FP32), 2)


def _sized(f, *arg_types):
    m = Module()
    m.add(f, ctx=fp.FP32, arg_types=list(arg_types))
    return Specialize.apply(m, size_key=True).get(f.name).func


def _reads(ast) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_list_ref(self, e, ctx):
            nonlocal n
            n += 1
            super()._visit_list_ref(e, ctx)

    _C()._visit_function(ast, None)
    return n


def _bound(f, *arg_types, args):
    g = _sized(f, *arg_types)
    out = BindElements.apply(g.ast)
    assert repr(g.with_ast(out)(*args)) == repr(g(*args))
    return out


class TestBindElements:
    def test_two_reads_become_one_name(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs, ys):
            p = xs[0] * ys[0]
            x = xs[0]
            return p + x

        out = _bound(f, _L2, _L2, args=([1.5, 2.5], [3.0, 4.0]))
        assert _reads(out) == 2      # the binding of `xs[0]`, and `ys[0]`

    def test_a_list_bound_by_an_assignment(self):
        """... bound where the list is, which every read comes after."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            zs = [x + 1 for x in xs]
            if zs[1] > 0:
                t = zs[1]
            else:
                t = -zs[1]
            return t

        out = _bound(f, _L2, args=([1.5, -2.5],))
        assert _reads(out) == 1

    def test_not_a_store_anywhere(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            a = xs[0]
            xs[0] = 5.0
            return a + xs[0]

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 2

    def test_not_a_store_through_a_copy(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            ys = xs
            a = xs[0]
            ys[0] = 5.0
            return a + xs[0]

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 2

    def test_not_a_store_into_a_row(self):
        """`xss[0][1] = v` replaces a cell of the row, not of `xss`."""
        @fp.fpy(ctx=fp.FP32)
        def f(xss):
            row = xss[0]
            a = row[1]
            xss[0][1] = 5.0
            return a + row[1]

        LL = ListType(_L2, 2)
        out = _bound(f, LL, args=([[1.5, 2.5], [3.0, 4.0]],))
        assert _reads(out) == 3

    def test_not_one_read(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            return xs[0] + xs[1]

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 2

    def test_not_an_index_that_varies(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs, i: int):
            return xs[i] * xs[i]

        out = _bound(f, _L2, fp.types.RealType(fp.INTEGER), args=([1.5, 2.5], 1))
        assert _reads(out) == 2

    def test_an_index_a_name_holds(self):
        """An unrolled loop leaves its index as a name each copy assigns."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            g = 1
            a = xs[g]
            b = xs[1]
            return a * b

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 1

    def test_through_a_copy(self):
        """Inlining binds a callee's parameter to the caller's list by name."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            ys = xs
            return xs[0] * ys[0]

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 1

    def test_not_a_store_through_the_copy(self):
        @fp.fpy(ctx=fp.FP32)
        def f(xs):
            ys = xs
            a = xs[0]
            ys[0] = 5.0
            return a + ys[0]

        assert _reads(_bound(f, _L2, args=([1.5, 2.5],))) == 2
