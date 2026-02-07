import fpy2 as fp
import itertools
import unittest


from fpy2.backend.mpfx import ElimRound


@fp.fpy(ctx=fp.REAL)
def _example_round_1(x: fp.Real) -> fp.Real:
    with fp.MX_E5M2:
        return fp.round(x)

@fp.fpy(ctx=fp.REAL)
def _example_round_1_expect(x: fp.Real) -> fp.Real:
    with fp.MX_E5M2:
        return fp.cast(x)

@fp.fpy(ctx=fp.REAL)
def _example_add_1(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        return x + y

@fp.fpy(ctx=fp.REAL)
def _example_add_1_expect(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        with fp.REAL:
            t = x + y
        return fp.cast(t)

@fp.fpy(ctx=fp.REAL)
def _example_mul_1(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        return x * y

@fp.fpy(ctx=fp.REAL)
def _example_mul_1_expect(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        with fp.REAL:
            t = x * y
        return fp.cast(t)


class TestElimRound(unittest.TestCase):

    def test_elim_round(self):
        ast = _example_round_1.ast
        expect_ast = _example_round_1_expect.ast

        arg_types = [fp.types.RealType(fp.MX_E2M1)]
        ast = fp.transform.Monomorphize.apply_by_arg(ast, None, arg_types)
        expect_ast = fp.transform.Monomorphize.apply_by_arg(expect_ast, None, arg_types)

        opt = ElimRound.apply(ast)
        opt.name = expect_ast.name

        self.assertTrue(opt.is_equiv(expect_ast), f'expect:\n{expect_ast.format()}\nactual:\n{opt.format()}')

    def test_elim_add(self):
        ast = _example_add_1.ast
        expect_ast = _example_add_1_expect.ast

        arg_types = [fp.types.RealType(fp.MX_E4M3), fp.types.RealType(fp.MX_E4M3)]
        ast = fp.transform.Monomorphize.apply_by_arg(ast, None, arg_types)
        expect_ast = fp.transform.Monomorphize.apply_by_arg(expect_ast, None, arg_types)

        opt = ElimRound.apply(ast)
        opt.name = expect_ast.name

        self.assertTrue(opt.is_equiv(expect_ast), f'expect:\n{expect_ast.format()}\nactual:\n{opt.format()}')

    def test_elim_mul(self):
        ast = _example_mul_1.ast
        expect_ast = _example_mul_1_expect.ast

        arg_types = [fp.types.RealType(fp.FP16), fp.types.RealType(fp.FP16)]
        ast = fp.transform.Monomorphize.apply_by_arg(ast, None, arg_types)
        expect_ast = fp.transform.Monomorphize.apply_by_arg(expect_ast, None, arg_types)

        opt = ElimRound.apply(ast)
        opt.name = expect_ast.name

        self.assertTrue(opt.is_equiv(expect_ast), f'expect:\n{expect_ast.format()}\nactual:\n{opt.format()}')
