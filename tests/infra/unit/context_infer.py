"""
Context inference tests
"""

import fpy2 as fp

from fpy2.analysis import ContextInfer
from fpy2.transform import ContextInline

from .defs import tests, examples

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

_unit_ignore = [
    'test_context_expr1',
    'test_context_expr2',
    'test_context8',
    'keep_p_1'
]

def _test_tcheck_unit():
    for core in tests + examples:
        assert isinstance(core, fp.Function)
        if core.name in _unit_ignore:
            continue

        ast = ContextInline.apply(core.ast, core.env)
        info = ContextInfer.infer(ast)
        print(ast.name, info.func_ty)

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    ast = ContextInline.apply(obj.ast, obj.env)
                    info = ContextInfer.infer(ast)
                    print(ast.name, info.func_ty)
                case fp.Primitive():
                    ctx = ContextInfer.infer_primitive(obj)
                    print(obj.name, ctx)

def test_context_infer():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_context_infer()
