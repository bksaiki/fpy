"""
Context inference tests
"""

import fpy2 as fp

from fpy2.analysis import ContextInfer
from fpy2.transform import ConstFold

from ..examples import all_tests


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
    'example_static_context1',
    'example_static_context2',
    'keep_p_1'
]

def _test_tcheck_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        if core.name in _unit_ignore:
            continue

        ast = ConstFold.apply(core.ast, enable_op=True)
        info = ContextInfer.infer(ast)
        print(ast.name, info.fn_type)

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    ast = ConstFold.apply(obj.ast, enable_op=True)
                    print(ast.name)
                    info = ContextInfer.infer(ast)
                    print(ast.name, info.fn_type)
                case fp.Primitive():
                    ctx = ContextInfer.infer_primitive(obj)
                    print(obj.name, ctx)

def test_context_infer():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_context_infer()
