"""
Context inference tests
"""

import fpy2 as fp

from fpy2.analysis import ContextInfer
from fpy2.transform import ContextInline

from .defs import tests, examples

_modules = [
    fp.libraries.core
]

def _test_tcheck_unit():
    for core in tests + examples:
        assert isinstance(core, fp.Function)
        ast = ContextInline.apply(core.ast, core.env)
        info = ContextInfer.infer(ast)
        print(ast.name, info.ret_ctx)

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                ast = ContextInline.apply(obj.ast, obj.env)
                info = ContextInfer.infer(ast)
                print(ast.name, info.ret_ctx)

def test_context_infer():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_context_infer()
