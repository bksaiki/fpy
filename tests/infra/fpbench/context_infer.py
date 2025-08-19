"""
Context inference tests
"""

import fpy2 as fp

from fpy2.analysis import ContextInfer
from fpy2.transform import ContextInline

from .fetch import fetch_cores

_skip_cores = [
    'even-int',
    'forward-euler-3d',
    'midpoint-3d',
    'ralston-3d',
    'rk4-step-3d'
]

def test_context_infer():
    for core in fetch_cores().all_cores():
        if core.name not in _skip_cores:
            print('tcheck', core.name, core.ident)
            fn = fp.Function.from_fpcore(core, ignore_unknown=True)
            ast = ContextInline.apply(fn.ast, fn.env)
            ContextInfer.infer(ast)

if __name__ == '__main__':
    test_context_infer()
