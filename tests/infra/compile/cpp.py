"""
Compilation tests for C++
"""

import fpy2 as fp

from ..unit.defs import tests, examples

def _test_unit():
    for test in tests:
        compiler = fp.CppBackend()
        arg_ctxs = tuple(fp.FP64 for _ in test.args)
        s = compiler.compile(test, ctx=fp.FP64, arg_ctxs=arg_ctxs)
        print('\n'.join(compiler.headers()))
        print(compiler.helpers())
        print(s)

    for example in examples:
        compiler = fp.CppBackend()
        arg_ctxs = tuple(fp.FP64 for _ in example.args)
        s = compiler.compile(example, ctx=fp.FP64, arg_ctxs=arg_ctxs)
        print('\n'.join(compiler.headers()))
        print(compiler.helpers())
        print(s)


if __name__ == '__main__':
    _test_unit()
