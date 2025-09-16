"""
Compilation tests for C++
"""

import fpy2 as fp

from ..unit.defs import tests, examples

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

_test_ignore = [
    'test_context_expr1',
    'test_context1',
    'test_context2',
    'test_context3',
    'test_context4',
    'test_context5',
]

_example_ignore = [
    'fma_ctx',
    'keep_p_1'
]

_library_ignore = [
    '_modf_spec',
    'isinteger',
    '_ldexp_spec'
]


def _test_unit():
    compiler = fp.CppBackend(unsafe_allow_int=True)
    for test in tests:
        if test.name in _test_ignore:
            continue

        arg_ctxs = tuple(fp.FP64 for _ in test.args)
        s = compiler.compile(test, ctx=fp.FP64, arg_ctxs=arg_ctxs)
        print('\n'.join(compiler.headers()))
        print(compiler.helpers())
        print(s)

    for example in examples:
        if example.name in _example_ignore:
            continue

        arg_ctxs = tuple(fp.FP64 for _ in example.args)
        s = compiler.compile(example, ctx=fp.FP64, arg_ctxs=arg_ctxs)
        print('\n'.join(compiler.headers()))
        print(compiler.helpers())
        print(s)

def _test_library():
    compiler = fp.CppBackend(unsafe_allow_int=True)
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function) and obj.name not in _library_ignore:
                arg_ctxs = tuple(fp.FP64 for _ in obj.args)
                s = compiler.compile(obj, ctx=fp.FP64, arg_ctxs=arg_ctxs)
                print('\n'.join(compiler.headers()))
                print(compiler.helpers())
                print(s)

def test_cpp():
    _test_unit()
    _test_library()


if __name__ == '__main__':
    test_cpp()
