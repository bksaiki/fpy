"""
Compilation tests for C++
"""

import tempfile
import fpy2 as fp

from pathlib import Path

from ..unit.defs import tests, examples

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    # fp.libraries.matrix
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
    # core
    '_modf_spec',
    'isinteger',
    '_ldexp_spec',
    # eft
    'ideal_2sum',
    'ideal_2mul',
    'fast_2sum', # isnar
    'classic_2mul', # max_p
    'ideal_fma',
    'classic_2fma', # relies on `fast_2sum`
]



def _test_unit_tests(output_dir: Path, name: str, funcs: list[fp.Function], ignore: list[str]):
    compiler = fp.CppBackend(unsafe_allow_int=True)
    with open(output_dir / f'{name}.cpp', 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        for func in funcs:
            if func.name in ignore:
                continue

            arg_ctxs = tuple(fp.FP64 for _ in func.args)
            s = compiler.compile(func, ctx=fp.FP64, arg_ctxs=arg_ctxs)
            print(s, file=f)
            print(file=f)


def _test_unit(output_dir: Path):
    _test_unit_tests(output_dir, 'unit_tests', tests, _test_ignore)
    _test_unit_tests(output_dir, 'unit_examples', examples, _example_ignore)

def _test_library(output_dir: Path, name: str, mod, ignore: list[str]):
    compiler = fp.CppBackend(unsafe_allow_int=True)
    with open(output_dir / f'library_{name}.cpp', 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function) and obj.name not in ignore:
                arg_ctxs = tuple(fp.FP64 for _ in obj.args)
                s = compiler.compile(obj, ctx=fp.FP64, arg_ctxs=arg_ctxs)
                print(s, file=f)
                print(file=f)

def _test_libraries(output_dir: Path):
    for mod in _modules:
        _test_library(output_dir, mod.__name__.split('.')[-1], mod, _library_ignore)

def test_cpp(delete: bool = True):
    with tempfile.TemporaryDirectory(delete=delete) as dir_str:
        output_dir = Path(dir_str)
        print(f"Running C++ tests with output under `{output_dir}`")
        _test_unit(output_dir)
        _test_libraries(output_dir)


if __name__ == '__main__':
    test_cpp(delete=False)
