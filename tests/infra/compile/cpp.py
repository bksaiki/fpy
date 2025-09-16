"""
Compilation tests for C++
"""

import argparse
import fpy2 as fp
import hashlib
import shutil
import subprocess
import tempfile

from pathlib import Path
from types import ModuleType

from ..unit.defs import tests, examples

###########################################################
# Compilation

_CPP_CMD = ['cc']
_CPP_OPTIONS = ['-std=c++11', '-O0', '-Wall', '-Wextra']

def _compile(output_dir: Path, prefix: str, compiler: fp.CppBackend, func: fp.Function):
    name = hashlib.md5(func.name.encode()).hexdigest()
    cpp_path = output_dir / f'{prefix}_{name}.cpp'
    print(f"Compiling `{func.name}` to `{cpp_path}`")
    with open(cpp_path, 'w') as f:
        # emit headers and helpers
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)

        # compile function
        arg_ctxs = tuple(fp.FP64 for _ in func.args)
        s = compiler.compile(func, ctx=fp.FP64, arg_ctxs=arg_ctxs)
        print(s, file=f)
        print(file=f)

    return cpp_path


def _compile_obj(cpp_path: Path):
    obj_file = cpp_path.with_suffix('.o')
    cmd = _CPP_CMD + _CPP_OPTIONS + ['-c', '-o', str(obj_file), str(cpp_path)]
    cmd_str = ' '.join(cmd)
    print(f"Compiling `{cpp_path}` with command: `{cmd_str}`")
    subprocess.run(cmd, check=True)

###########################################################
# Unit tests

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

def _test_unit_tests(
    output_dir: Path,
    prefix: str,
    funcs: list[fp.Function],
    ignore: list[str],
    *,
    no_cc: bool = False
):
    compiler = fp.CppBackend(unsafe_allow_int=True)
    for func in funcs:
        if func.name in ignore:
            continue

        # compile function to C++ file
        cpp_path = _compile(output_dir, prefix, compiler, func)

        # try to compile with C++ compiler
        if not no_cc:
            _compile_obj(cpp_path)

def _test_unit(output_dir: Path, no_cc: bool = False):
    _test_unit_tests(output_dir, 'unit_tests', tests, _test_ignore, no_cc=no_cc)
    _test_unit_tests(output_dir, 'unit_examples', examples, _example_ignore, no_cc=no_cc)

###########################################################
# Libraries

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    # fp.libraries.matrix
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

def _test_library(output_dir: Path, prefix: str, mod: ModuleType, ignore: list[str], no_cc: bool = False):
    compiler = fp.CppBackend(unsafe_allow_int=True)
    cpp_path = output_dir / f'library_{prefix}.cpp'
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        for func in mod.__dict__.values():
            if isinstance(func, fp.Function) and func.name not in ignore:
                arg_ctxs = tuple(fp.FP64 for _ in func.args)
                s = compiler.compile(func, ctx=fp.FP64, arg_ctxs=arg_ctxs)
                print(s, file=f)
                print(file=f)

    if not no_cc:
        _compile_obj(cpp_path)


def _test_libraries(output_dir: Path, no_cc: bool = False):
    for mod in _modules:
        name = mod.__name__.split('.')[-1]
        _test_library(output_dir, name, mod, _library_ignore, no_cc=no_cc)

###########################################################
# Main tester

def test_cpp(delete: bool = True, no_cc: bool = False):
    dir_str = tempfile.TemporaryDirectory(prefix='tmp_fpy_cpp')
    output_dir = Path(dir_str.name)
    # with tempfile.TemporaryDirectory(prefix='tmp_fpy_cpp', delete=delete) as dir_str:

    print(f"Running C++ tests with output under `{output_dir}`")
    _test_unit(output_dir, no_cc=no_cc)
    _test_libraries(output_dir, no_cc=no_cc)
    if delete:
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run C++ compilation tests for fpy2")
    parser.add_argument('--keep', action='store_true', help="Keep temporary files (do not delete)")
    parser.add_argument('--no-cc', action='store_true', help="Do not run the C++ compiler, only emit C++ files")
    args = parser.parse_args()

    # arguments
    delete: bool = not args.keep
    no_cc: bool = args.no_cc

    # run test
    test_cpp(delete=delete, no_cc=no_cc)
