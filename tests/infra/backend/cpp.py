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

from ..examples import all_unit_tests, all_example_tests

###########################################################
# Compilation

_CPP_CMD = ['cc']
_CPP_OPTIONS = ['-std=c++11', '-O0', '-Wall', '-Wextra']

def _inst_type(ty: fp.types.Type):
    match ty:
        case fp.types.BoolType() | fp.types.ContextType():
            return ty
        case fp.types.VarType() | fp.types.RealType():
            return fp.types.RealType(fp.FP64)
        case fp.types.TupleType():
            return fp.types.TupleType(*[ _inst_type(elt) for elt in ty.elts ])
        case fp.types.ListType():
            return fp.types.ListType(_inst_type(ty.elt))
        case _:
            raise ValueError(f'Cannot instantiate type: {ty.format()}')


def _compile(output_dir: Path, prefix: str, compiler: fp.CppBackend, func: fp.Function):
    # substitute context variables with `FP64`
    ty_info = fp.analysis.TypeInfer.check(func.ast)
    arg_types = [ _inst_type(ty) for ty in ty_info.arg_types ]

    name = hashlib.md5(func.name.encode()).hexdigest()
    cpp_path = output_dir / f'{prefix}_{name}.cpp'
    print(f"Compiling `{func.name}` to `{cpp_path}`")
    with open(cpp_path, 'w') as f:
        # emit headers and helpers
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)

        # compile function
        s = compiler.compile(func, ctx=fp.FP64, arg_types=arg_types)
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
    # unrounded literals
    # 'test_integer1',
    # 'test_integer2',
    # 'test_decnum1',
    'test_decnum2',
    'test_hexnum1',
    'test_hexnum2', # TODO: implement
    'test_rational1',
    'test_rational2', # TODO: implement
    # 'test_digits1',
    # 'test_digits2',
    # 'test_digits3',
    'test_digits4',
    'test_digits5', # TODO: implement
    'test_let3',
    'test_augassign1',
    'test_augassign2',
    'test_augassign3',
    'test_augassign4',
    # 'test_ife2',
    # 'test_ife3',
    'test_ife4',
    'test_ife5',
    # 'test_tuple2',
    # 'test_tuple3',
    'test_tuple4',
    'test_tuple5',
    'test_tuple6',
    'test_list_comp1',
    'test_list_comp2',
    'test_list_comp3',
    'test_if3',
    'test_if7',
    'test_while4',
    'test_while5',
    'test_while6',
    'test_while7',
    'test_for2',
    'test_for3',
    'test_for4',
    'test_for5',
    # empty list is not monomorphic
    'test_list1',
    'test_list_len1',
    'test_list_dim1',
    'test_list_size1',
    'test_enumerate1', 
    # context expressions
    'test_context_expr1',
    'test_context_expr2',
    # unsupported contexts
    'test_context2',
    'test_context3',
    'test_context4',
    'test_context5',
    'test_context7',
    'test_context8',
    # assertion messages
    'test_assert2',
    'test_assert3',
    # not monomorphic
    'test_meta_inner'
]

_example_ignore = [
    'fma_ctx',
    'dpN',
    'example_static_context1',
    'example_static_context2',
    'example_fold_op1',
    'example_fold_op2',
    'example_fold_op3',
    'example_fold_op4',
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
    compiler = fp.CppBackend(unsafe_finitize_int=True, unsafe_cast_int=True)
    for func in funcs:
        if func.name in ignore:
            continue

        # compile function to C++ file
        cpp_path = _compile(output_dir, prefix, compiler, func)

        # try to compile with C++ compiler
        if not no_cc:
            _compile_obj(cpp_path)

def _test_unit(output_dir: Path, no_cc: bool = False):
    _test_unit_tests(output_dir, 'unit_tests', all_unit_tests(), _test_ignore, no_cc=no_cc)
    _test_unit_tests(output_dir, 'unit_examples', all_example_tests(), _example_ignore, no_cc=no_cc)

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
    compiler = fp.CppBackend(unsafe_finitize_int=True, unsafe_cast_int=True)
    cpp_path = output_dir / f'library_{prefix}.cpp'
    print(f"Compiling library `{mod.__name__}` to `{cpp_path}`")
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        for func in mod.__dict__.values():
            if isinstance(func, fp.Function) and func.name not in ignore:
                # substitute context variables with `FP64`
                ty_info = fp.analysis.TypeInfer.check(func.ast)
                arg_types = [ _inst_type(ty) for ty in ty_info.arg_types ]

                # compile
                s = compiler.compile(func, ctx=fp.FP64, arg_types=arg_types)
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

def test_compile_cpp(delete: bool = True, no_cc: bool = False):
    dir_str = tempfile.mkdtemp(prefix='tmp_fpy_cpp')
    output_dir = Path(dir_str)
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
    test_compile_cpp(delete=delete, no_cc=no_cc)
