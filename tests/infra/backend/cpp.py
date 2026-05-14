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


def _compile(output_dir: Path, prefix: str, compiler: fp.CppCompiler, func: fp.Function):
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
    'test_neg2',
    'test_abs1',
    'test_add1',
    'test_sub1',
    'test_mul1',
    'test_div1',
    'test_mod1',
    'test_augassign1',
    'test_augassign2',
    'test_augassign3',
    'test_augassign4',
    'test_augassign5',
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
    'test_meta_inner',
    # unsupported operations
    'test_declcontext',
    'test_empty1',
    'test_empty2',
    'test_empty3',
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
) -> list[tuple[str, str, str]]:
    """Compile each non-ignored function in *funcs* through the cpp
    backend (and, when ``no_cc`` is False, through ``cc``).  Returns
    a list of ``(group, name, error)`` tuples describing the
    failures; an empty list means everything compiled.  Failures
    are also printed inline for human-friendly output.

    Continues past failures so a single run reports every
    regression, but does *not* mask them — the caller is expected
    to aggregate the returned list and raise / exit non-zero when
    it's non-empty.
    """
    compiler = fp.CppCompiler(unsafe_cast_int=True)
    failures: list[tuple[str, str, str]] = []
    for func in funcs:
        if func.name in ignore:
            continue

        try:
            cpp_path = _compile(output_dir, prefix, compiler, func)
        except fp.backend.CppCompileError as e:
            print(f'  FAILED `{func.name}`: {e}')
            failures.append((prefix, func.name, str(e)))
            continue

        # try to compile with C++ compiler
        if not no_cc:
            _compile_obj(cpp_path)
    if failures:
        print(f'\n{len(failures)} failures in `{prefix}`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')
    return failures


def _test_unit(output_dir: Path, no_cc: bool = False) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    failures += _test_unit_tests(output_dir, 'unit_tests', all_unit_tests(), _test_ignore, no_cc=no_cc)
    failures += _test_unit_tests(output_dir, 'unit_examples', all_example_tests(), _example_ignore, no_cc=no_cc)
    return failures

###########################################################
# Libraries

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

_library_ignore = [
    # core
    'logb', # deprecated
    '_modf_spec',
    'isinteger',
    '_ldexp_spec',
    'tree_sum',
    # eft
    'ideal_2sum',
    'ideal_2mul',
    'fast_2sum', # isnar
    'classic_2mul', # max_p
    'ideal_fma',
    'classic_2fma', # relies on `fast_2sum`
    # matrix
]

def _test_library(
    output_dir: Path, prefix: str, mod: ModuleType,
    ignore: list[str], no_cc: bool = False,
) -> list[tuple[str, str, str]]:
    """Compile a library module's non-ignored functions into one
    translation unit.  Returns ``(group, name, error)`` tuples for
    every function that failed to register with the unit.  Empty
    list = clean run.
    """
    compiler = fp.CppCompiler(unsafe_cast_int=True)
    cpp_path = output_dir / f'library_{prefix}.cpp'
    print(f"Compiling library `{mod.__name__}` to `{cpp_path}`")
    group = f'library_{prefix}'
    failures: list[tuple[str, str, str]] = []
    # One translation unit per library — shares the specialization
    # cache so a callee referenced from multiple library functions
    # is emitted exactly once (no ODR redefinitions).
    unit = compiler.unit()
    for func in mod.__dict__.values():
        if isinstance(func, fp.Function) and func.name not in ignore:
            ty_info = fp.analysis.TypeInfer.check(func.ast)
            arg_types = [_inst_type(ty) for ty in ty_info.arg_types]
            try:
                unit.add(func, ctx=fp.FP64, arg_types=arg_types)
            except fp.backend.CppCompileError as e:
                print(f'  FAILED `{func.name}`: {e}')
                failures.append((group, func.name, str(e)))

    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        print(unit.render(), file=f)
        print(file=f)

    if failures:
        print(f'\n{len(failures)} failures in `{group}`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')

    if not no_cc:
        _compile_obj(cpp_path)
    return failures


def _test_libraries(output_dir: Path, no_cc: bool = False) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for mod in _modules:
        name = mod.__name__.split('.')[-1]
        failures += _test_library(output_dir, name, mod, _library_ignore, no_cc=no_cc)
    return failures

###########################################################
# Targeted regression tests
#
# Hand-picked FPy programs that exercise a specific cpp-backend
# feature or guard against a known regression.  Add a new entry
# here when fixing a bug whose minimal reproducer doesn't naturally
# belong in the upstream corpus.  Each entry compiles via the same
# pipeline + ``cc`` invocation the unit-test corpus uses; a
# compilation failure surfaces as a test failure.


@fp.fpy
def _regression_quant_dot_real_widen(
    xs: list[fp.Real], ys: list[fp.Real],
) -> fp.Real:
    """Quantize two FP64 lists into SINT8 elements, compute their
    elementwise products under ``with fp.REAL:`` (the cpp backend
    proves the exact ``int8 * int8`` product fits in ``int16_t``
    losslessly), then accumulate into FP32.  Pins:

    - ``[fp.round(...) for ... in xs]`` under SINT8 lowers to a
      ``static_cast<int8_t>`` push-back loop.
    - ``[xq * yq for ... in zip(...)]`` under REAL invokes the
      lossless-widening dispatch in
      :meth:`fpy2.backend.cpp.emitter.CppEmitter._try_widen_binary`:
      the ``Mul`` widens both operands to ``int16_t``.
    - ``sum(prods)`` under FP32 lowers to ``std::accumulate`` with
      a ``float`` accumulator, taking advantage of the lossless
      ``int16 → float`` implicit conversion.
    """
    with fp.SINT8:
        xqs = [fp.round(x) for x in xs]
        yqs = [fp.round(y) for y in ys]
    with fp.REAL:
        prods = [xq * yq for xq, yq in zip(xqs, yqs)]
    with fp.FP32:
        return sum(prods)


_regression_funcs: list[fp.Function] = [
    _regression_quant_dot_real_widen,
]


def _test_regressions(
    output_dir: Path, no_cc: bool = False,
) -> list[tuple[str, str, str]]:
    """Compile each regression function as a one-function
    translation unit, identical to the corpus path.  Failures are
    accumulated and returned for the top-level harness to surface."""
    return _test_unit_tests(
        output_dir, 'regressions', _regression_funcs, ignore=[],
        no_cc=no_cc,
    )

###########################################################
# Main tester

class CppInfraFailure(AssertionError):
    """One or more non-ignored functions failed to compile through
    the cpp backend.  Raised at the end of :func:`test_compile_cpp`
    so CI surfaces regressions reliably (failures aren't swallowed
    by per-function ``try/except`` blocks)."""


def test_compile_cpp(delete: bool = True, no_cc: bool = False):
    dir_str = tempfile.mkdtemp(prefix='tmp_fpy_cpp')
    output_dir = Path(dir_str)

    print(f"Running C++ tests with output under `{output_dir}`")
    failures: list[tuple[str, str, str]] = []
    failures += _test_unit(output_dir, no_cc=no_cc)
    failures += _test_libraries(output_dir, no_cc=no_cc)
    failures += _test_regressions(output_dir, no_cc=no_cc)

    if delete:
        shutil.rmtree(output_dir)

    if failures:
        # Print a single consolidated summary so the failing names
        # are easy to find in CI logs.
        print(f'\n=== cpp infra: {len(failures)} compile failure(s) ===')
        for group, name, msg in failures:
            print(f'  [{group}] {name}: {msg}')
        raise CppInfraFailure(
            f'{len(failures)} cpp-backend compile failure(s); '
            f'see output above for details'
        )


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
