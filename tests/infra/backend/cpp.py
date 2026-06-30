"""
Compilation tests for C++
"""

import argparse
import fpy2 as fp
import hashlib
import math
import shutil
import struct
import subprocess
import tempfile

from pathlib import Path
from types import ModuleType

from fpy2.ast.visitor import DefaultVisitor

from ..examples import all_unit_tests, all_example_tests

# Test mode: how far each function is taken.
#   'emit'    — emit the C++ source only (no compiler invoked)
#   'compile' — emit + compile the C++ (``cc -c``); the default
#   'run'     — emit + compile + execute eligible functions and check the
#               output bit-matches the FPy interpreter
_MODES = ('emit', 'compile', 'run')

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
# Differential execution (`run` mode)
#
# Beyond compiling, `run` mode executes a compiled function on concrete
# inputs and asserts the result bit-matches the FPy interpreter (the
# oracle).  Only functions whose operations are all IEEE-correctly-rounded
# are eligible: C++ `<cmath>` transcendentals are only faithfully rounded,
# so they would spuriously fail a bit-exact check.  Ineligible functions
# are still compiled (the `compile` path), just not executed.

# A C++ compiler driver that links the standard library (the object-only
# `cc -c` path used elsewhere does not link, so it cannot build an exe).
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

# Operators NOT correctly-rounded in C++ `<cmath>`; using any of these
# disqualifies a function from bit-exact execution comparison.
_NON_CR_OPS = frozenset([
    'Acos', 'Asin', 'Atan', 'Atan2', 'Cos', 'Sin', 'Tan',
    'Acosh', 'Asinh', 'Atanh', 'Cosh', 'Sinh', 'Tanh',
    'Exp', 'Exp2', 'Expm1', 'Log', 'Log10', 'Log1p', 'Log2',
    'Erf', 'Erfc', 'Lgamma', 'Tgamma', 'Pow', 'Cbrt', 'Hypot',
])

# Functions that compile and use only correctly-rounded ops but still
# diverge from the interpreter for a known reason (populate as discovered,
# e.g. exact-rational decimal literals rounded once in the interpreter vs.
# per-literal in C++).
_run_ignore: list[str] = []


class _OpScan(DefaultVisitor):
    """Collect operator/call nodes that disqualify bit-exact execution."""

    def __init__(self):
        super().__init__()
        self.bad: set[str] = set()

    def _visit_expr(self, e, ctx):
        name = type(e).__name__
        if name in _NON_CR_OPS or name == 'Call':
            self.bad.add(name)
        return super()._visit_expr(e, ctx)


def _exec_eligible(func: fp.Function) -> bool:
    """True iff *func* can be executed and bit-exactly compared: it takes
    no arguments (phase 1), uses only correctly-rounded operations, and
    calls no user-defined function."""
    if len(func.ast.args) != 0:
        return False
    if func.name in _run_ignore:
        return False
    scan = _OpScan()
    scan._visit_function(func.ast, None)
    return not scan.bad


def _emit_print(expr: str, value, lines: list[str], counter: list[int]) -> None:
    """Emit C++ statements that print *expr* — whose runtime value mirrors
    the interpreter *value* — as a whitespace-separated token stream that
    :func:`_compare` reads back structurally."""
    if isinstance(value, bool):
        lines.append(f'  std::printf("%d ", (int)({expr}));')
    elif isinstance(value, list):
        lines.append(f'  std::printf("%zu ", (size_t)(({expr}).size()));')
        if value:  # homogeneous: one representative element shape
            i = counter[0]
            counter[0] += 1
            elt = f'__e{i}'
            # ``auto`` (by value), not ``auto&``: ``std::vector<bool>`` yields
            # proxy rvalues that a non-const reference cannot bind to.
            lines.append(f'  for (auto {elt} : ({expr})) {{')
            _emit_print(elt, value[0], lines, counter)
            lines.append('  }')
    elif isinstance(value, tuple):
        for i, elt in enumerate(value):
            _emit_print(f'std::get<{i}>({expr})', elt, lines, counter)
    else:  # real scalar
        lines.append(f'  std::printf("%a ", (double)({expr}));')


def _compare(value, toks: list[str], pos: list[int]) -> str | None:
    """Consume tokens from *toks* (cursor *pos*) per *value*'s structure,
    comparing bit-exactly.  Returns an error description on mismatch, else
    ``None``."""
    def take() -> str:
        t = toks[pos[0]]
        pos[0] += 1
        return t

    if isinstance(value, bool):
        t = take()
        if int(t) != int(value):
            return f'bool {int(value)} != {t}'
    elif isinstance(value, list):
        n = int(take())
        if n != len(value):
            return f'list length {len(value)} != {n}'
        for elt in value:
            err = _compare(elt, toks, pos)
            if err:
                return err
    elif isinstance(value, tuple):
        for elt in value:
            err = _compare(elt, toks, pos)
            if err:
                return err
    else:  # real scalar
        cpp = float.fromhex(take())
        if not _float_bit_eq(float(value), cpp):
            return f'real {float(value).hex()} != {cpp.hex()}'
    return None


def _float_bit_eq(a: float, b: float) -> bool:
    """Bit-exact equality, treating all NaNs as equal (their payload/sign
    is not contractually preserved) but distinguishing +0/-0."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return struct.pack('<d', a) == struct.pack('<d', b)


def _emit_driver(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler,
    func: fp.Function, arg_types: list, value,
) -> Path:
    """Write a self-contained translation unit: headers, helpers, the
    compiled function, and a ``main`` that calls it (no arguments — phase 1)
    and prints the result for :func:`_compare`."""
    name = hashlib.md5(func.name.encode()).hexdigest()
    cpp_path = output_dir / f'{prefix}_{name}_run.cpp'
    body = compiler.compile(func, ctx=fp.FP64, arg_types=arg_types)
    main_lines = ['int main() {', f'  auto __ret = {func.name}();']
    _emit_print('__ret', value, main_lines, [0])
    main_lines.append(r'  std::printf("\n");')
    main_lines.append('  return 0;')
    main_lines.append('}')
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print('#include <cstdio>', file=f)
        print(compiler.helpers(), file=f)
        print(body, file=f)
        print('\n'.join(main_lines), file=f)
    return cpp_path


def _build_and_run(cpp_path: Path) -> str:
    """Compile *cpp_path* to an executable (linking libstdc++) and run it,
    returning its stdout."""
    exe = cpp_path.with_suffix('.exe')
    assert _CXX is not None
    cmd = [_CXX, *_CPP_OPTIONS, '-o', str(exe), str(cpp_path)]
    print(f"Building `{cpp_path}` with command: `{' '.join(cmd)}`")
    subprocess.run(cmd, check=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return result.stdout


def _run_and_check(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler, func: fp.Function,
) -> str | None:
    """Run *func* through the interpreter and the compiled binary on the
    same (empty) input and compare.  Returns ``None`` on agreement, an
    error string on mismatch, or ``'skip'`` when the interpreter rejects
    the input (so there is nothing to compare against)."""
    ty_info = fp.analysis.TypeInfer.check(func.ast)
    arg_types = [_inst_type(ty) for ty in ty_info.arg_types]
    try:
        expected = func(ctx=fp.FP64)
    except Exception:
        # the oracle rejects this input — cannot compare (no C++ run)
        return 'skip'
    driver = _emit_driver(output_dir, prefix, compiler, func, arg_types, expected)
    out = _build_and_run(driver)
    return _compare(expected, out.split(), [0])

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
    mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """Take each non-ignored function in *funcs* through the cpp backend up
    to *mode* (``'emit'`` / ``'compile'`` / ``'run'``).  Returns a list of
    ``(group, name, error)`` tuples describing the failures; an empty list
    means everything succeeded.  Failures are also printed inline.

    In ``'run'`` mode, execution-eligible functions (see
    :func:`_exec_eligible`) are compiled to an executable and their output
    is checked bit-for-bit against the interpreter; everything else is
    compiled as in ``'compile'`` mode.

    Continues past failures so a single run reports every regression, but
    does *not* mask them — the caller aggregates the returned list and
    exits non-zero when it's non-empty.
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

        if mode == 'emit':
            continue

        # In `run` mode, execute eligible functions and bit-compare against
        # the interpreter; otherwise just object-compile.
        if mode == 'run' and _CXX is not None and _exec_eligible(func):
            try:
                err = _run_and_check(output_dir, prefix, compiler, func)
            except subprocess.CalledProcessError as e:
                print(f'  FAILED `{func.name}`: execution error: {e}')
                failures.append((prefix, func.name, f'execution error: {e}'))
                continue
            if err == 'skip':
                # interpreter rejects the input; fall back to object-compile
                _compile_obj(cpp_path)
            elif err is not None:
                print(f'  MISMATCH `{func.name}`: {err}')
                failures.append((prefix, func.name, f'output mismatch: {err}'))
        else:
            _compile_obj(cpp_path)
    if failures:
        print(f'\n{len(failures)} failures in `{prefix}`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')
    return failures


def _test_unit(output_dir: Path, mode: str = 'compile') -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    failures += _test_unit_tests(output_dir, 'unit_tests', all_unit_tests(), _test_ignore, mode=mode)
    failures += _test_unit_tests(output_dir, 'unit_examples', all_example_tests(), _example_ignore, mode=mode)
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
    ignore: list[str], mode: str = 'compile',
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
    # One module per library — sharing it with `compile_module` gives
    # cross-function specialization dedup so a callee referenced from
    # multiple library functions is emitted exactly once.  Validate
    # each candidate by compiling it in isolation first; survivors go
    # into the combined module that we emit.
    accepted: list[tuple[fp.Function, list]] = []
    for func in mod.__dict__.values():
        if isinstance(func, fp.Function) and func.name not in ignore:
            ty_info = fp.analysis.TypeInfer.check(func.ast)
            arg_types = [_inst_type(ty) for ty in ty_info.arg_types]
            probe = fp.Module()
            probe.add(func, ctx=fp.FP64, arg_types=arg_types)
            try:
                compiler.compile_module(probe)
            except fp.backend.CppCompileError as e:
                print(f'  FAILED `{func.name}`: {e}')
                failures.append((group, func.name, str(e)))
                continue
            accepted.append((func, arg_types))

    combined = fp.Module()
    for func, arg_types in accepted:
        combined.add(func, ctx=fp.FP64, arg_types=arg_types)

    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        print(compiler.compile_module(combined), file=f)
        print(file=f)

    if failures:
        print(f'\n{len(failures)} failures in `{group}`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')

    # Libraries compile as one translation unit; execution comparison
    # (phase 1, nullary) does not apply, so 'run' behaves like 'compile'.
    if mode != 'emit':
        _compile_obj(cpp_path)
    return failures


def _test_libraries(output_dir: Path, mode: str = 'compile') -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for mod in _modules:
        name = mod.__name__.split('.')[-1]
        failures += _test_library(output_dir, name, mod, _library_ignore, mode=mode)
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


@fp.fpy
def _regression_empty_range() -> fp.Real:
    """Empty/flipped/negative ranges compile and behave as empty
    iterables (mirroring Python).  Pins:

    - ``range(-5)`` (statically-known empty Range1): the loop body is
      never executed but is still emitted, so its inner definitions
      need format/storage bounds — a negative iteration count must be
      treated as zero, not skipped.
    - ``range(5, 0)`` (statically-known flipped Range2): emitted as an
      empty ``for`` loop.
    """
    with fp.SINT32:
        acc = 0
        for _ in range(-5):
            acc = acc + 1
        for _ in range(5, 0):
            acc = acc + 1
        return acc


_regression_funcs: list[fp.Function] = [
    _regression_quant_dot_real_widen,
    _regression_empty_range,
]


def _test_regressions(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """Take each regression function through the same pipeline as the
    corpus.  Failures are accumulated and returned for the top-level
    harness to surface."""
    return _test_unit_tests(
        output_dir, 'regressions', _regression_funcs, ignore=[],
        mode=mode,
    )

###########################################################
# Main tester

class CppInfraFailure(AssertionError):
    """One or more non-ignored functions failed through the cpp backend
    (compile error, or — in ``run`` mode — an output mismatch against the
    interpreter).  Raised at the end of :func:`test_compile_cpp` so CI
    surfaces regressions reliably (failures aren't swallowed by
    per-function ``try/except`` blocks)."""


def test_compile_cpp(delete: bool = True, mode: str = 'compile'):
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
    if mode == 'run' and _CXX is None:
        print('WARNING: no C++ compiler driver (c++/g++/clang++) found; '
              "falling back to 'compile' mode (no execution).")

    dir_str = tempfile.mkdtemp(prefix='tmp_fpy_cpp')
    output_dir = Path(dir_str)

    print(f"Running C++ tests (mode={mode}) with output under `{output_dir}`")
    failures: list[tuple[str, str, str]] = []
    failures += _test_unit(output_dir, mode=mode)
    failures += _test_libraries(output_dir, mode=mode)
    failures += _test_regressions(output_dir, mode=mode)

    if delete:
        shutil.rmtree(output_dir)

    if failures:
        # Print a single consolidated summary so the failing names
        # are easy to find in CI logs.
        print(f'\n=== cpp infra: {len(failures)} failure(s) ===')
        for group, name, msg in failures:
            print(f'  [{group}] {name}: {msg}')
        raise CppInfraFailure(
            f'{len(failures)} cpp-backend failure(s); '
            f'see output above for details'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run C++ compilation tests for fpy2")
    parser.add_argument('--keep', action='store_true', help="Keep temporary files (do not delete)")
    parser.add_argument(
        '--mode', choices=_MODES, default='compile',
        help="emit: write C++ only; compile (default): also object-compile; "
             "run: also execute eligible functions and bit-compare vs. the interpreter",
    )
    parser.add_argument('--no-cc', action='store_true', help="Alias for --mode emit (emit C++ only)")
    args = parser.parse_args()

    # arguments
    delete: bool = not args.keep
    mode: str = 'emit' if args.no_cc else args.mode

    # run test
    test_compile_cpp(delete=delete, mode=mode)
