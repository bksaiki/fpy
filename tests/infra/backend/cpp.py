"""
Compilation tests for C++
"""

import argparse
import fpy2 as fp
import hashlib
import math
import random
import shutil
import signal
import struct
import subprocess
import tempfile

from collections import Counter
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


def _compile(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler, func: fp.Function,
    arg_types: list | None = None,
):
    # Use the caller's explicit argument types when given (e.g. a
    # low-precision element type); otherwise substitute context variables
    # with `FP64`.
    if arg_types is None:
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
_run_ignore: list[str] = [
    # Quantized dot product that collapses to zero: the interpreter yields
    # -0.0 and C++ `std::accumulate` (seeded with +0.0) yields +0.0 —
    # numerically equal, a signed-zero-from-reduction edge.  This function
    # exists to pin widening codegen (which compiles), not exact execution.
    '_regression_quant_dot_real_widen',
]


class _OpScan(DefaultVisitor):
    """Scan one function body: collect non-correctly-rounded ops and the
    resolved targets of any calls (so the caller can recurse the call
    graph)."""

    def __init__(self):
        super().__init__()
        self.bad: set[str] = set()    # non-correctly-rounded op names
        self.callees: list = []       # ``Call.fn`` targets

    def _visit_expr(self, e, ctx):
        name = type(e).__name__
        if name in _NON_CR_OPS:
            self.bad.add(name)
        elif name == 'Call':
            self.callees.append(e.fn)
        return super()._visit_expr(e, ctx)


def _exec_skip_reason(func: fp.Function) -> str | None:
    """Why *func* can't be executed and bit-exactly compared, or ``None``
    if it can.  The reason categories drive the coverage summary.

    A function is eligible iff its whole (transitive) call graph uses only
    correctly-rounded operations, every call resolves to an analyzable FPy
    function (the emitted translation unit includes those callees), and the
    entry's argument types are synthesizable."""
    if func.name in _run_ignore:
        return 'run-ignored'
    try:
        ty_info = fp.analysis.TypeInfer.check(func.ast)
        arg_types = [_inst_type(ty) for ty in ty_info.arg_types]
    except Exception:
        return 'type-error'
    if not all(_generatable(ty) for ty in arg_types):
        return 'arg-type'

    # Walk the call graph: a transcendental anywhere, or a call we can't
    # resolve to an FPy function, makes the whole thing ineligible.
    seen: set[int] = {id(func.ast)}
    work = [func.ast]
    while work:
        scan = _OpScan()
        scan._visit_function(work.pop(), None)
        if scan.bad:
            return 'transcendental'
        for callee in scan.callees:
            if not isinstance(callee, fp.Function):
                return 'calls'  # foreign / unresolved callee — can't verify
            if id(callee.ast) not in seen:
                seen.add(id(callee.ast))
                work.append(callee.ast)
    return None


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
        tok = take().lower()
        # ``printf("%a", ...)`` prints "nan"/"-nan"/"nan(0x..)" and
        # "inf"/"-inf" for non-finite values; finite values are hex floats.
        if 'nan' in tok:
            cpp = math.nan
        elif 'inf' in tok:
            cpp = -math.inf if tok.startswith('-') else math.inf
        else:
            cpp = float.fromhex(tok)
        if not _float_bit_eq(float(value), cpp):
            return f'real {float(value).hex()} != {tok}'
    return None


def _float_bit_eq(a: float, b: float) -> bool:
    """Bit-exact equality, treating all NaNs as equal (their payload/sign
    is not contractually preserved) but distinguishing +0/-0."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return struct.pack('<d', a) == struct.pack('<d', b)


# ---- input synthesis (phase 2) ---------------------------------------
#
# Inputs are exact FP64 doubles, so they round-trip identically to the
# interpreter (which receives the same Python floats) and to C++ (which
# parses the same decimal/`numeric_limits` literals).  Generation is
# deterministic (no flaky CI); samples the interpreter rejects are skipped,
# and since all of a function's samples run in one executable, a generous
# sample count is essentially free.

# Reals spanning sign, zero (both signs), special values, a subnormal, and
# a "random-looking" double.  Magnitudes stay <= 100 so a value used as a
# loop bound can't blow up the interpreter; overflow behavior is covered by
# the explicit infinities.
_REAL_POOL = (
    0.0, -0.0,
    1.0, -1.0, 2.5, -3.0, 0.5, -0.25, 0.1, 100.0,
    1e-10, 5e-324,                          # tiny, smallest subnormal
    float('inf'), float('-inf'), float('nan'),
)
# List lengths cycled per sample (a single length per sample keeps multiple
# list arguments equal-length, as ``zip`` requires).  Includes 0 and 1 to
# exercise empty/singleton handling.
_LIST_LENS = (3, 1, 2, 0, 4)
_N_SAMPLES = len(_REAL_POOL)

# Random-distribution sampling, layered on top of the curated sweep above.
# Seeded with a fixed value per function, so it is fully reproducible (a
# failure always reproduces — no flaky CI).  Magnitudes stay modest so a
# value used as a loop bound can't blow up the interpreter.
_INT_N = 64           # integer-mode reals: uniform in [-_INT_N, _INT_N]
_FLOAT_STD = 8.0      # float-mode reals: normal(0, _FLOAT_STD) ...
_FLOAT_CAP = 128.0    # ... clamped to [-_FLOAT_CAP, _FLOAT_CAP]
_MAX_RANDOM_LEN = 5   # upper bound on a random list length
_RANDOM_SAMPLES = 16  # per arg-taking function (half integer-, half float-mode)


class _OracleTimeout(Exception):
    """The interpreter ran too long on a sampled input."""


def _interp(func: fp.Function, inputs: list, seconds: float = 2.0):
    """Run the interpreter with a wall-clock timeout.

    Special inputs (e.g. ``inf`` as a loop bound) can make a data-dependent
    loop in the interpreter run forever; ``SIGALRM`` interrupts it between
    bytecodes, raising :class:`_OracleTimeout` so the caller skips the
    sample.  Main-thread only (the harness is)."""
    def _handler(signum, frame):
        raise _OracleTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func(*inputs, ctx=fp.FP64)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _generatable(ty) -> bool:
    """Whether :func:`_gen_value` can synthesize an input of type *ty*
    (after context instantiation)."""
    match ty:
        case fp.types.RealType() | fp.types.BoolType():
            return True
        case fp.types.ListType():
            return _generatable(ty.elt)
        case fp.types.TupleType():
            return all(_generatable(elt) for elt in ty.elts)
        case _:
            return False


def _gen_value(ty, vseed: int, lseed: int):
    """Synthesize a Python input value of (instantiated) type *ty*.

    *vseed* selects scalar values (offset by position for sub-elements so
    components differ); *lseed* — constant across all arguments of one
    sample — selects list lengths, so paired lists stay equal-length."""
    match ty:
        case fp.types.RealType():
            return _REAL_POOL[vseed % len(_REAL_POOL)]
        case fp.types.BoolType():
            return bool(vseed % 2)
        case fp.types.ListType():
            n = _LIST_LENS[lseed % len(_LIST_LENS)]
            return [_gen_value(ty.elt, vseed + j, lseed) for j in range(n)]
        case fp.types.TupleType():
            return tuple(
                _gen_value(elt, vseed + j, lseed)
                for j, elt in enumerate(ty.elts)
            )
        case _:
            raise ValueError(f'cannot generate input for type: {ty.format()}')


def _rand_value(ty, rng: random.Random, list_len: int, int_mode: bool):
    """Draw a random input of (instantiated) type *ty*.

    *int_mode* selects the real distribution: uniform integers in
    ``[-_INT_N, _INT_N]`` (exercises loop bounds / exact-integer arithmetic)
    versus a clamped normal (general floating-point).  *list_len* is shared
    across a sample's arguments so paired lists stay equal-length."""
    match ty:
        case fp.types.RealType():
            if int_mode:
                return float(rng.randint(-_INT_N, _INT_N))
            return max(-_FLOAT_CAP, min(_FLOAT_CAP, rng.gauss(0.0, _FLOAT_STD)))
        case fp.types.BoolType():
            return rng.random() < 0.5
        case fp.types.ListType():
            return [_rand_value(ty.elt, rng, list_len, int_mode) for _ in range(list_len)]
        case fp.types.TupleType():
            return tuple(_rand_value(elt, rng, list_len, int_mode) for elt in ty.elts)
        case _:
            raise ValueError(f'cannot generate input for type: {ty.format()}')


def _cpp_type(ty) -> str:
    """C++ storage type for an (instantiated) argument type — must match the
    backend's choice (FP64 real -> ``double``, etc.)."""
    match ty:
        case fp.types.RealType():
            return 'double'
        case fp.types.BoolType():
            return 'bool'
        case fp.types.ListType():
            return f'std::vector<{_cpp_type(ty.elt)}>'
        case fp.types.TupleType():
            return f'std::tuple<{", ".join(_cpp_type(elt) for elt in ty.elts)}>'
        case _:
            raise ValueError(f'no C++ type for: {ty.format()}')


def _cpp_literal(value, ty) -> str:
    """C++ literal for *value* of (instantiated) type *ty*."""
    match ty:
        case fp.types.RealType():
            v = float(value)
            if math.isnan(v):
                return 'std::numeric_limits<double>::quiet_NaN()'
            if math.isinf(v):
                inf = 'std::numeric_limits<double>::infinity()'
                return f'-{inf}' if v < 0 else inf
            # ``repr`` is the shortest round-tripping decimal; C++ parses it
            # (correctly-rounded) to the identical double.  Decimal — not a
            # hex-float literal — keeps the driver valid under C++11, which
            # the harness compiles with (hex floats are C++17).
            return repr(v)
        case fp.types.BoolType():
            return 'true' if value else 'false'
        case fp.types.ListType():
            elts = ', '.join(_cpp_literal(v, ty.elt) for v in value)
            return f'{_cpp_type(ty)}{{{elts}}}'
        case fp.types.TupleType():
            elts = ', '.join(_cpp_literal(v, e) for v, e in zip(value, ty.elts))
            return f'std::make_tuple({elts})'
        case _:
            raise ValueError(f'no C++ literal for: {ty.format()}')


def _emit_driver(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler,
    func: fp.Function, arg_types: list, samples: list,
) -> Path:
    """Write a self-contained translation unit: headers, helpers, the
    compiled function, and a ``main`` that calls it once per sample and
    prints each result (one line per sample) for :func:`_compare`.

    *samples* is a list of ``(inputs, expected)``; each ``inputs`` is the
    list of Python argument values to pass (empty for a nullary function).
    """
    name = hashlib.md5(func.name.encode()).hexdigest()
    cpp_path = output_dir / f'{prefix}_{name}_run.cpp'
    body = compiler.compile(func, ctx=fp.FP64, arg_types=arg_types)
    counter = [0]
    main_lines = ['int main() {']
    for inputs, expected in samples:
        args = ', '.join(_cpp_literal(v, t) for v, t in zip(inputs, arg_types))
        # Each sample in its own scope; one printed line per sample.
        main_lines.append('  {')
        main_lines.append(f'    auto __ret = {func.name}({args});')
        _emit_print('__ret', expected, main_lines, counter)
        main_lines.append(r'    std::printf("\n");')
        main_lines.append('  }')
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
    # Timeout guards against a codegen bug producing an infinite loop; the
    # interpreter ran first (and fast) on the same inputs, so a hang here is
    # a divergence, surfaced as a SubprocessError failure.
    result = subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True, timeout=10,
    )
    return result.stdout


def _run_and_check(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler, func: fp.Function,
) -> str | None:
    """Run *func* through the interpreter and the compiled binary on the
    same synthesized inputs and compare bit-for-bit.  Returns ``None`` on
    agreement, an error string on mismatch, or ``'skip'`` when the
    interpreter rejects every sampled input (nothing to compare against)."""
    ty_info = fp.analysis.TypeInfer.check(func.ast)
    arg_types = [_inst_type(ty) for ty in ty_info.arg_types]

    # Generate deterministic samples; keep only those the oracle accepts so
    # the compiled binary is never run on inputs the interpreter rejects.
    n = _N_SAMPLES if arg_types else 1
    samples: list[tuple[list, object]] = []
    for k in range(n):
        # `lseed=k` is shared across args (paired lists stay equal-length);
        # `vseed=k + i` decorrelates values between argument positions.
        inputs = [_gen_value(ty, k + i, k) for i, ty in enumerate(arg_types)]
        try:
            expected = _interp(func, inputs)
        except Exception:
            # interpreter rejected the input (domain error) or ran too long
            continue
        samples.append((inputs, expected))

    # Random samples (fixed seed -> reproducible): alternate integer-mode
    # (uniform) and float-mode (clamped normal); one list length per sample.
    if arg_types:
        rng = random.Random(0)
        for s in range(_RANDOM_SAMPLES):
            list_len = rng.randint(0, _MAX_RANDOM_LEN)
            inputs = [_rand_value(ty, rng, list_len, s % 2 == 0) for ty in arg_types]
            try:
                expected = _interp(func, inputs)
            except Exception:
                continue
            samples.append((inputs, expected))

    if not samples:
        return 'skip'

    driver = _emit_driver(output_dir, prefix, compiler, func, arg_types, samples)
    out = _build_and_run(driver)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) != len(samples):
        return f'expected {len(samples)} output line(s), got {len(lines)}'
    for (inputs, expected), line in zip(samples, lines):
        err = _compare(expected, line.split(), [0])
        if err:
            return f'inputs={inputs}: {err}'
    return None

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
    cov: 'Counter[str] | None' = None,
) -> list[tuple[str, str, str]]:
    """Take each non-ignored function in *funcs* through the cpp backend up
    to *mode* (``'emit'`` / ``'compile'`` / ``'run'``).  Returns a list of
    ``(group, name, error)`` tuples describing the failures; an empty list
    means everything succeeded.  Failures are also printed inline.

    In ``'run'`` mode, execution-eligible functions (see
    :func:`_exec_skip_reason`) are compiled to an executable and their
    output is checked bit-for-bit against the interpreter; everything else
    is compiled as in ``'compile'`` mode.  When *cov* is given, each
    function's outcome is tallied into it for the coverage summary.

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

        if mode != 'run':
            _compile_obj(cpp_path)
            continue

        # `run` mode: execute eligible functions and bit-compare against the
        # interpreter; object-compile the rest, tallying why each is or
        # isn't executed (see the coverage summary in `test_compile_cpp`).
        reason = 'no-compiler' if _CXX is None else _exec_skip_reason(func)
        if reason is not None:
            if cov is not None:
                cov[reason] += 1
            _compile_obj(cpp_path)
            continue

        try:
            err = _run_and_check(output_dir, prefix, compiler, func)
        except subprocess.SubprocessError as e:
            # build failure, nonzero exit (e.g. assert/SIGSEGV), or timeout
            print(f'  FAILED `{func.name}`: execution error: {e}')
            failures.append((prefix, func.name, f'execution error: {e}'))
            if cov is not None:
                cov['exec-error'] += 1
            continue
        if err == 'skip':
            # interpreter rejected every sampled input; nothing to compare
            if cov is not None:
                cov['uncovered'] += 1
            _compile_obj(cpp_path)
        elif err is not None:
            print(f'  MISMATCH `{func.name}`: {err}')
            failures.append((prefix, func.name, f'output mismatch: {err}'))
            if cov is not None:
                cov['mismatch'] += 1
        else:
            if cov is not None:
                cov['executed'] += 1
    if failures:
        print(f'\n{len(failures)} failures in `{prefix}`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')
    return failures


def _test_unit(
    output_dir: Path, mode: str = 'compile', cov: 'Counter[str] | None' = None,
) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    failures += _test_unit_tests(output_dir, 'unit_tests', all_unit_tests(), _test_ignore, mode=mode, cov=cov)
    failures += _test_unit_tests(output_dir, 'unit_examples', all_example_tests(), _example_ignore, mode=mode, cov=cov)
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


@fp.fpy
def _regression_call_helper(x: fp.Real) -> fp.Real:
    """Callee for :func:`_regression_calls_user_fn`."""
    with fp.FP64:
        return x * x + 1.0


@fp.fpy
def _regression_calls_user_fn(x: fp.Real, y: fp.Real) -> fp.Real:
    """A caller of another FPy function — exercises multi-function
    execution: eligibility recurses the (exact-op) call graph, and the
    emitted translation unit includes the specialized callee, so the
    binary links, runs, and bit-matches the interpreter."""
    with fp.FP64:
        return _regression_call_helper(x) + _regression_call_helper(y)


_regression_funcs: list[fp.Function] = [
    _regression_quant_dot_real_widen,
    _regression_empty_range,
    _regression_calls_user_fn,
]


@fp.fpy
def _regression_blocked_dot_e4m3(xs):
    """Blocked reduction over a low-precision tensor.  Pins several things
    at once:

    - a strided ``range(0, len(xs), 32)``,
    - a slice ``xs[i:i+32]`` and ``sum`` of it under ``with fp.REAL``,
      whose accumulation widens losslessly to the element's wider storage
      (only well-defined for a low-precision element type — hence the
      explicit ``MX_E4M3`` argument), and
    - the read-only-aggregate optimizations: ``xs`` is a ``const&`` param,
      the slice/sum operands bind by reference rather than copying.
    """
    acc = 0
    for i in range(0, len(xs), 32):
        with fp.REAL:
            slc = xs[i:i+32]
            slc_acc = sum(slc)
        with fp.FP32:
            acc += slc_acc
    return acc


# Regressions needing a specific (non-FP64) argument instantiation — e.g.
# a low-precision element type whose REAL-context reduction widens
# losslessly.  These are compiled (and ``cc``-checked) with the given arg
# types; they are not run-mode executed (custom-typed inputs aren't
# synthesized by the differential harness).
_typed_regression_funcs: list[tuple[fp.Function, list]] = [
    (
        _regression_blocked_dot_e4m3,
        [fp.types.ListType(fp.types.RealType(fp.MX_E4M3))],
    ),
]


def _test_typed_regressions(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """Compile each explicitly-typed regression with its given arg types
    (bypassing the FP64 instantiation), then object-compile unless in
    ``emit`` mode.  No execution — these carry non-synthesizable arg types."""
    compiler = fp.CppCompiler(unsafe_cast_int=True)
    failures: list[tuple[str, str, str]] = []
    for func, arg_types in _typed_regression_funcs:
        try:
            cpp_path = _compile(
                output_dir, 'typed_regressions', compiler, func, arg_types,
            )
        except fp.backend.CppCompileError as e:
            print(f'  FAILED `{func.name}`: {e}')
            failures.append(('typed_regressions', func.name, str(e)))
            continue
        if mode != 'emit':
            _compile_obj(cpp_path)
    if failures:
        print(f'\n{len(failures)} failures in `typed_regressions`:')
        for _, name, msg in failures:
            print(f'  - {name}: {msg}')
    return failures


def _test_regressions(
    output_dir: Path, mode: str = 'compile', cov: 'Counter[str] | None' = None,
) -> list[tuple[str, str, str]]:
    """Take each regression function through the same pipeline as the
    corpus.  Failures are accumulated and returned for the top-level
    harness to surface."""
    return _test_unit_tests(
        output_dir, 'regressions', _regression_funcs, ignore=[],
        mode=mode, cov=cov,
    )

###########################################################
# Main tester

class CppInfraFailure(AssertionError):
    """One or more non-ignored functions failed through the cpp backend
    (compile error, or — in ``run`` mode — an output mismatch against the
    interpreter).  Raised at the end of :func:`test_compile_cpp` so CI
    surfaces regressions reliably (failures aren't swallowed by
    per-function ``try/except`` blocks)."""


_COV_LABELS = {
    'uncovered': 'uncovered (interpreter rejected all sampled inputs)',
    'transcendental': 'ineligible: uses non-correctly-rounded op(s)',
    'calls': 'ineligible: calls an unanalyzable/foreign function',
    'arg-type': 'ineligible: unsupported argument type',
    'run-ignored': 'ineligible: in _run_ignore (known divergence)',
    'type-error': 'ineligible: type inference failed',
    'no-compiler': 'skipped: no C++ compiler driver found',
    'mismatch': 'FAIL: output mismatch',
    'exec-error': 'FAIL: build/run error',
}


def _print_coverage(cov: 'Counter[str]') -> None:
    """Print the ``run``-mode execution-coverage breakdown: how many
    functions were bit-compared vs. only compiled, and why."""
    executed = cov.get('executed', 0)
    total = executed + sum(cov.get(k, 0) for k in _COV_LABELS)
    print(f'\n=== cpp exec coverage: {executed}/{total} bit-compared '
          f'against the interpreter ===')
    for key, label in _COV_LABELS.items():
        n = cov.get(key, 0)
        if n:
            print(f'  {n:4d}  {label}')


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
    cov: Counter[str] = Counter()
    failures += _test_unit(output_dir, mode=mode, cov=cov)
    failures += _test_libraries(output_dir, mode=mode)
    failures += _test_regressions(output_dir, mode=mode, cov=cov)
    failures += _test_typed_regressions(output_dir, mode=mode)

    if delete:
        shutil.rmtree(output_dir)

    if mode == 'run':
        _print_coverage(cov)

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
