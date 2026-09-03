"""
Compilation tests for C++
"""

import argparse
import hashlib
import math
import random
import re
import shutil
import signal
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType

import fpy2 as fp
from fpy2.ast.visitor import DefaultVisitor

from ..examples import all_example_tests, all_unit_tests

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


def corpus():
    """Every function the backend's corpus gates run over.

    One definition: three profile tests sweep the same set, and a difference
    between them would be a silent gap rather than a failure.
    """
    import importlib

    from ..examples import all_example_tests, all_unit_tests

    yield from all_unit_tests()
    yield from all_example_tests()
    for name in ('core', 'eft', 'vector', 'matrix'):
        mod = importlib.import_module(f'fpy2.libraries.{name}')
        for f in mod.__dict__.values():
            if isinstance(f, fp.Function) and f.name not in _library_ignore:
                yield f


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

# The compiler every corpus-shaped driver uses.  The corpus deliberately
# contains shared lists, which the strict default refuses, so these drivers
# compile in ALLOW mode; strict refusals are pinned by the unit tests.
_CORPUS_COMPILER = fp.CppCompiler(
    unsafe_cast_int=True, unbox=fp.CppCompiler.UnboxMode.ALLOW,
)

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


def _emit_print(
    expr: str, value, cty, lines: list[str], counter: list[int],
) -> None:
    """Emit C++ statements that print *expr* — whose runtime value mirrors
    the interpreter *value* — as a whitespace-separated token stream that
    :func:`_compare` reads back structurally.

    *cty* is the emitted storage type: whether a list is reached through a
    handle.
    """
    if isinstance(value, bool):
        lines.append(f'  std::printf("%d ", (int)({expr}));')
    elif isinstance(value, list):
        seq = f'(*({expr}))' if getattr(cty, 'boxed', False) else f'({expr})'
        lines.append(f'  std::printf("%zu ", (size_t){seq}.size());')
        if value:  # homogeneous: one representative element shape
            i = counter[0]
            counter[0] += 1
            elt = f'__e{i}'
            # ``auto`` (by value), not ``auto&``: ``std::vector<bool>`` yields
            # proxy rvalues that a non-const reference cannot bind to.
            lines.append(f'  for (auto {elt} : {seq}) {{')
            _emit_print(elt, value[0], cty.elt, lines, counter)
            lines.append('  }')
    elif isinstance(value, tuple):
        for i, elt in enumerate(value):
            _emit_print(
                f'std::get<{i}>({expr})', elt, cty.elts[i], lines, counter,
            )
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


def _round_to_format(value, ty):
    """*value*, rounded to whatever format *ty* declares.

    The driver declares each parameter at the backend's storage type, so an
    FP32 parameter rounds whatever literal it is handed — while the interpreter
    would keep the value unrounded.  The two would then disagree for a reason
    that is not a bug.  Rounding here means both sides start from the same
    number.

    Identity for FP64, which is every corpus program: the pool is already
    binary64, and the round preserves ``-0.0`` and NaN.
    """
    match ty:
        case fp.types.RealType():
            return float(ty.ctx.round(value))
        case fp.types.ListType():
            return [_round_to_format(v, ty.elt) for v in value]
        case fp.types.TupleType():
            return tuple(
                _round_to_format(v, t) for v, t in zip(value, ty.elts)
            )
        case _:
            return value


def _cpp_type(ty) -> str:
    """C++ storage type for an (instantiated) argument type — must match the
    backend's choice (FP64 real -> ``double``, etc.)."""
    match ty:
        case fp.types.RealType():
            return 'double'
        case fp.types.BoolType():
            return 'bool'
        case fp.types.ListType():
            # must match `CppList.format()`: a list is a shared handle, not a
            # bare vector
            return f'std::shared_ptr<std::vector<{_cpp_type(ty.elt)}>>'
        case fp.types.TupleType():
            return f'std::tuple<{", ".join(_cpp_type(elt) for elt in ty.elts)}>'
        case _:
            raise ValueError(f'no C++ type for: {ty.format()}')


def _boxed_list_literal(elt: str, elts: str) -> str:
    """The boxed-list literal idiom, matching the emitter's spelling:
    `make_shared` cannot deduce a braced init-list, so the inner vector is
    spelled and moved in."""
    return (
        f'std::make_shared<std::vector<{elt}>>'
        f'(std::vector<{elt}>{{{elts}}})'
    )


def _cpp_value(value, cty) -> str:
    """C++ initializer for *value* at emitted storage type *cty*.

    Built from the *storage* type rather than the FPy type: whether a list is a
    handle or a bare vector is the backend's choice (see
    ``fpy2.backend.cpp.unbox``), and the driver has to match it exactly.
    """
    from fpy2.backend.cpp.types import CppList, CppScalar, CppTuple
    match cty:
        case CppList():
            elts = ', '.join(_cpp_value(v, cty.elt) for v in value)
            if cty.boxed:
                return _boxed_list_literal(cty.elt.format(), elts)
            if cty.size is not None:
                # a fixed-size list: the driver's value must match the
                # length the signature promised
                assert len(value) == cty.size, (
                    f'driver value of length {len(value)} for {cty.format()}'
                )
                if not elts:
                    return f'{cty.format()}{{}}'
                return f'{cty.format()}{{{{{elts}}}}}'
            return f'{cty.format()}{{{elts}}}'
        case CppTuple():
            elts = ', '.join(
                _cpp_value(v, e) for v, e in zip(value, cty.elts)
            )
            return f'std::make_tuple({elts})'
        case CppScalar.BOOL:
            return 'true' if value else 'false'
        case _:
            # Spelled at the *target* scalar, not at ``double``: these go inside
            # braced initializers, which reject a narrowing conversion.  A
            # `double` infinity handed to a `std::vector<float>{...}` is a hard
            # error even though the value is representable.
            t = cty.format()
            v = float(value)
            if not cty.is_float():
                # An integral `numeric_limits` has no NaN or infinity —
                # `quiet_NaN()` returns 0 — so a driver built this way would
                # pass 0 and the mismatch would read as agreement.  Unreachable
                # today (no emitted signature narrows a *parameter* to an
                # integer), so refuse rather than guess.
                raise ValueError(
                    f'cannot build a driver value for integer storage {t}: '
                    f'{value!r} (see _cpp_value)'
                )
            if math.isnan(v):
                return f'std::numeric_limits<{t}>::quiet_NaN()'
            if math.isinf(v):
                inf = f'std::numeric_limits<{t}>::infinity()'
                return f'-{inf}' if v < 0 else inf
            # ``repr`` is the shortest round-tripping decimal; C++ parses it
            # (correctly-rounded) to the identical double.  Decimal — not a
            # hex-float literal — keeps the driver valid under C++11.  A cast
            # for a narrower target, for the same braced-init reason; the value
            # is representable there (see ``_round_to_format``), so it is exact.
            return repr(v) if t == 'double' else f'static_cast<{t}>({v!r})'


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
            return _boxed_list_literal(_cpp_type(ty.elt), elts)
        case fp.types.TupleType():
            elts = ', '.join(_cpp_literal(v, e) for v, e in zip(value, ty.elts))
            return f'std::make_tuple({elts})'
        case _:
            raise ValueError(f'no C++ literal for: {ty.format()}')


def _emit_driver(
    output_dir: Path, prefix: str, compiler: fp.CppCompiler,
    func: fp.Function, arg_types: list, samples: list,
    ctx=fp.FP64, suffix: str = '',
) -> Path:
    """Write a self-contained translation unit: headers, helpers, the
    compiled function, and a ``main`` that calls it once per sample and
    prints each result (one line per sample) for :func:`_compare`.

    *samples* is a list of ``(inputs, expected)``; each ``inputs`` is the
    list of Python argument values to pass (empty for a nullary function).

    *suffix* distinguishes several instantiations of one function, which
    otherwise hash to the same file name.
    """
    name = hashlib.md5((func.name + suffix).encode()).hexdigest()
    cpp_path = output_dir / f'{prefix}_{name}_run.cpp'
    body = compiler.compile(func, ctx=ctx, arg_types=arg_types)
    params, ret_ty = compiler.signature(
        func, ctx=ctx, arg_types=arg_types,
    )
    counter = [0]
    main_lines = ['int main() {']
    for inputs, expected in samples:
        # Each sample in its own scope; one printed line per sample.
        main_lines.append('  {')
        # Bind every argument to a named local: a non-const reference
        # parameter cannot bind to a prvalue.
        names = []
        for i, (v, cty) in enumerate(zip(inputs, params)):
            names.append(f'__a{i}')
            main_lines.append(
                f'    {cty.format()} __a{i} = {_cpp_value(v, cty)};'
            )
        main_lines.append(
            f'    auto __ret = {func.name}({", ".join(names)});'
        )
        _emit_print('__ret', expected, ret_ty, main_lines, counter)
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
    *, ctx=fp.FP64, arg_types: list | None = None, suffix: str = '',
    stats: dict | None = None,
) -> str | None:
    """Run *func* through the interpreter and the compiled binary on the
    same synthesized inputs and compare bit-for-bit.  Returns ``None`` on
    agreement, an error string on mismatch, or ``'skip'`` when the
    interpreter rejects every sampled input (nothing to compare against).

    *arg_types* overrides the default all-FP64 instantiation, so a caller can
    execute a program whose lists are at some other format — a shape no corpus
    program has.  Inputs are rounded to whatever formats it declares; see
    :func:`_round_to_format`.
    """
    if arg_types is None:
        ty_info = fp.analysis.TypeInfer.check(func.ast)
        arg_types = [_inst_type(ty) for ty in ty_info.arg_types]

    # Generate deterministic samples; keep only those the oracle accepts so
    # the compiled binary is never run on inputs the interpreter rejects.
    n = _N_SAMPLES if arg_types else 1
    samples: list[tuple[list, object]] = []
    for k in range(n):
        # `lseed=k` is shared across args (paired lists stay equal-length);
        # `vseed=k + i` decorrelates values between argument positions.
        inputs = [
            _round_to_format(_gen_value(ty, k + i, k), ty)
            for i, ty in enumerate(arg_types)
        ]
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
            inputs = [
                _round_to_format(
                    _rand_value(ty, rng, list_len, s % 2 == 0), ty,
                )
                for ty in arg_types
            ]
            try:
                expected = _interp(func, inputs)
            except Exception:
                continue
            samples.append((inputs, expected))

    if stats is not None:
        # How much was actually compared, not just how many programs ran: a
        # caller's coverage floor is meaningless if every surviving sample
        # happens to pass an empty list.
        stats['samples'] = stats.get('samples', 0) + len(samples)
        stats['nonempty'] = stats.get('nonempty', 0) + sum(
            1 for inputs, _ in samples
            if any(isinstance(v, list) and v for v in inputs)
        )

    if not samples:
        return 'skip'

    driver = _emit_driver(
        output_dir, prefix, compiler, func, arg_types, samples,
        ctx=ctx, suffix=suffix,
    )
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
    # an empty list leaves a type variable in the signature, which would
    # need a C++ template (the cases where it stays internal now compile)
    'test_list1',
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
    # `i * n` is an int64 product under FP64, which the lossy-implicit-cast
    # guard refuses -- nothing to do with `empty` (`test_empty1` compiles).
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
    compiler = _CORPUS_COMPILER
    failures: list[tuple[str, str, str]] = []
    for func in funcs:
        if func.name in ignore or not _selected(func.name):
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
    'ldexp', # relies on `isinteger`
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
    compiler = _CORPUS_COMPILER
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
        if (
            isinstance(func, fp.Function)
            and func.name not in ignore
            and _selected(func.name)
        ):
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

    if not accepted:
        return failures

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


CPP_INTEROP: str = '''\
namespace fpy {

// Interop: a program holding `std::vector` converts here.  A flat vector can be
// shared or copied; a nested one can only be copied.  Spelled in the same
// standard-library types the compiler emits -- there is no runtime alias.

// Must not outlive `v`.
template <typename T>
inline std::shared_ptr<std::vector<T>> borrow(std::vector<T>& v) {
    return std::shared_ptr<std::vector<T>>(&v, [](std::vector<T>*) {});
}

template <typename T>
inline std::shared_ptr<std::vector<T>> copy_in(const std::vector<T>& v) {
    return std::make_shared<std::vector<T>>(v);
}

template <typename T>
inline std::shared_ptr<std::vector<std::shared_ptr<std::vector<T>>>>
copy_in(const std::vector<std::vector<T>>& vs) {
    auto out = std::make_shared<
        std::vector<std::shared_ptr<std::vector<T>>>>(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i)
        (*out)[i] = copy_in(vs[i]);
    return out;
}

// Fixed-size interop: a kernel with a proven length takes `std::array`.
// `K` is spelled at the call site (`fpy::copy_in_sized<3>(v)`); the length
// check is the caller's obligation made explicit.
template <std::size_t K, typename T>
inline std::array<T, K> copy_in_sized(const std::vector<T>& v) {
    assert(v.size() == K);
    std::array<T, K> out{};
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

template <typename T, std::size_t K>
inline std::vector<T> copy_out(const std::array<T, K>& xs) {
    return std::vector<T>(xs.begin(), xs.end());
}

template <typename T>
inline std::vector<T> copy_out(const std::shared_ptr<std::vector<T>>& xs) {
    return *xs;
}

template <typename T>
inline std::vector<std::vector<T>> copy_out(
    const std::shared_ptr<std::vector<std::shared_ptr<std::vector<T>>>>& xss
) {
    std::vector<std::vector<T>> out(xss->size());
    for (std::size_t i = 0; i < xss->size(); ++i)
        out[i] = *(*xss)[i];
    return out;
}

}  // namespace fpy
'''
"""Conversions a *caller* needs to hand a ``std::vector`` to a generated kernel.

Not part of the runtime, because nothing emitted names them: the emitter never
produces a ``borrow`` / ``copy_in`` / ``copy_out`` call, so carrying them in
every translation unit would be surface no generated program uses.  They live
here, with the tests that exercise the boundary.

Two properties the ABI tests pin:

- A ``vector<vector<T>>`` can only be copied.  The caller stores rows by value
  where a list stores handles, so no arrangement makes a write through either
  side visible to the other.
- A borrowed handle must not outlive its vector.  That holds because a callee
  cannot retain one: FPy has no globals, and captures are materialized before
  compilation.
"""


def _build_and_run_driver(
    cpp_path: Path, group: str, name: str,
) -> list[tuple[str, str, str]]:
    """Build a driver translation unit and run it; a nonzero exit is a failure.

    The driver `main`s assert what the differential harness cannot: they inspect
    state a kernel leaves behind rather than the value it returns.
    """
    exe = cpp_path.with_suffix('.exe')
    try:
        subprocess.run(
            [_CXX, *_CPP_OPTIONS, '-o', str(exe), str(cpp_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'  FAILED to build: {e.stderr[-400:]}')
        return [(group, name, f'build failed: {e.stderr[-200:]}')]

    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  FAILED: {r.stdout.strip()} {r.stderr.strip()}')
        return [(group, name, f'assertion failed: {r.stdout.strip()}')]
    return []


def _test_runtime(output_dir: Path, mode: str = 'compile') -> list[tuple[str, str, str]]:
    """Compile and run a self-test of the boxed-list idioms the emitter spells.

    There is no list runtime -- boxed lists are emitted as
    ``std::shared_ptr<std::vector<T>>`` at the use site -- so this pins the
    sharing contract those spellings provide: copying a handle shares its
    elements, an element of a nested list is the same object, and copying the
    range is the opt-out.
    """
    group = 'runtime'
    compiler = fp.CppCompiler()
    cpp_path = output_dir / 'runtime_selftest.cpp'
    print(f'Compiling runtime self-test to `{cpp_path}`')
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print(compiler.helpers(), file=f)
        print(_RUNTIME_SELFTEST, file=f)

    if mode == 'emit':
        return []
    if _CXX is None:
        print('  SKIPPED (no C++ compiler driver)')
        return []

    exe = cpp_path.with_suffix('.exe')
    cmd = [_CXX] + _CPP_OPTIONS + ['-o', str(exe), str(cpp_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f'  FAILED to build: {e.stderr[-400:]}')
        return [(group, 'runtime_selftest', f'build failed: {e.stderr[-200:]}')]

    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  FAILED: {r.stdout.strip()} {r.stderr.strip()}')
        return [(group, 'runtime_selftest', f'assertion failed: {r.stdout.strip()}')]
    return []


_RUNTIME_SELFTEST: str = """\
#include <cstdio>

typedef std::shared_ptr<std::vector<double>> list_d;
typedef std::shared_ptr<std::vector<list_d>> list_dd;

int main() {
    // copying a handle shares the elements
    list_d xs = std::make_shared<std::vector<double>>(
        std::vector<double>{1.0, 2.0, 3.0});
    list_d ys = xs;
    (*ys)[0] = 99.0;
    assert((*xs)[0] == 99.0);

    // ...and an explicit copy is the opt-out -- the idiom the emitter emits
    // for `xs[:]`
    list_d zs = std::make_shared<std::vector<double>>(xs->begin(), xs->end());
    (*zs)[0] = 7.0;
    assert((*xs)[0] == 99.0);

    // an element of a nested list is the same object, at every slot
    list_dd m = std::make_shared<std::vector<list_d>>(
        std::vector<list_d>{xs, xs});
    (*(*m)[0])[1] = 42.0;
    assert((*xs)[1] == 42.0);
    assert((*(*m)[1])[1] == 42.0);

    // a projection shares, and survives its container's slot being replaced
    list_d row = (*m)[0];
    (*m)[0] = zs;
    (*row)[2] = 5.0;
    assert((*xs)[2] == 5.0);
    assert((*(*m)[0])[2] != 5.0);

    // range-for over the pointee, and size
    list_d acc = std::make_shared<std::vector<double>>(3);
    std::size_t i = 0;
    for (double v : *xs) { (*acc)[i] = v; ++i; }
    assert(i == 3 && acc->size() == 3);

    // contiguous storage for C interop
    double* raw = xs->data();
    raw[0] = -1.0;
    assert((*xs)[0] == -1.0);

    // refcounting: the last owner keeps the elements alive
    {
        list_d tmp = xs;
        (*tmp)[0] = 3.5;
    }
    assert((*xs)[0] == 3.5);

    std::printf("runtime self-test OK\\n");
    return 0;
}
"""


@fp.fpy
def _abi_scale_in_place(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    """Mutates its argument and returns a value."""
    with fp.FP64:
        acc = 0.0
        for i in range(len(xs)):
            xs[i] = xs[i] * k
            acc = acc + xs[i]
        return acc


@fp.fpy
def _abi_fresh_result(xs: list[fp.Real]) -> list[fp.Real]:
    """Returns a new list, so its handle is the sole owner."""
    with fp.FP64:
        return [x * 2 for x in xs]


@fp.fpy
def _abi_row_element_write(xss: list[list[fp.Real]]) -> fp.Real:
    """Writes an *element* of a row: reaches the caller's buffer directly."""
    with fp.FP64:
        for row in xss:
            row[0] = 99
        return xss[0][0]


@fp.fpy
def _abi_nested_sum(xss: list[list[fp.Real]]) -> fp.Real:
    """Reads a nested list without ever *naming* a row, so both levels unbox."""
    with fp.FP64:
        acc = 0.0
        for i in range(len(xss)):
            for j in range(len(xss[i])):
                acc = acc + xss[i][j]
        return acc


@fp.fpy
def _abi_nested_write(xss: list[list[fp.Real]]) -> fp.Real:
    """Writes through a fully-unboxed nested parameter."""
    with fp.FP64:
        xss[0][0] = 99
        return xss[0][0]


def _test_abi(output_dir: Path, mode: str = 'compile') -> list[tuple[str, str, str]]:
    """Compile kernels and call them the way an embedding program would.

    Nothing in the corpus covers handing a kernel storage the *caller* owns —
    the differential driver builds fresh boxed-list arguments — so the
    ``fpy::`` conversions are pinned here: ``borrow`` shares a flat vector,
    ``copy_in`` does not, ``copy_out`` reads a result back, and a
    ``vector<vector<T>>`` can only be copied, so a kernel's write must *not*
    reach the caller.
    """
    group = 'abi'
    # Boxed on purpose, whatever the default: these helpers exist to convert
    # *to* a handle, so a signature that has no handle has nothing to pin.  The
    # native path is `_test_abi_native`.
    compiler = fp.CppCompiler(unbox=fp.CppCompiler.UnboxMode.NEVER)
    cpp_path = output_dir / 'abi_boundary.cpp'
    print(f'Compiling ABI boundary test to `{cpp_path}`')
    lst = fp.types.ListType(fp.types.RealType(fp.FP64))
    real = fp.types.RealType(fp.FP64)
    nested = fp.types.ListType(lst)
    module = fp.Module()
    module.add(_abi_scale_in_place, ctx=fp.FP64, arg_types=[lst, real])
    module.add(_abi_fresh_result, ctx=fp.FP64, arg_types=[lst])
    module.add(_abi_row_element_write, ctx=fp.FP64, arg_types=[nested])
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print('#include <cstdio>', file=f)
        print(compiler.helpers(), file=f)
        print(CPP_INTEROP, file=f)
        print(compiler.compile_module(module), file=f)
        print(_ABI_MAIN, file=f)

    if mode == 'emit':
        return []
    if _CXX is None:
        print('  SKIPPED (no C++ compiler driver)')
        return []

    return _build_and_run_driver(cpp_path, group, 'abi_boundary')


_RTZ_64 = fp.IEEEContext(11, 64, fp.RM.RTZ)
_RTP_64 = fp.IEEEContext(11, 64, fp.RM.RTP)


@fp.fpy
def _fenv_early_return(x: fp.Real) -> fp.Real:
    """Returns from *inside* a rounding scope.

    The restore a scope emits after its body is unreachable from here, so
    without a restore before the return the mode escapes into the caller.
    """
    with _RTZ_64:
        y = x + 1.0
        return y


@fp.fpy
def _fenv_return_after_scope(x: fp.Real) -> fp.Real:
    """The counterweight: execution leaves the scope normally.

    The restore has to still happen at the end of the block, or `z` is computed
    under the scope's mode instead of the function's.
    """
    with _RTZ_64:
        y = x + 1.0
    z = y + 1.0
    return z


@fp.fpy
def _fenv_nested_early_return(x: fp.Real) -> fp.Real:
    """Two scopes deep, so the restore has to reach past both."""
    with _RTZ_64:
        with _RTP_64:
            y = x + 1.0
            return y


def _test_fenv(output_dir: Path, mode: str = 'compile') -> list[tuple[str, str, str]]:
    """A kernel must not change the caller's rounding mode.

    The differential driver cannot see this: it compares a kernel's *own*
    result, and a leaked mode is correct there and wrong everywhere after.  So
    the check is a driver that inspects `fegetround` across a call, and computes
    a sum whose value differs between the modes involved.
    """
    group = 'fenv'
    compiler = fp.CppCompiler()
    cpp_path = output_dir / 'fenv_boundary.cpp'
    print(f'Compiling fenv boundary test to `{cpp_path}`')
    real = fp.types.RealType(fp.FP64)
    module = fp.Module()
    module.add(_fenv_early_return, ctx=fp.FP64, arg_types=[real])
    module.add(_fenv_return_after_scope, ctx=fp.FP64, arg_types=[real])
    module.add(_fenv_nested_early_return, ctx=fp.FP64, arg_types=[real])
    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print('#include <cstdio>', file=f)
        print(compiler.helpers(), file=f)
        print(compiler.compile_module(module), file=f)
        print(_FENV_MAIN, file=f)

    if mode == 'emit':
        return []
    if _CXX is None:
        print('  SKIPPED (no C++ compiler driver)')
        return []

    return _build_and_run_driver(cpp_path, group, 'fenv_boundary')


_FENV_MAIN: str = """\
// 1.0 + 1.5e-16 is just past half an ulp of 1.0, so RNE rounds it up and RTZ
// truncates -- the cheapest expression that tells the two modes apart.
static bool nearest_is_live() {
    volatile double a = 1.0, b = 1.5e-16;
    return (a + b) != 1.0;
}

int main() {
    std::fesetround(FE_TONEAREST);
    if (!nearest_is_live()) {
        printf("precondition: FE_TONEAREST not in effect\\n");
        return 1;
    }

    volatile double r = _fenv_early_return(1.0);
    (void)r;
    if (std::fegetround() != FE_TONEAREST || !nearest_is_live()) {
        printf("_fenv_early_return leaked its rounding mode\\n");
        return 1;
    }

    r = _fenv_nested_early_return(1.0);
    (void)r;
    if (std::fegetround() != FE_TONEAREST || !nearest_is_live()) {
        printf("_fenv_nested_early_return leaked its rounding mode\\n");
        return 1;
    }

    r = _fenv_return_after_scope(1.0);
    (void)r;
    if (std::fegetround() != FE_TONEAREST || !nearest_is_live()) {
        printf("_fenv_return_after_scope leaked its rounding mode\\n");
        return 1;
    }

    // ...and the scope still applied while it was open: 1.0 + 1.0 is exact, so
    // both statements give 3 under any mode -- this pins the value, not the mode
    if (_fenv_return_after_scope(1.0) != 3.0) {
        printf("_fenv_return_after_scope computed %g, wanted 3\\n",
               _fenv_return_after_scope(1.0));
        return 1;
    }
    return 0;
}
"""


_ABI_MAIN: str = """\
int main() {
    // borrow: shares the caller's buffer, so the kernel's writes land in it
    {
        std::vector<double> v(2, 1.0);
        v[1] = 2.0;
        const double* buf = v.data();
        double acc = _abi_scale_in_place(fpy::borrow(v), 3.0);
        assert(acc == 9.0);
        assert(v[0] == 3.0 && v[1] == 6.0);
        assert(v.data() == buf);          // shared, not reallocated
    }
    // copy_in: the caller's vector is untouched
    {
        std::vector<double> v(2, 1.0);
        v[1] = 2.0;
        double acc = _abi_scale_in_place(fpy::copy_in(v), 3.0);
        assert(acc == 9.0);
        assert(v[0] == 1.0 && v[1] == 2.0);
    }
    // copy_out: read a result back into a native vector
    {
        std::vector<double> v(2, 1.0);
        v[1] = 2.0;
        std::vector<double> out = fpy::copy_out(_abi_fresh_result(fpy::borrow(v)));
        assert(out.size() == 2 && out[0] == 2.0 && out[1] == 4.0);
    }
    // A nested vector is copied in, so the kernel's write does *not* reach the
    // caller -- there is no faithful way to share rows, so none is offered.
    {
        std::vector<std::vector<double> > m(2, std::vector<double>(2, 1.0));
        m[1][0] = 2.0;
        double got = _abi_row_element_write(fpy::copy_in(m));
        assert(got == 99.0);
        assert(m[0][0] == 1.0 && m[1][0] == 2.0);
    }
    // ...and copied out row by row.
    {
        std::vector<std::vector<double> > m(1, std::vector<double>(2, 1.0));
        auto h = fpy::copy_in(m);
        double got = _abi_row_element_write(h);
        assert(got == 99.0);
        std::vector<std::vector<double> > out = fpy::copy_out(h);
        assert(out.size() == 1 && out[0][0] == 99.0);
        assert(m[0][0] == 1.0);
    }
    std::printf("abi boundary OK\\n");
    return 0;
}
"""


def _test_abi_native(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """The other boundary: a kernel whose lists are proven unshared takes the
    caller's ``std::vector`` directly — no ``copy_in``, no per-row conversion.

    Opposite of :func:`_test_abi`, where a copy protects the caller's data;
    here the write lands.
    """
    group = 'abi'
    compiler = fp.CppCompiler(unbox=fp.CppCompiler.UnboxMode.ALLOW)
    cpp_path = output_dir / 'abi_native.cpp'
    print(f'Compiling native ABI test to `{cpp_path}`')
    nested = fp.types.ListType(fp.types.ListType(fp.types.RealType(fp.FP64)))
    module = fp.Module()
    module.add(_abi_nested_sum, ctx=fp.FP64, arg_types=[nested])
    module.add(_abi_nested_write, ctx=fp.FP64, arg_types=[nested])

    # the claim under test, checked rather than assumed
    for f in (_abi_nested_sum, _abi_nested_write):
        params, _ = compiler.signature(f, ctx=fp.FP64, arg_types=[nested])
        got = params[0].format()
        if got != 'std::vector<std::vector<double>>':
            return [(
                group, 'abi_native',
                f'{f.name} did not unbox: took `{got}`',
            )]

    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print('#include <cstdio>', file=f)
        print(compiler.helpers(), file=f)
        print(CPP_INTEROP, file=f)
        print(compiler.compile_module(module), file=f)
        print(_ABI_NATIVE_MAIN, file=f)

    if mode == 'emit':
        return []
    if _CXX is None:
        print('  SKIPPED (no C++ compiler driver)')
        return []

    exe = cpp_path.with_suffix('.exe')
    try:
        subprocess.run(
            [_CXX, *_CPP_OPTIONS, '-o', str(exe), str(cpp_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'  FAILED to build: {e.stderr[-400:]}')
        return [(group, 'abi_native', f'build failed: {e.stderr[-200:]}')]

    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  FAILED: {r.stdout.strip()} {r.stderr.strip()}')
        return [(group, 'abi_native', f'assertion failed: {r.stdout.strip()}')]
    return []


_ABI_NATIVE_MAIN: str = """\
int main() {
    // The caller's own nested vector, passed with no conversion whatsoever.
    std::vector<std::vector<double> > m(2, std::vector<double>(2, 1.0));
    m[1][1] = 2.0;
    const std::vector<double>* row0 = m.data();

    assert(_abi_nested_sum(m) == 5.0);
    assert(m.data() == row0);          // read-only: nothing was copied

    // a write reaches the caller -- FPy's semantics
    assert(_abi_nested_write(m) == 99.0);
    assert(m[0][0] == 99.0);

    std::printf("abi native OK\\n");
    return 0;
}
"""


@fp.fpy
def _abi_sized_dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    with fp.FP64:
        acc = 0.0
        for i in range(len(xs)):
            acc = acc + xs[i] * ys[i]
        return acc


@fp.fpy
def _abi_sized_scaled(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        return [k * x for x in xs]


def _test_abi_sized(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """The fixed-size boundary: ``arg_types`` carrying a length compile to
    ``std::array`` parameters, and a caller holding vectors converts with
    ``copy_in_sized`` / reads a fresh array result back with ``copy_out``.
    """
    group = 'abi'
    compiler = fp.CppCompiler()
    cpp_path = output_dir / 'abi_sized.cpp'
    print(f'Compiling sized ABI test to `{cpp_path}`')
    sized = fp.types.ListType(fp.types.RealType(fp.FP64), 3)
    real = fp.types.RealType(fp.FP64)
    module = fp.Module()
    module.add(_abi_sized_dot, ctx=fp.FP64, arg_types=[sized, sized])
    module.add(_abi_sized_scaled, ctx=fp.FP64, arg_types=[sized, real])

    # the claim under test, checked rather than assumed
    params, _ = compiler.signature(
        _abi_sized_dot, ctx=fp.FP64, arg_types=[sized, sized], module=module,
    )
    got = params[0].format()
    if got != 'std::array<double, 3>':
        return [(
            group, 'abi_sized',
            f'_abi_sized_dot did not take an array: `{got}`',
        )]

    with open(cpp_path, 'w') as f:
        print('\n'.join(compiler.headers()), file=f)
        print('#include <cstdio>', file=f)
        print(compiler.helpers(), file=f)
        print(CPP_INTEROP, file=f)
        print(compiler.compile_module(module), file=f)
        print(_ABI_SIZED_MAIN, file=f)

    if mode == 'emit':
        return []
    if _CXX is None:
        print('  SKIPPED (no C++ compiler driver)')
        return []

    return _build_and_run_driver(cpp_path, group, 'abi_sized')


_ABI_SIZED_MAIN: str = """\
int main() {
    // A caller holding vectors converts at the boundary, length checked.
    std::vector<double> v{1.0, 2.0, 3.0};
    std::array<double, 3> a = fpy::copy_in_sized<3>(v);
    std::array<double, 3> b{{2.0, 2.0, 2.0}};

    assert(_abi_sized_dot(a, b) == 12.0);

    // ...or passes its own array directly -- no conversion at all.
    assert(_abi_sized_dot(b, b) == 12.0);

    // a fresh fixed-size result reads back into a vector
    std::vector<double> out = fpy::copy_out(_abi_sized_scaled(a, 10.0));
    assert(out.size() == 3 && out[0] == 10.0 && out[2] == 30.0);

    std::printf("abi sized OK\\n");
    return 0;
}
"""


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
    - ``sum(prods)`` under FP32 lowers to ``std::accumulate`` seeded with the
      first element cast to ``float``, taking advantage of the lossless
      ``int16 → float`` conversion.  That the conversion is exact is what makes
      the fold uni-precision at ``float`` -- ``init + *first`` converts to the
      common type, which is the accumulator only because the element fits it --
      and so equal to the interpreter's per-addition rounding.

    Was in ``_run_ignore`` while ``sum`` seeded from a typed zero: this product
    collapses to zero, and the interpreter's ``-0.0`` came out ``+0.0``.
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


@fp.fpy
def _regression_any_bool_list(bs: list[bool]) -> bool:
    """``any(bs)`` over a ``list[bool]`` argument -> ``std::any_of``.

    Pins the boolean reduction end to end: a ``std::vector<bool>`` parameter
    (whose proxy-reference specialization is easy to get wrong), and the empty
    case — ``_LIST_LENS`` includes ``0``, so ``any([]) == false`` is checked
    against the interpreter on every run rather than by hand.
    """
    return any(bs)


@fp.fpy
def _regression_all_bool_list(bs: list[bool]) -> bool:
    """``all(bs)`` over a ``list[bool]`` argument -> ``std::all_of``.

    The ``all`` half of :func:`_regression_any_bool_list`; separate because a
    single boolean result cannot distinguish a bug in one operator from a bug
    in the other.  Also pins the *other* empty-list identity,
    ``all([]) == true``.
    """
    return all(bs)


@fp.fpy
def _regression_any_over_comprehension(xs: list[fp.Real]) -> bool:
    """The idiomatic ``any([pred for x in xs])``.

    ``ReduceFusion`` rewrites this to an accumulator loop in the default
    pipeline, so what runs here is the *fused* form — this is the differential
    check that fusion preserves behaviour on generated code, empty list
    included (``_LIST_LENS`` covers 0).
    """
    with fp.FP64:
        return any([x < 0 for x in xs])


###########################################################
# List-aliasing regressions
#
# FPy lists are shared: assignment aliases, `xs[i] = e` mutates the object, and
# passing/returning/projecting carries the identity along.  `std::vector` is a
# value type, so every place the backend copies a list is a place the generated
# code can disagree with the interpreter.
#
# One function per *route* by which a list can reach a copy.  All of them are
# executed and bit-compared against the interpreter on every `--mode run`; the
# eleven that once diverged were the acceptance criterion for representing a
# list as a shared handle.  Each guards the empty case because `_LIST_LENS`
# includes 0.


@fp.fpy
def _regression_alias_then_mutate(xs: list[fp.Real]) -> fp.Real:
    """Route: a bare `ys = xs` alias, then a write through the alias."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        ys = xs
        ys[0] = 99
        return xs[0]


@fp.fpy
def _regression_alias_readonly(xs: list[fp.Real]) -> fp.Real:
    """Pin: a read-only alias.  Nothing is written, so a copy is unobservable
    and the backend is free to pick `const&`.  Guards against a fix for the
    route above that pessimizes this one."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        ys = xs
        return ys[0] + xs[0]


@fp.fpy
def _regression_writes_its_arg(zs: list[fp.Real]) -> fp.Real:
    """Callee for :func:`_regression_callee_mutates_param`."""
    with fp.FP64:
        zs[0] = 99
        return 0


@fp.fpy
def _regression_callee_mutates_param(xs: list[fp.Real]) -> fp.Real:
    """Route: a callee writes its list parameter; FPy shares, so the caller
    must see it."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        v = _regression_writes_its_arg(xs)
        return xs[0] + v


@fp.fpy
def _regression_returns_its_argument(zs: list[fp.Real]) -> list[fp.Real]:
    """Callee for :func:`_regression_returned_list_aliases`."""
    return zs


@fp.fpy
def _regression_returned_list_aliases(xs: list[fp.Real]) -> fp.Real:
    """Route: a returned list keeps its identity, so the caller's binding
    aliases the list it passed in."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        ys = _regression_returns_its_argument(xs)
        ys[0] = 99
        return xs[0]


@fp.fpy
def _regression_projected_element(xss: list[list[fp.Real]]) -> fp.Real:
    """Route: projection *out of* a container.  An element lives inside its
    container, so `inner` is that element, not a copy of it."""
    with fp.FP64:
        if len(xss) == 0:
            return 0
        inner = xss[0]
        inner[0] = 99
        return xss[0][0]


@fp.fpy
def _regression_loop_variable(xss: list[list[fp.Real]]) -> fp.Real:
    """Route: the same projection through a loop variable."""
    with fp.FP64:
        if len(xss) == 0:
            return 0
        for row in xss:
            row[0] = 99
        return xss[0][0]


@fp.fpy
def _regression_list_into_list(xs: list[fp.Real]) -> fp.Real:
    """Route: construction *into* a container.  `std::vector` owns its
    elements, so the copy happens at construction, before any name exists that
    could have been a reference instead."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        zss = [xs]
        zss[0][0] = 99
        return xs[0]


@fp.fpy
def _regression_list_into_tuple(xs: list[fp.Real]) -> fp.Real:
    """Route: the same construction into a tuple.  Tuples are immutable, so
    copying one is normally unobservable — but not when it holds a list."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        t = (xs, 1.0)
        ys = fp.fst(t)
        ys[0] = 99
        return xs[0]


@fp.fpy
def _regression_comprehension_of_rows(xss: list[list[fp.Real]]) -> fp.Real:
    """Route: `[row for row in xss]` is a new *outer* list over the *same*
    inner lists.  Contrast `_regression_comprehension_deep_copy`."""
    with fp.FP64:
        if len(xss) == 0:
            return 0
        yss = [row for row in xss]
        yss[0][0] = 99
        return xss[0][0]


@fp.fpy
def _regression_comprehension_deep_copy(xss: list[list[fp.Real]]) -> fp.Real:
    """Pin: one character of difference from the route above — the element is a
    fresh comprehension, so this really is a deep copy and `xss` is untouched."""
    with fp.FP64:
        if len(xss) == 0:
            return 0
        yss = [[x for x in row] for row in xss]
        yss[0][0] = 99
        return xss[0][0]


@fp.fpy
def _regression_nested_slice(xss: list[list[fp.Real]]) -> fp.Real:
    """Route: slicing is *shallow* — a fresh outer list over the same inner
    lists.  The C++ range constructor copies every element."""
    with fp.FP64:
        if len(xss) == 0:
            return 0
        yss = xss[0:1]
        yss[0][0] = 99
        return xss[0][0]


@fp.fpy
def _regression_flat_slice(xs: list[fp.Real]) -> fp.Real:
    """Pin: a slice of a *flat* list has scalar elements, so one level of copy
    is all there is and C++ agrees."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        ys = xs[0:1]
        ys[0] = 99
        return xs[0]


@fp.fpy
def _regression_one_list_two_indices(xs: list[fp.Real]) -> fp.Real:
    """Route: one list placed at two indices.  `x[0]` and `x[1]` are one object
    in FPy and two independent vectors in C++."""
    with fp.FP64:
        if len(xs) == 0:
            return 0
        a = [xs[0], xs[0]]
        x = [a, a]
        x[0][0] = 99
        return x[1][0]


@fp.fpy
def _regression_enumerate_row_write(xss: list[list[fp.Real]]) -> fp.Real:
    """Route: a row aliased out of an `enumerate` and written through.  Both
    lowerings must keep that write visible in `xss`, for different reasons, so
    this pins the pair rather than one code path: `optimize=False` materializes
    a `vector<tuple<I, T>>` and reaches the row through a copy synthesized by
    codegen that appears nowhere in the AST, while `optimize=True` (the
    default) rewrites to an ordinary `row = _src0[i]`.  A boxed list is a
    handle, so both copies share elements.
    """
    with fp.FP64:
        if len(xss) == 0:
            return 0
        acc = 0.0
        for (i, row) in enumerate(xss):
            row[0] = 99
            acc = acc + 1.0
        return xss[0][0] + acc


@fp.fpy
def _regression_conditional_alias(
    xs: list[fp.Real], zs: list[fp.Real], c: fp.Real,
) -> fp.Real:
    """Pin: one name aliasing a *different* list on each path.  No C++
    reference can stand for both, so the backend must either hoist a variable
    (as it does today) or refuse — but never silently rename one to the
    other."""
    with fp.FP64:
        if len(xs) == 0 or len(zs) == 0:
            return 0
        if c > 0:
            ys = xs
        else:
            ys = zs
        return ys[0]


@fp.fpy
def _regression_replaced_slot(
    xss: list[list[fp.Real]], ys: list[fp.Real],
) -> fp.Real:
    """Pin: a projection names an *object*, not a slot.  After `xss[0] = ys`,
    `row` is still the detached old list — a C++ reference would follow the
    slot instead."""
    with fp.FP64:
        if len(xss) == 0 or len(ys) == 0:
            return 0
        row = xss[0]
        xss[0] = ys
        row[0] = 99
        return xss[0][0] + ys[0]


@fp.fpy
def _regression_signed_zero_from_neg() -> fp.Real:
    """A *statically-known* zero turned negative.

    `format_infer` bounds a literal by the exact set of its values, and a
    `Fraction` has no signed zero -- so `{0}` used to survive a negation, the
    C++ backend stored it as `uint8_t`, and this returned `+0.0` where the
    interpreter returns `-0.0`.

    The whole family survived on coverage: `_float_bit_eq` has always
    distinguished +0 from -0, but no program made a known zero go negative.
    `x` is unused on purpose -- the point is that the zero is a compile-time
    constant, which is what produces the set bound.
    """
    with fp.FP64:
        z = 0.0
        return -z


@fp.fpy
def _regression_signed_zero_from_mul() -> fp.Real:
    """The same via multiplication: IEEE makes a product's zero sign the XOR of
    the operand signs, so `0.0 * -1.0` is `-0.0`."""
    with fp.FP64:
        z = 0.0
        return z * -1.0


@fp.fpy
def _regression_positive_zero_still_narrows() -> fp.Real:
    """The other direction, so the fix above cannot be "widen everything":
    `+0.0` really is the integer 0 and keeps its precise bound."""
    with fp.FP64:
        z = 0.0
        return abs(z) + 0.0


_regression_funcs: list[fp.Function] = [
    _regression_quant_dot_real_widen,
    _regression_empty_range,
    _regression_signed_zero_from_neg,
    _regression_signed_zero_from_mul,
    _regression_positive_zero_still_narrows,
    _regression_calls_user_fn,
    _regression_any_bool_list,
    _regression_all_bool_list,
    _regression_any_over_comprehension,
    _regression_alias_then_mutate,
    _regression_alias_readonly,
    _regression_callee_mutates_param,
    _regression_returned_list_aliases,
    _regression_projected_element,
    _regression_loop_variable,
    _regression_list_into_list,
    _regression_list_into_tuple,
    _regression_comprehension_of_rows,
    _regression_comprehension_deep_copy,
    _regression_nested_slice,
    _regression_flat_slice,
    _regression_one_list_two_indices,
    _regression_enumerate_row_write,
    _regression_conditional_alias,
    _regression_replaced_slot,
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
@fp.fpy
def _regression_big_literal_in_a_list(y: fp.Real) -> fp.Real:
    """An integral literal past `long long`, reached through a list element.

    Storage selection refuses such a value for a scalar, but an element never
    asks -- so this compiled to a 301-digit token that gcc folded to `0`, where
    the interpreter gives 1e300.  Executed, so the harness bit-compares it.
    """
    with fp.FP64:
        zs = [1e300, y]
        zs[1] = 1e300
        return zs[0] + zs[1]


@fp.fpy(ctx=fp.FP32)
def _regression_fp32_literal_call_arg(y: fp.Real, z: fp.Real) -> fp.Real:
    """Literal arguments to type-deduced C++ calls, under FP32.

    A literal's storage comes from its value, so `1.5` matches the `float`
    signature while the token is a C++ `double`.  `fpy::max` is a template, so
    that combination does not deduce and the emitted C++ *does not compile* --
    which is what this regression catches, since the harness compiles every
    corpus program.  `std::fma` is the silent half: it would pick the `double`
    overload and round twice.  Only FP32 shows either; an FP64 literal already
    has the right type.
    """
    a = fp.fmax(y, 1.5)
    b = fp.fma(y, z, 0.25)
    return a + b


_typed_regression_funcs: list[tuple[fp.Function, list]] = [
    (
        _regression_blocked_dot_e4m3,
        [fp.types.ListType(fp.types.RealType(fp.MX_E4M3))],
    ),
    (
        _regression_fp32_literal_call_arg,
        [fp.types.RealType(fp.FP32), fp.types.RealType(fp.FP32)],
    ),
]


def _test_typed_regressions(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """Compile each explicitly-typed regression with its given arg types
    (bypassing the FP64 instantiation), then object-compile unless in
    ``emit`` mode.  No execution — these carry non-synthesizable arg types."""
    compiler = _CORPUS_COMPILER
    failures: list[tuple[str, str, str]] = []
    for func, arg_types in _typed_regression_funcs:
        if not _selected(func.name):
            continue
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
# Generated format matrix
#
# Every function above is instantiated at FP64, because `_inst_type` is what
# fills in a free `Real`.  So no executed program has ever had a list at
# another format -- and the join/widening machinery in `storage_infer` and
# `emitter._emit_at` exists precisely to reconcile two formats meeting at one
# place.  `tests/unit/backend/cpp/test_generated_typecheck.py` covers the same
# matrix for *ill-typed emission*; this covers it for *wrong answers*, which is
# the class only execution can catch.


@fp.fpy
def _gen_return_param_or_literal(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def _gen_ternary_param(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        zs = xs if c > 0 else [y, y]
        return zs[0]


@fp.fpy
def _gen_write_then_return(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            xs[0] = y
        return xs


@fp.fpy
def _gen_nested_param_sum(xss: list[list[fp.Real]], y: fp.Real) -> fp.Real:
    with fp.FP64:
        acc = y
        for row in xss:
            for x in row:
                acc = acc + x
        return acc


@fp.fpy
def _gen_row_write(xss: list[list[fp.Real]], y: fp.Real) -> fp.Real:
    with fp.FP64:
        xss[0][0] = y
        return xss[0][0]


@fp.fpy
def _gen_alias_then_mutate(xs: list[fp.Real], y: fp.Real) -> fp.Real:
    with fp.FP64:
        ys = xs
        ys[0] = y
        return xs[0]


@fp.fpy
def _gen_list_into_tuple(xs: list[fp.Real], y: fp.Real) -> fp.Real:
    with fp.FP64:
        t = (xs, y)
        zs = fp.fst(t)
        zs[0] = y
        return xs[0]


@fp.fpy
def _gen_comprehension_of_rows(
    xss: list[list[fp.Real]], y: fp.Real,
) -> fp.Real:
    with fp.FP64:
        rows = [row for row in xss]
        rows[0][0] = y
        return xss[0][0]


# ``'flat'`` takes one ``list[Real]``, ``'nested'`` one ``list[list[Real]]``;
# the remaining parameters are scalars, in declaration order.
_generated_funcs: list[tuple[fp.Function, str, int]] = [
    (_gen_return_param_or_literal, 'flat', 2),
    (_gen_ternary_param, 'flat', 2),
    (_gen_write_then_return, 'flat', 2),
    (_gen_nested_param_sum, 'nested', 1),
    (_gen_row_write, 'nested', 1),
    (_gen_alias_then_mutate, 'flat', 1),
    (_gen_list_into_tuple, 'flat', 1),
    (_gen_comprehension_of_rows, 'nested', 1),
]

_GEN_FORMATS = (fp.FP32, fp.FP64)

# Instantiations known to disagree with the interpreter, keyed by label so the
# *other* combinations of the same shape still run -- this shape passes at 6 of
# its 8.  Strict: an entry that starts agreeing is reported as a failure, so a
# fix cannot leave a stale suppression behind.  That is how this one survived
# unnoticed in the first place.
_generated_xfail: dict[str, str] = {
    # Empty, and worth keeping that way: an entry here says the compiler
    # produces a *wrong answer* for some instantiation, which the correctness
    # criterion does not allow.  A shape the backend cannot handle belongs in
    # the "not compared" list below -- refused, not divergent.
    #
    # Note a refusal never reaches the strict check, so an entry that starts
    # being refused rather than diverging goes stale silently.  Re-read this
    # list when a refusal appears for something listed here.
}


def _test_generated(
    output_dir: Path, mode: str = 'compile',
) -> list[tuple[str, str, str]]:
    """Execute each shape at every combination of list-element and scalar
    format, bit-comparing against the interpreter.

    Skipped outside ``run`` mode: the point is the comparison, and
    ``test_generated_typecheck.py`` already covers emission over a wider matrix.
    A *refusal* is a legitimate answer here too -- a shared list cannot change
    element type -- so a ``CppCompileError`` is recorded, not failed.  Refusals
    are printed: a shape quietly regressing into one is a loss of coverage that
    no failure count would show.

    There is deliberately no outer-context axis.  Every shape pins its context
    with ``with fp.FP64:``, so varying it produced byte-identical programs and
    only inflated the counts.  Making it mean something needs two changes
    together -- drop the ``with`` from some shapes *and* pass ``ctx`` into
    ``_interp``, which hardcodes ``fp.FP64`` -- or the oracle and the binary
    disagree for a reason that is not a bug.
    """
    if mode != 'run' or _CXX is None:
        return []
    compiler = _CORPUS_COMPILER
    failures: list[tuple[str, str, str]] = []
    xfailed: list[str] = []
    refused: list[str] = []
    stats: dict[str, int] = {}
    ran = 0
    for func, sig, n_scalars in _generated_funcs:
        if not _selected(func.name):
            continue
        for elt_fmt in _GEN_FORMATS:
            for scalar_fmt in _GEN_FORMATS:
                elt = fp.types.RealType(elt_fmt)
                lst = (
                    fp.types.ListType(elt) if sig == 'flat'
                    else fp.types.ListType(fp.types.ListType(elt))
                )
                arg_types = [
                    lst,
                    *[fp.types.RealType(scalar_fmt)] * n_scalars,
                ]
                tag = f'{elt_fmt.nbits}_{scalar_fmt.nbits}'
                label = f'{func.name}[{tag}]'
                try:
                    err = _run_and_check(
                        output_dir, 'generated', compiler, func,
                        arg_types=arg_types, suffix=tag, stats=stats,
                    )
                except fp.backend.CppCompileError as e:
                    # A refusal is an answer, not a failure -- but say so.
                    refused.append(f'{label}: {str(e).split(": ")[-1][:70]}')
                    continue
                except Exception as e:
                    failures.append(('generated', label, str(e)))
                    continue
                if err == 'skip':
                    refused.append(f'{label}: no sample the interpreter accepts')
                    continue
                known = _generated_xfail.get(label)
                if known is not None:
                    if err is None:
                        failures.append((
                            'generated', label,
                            f'now agrees with the interpreter -- remove the '
                            f'_generated_xfail entry ({known})',
                        ))
                    else:
                        xfailed.append(f'{label}: {known}')
                    continue
                if err is not None:
                    failures.append(('generated', label, err))
                else:
                    ran += 1
    print(
        f'=== generated format matrix: {ran} instantiations, '
        f'{stats.get("samples", 0)} samples bit-compared '
        f'({stats.get("nonempty", 0)} with a non-empty list) ==='
    )
    for note in xfailed:
        print(f'     known divergence: {note}')
    for note in refused:
        print(f'     not compared: {note}')
    # A `-k` run tests a subset on purpose, so the minimums below do not apply.
    if _select is not None:
        return failures
    if stats.get('nonempty', 0) < 200:
        failures.append((
            'generated', '<coverage>',
            f'only {stats.get("nonempty", 0)} samples with a non-empty list '
            f'were compared; the counts above can stay high while every '
            f'comparison is on an empty list, which touches none of the '
            f'join/widening/aliasing behaviour this stage exists for',
        ))
    if ran < 20:
        failures.append((
            'generated', '<coverage>',
            f'only {ran} instantiations executed; the matrix has stopped '
            f'covering anything (refusals and skips are not failures, so this '
            f'check is what keeps the stage honest)',
        ))
    return failures


###########################################################
# Name filter

_select: 're.Pattern[str] | None' = None
"""``-k`` filter set by the CLI.  ``None`` — the default, and what the pytest
entry point uses — tests everything."""


def _selected(name: str) -> bool:
    """Whether *name* passes the ``-k`` filter (a regex, searched not anchored,
    so a plain substring works)."""
    return _select is None or _select.search(name) is not None


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
    failures += _test_runtime(output_dir, mode=mode)
    failures += _test_abi(output_dir, mode=mode)
    failures += _test_abi_native(output_dir, mode=mode)
    failures += _test_abi_sized(output_dir, mode=mode)
    failures += _test_fenv(output_dir, mode=mode)
    failures += _test_unit(output_dir, mode=mode, cov=cov)
    failures += _test_libraries(output_dir, mode=mode)
    failures += _test_regressions(output_dir, mode=mode, cov=cov)
    failures += _test_typed_regressions(output_dir, mode=mode)
    failures += _test_generated(output_dir, mode=mode)

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
    parser.add_argument(
        '-k', dest='select', metavar='PATTERN', default=None,
        help="Only test functions whose name matches PATTERN (a regex, searched "
             "anywhere in the name), like pytest's -k",
    )
    args = parser.parse_args()
    if args.select is not None:
        _select = re.compile(args.select)
        print(f'Filtering to function names matching `{args.select}`')

    # arguments
    delete: bool = not args.keep
    mode: str = 'emit' if args.no_cc else args.mode

    # run test
    test_compile_cpp(delete=delete, mode=mode)
