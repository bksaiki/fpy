import argparse
import functools
import importlib
import itertools
import math
import multiprocessing
import random
import signal
import time
from fractions import Fraction
from pathlib import Path

import titanfp.fpbench.fpcast as fpc
from titanfp.titanic.gmpmath import SignedOverflow, SignedUnderflow

import fpy2 as fp
from fpy2.interpret import get_default_interpreter
from infra.fpbench import module_name, port_names, read_cores

from .fetch import benchmarks_dir
from .shim import fpy_to_mpmf, mpmf_interpreter, mpmf_to_fpy

# `_rand_real` draws an argument without a bound from `[-_REAL_RANGE, _REAL_RANGE]`
_REAL_RANGE = 1024

# `_rand_int` draws an argument without a bound from `[-_INT_RANGE, _INT_RANGE]`;
# integer arguments often count loop iterations
_INT_RANGE = 100

# seconds an input may take before it is skipped (loops may not terminate)
_INPUT_TIME = 5.0

###########################################################
# Input generation

def _bounds(pre: fpc.Expr | None, names: list[str]):
    """Constant bounds `[lo, hi]` on each argument from the conjuncts of `pre`."""
    bounds: dict[str, list[Fraction | None]] = {name: [None, None] for name in names}

    def number(e: fpc.Expr):
        match e:
            case fpc.Integer() | fpc.Decnum():
                return Fraction(e.value)
            case fpc.Rational():
                return Fraction(e.p, e.q)
            case _:
                return None

    def visit(e: fpc.Expr | None):
        match e:
            case fpc.And():
                for child in e.children:
                    visit(child)
            case fpc.LT() | fpc.LEQ() | fpc.GT() | fpc.GEQ():
                args = e.children if isinstance(e, fpc.LT | fpc.LEQ) else e.children[::-1]
                # each adjacent pair `a <= b` may bound a variable by a constant
                for a, b in itertools.pairwise(args):
                    if isinstance(b, fpc.Var) and b.value in bounds and (lo := number(a)) is not None:
                        bounds[b.value][0] = lo
                    if isinstance(a, fpc.Var) and a.value in bounds and (hi := number(b)) is not None:
                        bounds[a.value][1] = hi

    visit(pre)
    return bounds

def _float_bounds(lo: Fraction | None, hi: Fraction | None, ctx: fp.Context):
    return (
        ctx.maxval(True) if lo is None else ctx.round(lo),
        ctx.maxval() if hi is None else ctx.round(hi)
    )

def _rand_bits(lo: Fraction | None, hi: Fraction | None, ctx: fp.Context, rng: random.Random):
    """a float in `[lo, hi]`, each equally likely"""
    a, b = _float_bounds(lo, hi, ctx)
    return ctx.from_ordinal(rng.randint(ctx.to_ordinal(a), ctx.to_ordinal(b)))

def _rand_edge(lo: Fraction | None, hi: Fraction | None, ctx: fp.Context, rng: random.Random):
    """an end of `[lo, hi]`, its neighbor outside it, or a special value"""
    a, b = _float_bounds(lo, hi, ctx)
    below = ctx.from_ordinal(ctx.to_ordinal(a, True) - 1, True)
    above = ctx.from_ordinal(ctx.to_ordinal(b, True) + 1, True)
    specials = [ctx.round(x) for x in (0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan)]
    return rng.choice([a, b, below, above, ctx.minval(), ctx.minval(True), *specials])

def _rand_real(lo: Fraction | None, hi: Fraction | None, ctx: fp.Context, rng: random.Random):
    """a real in `[lo, hi]`, each equally likely, rounded"""
    a = -_REAL_RANGE if lo is None else lo
    b = _REAL_RANGE if hi is None else hi
    return ctx.round(a + (b - a) * Fraction(rng.random()))

def _rand_subnormal(lo: Fraction | None, hi: Fraction | None, ctx: fp.Context, rng: random.Random):
    """standard normal scaled so that 1 becomes the smallest normal number"""
    return ctx.round(rng.gauss(0.0, 1.0) * 2.0 ** ctx.emin)

GENS = [_rand_bits, _rand_edge, _rand_real, _rand_subnormal]

def _rand_int(lo: Fraction | None, hi: Fraction | None, rng: random.Random):
    """
    an integer in `[lo - 1, hi + 1]`, for an `integer` argument;
    FPCore leaves non-integers undefined
    """
    a = -_INT_RANGE if lo is None else math.ceil(lo) - 1
    b = _INT_RANGE if hi is None else math.floor(hi) + 1
    return fp.Float.from_int(rng.randint(a, b))

###########################################################
# Evaluation

class _Timeout(Exception):
    """An input ran out of time."""

def _timeout(signum, frame):
    raise _Timeout()

def _same(x, y) -> bool:
    """Bit-for-bit equality: NaNs are equal, but +0 and -0 are not."""
    match x, y:
        case fp.Float(), fp.Float():
            return (x.isnan and y.isnan) or (x == y and x.s == y.s)
        case list(), list():
            return len(x) == len(y) and all(map(_same, x, y))
        case _:
            return x == y

def _titanic(rt, core: fpc.FPCore, e: fpc.Expr, args: list[fp.Float]):
    """Titanic's value of `e`, the body, `:pre`, or `:spec` of `core`, on `args`."""
    ctx = rt.arg_ctx(core, [fpy_to_mpmf(x) for x in args])
    return mpmf_to_fpy(rt.evaluate(e, ctx))

def _fpy_prop(port: fp.Function, prop, args: list[fp.Float], ctx: fp.Context):
    """Evaluates the `pre` or `spec` of `port`: an FPy expression or function."""
    if isinstance(prop, fp.Function):
        return prop(*args)
    env = {arg.name: x for arg, x in zip(port.args, args)}
    return get_default_interpreter().eval_expr(prop, env, ctx)

def _compare(rt, core: fpc.FPCore, port: fp.Function, args: list[fp.Float], ctx: fp.Context):
    """
    Compares `port` against `core` on `args`.
    Returns whether `args` satisfies `:pre` and the first mismatch (if any).
    """
    meta = port.ast.meta

    # (check, Titanic's value, FPy's value)
    checks = [('body', _titanic(rt, core, core.e, args), functools.partial(port, *args))]
    sat = True
    if core.pre is not None:
        sat = _titanic(rt, core, core.pre, args)
        checks.append(('pre', sat, functools.partial(_fpy_prop, port, meta.pre, args, ctx)))
    if core.spec is not None and sat:
        expect = _titanic(rt, core, core.spec, args)
        checks.append(('spec', expect, functools.partial(_fpy_prop, port, meta.spec, args, ctx)))

    for what, expect, run in checks:
        try:
            actual = run()
        except _Timeout:
            raise
        except Exception as e:
            actual = e
        if not _same(expect, actual):
            return sat, f'{what} on ({", ".join(map(str, args))}): expected {expect}, got {actual}'
    return sat, None

###########################################################
# Driver

_rt = None

def _init_worker():
    global _rt
    _rt = mpmf_interpreter()
    signal.signal(signal.SIGALRM, _timeout)

@functools.cache
def _cores(path: Path):
    return read_cores(path)

def _test_core(path: Path, index: int, trials: int, budget: float, seed: int):
    """Compares a port against its core on up to `trials` inputs within `budget` seconds."""
    cores = _cores(path)
    core, name = cores[index], port_names(cores)[index]
    port = getattr(importlib.import_module(f'examples.fpbench.benchmarks.{module_name(path)}'), name)
    ctx = port.ast.ctx or fp.FP64
    names = [arg for arg, _, _ in core.inputs]
    ints = {arg for arg, props, _ in core.inputs if str(props.get('precision')) == 'integer'}
    bounds = _bounds(core.pre, names)
    rng = random.Random(f'{seed}:{path.name}:{name}')

    compared = in_pre = skipped = 0
    deadline = time.monotonic() + budget
    for i in range(trials):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        gen = GENS[i % len(GENS)]
        args = [
            _rand_int(*bounds[arg], rng) if arg in ints else gen(*bounds[arg], ctx, rng)
            for arg in names
        ]
        signal.setitimer(signal.ITIMER_REAL, min(remaining, _INPUT_TIME))
        try:
            sat, mismatch = _compare(_rt, core, port, args, ctx)
        except (_Timeout, SignedOverflow, SignedUnderflow):
            # ran out of time, or Titanic cannot evaluate the input
            skipped += 1
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        if mismatch is not None:
            return f'FAIL {path.name} {name}: {mismatch}'
        compared += 1
        in_pre += sat

    if compared == 0:
        return f'skip {path.name} {name}: no input finished ({skipped} skipped)'
    return f'ok   {path.name} {name}: {compared} inputs ({in_pre} satisfy :pre, {skipped} skipped)'

def test_fpbench_examples(trials: int = 1000, seed: int = 0, budget: float = 30.0, jobs: int = 1):
    """
    Checks every port against its FPBench source on up to `trials` inputs,
    in `jobs` worker processes.  Each core gets at most `budget` seconds.
    """
    tasks = [
        (path, i, trials, budget, seed)
        for path in sorted(benchmarks_dir.glob('*.fpcore'))
        for i in range(len(_cores(path)))
    ]

    failures: list[str] = []
    with multiprocessing.Pool(jobs, initializer=_init_worker) as pool:
        for line in pool.starmap(_test_core, tasks):
            print(line)
            if line.startswith('FAIL'):
                failures.append(line)

    if failures:
        raise AssertionError(f'{len(failures)} ports disagree with FPBench:\n' + '\n'.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='differential testing of the FPBench ports')
    parser.add_argument('--trials', type=int, default=1000, help='inputs per core (default 1000)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=float, default=30.0, help='seconds per core (default 30)')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='worker processes (default 1)')
    args = parser.parse_args()
    test_fpbench_examples(args.trials, args.seed, args.budget, args.jobs)
