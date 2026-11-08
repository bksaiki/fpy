"""
Grade MMA designs as Grading the BLAS (GtB) grades a GEMM: each design of
`sweep.grid()` multiplies GtB's Test 1c and 2c matrices, `N` by `N`, scored
by the worst componentwise backward error `|C^ - C| / (|A| |B|)` over the
entries GtB scores. Each design runs on the CPU as FPy's C++ for it, chained
by `fpy2_models`' `<op>_gemm`.

The table, `<results>/blas/blas-grid.csv` (`plots.py`), has a row per grid
point and case (test, `n`, `s`): the point's columns, its error, the FP32
FMA loop's error on the same matrices, and the seconds the point took in FPy
(to C++), in the C++ compiler, and to grade. Its provenance has each stage's
wall time.

    python evaluation/blas/explore.py
"""

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[2]
MMASIM = ROOT / 'examples' / 'mmasim'
COMMON = ROOT / 'evaluation' / 'common' / 'fpy2_models'
# `serve`, for `sweep` and `core`; `examples/mmasim`, for `compile` and `generator`
sys.path[:0] = [str(MMASIM / 'serve'), str(MMASIM), str(COMMON)]

import compile as mm
import generate_models
import generator
import sweep
from core import outputs, quant
from generator.catalog import CATALOG

import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend.triton.normalize import normalize_module
from fpy2.transform import FreeVarElim
from fpy2.transform.const_fold import ConstFold

RESULTS = ROOT / 'evaluation' / 'results'
# fpy2_models' flags (setup.py): the models set rounding modes with `fesetround`
CXXFLAGS = ['-O3', '-frounding-math', '-ffp-contract=off',
            '-DTORCH_TARGET_VERSION=0x020a000000000000']
SEED = 0
N = 1024
FP32_UN = 2.0 ** -126
FORMATS = {
    # name: (context, s)
    'BF16': (fp.BF16, [0, 11, 23, 34, 46, 57]),
    'E4M3': (fp.MX_E4M3, list(range(7))),
    # FNUZ tops out at 240, below 6 * 2^6
    'E4M3FNUZ': (fp.S1E4M3, list(range(6))),
}
# graded on every format's matrices, beside each point
FMA = CATALOG['fp32 (fma)']


def to_cpp(d: generator.Design) -> dict:
    """d in C++, with its entry's signature as `compile.py`'s manifest lists
    it, and the seconds that took; or why FPy refused it."""
    name = generator.name(d)
    t0 = time.perf_counter()
    try:
        func, arg_types = generator.build(d)
        mod = fp.Module()
        mod.add(st.monomorphize(func, args=arg_types))
        # free variables removed, inlined and folded first, as the Triton backend
        # does: the C++ backend refuses the generator's helpers otherwise
        mod = normalize_module(mod.map(lambda _, f: FreeVarElim.apply(f)))
        mod = mod.map(lambda _, f: ConstFold.apply(f)).map(mm._prepare)
        compiler = fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.ROUNDINGS)
        src = compiler.compile_module(mod)
        params, result = compiler.signature(mod.public()[0], module=mod)
        out = {'src': src, 'model': {'name': name, 'file': mm._filename(name),
                                     'namespace': f'fpy_models::{mm._namespace(name)}',
                                     'entry': func.name,
                                     'params': [mm._cpp_type(p) for p in params],
                                     'result': mm._cpp_type(result)}}
    except Exception as e:  # noqa: BLE001 -- a refusal is a result
        out = {'reason': f'{type(e).__name__}: {e}'.splitlines()[0][:300]}
    return out | {'seconds': time.perf_counter() - t0}


def build(compiled: list[dict], out: Path, jobs: int) -> dict[str, float]:
    """Each compiled design loaded as torch.ops.fpy2_models.<op> and <op>_gemm,
    its translation units written as `generate_models.py` writes them, and
    built from scratch; the seconds the C++ compiler took on each, by
    operator."""
    os.environ['MAX_JOBS'] = str(jobs)
    shutil.rmtree(out, ignore_errors=True)
    (out / 'models').mkdir(parents=True)
    sources = []
    for c in compiled:
        m = c['model']
        sources.append(out / f'{op(m)}.cpp')
        (out / 'models' / m['file']).write_text(mm._translation_unit(m['name'], c['src']))
        sources[-1].write_text(generate_models._operator(op(m), m))
    cpp_extension.load('fpy2_blas', [str(s) for s in sources], extra_cflags=CXXFLAGS,
                       extra_include_paths=[str(COMMON / 'fpy2_models' / 'csrc')],
                       build_directory=str(out), is_python_module=False)
    # ninja's log: a line per output, `start  end  mtime  output  hash`, in milliseconds
    lines = (line.split('\t') for line in (out / '.ninja_log').read_text().splitlines()[1:])
    return {Path(o).stem: (int(end) - int(start)) / 1000 for start, end, _, o, _ in lines if o.endswith('.o')}


def op(model: dict) -> str:
    """model's operator name: `gen.<digest>` as `gen_<digest>`."""
    return model['name'].replace('.', '_')


def unit_scales(model: dict, n: int) -> tuple:
    """A scaled model's scales for rows of n, all 1, so that its inputs are its
    format's values as they are; none for an unscaled model."""
    a, _, _, *scales = model['params']
    if not scales:
        return ()
    ones = torch.ones(n, n // a['size'] * scales[0].get('size', 1))
    return ones, ones


def _mask(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    """k of n positions, at random."""
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.randperm(n, generator=g)[:k]] = True
    return m


# Tests 1c and 2c, as GtB's `include/init_matrices.hpp` builds them. GtB
# seeds a fresh generator for each step, so every s draws the same values.

def test_1c(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A, B, and the rows and columns of the target entries."""
    g = torch.Generator().manual_seed(SEED)
    A, B = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
    g = torch.Generator().manual_seed(SEED)
    rows, cols = torch.randperm(n, generator=g)[:n // 2], torch.randperm(n, generator=g)[:n // 2]
    tiny = _mask(n // 2, n // 4, g)
    for r, c, has_tiny in zip(rows.tolist(), cols.tolist(), tiny.tolist()):
        zero_a = _mask(n, n // 2, g)
        t = int(torch.randint(n, (), generator=g))
        x, y = A[r, t] * 2.0 ** -s, B[t, c] * 2.0 ** -s
        # every product is 0 * huge ...
        A[r, zero_a], B[zero_a, c] = 0.0, B[zero_a, c] * 2.0 ** s
        A[r, ~zero_a], B[~zero_a, c] = A[r, ~zero_a] * 2.0 ** s, 0.0
        # ... but, for half the targets, one tiny one
        if has_tiny:
            A[r, t], B[t, c] = x, y
    return A, B, rows, cols


def test_2b(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Products from about 2^-2s to 2^2s in every entry."""
    g = torch.Generator().manual_seed(SEED)
    x = 1 + torch.rand(n, generator=g)
    # from -s + 2s/n to s, truncated as GtB's conversion to int does
    e = torch.trunc((torch.arange(n) - n // 2 + 1) * (s / (n // 2)))
    e = e[torch.randperm(n, generator=g)]
    a0, b0 = torch.ldexp(x, e), torch.ldexp(x, -e)
    shift = (torch.arange(n) - torch.arange(n)[:, None]) % n  # (j - i) mod n
    A, B = a0[shift], b0[shift.T]
    g = torch.Generator().manual_seed(SEED)
    p1, p2, p3 = (torch.randperm(n, generator=g) for _ in range(3))
    return A[p1][:, p3], B[p3][:, p2]


def test_2c(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Test 2b's, a quarter of the values in a quarter of the rows and columns negated."""
    A, B = test_2b(n, s)
    g = torch.Generator().manual_seed(SEED)
    for i in _mask(n, n // 4, g).nonzero().flatten().tolist():
        A[i, _mask(n, n // 4, g)] *= -1
    for j in _mask(n, n // 4, g).nonzero().flatten().tolist():
        B[_mask(n, n // 4, g), j] *= -1
    S = torch.where(torch.randint(2, (n,), generator=g) == 0, -1.0, 1.0)
    return A * S, B * S[:, None]


@dataclass
class Case:
    """One test at one `n` and `s`: the inputs (FP32, values in the format), the
    entries GtB scores, their exact values, and the size of their products."""
    test: str
    n: int
    s: int
    A: torch.Tensor
    Bt: torch.Tensor
    I: torch.Tensor
    J: torch.Tensor
    exact: torch.Tensor
    size: torch.Tensor


def case(test: str, n: int, s: int, fmt) -> Case:
    """Test 1c or 2c at n and s, its matrices rounded to fmt."""
    rnd = quant.Operand(fmt)
    if test == '1c':
        A, B, I, J = test_1c(n, s)
        A, B = quant.quantize(A, rnd).elements, quant.quantize(B, rnd).elements
    else:
        A, B = (quant.quantize(X, rnd).elements for X in test_2c(n, s))
        # GtB scores only the entries with no negative product
        mixed = (A < 0).double() @ (B > 0).double() + (A > 0).double() @ (B < 0).double()
        I, J = (mixed == 0).nonzero().unbind(1)
    # in FP64, off by at most n 2^-53 of |A| |B|: far below any design's error
    exact = (A.double() @ B.double())[I, J]
    size = (A.double().abs() @ B.double().abs())[I, J]
    if test == '1c':
        # below FP32's normals, even a correctly rounded result has a large relative error
        keep = (exact == 0) | (exact.abs() >= FP32_UN)
        I, J, exact, size = I[keep], J[keep], exact[keep], size[keep]
    return Case(test, n, s, A, B.T.contiguous(), I, J, exact, size)


def error(model: dict, c: Case) -> float:
    """model's worst componentwise backward error on c; an entry whose
    products are all 0 must come out exactly 0."""
    gemm = getattr(torch.ops.fpy2_models, f'{op(model)}_gemm')
    got = gemm(c.A, c.Bt, c.I, c.J, *unit_scales(model, c.n)).double()
    err = (got - c.exact).abs()
    return torch.where(c.size > 0, err / c.size, torch.where(err == 0, 0.0, math.inf)).max().item()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="the outputs' root (default: %(default)s)")
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                    help='processes compiling the designs, and threads grading them '
                    '(default: every CPU)')
    args = ap.parse_args(argv)
    # the threads `<op>_gemm` grades on
    torch.set_num_threads(args.jobs)

    pts = sweep.grid()
    designs = [FMA, *(p.design for p in pts)]
    # each stage's wall time
    wall = {}
    t0 = time.perf_counter()
    compiled = dict(zip(designs, mm.in_processes(to_cpp, designs, args.jobs)))
    wall['fpy'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cxx = build([c for c in compiled.values() if 'src' in c], args.results / 'blas' / 'build', args.jobs)
    wall['cxx'] = time.perf_counter() - t0
    models = {d: c['model'] for d, c in compiled.items() if 'src' in c}
    t0 = time.perf_counter()

    rows = []
    for f, (fmt, ss) in FORMATS.items():
        cases = [case(test, N, s, fmt) for test in ('1c', '2c') for s in ss]
        fma = [error(models[FMA], c) for c in cases]
        mine = [p for p in pts if p.design.a == fmt]
        for p in mine:
            cpp = compiled[p.design]
            point = {'anchor': p.anchor, 'format': f, 'value': p.value, 'is_anchor': p.is_anchor,
                     'status': 'refused' if 'reason' in cpp else 'ok',
                     'reason': cpp.get('reason', ''), 'design': generator.name(p.design),
                     **sweep._about(p.design), 'fpy_seconds': cpp['seconds']}
            if 'reason' in cpp:
                rows.append(point)
                continue
            point['cxx_seconds'] = cxx[op(cpp['model'])]
            t1 = time.perf_counter()
            errs = [error(models[p.design], c) for c in cases]
            point['seconds'] = time.perf_counter() - t1
            rows += [point | {'test': c.test, 'n': c.n, 's': c.s, 'error': e, 'fma_error': b}
                     for c, e, b in zip(cases, errs, fma)]
            print(f'{p.anchor} {p.value}: worst {max(errs):.3g} in {point["seconds"]:.1f}s', flush=True)

    wall['grade'] = time.perf_counter() - t0

    table = args.results / 'blas' / 'blas-grid.csv'
    outputs.write_csv(table, rows)
    outputs.write_json(table.with_suffix('.provenance.json'), {
        'argv': sys.argv, **outputs.commit(), 'machine': platform.platform(),
        'cxx': subprocess.run([os.environ.get('CXX', 'c++'), '--version'], capture_output=True,
                              text=True, check=True).stdout.splitlines()[0],
        'settings': {k: v if isinstance(v, int) else str(v) for k, v in vars(args).items()},
        'seconds': wall | {'total': sum(wall.values())}})
    print(f'{len(pts)} points, {sum(r["status"] == "refused" for r in rows)} refused: {table}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
