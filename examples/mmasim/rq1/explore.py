"""
Grade MMA designs, real and swept, with Grading the BLAS's Tests 1c and 2c.

    python examples/mmasim/rq1/explore.py                   # -> output/rq1/<format>.json
    python examples/mmasim/rq1/explore.py --formats E4M3 --jobs 32
"""

import argparse
import ctypes
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

MMASIM = Path(__file__).resolve().parents[1]
ROOT = MMASIM.parents[1]
sys.path.insert(0, str(MMASIM))

import compile as mm
import generator
from generator.catalog import CATALOG
from serve.core import quant

import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend.triton.normalize import normalize_module
from fpy2.transform.const_fold import ConstFold

OUT = ROOT / 'output' / 'rq1'
GEMM = Path(__file__).with_name('gemm.hpp')
CXX = os.environ.get('CXX', 'c++')
# designs per shared library: one C++ compile reads the headers once
BATCH = 50

# parallelism comes from processes and the GEMM loop's threads
torch.set_num_threads(1)

SEED = 0
NS = [32, 64, 128, 256, 512, 1024]
EPS = 2.0 ** -24
FP32_UN = 2.0 ** -126

FORMATS = {
    # name: (context, the sweep's anchor, s)
    'BF16': (fp.BF16, 'nv.hopper.bf16.f32', [0, 11, 23, 34, 46, 57]),
    'FP16': (fp.FP16, 'nv.hopper.f16.f32', [0, 3, 5, 8, 10, 13]),
    'E4M3': (fp.MX_E4M3, 'nv.hopper.e4m3.f32', list(range(7))),
    'E5M2': (fp.MX_E5M2, 'nv.hopper.e5m2.f32', [0, 3, 5, 8, 10, 13]),
}
SWEEP = {
    'F': [8, 10, 13, 16, 19, 22, 25, 28, 32, 35],
    'L': [2, 4, 8, 16, 32],
    'rho_bits': [7, 10, 13, 16, 19, 23],
    'rho_mode': ['RTZ', 'RNE', 'RTN', 'RTP'],
}
BASELINE = 'fp32 (fma)'


def designs() -> list[tuple[str, dict, generator.Design]]:
    """Every (format, tag, design) graded: the sweep, the catalog's designs
    for the format, and the FP32 FMA loop."""
    out = []
    for fmt, (ctx, anchor, _) in FORMATS.items():
        for F, L, bits, mode in itertools.product(*SWEEP.values()):
            rho = fp.IEEEContext(8, 9 + bits, fp.RoundingMode[mode])  # FP32's exponent bits
            tag = {'run': 'sweep', 'F': F, 'L': L, 'rho_bits': bits, 'rho_mode': mode}
            out.append((fmt, tag, replace(CATALOG[anchor], F=F, L=L, rho=rho)))
        for name, d in CATALOG.items():
            if d.a == d.b == ctx and d.c == fp.FP32 and d.scales is None:
                out.append((fmt, {'run': 'real', 'catalog': name}, d))
        out.append((fmt, {'run': 'baseline'}, CATALOG[BASELINE]))
    return out


def to_cpp(d: generator.Design) -> dict:
    """*d* in C++, or why FPy refused it, and how long that took."""
    t0 = time.perf_counter()
    try:
        func, arg_types = generator.build(d)
        mod = fp.Module()
        mod.add(st.monomorphize(func, args=arg_types))
        # inlined and folded first: the C++ backend refuses the generator's helpers otherwise
        mod = normalize_module(mod).map(lambda _, f: ConstFold.apply(f)).map(mm._prepare)
        src = fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.ROUNDINGS).compile_module(mod)
        out = {'src': src, 'entry': func.name}
    except Exception as e:
        out = {'reason': f'{type(e).__name__}: {e}'.splitlines()[0][:300]}
    return out | {'secs': time.perf_counter() - t0}


def _cxx() -> tuple[str, list[str]]:
    """The C++ compiler's version, and the command that builds a library."""
    version = subprocess.run([CXX, '--version'], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    # the generated code sets rounding modes with `fesetround`
    strict = (['-ffp-model=strict'] if 'clang' in version
              else ['-frounding-math', '-ffp-contract=off'])
    # every library reuses the names `d0`, `d1`, ...: keep them private
    return version, [CXX, '-std=c++17', '-O2', *strict, '-fvisibility=hidden', '-shared',
                     '-fPIC', '-w']


_ENTRY = r'''
extern "C" __attribute__((visibility("default"))) void gemm_@I@(
        const double* A, const double* Bt, long K, const long* I, const long* J,
        double* out, long P, int threads) {
    gemm::dots(&d@I@::@ENTRY@, A, Bt, K, I, J, out, P, threads);
}
'''


def build_lib(cmd: list[str], tmp: Path, batch: list[dict]) -> Path:
    """One shared library for *batch* (`to_cpp` results), design `i` as `gemm_<i>`."""
    parts = [mm._translation_unit(''), GEMM.read_text()]
    for i, c in enumerate(batch):
        parts.append(f'namespace d{i} {{\n{c["src"]}\n}}')
        parts.append(_ENTRY.replace('@I@', str(i)).replace('@ENTRY@', c['entry']))
    cpp = Path(tempfile.mkstemp(suffix='.cpp', dir=tmp)[1])
    cpp.write_text('\n'.join(parts))
    subprocess.run([*cmd, '-o', str(cpp.with_suffix('.so')), str(cpp)], check=True)
    return cpp.with_suffix('.so')


def _gemm(lib: ctypes.CDLL, i: int):
    fn = getattr(lib, f'gemm_{i}')
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int]
    fn.restype = None
    return fn


def _mask(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    """*k* of *n* positions, at random."""
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.randperm(n, generator=g)[:k]] = True
    return m


# Tests 1c and 2c, as GtB's `include/init_matrices.hpp` builds them. GtB
# seeds a fresh generator for each step, so every s draws the same values.

def test_1c(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """*A*, *B*, and the rows and columns of the target entries."""
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
    """One test at one `n` and `s`: the inputs (FP64, values in the format), the
    entries scored, their exact values, and the size of their products."""
    test: str
    n: int
    s: int
    A: torch.Tensor
    Bt: torch.Tensor
    I: torch.Tensor
    J: torch.Tensor
    exact: torch.Tensor
    size: torch.Tensor


def _exact(A: torch.Tensor, Bt: torch.Tensor, I: torch.Tensor,
           J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Entries (`I[p]`, `J[p]`) of `A @ Bt.T`, exactly, and `sum |a_t b_t|` of each."""
    exact = torch.empty(len(I), dtype=torch.float64)
    size = torch.empty(len(I), dtype=torch.float64)
    lo = 0
    for i, count in zip(*(t.tolist() for t in torch.unique_consecutive(I, return_counts=True))):
        P = Bt[J[lo:lo + count]] * A[i]  # products of narrow values: exact in FP64
        # fsum rounds once: off by at most 2^-53 * size, under 2^-29 in f_A
        exact[lo:lo + count] = torch.tensor([math.fsum(p) for p in P.tolist()],
                                            dtype=torch.float64)
        size[lo:lo + count] = P.abs().sum(1)
        lo += count
    return exact, size


def _quantize(X: torch.Tensor, fmt: str) -> torch.Tensor:
    """*X* rounded to *fmt* (to nearest, ties to even), held in FP64."""
    q = quant.quantize(X, quant.Operand(FORMATS[fmt][0])).elements.double()
    # `FORMATS`' s takes every N(0, 1) draw below 6
    assert q.isfinite().all()
    return q


def build_case(job: tuple[str, str, int, int]) -> Case:
    """The `Case` of *job*: (format, test, `n`, `s`)."""
    fmt, test, n, s = job
    if test == '1c':
        A, B, I, J = test_1c(n, s)
        A, B = _quantize(A, fmt), _quantize(B, fmt)
    else:
        A, B = (_quantize(X, fmt) for X in test_2c(n, s))
        # GtB scores only the entries with no negative product
        mixed = (A < 0).double() @ (B > 0).double() + (A > 0).double() @ (B < 0).double()
        I, J = (mixed == 0).nonzero().unbind(1)
    Bt = B.T.contiguous()
    exact, size = _exact(A, Bt, I, J)
    if test == '1c':
        # below FP32's normals, even a correctly rounded result misses grade A
        keep = (exact == 0) | (exact.abs() >= FP32_UN)
        I, J, exact, size = I[keep], J[keep], exact[keep], size[keep]
    return Case(test, n, s, A, Bt, I.contiguous(), J.contiguous(), exact, size)


def f_a(got: torch.Tensor, c: Case) -> float:
    """GtB's `f_A`: the worst error, in units of `EPS` times the entry's size.
    An entry whose products are all 0 must come out exactly 0."""
    err = (got - c.exact).abs()
    r = torch.where(c.size > 0, err / c.size, torch.where(err == 0, 0.0, math.inf))
    return r.amax().item() / EPS


def grade(gemm, cases: list[Case], threads: int) -> dict:
    """`f_A` of design *gemm* on each of *cases*, by test, `n` and `s`."""
    fA: dict = {}
    for c in cases:
        got = torch.empty(len(c.I), dtype=torch.float64)
        gemm(c.A.data_ptr(), c.Bt.data_ptr(), c.n, c.I.data_ptr(), c.J.data_ptr(),
             got.data_ptr(), len(c.I), threads)
        fA.setdefault(c.test, {}).setdefault(c.n, {})[c.s] = f_a(got, c)
    return fA


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[1])
    ap.add_argument('--formats', nargs='*', choices=list(FORMATS), default=list(FORMATS),
                    help='input formats to grade (default: all)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1,
                    help='processes and threads (default: every CPU)')
    args = ap.parse_args(argv)
    cxx, cmd = _cxx()
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, check=True).stdout.strip()

    rows = [r for r in designs() if r[0] in args.formats]
    names = [generator.name(d) for _, _, d in rows]
    todo = dict(zip(names, (d for _, _, d in rows)))
    print(f'compiling {len(todo)} designs', flush=True)
    t0 = time.perf_counter()
    compiled = dict(zip(todo, mm.in_processes(to_cpp, todo.values(), args.jobs)))
    t_compile = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.jobs) as pool:
        t0 = time.perf_counter()
        ok = [n for n, c in compiled.items() if 'src' in c]
        batches = [ok[i:i + BATCH] for i in range(0, len(ok), BATCH)]
        libs = pool.map(partial(build_lib, cmd, Path(tmp)),
                        [[compiled[n] for n in b] for b in batches])
        gemms = {n: _gemm(lib, i) for lib, b in zip(map(ctypes.CDLL, libs), batches)
                 for i, n in enumerate(b)}
        t_build = time.perf_counter() - t0

        for fmt in args.formats:
            t0 = time.perf_counter()
            jobs = [(fmt, test, n, s) for test in ('1c', '2c') for n in NS for s in FORMATS[fmt][2]]
            cases = list(mm.in_processes(build_case, jobs, args.jobs))
            t_cases = time.perf_counter() - t0

            t0 = time.perf_counter()
            graded = {}
            for n in dict.fromkeys(n for n, r in zip(names, rows) if r[0] == fmt and n in gemms):
                t1 = time.perf_counter()
                graded[n] = (grade(gemms[n], cases, args.jobs), time.perf_counter() - t1)
                if len(graded) % 100 == 0:
                    print(f'[{fmt}] graded {len(graded)}', flush=True)

            out = {'format': fmt, 'commit': commit, 'machine': platform.platform(), 'cxx': cxx,
                   'cpus': os.cpu_count(), 'jobs': args.jobs,
                   'seconds': {'compile, all formats': t_compile, 'build, all formats': t_build,
                               'cases': t_cases, 'grade': time.perf_counter() - t0},
                   'designs': [{'name': n, 'tag': tag, 'about': generator.about(d),
                                'refused': compiled[n].get('reason'),
                                'compile_secs': compiled[n]['secs'],
                                'grade_secs': graded[n][1] if n in graded else None,
                                'fA': graded[n][0] if n in graded else None}
                               for n, (f, tag, d) in zip(names, rows) if f == fmt]}
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f'{fmt}.json').write_text(json.dumps(out, indent=1))
            print(f'[{fmt}] done -> {OUT / f"{fmt}.json"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
