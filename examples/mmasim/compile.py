"""
Compiles every MMA-Sim design to C++, reporting where each one stops.

A roadmap tracker rather than a test: the point is to see *which* refusal
each design hits and how the count moves.

    python examples/mmasim/compile.py           # one line per design
    python examples/mmasim/compile.py -v        # full error text
    python examples/mmasim/compile.py -e cdna2  # print the C++ of a design
    python examples/mmasim/compile.py -o out/   # write each one to out/
    python examples/mmasim/compile.py -j 8      # in 8 processes
"""

import argparse
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from models import amd, nv
from models.nv import RNE_FP16, RZ_E8M13, RZ_FP32
from models.utils import make_fma_dpa

import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend.cpp.utils import CPP_HEADERS, CPP_HELPERS
from fpy2.strategies import TransformDeclined
from fpy2.transform import CompToLoop, RescaleFixed, Simplify, ZipElim
from fpy2.types import Type

_L = fp.types.ListType
_R = fp.types.RealType


def _vecs(a_ctx, b_ctx, c_ctx, k):
    """The `(A, B, c)` argument types every model shares."""
    return [_L(_R(a_ctx), k), _L(_R(b_ctx), k), _R(c_ctx)]


def _t_chain(
    L: int, a_ctx: fp.EFloatContext, b_ctx: fp.EFloatContext, c_ctx: fp.EFloatContext,
    F: int, rho: fp.Context, k: int, **kw: Any,
) -> Callable[[], tuple[fp.Function, list[Type]]]:
    """A T-FDPA chain's builder, over vectors of length *k*."""
    return lambda: (nv.make_t_fdpa_chain(L, a_ctx, b_ctx, c_ctx, F, rho, **kw),
                    _vecs(a_ctx, b_ctx, c_ctx, k))


# (tag, A format, B format): each unordered pair once; a swapped pair is the
# same design with its arguments swapped
_FP8_PAIRS = [
    ('e4m3', fp.MX_E4M3, fp.MX_E4M3),
    ('e5m2', fp.MX_E5M2, fp.MX_E5M2),
    ('e4m3.e5m2', fp.MX_E4M3, fp.MX_E5M2),
]
# FP6 is left out: MMA-Sim has no reference for it
_F8F6F4_PAIRS = _FP8_PAIRS + [
    ('e2m1', fp.MX_E2M1, fp.MX_E2M1),
    ('e4m3.e2m1', fp.MX_E4M3, fp.MX_E2M1),
    ('e5m2.e2m1', fp.MX_E5M2, fp.MX_E2M1),
]
_CDNA3_FP8_PAIRS = [
    ('fp8', fp.S1E4M3, fp.S1E4M3),
    ('bf8', fp.S1E5M2, fp.S1E5M2),
    ('fp8.bf8', fp.S1E4M3, fp.S1E5M2),
]

# (name, builder) where the builder returns (function, argument types): one
# per row of the paper's Tables 4-6, a row listing several input types once
# per pair.  A row that builds the same function as another is listed once,
# its other architectures noted.
DESIGNS = [
    ('nv.volta.f16.f32', _t_chain(4, fp.FP16, fp.FP16, fp.FP32, 23, RZ_FP32, 8)),
    ('nv.volta.f16.f16', _t_chain(4, fp.FP16, fp.FP16, fp.FP16, 23, RNE_FP16, 8)),
    # also Ampere, Ada
    ('nv.turing.f16.f32', _t_chain(8, fp.FP16, fp.FP16, fp.FP32, 24, RZ_FP32, 16)),
    ('nv.turing.f16.f16', _t_chain(8, fp.FP16, fp.FP16, fp.FP16, 24, RNE_FP16, 16)),
    # also Ada
    ('nv.ampere.tf32.f32', _t_chain(4, fp.TF32, fp.TF32, fp.FP32, 24, RZ_FP32, 8)),
    ('nv.ampere.bf16.f32', _t_chain(8, fp.BF16, fp.BF16, fp.FP32, 24, RZ_FP32, 16)),
    *[(f'nv.ada.{tag}.f32', _t_chain(16, a, b, fp.FP32, 13, RZ_E8M13, 16, e_zero=-132))
      for tag, a, b in _FP8_PAIRS],
    *[(f'nv.ada.{tag}.f16', _t_chain(16, a, b, fp.FP16, 13, RNE_FP16, 16, e_zero=-21))
      for tag, a, b in _FP8_PAIRS],
    # also Blackwell, RTX Blackwell
    ('nv.hopper.tf32.f32', _t_chain(8, fp.TF32, fp.TF32, fp.FP32, 25, RZ_FP32, 16)),
    ('nv.hopper.bf16.f32', _t_chain(16, fp.BF16, fp.BF16, fp.FP32, 25, RZ_FP32, 32)),
    ('nv.hopper.f16.f32', _t_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32, 32,
                                   is_mma=False)),
    # also Blackwell tcgen05
    ('nv.hopper.f16.f16.wgmma', _t_chain(16, fp.FP16, fp.FP16, fp.FP16, 25, RNE_FP16, 32,
                                         is_mma=False)),
    # also Blackwell, RTX Blackwell
    ('nv.hopper.f16.f16.mma', _t_chain(16, fp.FP16, fp.FP16, fp.FP16, 25, RNE_FP16, 32)),
    *[(f'nv.hopper.{tag}.f32', _t_chain(32, a, b, fp.FP32, 13, RZ_E8M13, 32, e_zero=-133))
      for tag, a, b in _FP8_PAIRS],
    *[(f'nv.hopper.{tag}.f16', _t_chain(32, a, b, fp.FP16, 13, RNE_FP16, 32, e_zero=-133))
      for tag, a, b in _FP8_PAIRS],
    # Blackwell tcgen05, also RTX Blackwell mma
    *[(f'nv.blackwell.{tag}.f32', _t_chain(32, a, b, fp.FP32, 25, RZ_FP32, 32))
      for tag, a, b in _F8F6F4_PAIRS],
    *[(f'nv.blackwell.{tag}.f16.tcgen05', _t_chain(32, a, b, fp.FP16, 25, RNE_FP16, 32,
                                                   is_mma=False))
      for tag, a, b in _F8F6F4_PAIRS],
    *[(f'nv.rtx_blackwell.{tag}.f16.mma', _t_chain(32, a, b, fp.FP16, 25, RNE_FP16, 32))
      for tag, a, b in _F8F6F4_PAIRS],
    # also RTX Blackwell
    *[(f'nv.blackwell.mx.{tag}', lambda a=a, b=b: (
        nv.make_st_fdpa(a, b, fp.MX_E8M0, 25, RZ_FP32),
        _vecs(a, b, fp.FP32, 32) + [_R(fp.MX_E8M0), _R(fp.MX_E8M0)]))
      for tag, a, b in _F8F6F4_PAIRS],
    *[(f'nv.blackwell.{tag}', lambda s=s: (
        nv.make_gst_fdpa(16, s, 35, RZ_FP32),
        _vecs(fp.MX_E2M1, fp.MX_E2M1, fp.FP32, 64) + [_L(_R(s), 4), _L(_R(s), 4)]))
      for tag, s in [('nvfp4', fp.MX_E4M3), ('mxfp4', fp.MX_E8M0)]],
    ('amd.cdna1.bf16', lambda: (
        amd.make_e_fdpa(2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna1.f16', lambda: (
        amd.make_e_fdpa(4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna2.bf16', lambda: (
        amd.make_ftz_addmul(fp.BF16, 2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna2.bf16_1k', lambda: (
        amd.make_ftz_addmul(fp.BF16, 4), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna2.f16', lambda: (
        amd.make_ftz_addmul(fp.FP16, 4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna3.tf32', lambda: (
        amd.make_tr_fdpa(4, fp.TF32, fp.TF32),
        _vecs(fp.TF32, fp.TF32, fp.FP32, 8))),
    ('amd.cdna3.f16', lambda: (
        amd.make_tr_fdpa(8, fp.FP16, fp.FP16),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 8))),
    ('amd.cdna3.bf16', lambda: (
        amd.make_tr_fdpa(8, fp.BF16, fp.BF16),
        _vecs(fp.BF16, fp.BF16, fp.FP32, 8))),
    *[(f'amd.cdna3.{tag}', lambda a=a, b=b: (
        amd.make_gtr_fdpa(16, a, b), _vecs(a, b, fp.FP32, 16)))
      for tag, a, b in _CDNA3_FP8_PAIRS],
    # also NVIDIA DMMA, CDNA2/3 FP64
    ('fp64 (fma)', lambda: (
        make_fma_dpa(fp.FP64), _vecs(fp.FP64, fp.FP64, fp.FP64, 4))),
    # CDNA1-3 FP32
    ('fp32 (fma)', lambda: (
        make_fma_dpa(fp.FP32), _vecs(fp.FP32, fp.FP32, fp.FP32, 4))),
]


def _prepare(_module, func):
    """Put one function in the shape the backend needs.

    `zip_elim` precedes `comp_to_loop`, as it does in the backend: lowered
    first, a `zip` becomes a list of tuples, which no analysis reads through.
    `comp_to_loop` precedes `rescale_fixed`: the latter emits the scale-in and
    scale-out as statements, which a rounding inside a comprehension has no
    slot for.  A transform with nothing to do declines, which is not a
    failure.
    """
    for step in (ZipElim.apply, CompToLoop.apply, RescaleFixed.apply, Simplify.apply):
        try:
            func = step(func)
        except TransformDeclined:
            pass
    return func


def compile_design(build) -> str:
    """The C++ for one design; raises whatever refused it.

    Every function is prepared, not just the entry: a model's rounding at a
    run-time position usually sits in a helper, and a transform applied to
    the caller alone never reaches it.
    """
    func, arg_types = build()
    mod = fp.Module()
    mod.add(st.monomorphize(func, args=arg_types))
    mod = mod.map(_prepare)
    return fp.CppCompiler(unfold=fp.CppCompiler.UnfoldMode.ROUNDINGS).compile_module(mod)


def in_processes(fn: Callable, items: Iterable, jobs: int) -> Iterator:
    """*fn* of each of *items*, in order: in *jobs* processes if more than one.
    *fn* and its results must pickle."""
    if jobs <= 1:
        yield from map(fn, items)
        return
    pool = ProcessPoolExecutor(jobs)
    try:
        yield from pool.map(fn, items)
    finally:
        pool.shutdown(cancel_futures=True)


def _compile_named(name: str) -> tuple[str | None, str, str]:
    """The C++ for design *name*, or `None` and the refusal's type and text."""
    try:
        return compile_design(dict(DESIGNS)[name]), '', ''
    except Exception as ex:  # noqa: BLE001 -- any refusal is a result
        return None, type(ex).__name__, str(ex)


def _translation_unit(src: str) -> str:
    """*src* with the headers it needs, so the file builds on its own."""
    return '\n'.join(CPP_HEADERS) + '\n' + CPP_HELPERS + '\n' + src


def _filename(name: str) -> str:
    """*name* as a filename: `fp64 (fma)` -> `fp64_fma.cpp`."""
    keep = [c if (c.isalnum() or c in '._') else '_' for c in name]
    return ''.join(keep).strip('_').replace('__', '_') + '.cpp'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('filter', nargs='*', metavar='SUBSTRING',
                    help='only designs whose name contains one of these')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='full error text for a design that does not compile')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='compile in this many processes (default 1)')
    # where the C++ goes: stdout or files, not both
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument('-e', '--emit', action='store_true',
                      help='print the C++ of each design that compiles')
    dest.add_argument('-o', '--out', metavar='DIR', type=Path,
                      help='write each compiled design to DIR/<name>.cpp')
    args = ap.parse_args(argv)

    names = [
        name for name, _ in DESIGNS
        if not args.filter or any(f in name for f in args.filter)
    ]
    if not names:
        ap.error(f'no design matches {args.filter}')
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    width = max(len(name) for name in names)
    ok = 0
    for name, (src, kind, why) in zip(names, in_processes(_compile_named, names, args.jobs)):
        if src is None:
            detail = why if args.verbose else why.split('\n')[0][:110]
            print(f'{name:{width}}  {kind}: {detail}')
            continue
        ok += 1
        note = ''
        if args.out is not None:
            path = args.out / _filename(name)
            path.write_text(_translation_unit(src))
            note = f'  -> {path}'
        print(f'{name:{width}}  OK{note}')
        if args.emit:
            print(f'\n// ==== {name} ====\n{_translation_unit(src)}\n')
    print(f'\n{ok}/{len(names)} compile')
    return 0 if ok == len(names) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
