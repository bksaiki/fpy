"""
Compiles every MMA-Sim design to C++, reporting where each one stops.

A roadmap tracker rather than a test: most designs do not compile yet, and
the point is to see *which* refusal each one hits and how the count moves.

    python examples/mmasim/compile.py           # one line per design
    python examples/mmasim/compile.py -v        # full error text
    python examples/mmasim/compile.py -e cdna2  # print the C++ of a design
    python examples/mmasim/compile.py -o out/   # write each one to out/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import amd, nv
from models.nv import RZ_E8M13, RZ_FP32
from models.utils import make_fma_dpa

import fpy2 as fp
import fpy2.strategies as st
from fpy2.strategies import TransformDeclined
from fpy2.transform import CompToLoop, RescaleFixed, Simplify
from fpy2.backend.cpp.utils import CPP_HEADERS, CPP_HELPERS

_L = fp.types.ListType
_R = fp.types.RealType


def _vecs(a_ctx, b_ctx, c_ctx, k):
    """The `(A, B, c)` argument types every model shares."""
    return [_L(_R(a_ctx), k), _L(_R(b_ctx), k), _R(c_ctx)]


# (name, builder) where the builder returns (function, argument types).
# `k` matches what `table8_cases` pads its vectors to.
DESIGNS = [
    ('nv.volta.f16.f32', lambda: (
        nv.make_t_fdpa_chain(4, fp.FP16, fp.FP16, fp.FP32, 23, RZ_FP32),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 8))),
    ('nv.turing.f16.f32', lambda: (
        nv.make_t_fdpa_chain(8, fp.FP16, fp.FP16, fp.FP32, 24, RZ_FP32),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 16))),
    ('nv.ampere.tf32.f32', lambda: (
        nv.make_t_fdpa_chain(4, fp.TF32, fp.TF32, fp.FP32, 24, RZ_FP32),
        _vecs(fp.TF32, fp.TF32, fp.FP32, 8))),
    ('nv.ampere.bf16.f32', lambda: (
        nv.make_t_fdpa_chain(8, fp.BF16, fp.BF16, fp.FP32, 24, RZ_FP32),
        _vecs(fp.BF16, fp.BF16, fp.FP32, 16))),
    ('nv.ada.e5m2.f32', lambda: (
        nv.make_t_fdpa_chain(16, fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 13,
                             RZ_E8M13, e_zero=-132),
        _vecs(fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 16))),
    ('nv.hopper.f16.f32', lambda: (
        nv.make_t_fdpa_chain(16, fp.FP16, fp.FP16, fp.FP32, 25, RZ_FP32,
                             is_mma=False),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 32))),
    ('nv.blackwell.mxfp8', lambda: (
        nv.make_st_fdpa(fp.MX_E5M2, fp.MX_E5M2, fp.MX_E8M0, 25, RZ_FP32),
        _vecs(fp.MX_E5M2, fp.MX_E5M2, fp.FP32, 32)
        + [_R(fp.MX_E8M0), _R(fp.MX_E8M0)])),
    ('nv.blackwell.nvfp4', lambda: (
        nv.make_gst_fdpa(16, fp.MX_E4M3, 35, RZ_FP32),
        _vecs(fp.MX_E2M1, fp.MX_E2M1, fp.FP32, 64)
        + [_L(_R(fp.MX_E4M3), 4), _L(_R(fp.MX_E4M3), 4)])),
    ('amd.cdna1.bf16', lambda: (
        amd.make_e_fdpa(2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna1.f16', lambda: (
        amd.make_e_fdpa(4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna2.bf16', lambda: (
        amd.make_ftz_addmul(fp.BF16, 2), _vecs(fp.BF16, fp.BF16, fp.FP32, 4))),
    ('amd.cdna2.f16', lambda: (
        amd.make_ftz_addmul(fp.FP16, 4), _vecs(fp.FP16, fp.FP16, fp.FP32, 4))),
    ('amd.cdna3.f16', lambda: (
        amd.make_tr_fdpa(8, fp.FP16, fp.FP16),
        _vecs(fp.FP16, fp.FP16, fp.FP32, 8))),
    ('amd.cdna3.bf16', lambda: (
        amd.make_tr_fdpa(8, fp.BF16, fp.BF16),
        _vecs(fp.BF16, fp.BF16, fp.FP32, 8))),
    ('amd.cdna3.bf8', lambda: (
        amd.make_gtr_fdpa(16, fp.S1E5M2, fp.S1E5M2),
        _vecs(fp.S1E5M2, fp.S1E5M2, fp.FP32, 16))),
    ('fp64 (fma)', lambda: (
        make_fma_dpa(fp.FP64), _vecs(fp.FP64, fp.FP64, fp.FP64, 4))),
]


def _prepare(_module, func):
    """Put one function in the shape the backend needs.

    `comp_to_loop` precedes `rescale_fixed`: the latter emits the scale-in and
    scale-out as statements, which a rounding inside a comprehension has no
    slot for.  A transform with nothing to do declines, which is not a
    failure.
    """
    for step in (CompToLoop.apply, RescaleFixed.apply, Simplify.apply):
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
    # where the C++ goes: stdout or files, not both
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument('-e', '--emit', action='store_true',
                      help='print the C++ of each design that compiles')
    dest.add_argument('-o', '--out', metavar='DIR', type=Path,
                      help='write each compiled design to DIR/<name>.cpp')
    args = ap.parse_args(argv)

    designs = [
        (name, build) for name, build in DESIGNS
        if not args.filter or any(f in name for f in args.filter)
    ]
    if not designs:
        ap.error(f'no design matches {args.filter}')
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    width = max(len(name) for name, _ in designs)
    ok = 0
    for name, build in designs:
        try:
            src = compile_design(build)
        except Exception as ex:  # noqa: BLE001 -- any refusal is a result
            detail = str(ex) if args.verbose else str(ex).split('\n')[0][:110]
            print(f'{name:{width}}  {type(ex).__name__}: {detail}')
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
    print(f'\n{ok}/{len(designs)} compile')
    return 0 if ok == len(designs) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
