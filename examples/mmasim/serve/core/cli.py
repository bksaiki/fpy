"""
The options the scripts share, and what they do with them alike.
"""

import argparse
import hashlib
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import kernels, outputs, quant, swap

RESULTS = Path('results')
"""The outputs' root, from `examples/mmasim` (gitignored): `cache/` for
artifacts reused across runs, `plots/` for plots, a run's outputs in its own
directory."""


def add_args(ap: argparse.ArgumentParser, seed: bool = True) -> None:
    """The options every script shares: the model, designs to register, the
    outputs' root, and (with *seed*) the seed of its random subsets
    (`workloads.pick`)."""
    ap.add_argument('--model', choices=swap.MODELS, default=swap.MODEL,
                    help='the model, from the Hub, in BF16 (default: %(default)s)')
    ap.add_argument('--register', type=Path, action='append', default=[], metavar='FILE',
                    help='a Python file whose register() adds designs (kernels.register) '
                    'before anything runs; repeatable (default: none)')
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help='the outputs\' root: <results>/cache, <results>/plots (default: %(default)s)')
    if seed:
        ap.add_argument('--seed', type=int, default=0,
                        help='seed of the random subsets (default: %(default)s)')


def add_scheme_args(ap: argparse.ArgumentParser) -> None:
    """The options choosing a scheme, and listing the schemes and a scheme's
    matmuls."""
    ap.add_argument('--scheme', type=quant.scheme, default=swap.BF16,
                    help=f'one of {", ".join(quant.NAMES)} (default: {swap.BF16.name})')
    ap.add_argument('--list-schemes', action='store_true', help='list the schemes, and exit')
    ap.add_argument('--list-matmuls', action='store_true',
                    help="list the scheme's matmuls, and exit")


def add_runs(ap: argparse.ArgumentParser) -> None:
    """`--matmuls`, the runs to compare with R0 (checked by :func:`runs`)."""
    ap.add_argument('--matmuls', dest='runs', nargs='*', metavar='MATMUL',
                    help="matmuls besides fp32 (default: the scheme's exact matmul and "
                    'every design)')


def parse(ap: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """*argv* parsed, its `--register` files' designs added and kernels kept
    under the cache; or what a `--list-*` option asks for printed, and exit."""
    args = ap.parse_args(argv)
    for path in args.register:
        runpy.run_path(str(path))['register']()
    kernels.CACHE = cache(args, 'kernels')
    lines: list[str] = []
    if args.list_schemes:
        lines = [f'{n:16} {quant.describe(quant.scheme(n))}' for n in quant.NAMES]
    elif args.list_matmuls:
        lines = matmuls(args.scheme)
    if lines:
        print('\n'.join(lines))
        ap.exit()
    return args


def matmuls(scheme: quant.Scheme) -> list[str]:
    """*scheme*'s matmuls, a line each with what it is."""
    fp32, exact, rand, *designs = swap.modes(scheme)
    return [f'{fp32:24} R0, the model as trained',
            f"{exact:24} R1, the quantized operands' exact product",
            f'{rand:24} R1 moved an ulp up or down at random',
            *(f'{d:24} a design' for d in designs)]


def runs(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """`args.runs` checked against `args.scheme`'s (all but R0 if not
    given)."""
    known = swap.modes(args.scheme)[1:]
    if bad := [r for r in args.runs or () if r not in known]:
        ap.error(f'{args.scheme.name} has no matmul {", ".join(bad)} (only {", ".join(known)})')
    return args.runs or list(known)


def load(args: argparse.Namespace, designs: Sequence[str] = (),
         jobs: int | None = None) -> tuple[torch.nn.Module, swap.Run]:
    """`--model` under *args*' scheme (`swap.load`).  Under a block-scaled
    scheme, the fused kernels of the *designs* to run compile now, in *jobs*
    processes, over each `k` of the layers a design computes."""
    model, run = swap.load(args.model, args.scheme)
    designs = [d for d in designs if d in kernels.TILES]
    if designs and args.scheme.applied == 'instruction':
        kernels.precompile(designs, jobs, layer_ks(model, run))
    return model, run


def layer_ks(model: torch.nn.Module, run: swap.Run) -> list[int]:
    """The `k` of each linear layer of *model* that *run* computes by a design."""
    return sorted({m.in_features for m in model.modules()
                   if isinstance(m, torch.nn.Linear) and id(m.weight) not in run.ignore})


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    """What a result comes from: the command line, the commit, the
    `--register` files (by digest) and every registered design's
    description."""
    return {'argv': sys.argv, **outputs.commit(),
            'register': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.register},
            'registered': kernels.REGISTERED}


def cache(args: argparse.Namespace, *parts: str) -> Path:
    """*parts* under the outputs' cache, `<results>/cache`."""
    return args.results.joinpath('cache', *parts)

