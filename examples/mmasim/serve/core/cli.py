"""
The options the scripts share, and what they do with them alike.
"""

import argparse
import json
from pathlib import Path
from typing import Any, get_args

import torch

from . import checkpoints, kernels, quant, swap


def add_args(ap: argparse.ArgumentParser, seed: bool = True) -> None:
    """The options every script shares: the model, how designs split `k`,
    and (with *seed*) the seed of its random subsets (`workloads.pick`)."""
    ap.add_argument('--model', default=swap.MODEL,
                    help='the model, a master or a checkpoint (default: %(default)s)')
    ap.add_argument('--list-models', action='store_true',
                    help='list the models these scripts have run, and exit')
    ap.add_argument('--split-k', type=int, default=1,
                    help='slices of `k` each design runs from a zero accumulator '
                    '(default: %(default)s)')
    ap.add_argument('--combine', choices=get_args(kernels.Combine), default='linear',
                    help='how the slices sum in FP32: left to right, or pairwise '
                    '(default: %(default)s)')
    if seed:
        ap.add_argument('--seed', type=int, default=0,
                        help='seed of the random subsets (default: %(default)s)')


def add_scheme_args(ap: argparse.ArgumentParser) -> None:
    """The options choosing a scheme and a model's weights under it."""
    ap.add_argument('--scheme', type=quant.scheme, default=swap.BF16,
                    help=f'one of {", ".join(quant.SCHEMES)}, or fp8-row:fnuz, fp8-block:fnuz '
                    f'(default: {swap.BF16.name})')
    ap.add_argument('--list-schemes', action='store_true', help='list the schemes, and exit')
    ap.add_argument('--list-matmuls', action='store_true',
                    help="list the scheme's matmuls, and exit")
    ap.add_argument('--requantize', action='store_true',
                    help='allow a checkpoint in another scheme to be requantized, lossily')
    ap.add_argument('--baseline', dest='master', default=None,
                    help='the unquantized model a checkpoint is measured against '
                    '(default: its model card\'s base model)')


def add_runs(ap: argparse.ArgumentParser) -> None:
    """`--matmuls`, the runs to compare with R0 (checked by :func:`runs`)."""
    ap.add_argument('--matmuls', dest='runs', nargs='*', metavar='MATMUL',
                    help="matmuls besides fp32 (default: the scheme's exact matmul and "
                    'every design)')


def parse(ap: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """*argv* parsed, or what a `--list-*` option asks for printed, and exit."""
    args = ap.parse_args(argv)
    lines: list[str] = []
    if getattr(args, 'list_models', False):
        lines = [f'{m:34} {what}' for m, what in swap.MODELS.items()]
    elif getattr(args, 'list_schemes', False):
        names = [*quant.SCHEMES, 'fp8-row:fnuz', 'fp8-block:fnuz']
        lines = [f'{n:16} {quant.describe(quant.scheme(n))}' for n in names]
    elif getattr(args, 'list_matmuls', False):
        fp32, exact, *designs = swap.modes(args.scheme)
        lines = [f'{fp32:24} R0, the model as trained',
                 f"{exact:24} R1, the quantized operands' exact product",
                 *(f'{d:24} a design' for d in designs)]
    if lines:
        print('\n'.join(lines))
        ap.exit()
    return args


def runs(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """`args.runs` checked against `args.scheme`'s (all but R0 if not
    given)."""
    known = swap.modes(args.scheme)[1:]
    if bad := [r for r in args.runs or () if r not in known]:
        ap.error(f'{args.scheme.name} has no matmul {", ".join(bad)} (only {", ".join(known)})')
    return args.runs or list(known)


def load(args: argparse.Namespace, name: str | None = None,
         ) -> tuple[torch.nn.Module, swap.Run, dict[str, Any]]:
    """*name* (default `--model`) under *args*' scheme and splits:
    `checkpoints.for_scheme`'s."""
    return checkpoints.for_scheme(name or args.model, args.scheme, requantize=args.requantize,
                                  master=args.master, split_k=args.split_k, combine=args.combine)


def cached(path: Path, settings: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON at *path*, if it was made with *settings*."""
    if path.exists() and (got := json.loads(path.read_text()))['settings'] == settings:
        return got
    return None

