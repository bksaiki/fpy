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
    ap.add_argument('--model', default=swap.MODEL)
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=get_args(kernels.Combine), default='linear')
    if seed:
        ap.add_argument('--seed', type=int, default=0, help='seed of the random subsets')


def add_scheme_args(ap: argparse.ArgumentParser) -> None:
    """The options choosing a scheme and a model's weights under it."""
    ap.add_argument('--scheme', type=quant.scheme, default=swap.BF16,
                    help=f'one of {", ".join(quant.SCHEMES)}, or fp8-row:fnuz, fp8-block:fnuz')
    ap.add_argument('--requantize', action='store_true',
                    help='allow a checkpoint in another scheme to be requantized, lossily')
    ap.add_argument('--master', default=None, help='the unquantized model a checkpoint is '
                    'measured against (default: its model card\'s base model)')


def add_runs(ap: argparse.ArgumentParser) -> None:
    """`-r`, the runs to compare with R0 (checked by :func:`runs`)."""
    ap.add_argument('-r', '--runs', nargs='*',
                    help="runs besides fp32 (default: the scheme's exact run and every design)")


def runs(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """`args.runs` checked against `args.scheme`'s (all but R0 if not
    given)."""
    known = swap.modes(args.scheme)[1:]
    if bad := [r for r in args.runs or () if r not in known]:
        ap.error(f'{args.scheme.name} has no run {", ".join(bad)} (only {", ".join(known)})')
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

