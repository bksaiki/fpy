"""
The options the scripts share.
"""

import argparse
from typing import get_args

from . import kernels, quant, swap


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


def runs(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """`args.runs` checked against `args.scheme`'s (all but R0 if not
    given)."""
    known = swap.modes(args.scheme)[1:]
    if bad := [r for r in args.runs or () if r not in known]:
        ap.error(f'{args.scheme.name} has no run {", ".join(bad)} (only {", ".join(known)})')
    return args.runs or list(known)
