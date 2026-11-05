"""
MMA-Sim as a linear-layer backend: the same designs as FPy's kernels,
computed by MMA-Sim's PyTorch models.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from core import swap
from mmasim import MMAOperation
from mmasim.arithmetic import fdpa

OPERATIONS: dict[str, Callable[[], MMAOperation]] = {
    'nv.hopper.bf16.f32': lambda: fdpa.MMA_T_FDPA(F=25, rho='RZ-FP32', L_max=16, e_zero=-133),
    'amd.cdna3.bf16': lambda: fdpa.MMA_TR_FDPA(F=24, F2=31, rho='RNE-FP32', L_max=8),
}
"""Each supported design's MMA-Sim operation, with the parameters
`examples/mmasim/tests` checks it against."""

PREFIX = 'mmasim:'

_ELEMS = 1 << 24
"""Elements of `[rows, n, L]` per block of rows at most: bounds MMA-Sim's
intermediates, about 50 bytes each at peak."""


def operation(design: str) -> MMAOperation:
    """*design*'s MMA-Sim operation."""
    if design not in OPERATIONS:
        raise ValueError(f'{design!r} is not one of {tuple(OPERATIONS)}')
    return OPERATIONS[design]()


def matmul(a: torch.Tensor, w: torch.Tensor, design: str) -> torch.Tensor:
    """`a @ w.T` by *design*'s MMA-Sim operation, FP32 `[m, n]`, for BF16
    *a* `[m, k]` and *w* `[n, k]`, accumulating from `C = 0` along `k`."""
    op = operation(design)
    (m, _), (n, _) = a.shape, w.shape
    rows = max(1, _ELEMS // (n * op.L_max))
    b = w.T
    return torch.cat([op(a[i:i + rows], b, torch.zeros(min(rows, m - i), n, device=a.device))
                      for i in range(0, m, rows)])


@dataclass
class Run(swap.Run):
    """:class:`swap.Run` with modes `mmasim:<design>` (:data:`OPERATIONS`)
    under the `bf16` scheme."""

    def _matmul(self, qa, qw, w):
        if not self.mode.startswith(PREFIX):
            return super()._matmul(qa, qw, w)
        if self.scheme.name != swap.BF16.name:
            raise ValueError(f'{self.mode} takes the bf16 scheme, not {self.scheme.name}')
        return matmul(qa.elements.to(torch.bfloat16), self._prepare(w, qw, torch.bfloat16),
                      self.mode.removeprefix(PREFIX))


def _extend(run: swap.Run) -> Run:
    run.__class__ = Run   # the patched layers hold this instance
    return run


def patch(model: torch.nn.Module) -> Run:
    """:func:`swap.patch`, with a :class:`Run` that has the `mmasim` modes."""
    return _extend(swap.patch(model))


def load(name: str = swap.MODEL) -> tuple[torch.nn.Module, Run]:
    """:func:`swap.load`, with a :class:`Run` that has the `mmasim` modes."""
    model, run = swap.load(name)
    return model, _extend(run)
