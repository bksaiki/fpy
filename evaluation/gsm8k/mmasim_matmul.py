"""
MMA-Sim as a linear-layer backend: each design FPy's kernels compute, by
MMA-Sim's PyTorch model of it, built from the design's catalog parameters
as `examples/mmasim/tests` builds them.
"""

from dataclasses import dataclass

import generator
import torch
from core import kernels, quant, swap
from generator.catalog import CATALOG
from mmasim import MMABlockScaleOperation, MMAOperation
from mmasim.arithmetic import fdpa, fma, ftz_mul_add

import fpy2 as fp

REFUSED = ('amd.cdna1.bf16', 'amd.cdna1.f16')
"""The designs FPy's backends refuse: exact sums of 280-525 bits."""

DESIGNS = tuple(d for d in CATALOG if d not in REFUSED)
"""The designs both backends compute."""

PREFIX = 'mmasim:'

_ELEMS = 1 << 24
"""Elements of `[rows, cols, span]` per call at most: bounds MMA-Sim's
intermediates, about 50 bytes each at peak."""

_RHO = {(32, fp.RM.RTZ): 'RZ-FP32', (22, fp.RM.RTZ): 'RZ-E8M13', (32, fp.RM.RNE): 'RNE-FP32',
        (16, fp.RM.RNE): 'RNE-FP16'}
"""MMA-Sim's name of each output conversion, by width and rounding."""

DTYPES = quant.DTYPES | {fp.MX_E2M1: torch.float16, fp.MX_E8M0: torch.float8_e8m0fnu}
"""Each format's torch dtype as MMA-Sim takes it: E2M1 values in FP16, as
its tests pass them."""


def operation(design: str) -> MMAOperation | MMABlockScaleOperation:
    """*design*'s MMA-Sim operation."""
    d = CATALOG[design]
    if d.sum == 'fma':
        return fma.MMA_FMA()
    if d.sum == 'tree':
        return ftz_mul_add.MMA_FTZ_MUL_ADD(d.L)
    if d.sum == 'exact':
        return fdpa.MMA_E_FDPA(d.L)
    rho = _RHO[d.rho.nbits, d.rho.rm]
    if d.join == 'separate':
        op = fdpa.MMA_GTR_FDPA if d.groups == 'even-odd' else fdpa.MMA_TR_FDPA
        return op(F=d.F, F2=d.F2, rho=rho, L_max=d.L)
    e_zero = generator.zero_exponent(d)
    if d.G is not None:
        return fdpa.MMA_GST_FDPA(d.G, d.F, rho, d.L, e_zero)
    if d.scales is not None:
        return fdpa.MMA_ST_FDPA(d.F, rho, d.L, e_zero)
    return fdpa.MMA_T_FDPA(d.F, rho, d.L, e_zero)


def output(design: str) -> torch.dtype:
    """The dtype *design*'s operation returns."""
    d = CATALOG[design]
    return {16: torch.float16, 64: torch.float64}.get(d.rho.nbits, torch.float32)


def _span(design: str, k: int) -> int:
    """Elements along `k` MMA-Sim's operation holds per output at once:
    one FMA's, one instruction's, or all `k` (CDNA2's products, and TF32's
    truncation, are taken whole)."""
    d = CATALOG[design]
    if d.sum == 'tree' or d.a == fp.TF32:
        return k
    return 1 if d.sum == 'fma' else d.L


def matmul(design: str, a: torch.Tensor, w: torch.Tensor,
           scales: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """`a @ w.T` by *design*'s MMA-Sim operation from `C = 0`, `[m, n]` in
    its :func:`output` dtype, for *a* `[m, k]` and *w* `[n, k]` in
    :data:`DTYPES`; a block-scaled design takes *scales* `[m, k / b]` and
    `[n, k / b]`, one per block `b` of its scheme.  Computed in blocks of
    rows and columns where memory requires."""
    op = operation(design)
    (m, k), n = a.shape, w.shape[0]
    span = _span(design, k)
    cols = min(n, max(1, _ELEMS // span))
    rows = min(m, max(1, _ELEMS // (cols * span)))
    sa, sw = scales or (None, None)
    if rows == m and cols == n:
        return _call(op, design, a, w, sa, sw)
    y = torch.empty(m, n, dtype=output(design), device=a.device)
    for i in range(0, m, rows):
        for j in range(0, n, cols):
            y[i:i + rows, j:j + cols] = _call(op, design, a[i:i + rows], w[j:j + cols],
                                              None if sa is None else sa[i:i + rows],
                                              None if sw is None else sw[j:j + cols])
    return y


def _call(op: MMAOperation | MMABlockScaleOperation, design: str, a: torch.Tensor,
          w: torch.Tensor, sa: torch.Tensor | None, sw: torch.Tensor | None) -> torch.Tensor:
    c = torch.zeros(a.shape[0], w.shape[0], dtype=output(design), device=a.device)
    b = w.T
    if isinstance(op, fdpa.MMA_ST_FDPA):   # one scale per instruction, chained here
        L = op.L_max
        for t in range(0, a.shape[1], L):
            j = t // L
            c = op(a[:, t:t + L], b[t:t + L], c, sa[:, j:j + 1], sw[:, j:j + 1].T)
        return c
    if isinstance(op, fdpa.MMA_GST_FDPA):
        return op(a, b, c, sa, sw.T)
    return op(a, b, c)


def accumulate(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized,
               b: torch.Tensor | None = None) -> torch.Tensor:
    """`qa @ qw.T` by *design*'s MMA-Sim operation, as :func:`swap.accumulate`
    by its kernel: block scales applied by its instructions, *scheme*'s others
    not yet.  *b* is *qw*'s elements in :data:`DTYPES`, if converted already."""
    a = qa.elements.to(DTYPES[scheme.x.elements])
    if b is None:
        b = qw.elements.to(DTYPES[scheme.w.elements])
    scales = None
    if scheme.applied == 'instruction':
        dtype = DTYPES[scheme.x.scaling.fmt]
        scales = (qa.scales.to(dtype), qw.scales.to(dtype))
    return matmul(design, a, b, scales)


@dataclass
class Run(swap.Run):
    """:class:`swap.Run` with modes `mmasim:<design>` (:data:`DESIGNS`),
    under the design's scheme."""

    calls: list[tuple[quant.Quantized, quant.Quantized, torch.Tensor, torch.Tensor]] | None = None
    """While a list, each `mmasim:` call's operands and accumulator, appended."""

    def _matmul(self, qa, qw, w):
        if not self.mode.startswith(PREFIX):
            return super()._matmul(qa, qw, w)
        design = self.mode.removeprefix(PREFIX)
        if design not in DESIGNS or not kernels.applicable(design, self.scheme):
            raise ValueError(f'{self.mode} does not take the {self.scheme.name} scheme')
        acc = accumulate(design, self.scheme, qa, qw, self._held(w, qw))
        if self.calls is not None:
            self.calls.append((qa, qw, w, acc.clone()))
        return swap.epilogue(acc, self.scheme, qa, qw)

    def _held(self, w: torch.Tensor, qw: quant.Quantized) -> torch.Tensor:
        """*qw*'s elements in MMA-Sim's dtype, converted once per weight and
        kept apart from the kernel's (`_prepare`), so neither evicts the other."""
        dtype = DTYPES[self.scheme.w.elements]
        held = self.__dict__.setdefault('_mmasim_held', {})
        got = held.get(id(w))
        if got is None or got[0] is not qw or got[1] != dtype:
            got = held[id(w)] = (qw, dtype, qw.elements.to(dtype))
        return got[2]

    def check(self) -> dict[str, int]:
        """:attr:`calls`, recomputed by FPy's kernel for the same design and
        compared accumulator to accumulator, then emptied: the calls, their
        outputs, and the outputs that differ in any bit (every NaN equal)."""
        design = self.mode.removeprefix(PREFIX)
        held = kernels.storage(design)[1]
        out = {'calls': 0, 'outputs': 0, 'differing': 0}
        for qa, qw, w, y in self.calls or []:
            z = swap.accumulate(design, self.scheme, qa, qw, self._prepare(w, qw, held))
            out['calls'] += 1
            out['outputs'] += y.numel()
            out['differing'] += int((~same(y, z)).sum())
        if self.calls:
            self.calls.clear()
        return out


_BITS = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def same(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Where *y* and *z* are the same bits, or both NaN; of one dtype."""
    if y.dtype != z.dtype:
        raise ValueError(f'{y.dtype} and {z.dtype}')
    return (y.view(_BITS[y.dtype]) == z.view(_BITS[z.dtype])) | (y.isnan() & z.isnan())


def _extend(run: swap.Run) -> Run:
    run.__class__ = Run   # the patched layers hold this instance
    return run


def patch(model: torch.nn.Module) -> Run:
    """:func:`swap.patch`, with a :class:`Run` that has the `mmasim` modes."""
    return _extend(swap.patch(model))


def load(name: str) -> tuple[torch.nn.Module, Run]:
    """:func:`swap.load`, with a :class:`Run` that has the `mmasim` modes."""
    model, run = swap.load(name)
    return model, _extend(run)
