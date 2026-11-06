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

_SHARE = 0.6
"""Of the GPU's free memory, and PyTorch's cached but unused, the share one
MMA-Sim call's intermediates may take (:func:`_bytes`)."""

_BUDGET: int | None = None
"""Bytes per call instead, if set (tests)."""

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


def _bytes(design: str, k: int) -> int:
    """Bytes MMA-Sim's operation holds per output at its peak, as measured on
    the GPU: about 50 per element of an instruction (16 for GST's), 40 per
    FMA; CDNA2's products and TF32's truncation take all `k`, at about 20
    and 12 per element."""
    d = CATALOG[design]
    if d.sum == 'fma':
        return 40
    if d.sum == 'tree':
        return 20 * k
    return (16 if d.G is not None else 50) * d.L + (12 * k if d.a == fp.TF32 else 0)


def matmul(design: str, a: torch.Tensor, w: torch.Tensor,
           scales: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """`a @ w.T` by *design*'s MMA-Sim operation from `C = 0`, `[m, n]` in
    its :func:`output` dtype, for *a* `[m, k]` and *w* `[n, k]` in
    :data:`DTYPES`; a block-scaled design takes *scales* `[m, k / b]` and
    `[n, k / b]`, one per block `b` of its scheme.  Computed in blocks of
    rows and columns as large as free memory allows (:data:`_SHARE`):
    MMA-Sim is bound by launches, so fewer blocks are faster.  Out of
    memory, it retries in blocks half as large."""
    budget = _BUDGET or _free(a.device)
    while True:
        try:
            return _blocks(design, a, w, scales, budget)
        except torch.OutOfMemoryError:
            if budget < 1 << 20:
                raise
            torch.cuda.empty_cache()
            budget //= 2


def _free(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return int(_SHARE * (free + cached))


def _blocks(design: str, a: torch.Tensor, w: torch.Tensor,
            scales: tuple[torch.Tensor, torch.Tensor] | None, budget: int) -> torch.Tensor:
    op = operation(design)
    (m, k), n = a.shape, w.shape[0]
    per = _bytes(design, k)
    cols = min(n, max(1, budget // per))
    rows = min(m, max(1, budget // (cols * per)))
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

    calls: list[tuple[quant.Quantized, quant.Quantized, torch.Tensor, torch.Tensor | None]] | None = None
    """While a list, each design's call appended: its operands, and an
    `mmasim:` call's accumulator (`None` for FPy's kernel)."""

    def _matmul(self, qa, qw, w):
        if not self.mode.startswith(PREFIX):
            y = super()._matmul(qa, qw, w)
            if self.calls is not None and kernels.known(self.mode):
                self.calls.append((qa, qw, w, None))
            return y
        design = self.mode.removeprefix(PREFIX)
        if design not in DESIGNS or not kernels.applicable(design, self.scheme):
            raise ValueError(f'{self.mode} does not take the {self.scheme.name} scheme')
        acc = accumulate(design, self.scheme, qa, qw, self.held(w, qw))
        if self.calls is not None:
            self.calls.append((qa, qw, w, acc.clone()))
        return swap.epilogue(acc, self.scheme, qa, qw)

    def held(self, w: torch.Tensor, qw: quant.Quantized) -> torch.Tensor:
        """*qw*'s elements in MMA-Sim's dtype, converted once per weight and
        kept apart from the kernel's (`_prepare`), so neither evicts the other."""
        dtype = DTYPES[self.scheme.w.elements]
        held = self.__dict__.setdefault('_mmasim_held', {})
        got = held.get(id(w))
        if got is None or got[0] is not qw or got[1] != dtype:
            got = held[id(w)] = (qw, dtype, qw.elements.to(dtype))
        return got[2]

    def check(self) -> dict[str, int]:
        """:attr:`calls`' MMA-Sim accumulators, recomputed by FPy's kernel for
        the same design and compared, then emptied: the calls, their outputs,
        the outputs that differ in any bit (every NaN equal), and the finite
        ones."""
        design = self.mode.removeprefix(PREFIX)
        held = kernels.storage(design)[1]
        out = {'calls': 0, 'outputs': 0, 'differing': 0, 'finite': 0}
        for qa, qw, w, y in self.calls or []:
            z = swap.accumulate(design, self.scheme, qa, qw, self._prepare(w, qw, held))
            out['calls'] += 1
            out['outputs'] += y.numel()
            out['differing'] += int((~same(y, z)).sum())
            out['finite'] += int(y.isfinite().sum())
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
