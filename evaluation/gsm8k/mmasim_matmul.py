"""
MMA-Sim as a linear-layer backend: each design FPy's kernels compute, by
MMA-Sim's PyTorch model of it, built from the design's catalog parameters
as `examples/mmasim/tests` builds them.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass

import generator
import no_sync
import torch
import triton
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

EAGER_BUDGET = 3 << 30
"""Bytes of intermediates (:func:`_bytes`) one MMA-Sim call may hold: fixed,
so a call splits the same however much memory is free.
Every decode layer but `lm_head` fits in one block."""

_BUDGET: int | None = None
"""Bytes per call instead, if set (tests)."""

TUNED = False
"""Whether MMA-Sim is the tuned baseline (:func:`tune`)."""

_COMPILED: dict[str, Callable[..., torch.Tensor]] = {}
"""Each design's instruction (:func:`_step`), compiled (tuned)."""

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.fail_on_recompile_limit_hit = True   # never fall back to eager unseen
# compiled kernels round as eager does: no FMA contraction, flush to zero, or approximate division
torch._inductor.config.emulate_precision_casts = True
torch._inductor.config.eager_numerics.division_rounding = True
torch._inductor.config.eager_numerics.disable_ftz = True


def _fma_gpu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """MMA-Sim's `fma_gpu`, but IEEE: Triton's libdevice flushes FP32
    subnormals by default, MMA-Sim's CPU FMA does not."""
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    fma.fma_gpu_kernel[lambda meta: (triton.cdiv(n, meta['BLOCK']),)](
        a, b, c, out, n, BLOCK=1024, enable_reflect_ftz=False)
    return out


fma.fma_gpu = _fma_gpu   # released and tuned alike


def tune(on: bool = True) -> None:
    """Make MMA-Sim the tuned baseline, or as released: tuned, its
    arithmetic never waits for the GPU (`no_sync`) and runs under
    `torch.compile`, BF16 operands held in FP32 (Inductor compiles no BF16
    before Ampere; MMA-Sim treats both alike).  Bit for bit the same."""
    global TUNED
    no_sync.install(on)
    TUNED = on
    release_compiled()


def release_compiled() -> None:
    """Drop every compiled call: per design, as each adds to Dynamo's
    recompile count."""
    _COMPILED.clear()
    torch._dynamo.reset()


_RHO = {(32, fp.RM.RTZ): 'RZ-FP32', (22, fp.RM.RTZ): 'RZ-E8M13', (32, fp.RM.RNE): 'RNE-FP32',
        (16, fp.RM.RNE): 'RNE-FP16'}
"""MMA-Sim's name of each output conversion, by width and rounding."""

_PTX = {fp.FP16: 'f16', fp.BF16: 'bf16', fp.TF32: 'tf32', fp.MX_E4M3: 'e4m3', fp.MX_E5M2: 'e5m2',
        fp.MX_E3M2: 'e3m2', fp.MX_E2M3: 'e2m3', fp.MX_E2M1: 'e2m1'}
"""Each operand format's PTX type, which MMA-Sim's T- and ST-FDPA take."""

DTYPES = quant.DTYPES | {fp.MX_E3M2: torch.float16, fp.MX_E2M3: torch.float16,
                         fp.MX_E2M1: torch.float16, fp.MX_E8M0: torch.float8_e8m0fnu}
"""Each format's torch dtype as MMA-Sim takes it: FP6 and E2M1 values in
FP16, as its tests pass E2M1's; their PTX types give their exponent floors."""


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
        return fdpa.MMA_ST_FDPA(d.F, rho, d.L, e_zero, _PTX[d.a], _PTX[d.b])
    return fdpa.MMA_T_FDPA(d.F, rho, d.L, e_zero, _PTX[d.a], _PTX[d.b])


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
    rows and columns within :data:`EAGER_BUDGET`: MMA-Sim is bound by
    launches, so fewer blocks are faster.  Out of memory, it retries in
    blocks half as large."""
    budget = _BUDGET or EAGER_BUDGET
    while True:
        try:
            return _blocks(design, a, w, scales, budget)
        except torch.OutOfMemoryError:
            if budget < 1 << 20:
                raise
            torch.cuda.empty_cache()
            budget //= 2


def _operands(scheme: quant.Scheme, qa: quant.Quantized,
              qw: quant.Quantized) -> tuple[list[torch.Tensor], list[torch.dtype]]:
    """A call's operands, then any block scales, and the dtypes MMA-Sim takes
    them in."""
    ins = [qa.elements, qw.elements]
    dtypes = [_dtype(scheme.x.elements), _dtype(scheme.w.elements)]
    if scheme.applied == 'instruction':
        ins += [qa.scales, qw.scales]
        dtypes += [DTYPES[scheme.x.scaling.fmt]] * 2
    return ins, dtypes


def _dtype(fmt: fp.Context) -> torch.dtype:
    d = DTYPES[fmt]
    return torch.float32 if TUNED and d == torch.bfloat16 else d


def _blocks(design: str, a: torch.Tensor, w: torch.Tensor,
            scales: tuple[torch.Tensor, torch.Tensor] | None, budget: int) -> torch.Tensor:
    (m, k), n = a.shape, w.shape[0]
    per = _bytes(design, k)
    cols = min(n, max(1, budget // per))
    rows = min(m, max(1, budget // (cols * per)))
    sa, sw = scales or (None, None)
    if rows == m and cols == n:
        return _call(design, a, w, sa, sw)
    y = torch.empty(m, n, dtype=output(design), device=a.device)
    for i in range(0, m, rows):
        for j in range(0, n, cols):
            y[i:i + rows, j:j + cols] = _call(design, a[i:i + rows], w[j:j + cols],
                                              None if sa is None else sa[i:i + rows],
                                              None if sw is None else sw[j:j + cols])
    return y


def _width(op: MMAOperation | MMABlockScaleOperation) -> int:
    """Products per instruction of *op*."""
    if isinstance(op, fma.MMA_FMA):
        return 1
    if isinstance(op, ftz_mul_add.MMA_FTZ_MUL_ADD):
        return op.P
    return op.L_max


def _step(design: str) -> Callable[..., torch.Tensor]:
    """*design*'s operation on one instruction's operands, compiled: for its
    first shape, then for any `m` and `n` (Dynamo's automatic dynamic shapes;
    `dynamic=True` would make the instruction's width symbolic too, which
    miscompiles GTR-FDPA's even/odd split)."""
    if design not in _COMPILED:
        _COMPILED[design] = torch.compile(operation(design).__call__)
    return _COMPILED[design]


def _call(design: str, a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor | None,
          sw: torch.Tensor | None) -> torch.Tensor:
    """`a @ w.T` from `C = 0`, block scales *sa* `[m, k / b]` and *sw*
    `[n, k / b]` if any: tuned, an instruction at a time, as on the CPU
    (`cpu_performance`); as released, in one call, but ST-FDPA's one scale
    per instruction chained here."""
    op = operation(design)
    c = torch.zeros(a.shape[0], w.shape[0], dtype=output(design), device=a.device)
    b = w.T
    if not TUNED and not isinstance(op, fdpa.MMA_ST_FDPA):
        return op(a, b, c) if sa is None else op(a, b, c, sa, sw.T)
    step = _step(design) if TUNED else op
    k = a.shape[1]
    L = min(k, _width(op))
    block = None if sa is None else k // sa.shape[1]
    for t in range(0, k, L):
        if block is None:
            c = step(a[:, t:t + L], b[t:t + L], c)
        else:
            s = slice(t // block, (t + L) // block)
            c = step(a[:, t:t + L], b[t:t + L], c, sa[:, s], sw[:, s].T)
    return c


def accumulate(design: str, scheme: quant.Scheme, qa: quant.Quantized,
               qw: quant.Quantized) -> torch.Tensor:
    """`qa @ qw.T` by *design*'s MMA-Sim operation, as :func:`swap.accumulate`
    by its kernel: block scales applied by its instructions, *scheme*'s others
    not yet.  The operands are converted to :data:`DTYPES` per call, not kept."""
    ins, dtypes = _operands(scheme, qa, qw)
    a, b, *scales = (x.to(d) for x, d in zip(ins, dtypes, strict=True))
    return matmul(design, a, b, tuple(scales) or None)


@dataclass
class Run(swap.Run):
    """:class:`swap.Run` with modes `mmasim:<design>` (:data:`DESIGNS`),
    under the design's scheme."""

    calls: list[tuple[quant.Quantized, quant.Quantized, torch.Tensor, torch.Tensor | None]] | None = None
    """While a list, each design's call appended: its operands, and an
    `mmasim:` call's accumulator (`None` for FPy's kernel)."""

    def _matmul(self, qa: quant.Quantized, qw: quant.Quantized, w: torch.Tensor) -> torch.Tensor:
        if not self.mode.startswith(PREFIX):
            y = super()._matmul(qa, qw, w)
            if self.calls is not None and kernels.known(self.mode):
                self.calls.append((qa, qw, w, None))
            return y
        design = self.mode.removeprefix(PREFIX)
        if design not in DESIGNS or not kernels.applicable(design, self.scheme):
            raise ValueError(f'{self.mode} does not take the {self.scheme.name} scheme')
        acc = accumulate(design, self.scheme, qa, qw)
        if self.calls is not None:
            self.calls.append((qa, qw, w, acc.clone()))
        return swap.epilogue(acc, self.scheme, qa, qw)

    def release(self) -> None:
        """Drop every weight held for a design (quantized, in the kernel's
        storage) and the last input, and return the GPU memory: between
        designs, so they do not pile up."""
        self._weights.clear()
        self._prepared.clear()
        self._input = None
        release_compiled()
        gc.collect()
        torch.cuda.empty_cache()

    def drop_storage(self) -> None:
        """Free the kernel's copies of the weights, for MMA-Sim's intermediates; the
        check converts the few it needs."""
        self._prepared.clear()
        torch.cuda.empty_cache()

    def check(self) -> dict[str, int]:
        """:attr:`calls`' MMA-Sim accumulators, recomputed by FPy's kernel for
        the same design and compared, then emptied: the calls, their outputs,
        the outputs that differ in any bit (every NaN equal), and the finite
        ones."""
        design = self.mode.removeprefix(PREFIX)
        storage = kernels.storage(design)[1]
        out = {'calls': 0, 'outputs': 0, 'differing': 0, 'finite': 0}
        for qa, qw, _, y in self.calls or []:
            z = swap.accumulate(design, self.scheme, qa, qw, qw.elements.to(storage))   # uncached: room for MMA-Sim
            out['calls'] += 1
            out['outputs'] += y.numel()
            out['differing'] += int((~same(y, z)).sum())
            out['finite'] += int(y.isfinite().sum())
        if self.calls:
            self.calls.clear()
        return out


_BITS = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def same(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Where *y* and *z* are the same bits, both zero (FPy's semantics
    leaves zero unsigned), or both NaN; of one dtype."""
    if y.dtype != z.dtype:
        raise ValueError(f'{y.dtype} and {z.dtype}')
    return ((y.view(_BITS[y.dtype]) == z.view(_BITS[z.dtype])) | ((y == 0) & (z == 0))
            | (y.isnan() & z.isnan()))


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
