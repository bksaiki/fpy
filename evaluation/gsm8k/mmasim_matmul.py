"""
MMA-Sim as a linear-layer backend: each design FPy's kernels compute, by
MMA-Sim's PyTorch model of it, built from the design's catalog parameters
as `examples/mmasim/tests` builds them.
"""

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import generator
import mmasim_nosync
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

EAGER_BUDGET = 3 << 30
"""Bytes of intermediates (:func:`_bytes`) one MMA-Sim call without a graph
may hold: fixed, so a call splits the same however much memory is free.
Every decode layer but `lm_head` fits in one block."""

_BUDGET: int | None = None
"""Bytes per call instead, if set (tests)."""

TUNED = False
"""Whether MMA-Sim is the tuned baseline (:func:`tune`)."""

_REPLAYING = False
"""Whether calls replay their shape's graph (:func:`replaying`)."""

_GRAPHS: dict[tuple, tuple[torch.cuda.CUDAGraph, list[torch.Tensor], torch.Tensor]] = {}
"""Each call shape's captured graph, its input buffers, and its output."""

GRAPH_LIMIT = 2 << 30
"""A call shape gets a graph only if its intermediates (:func:`_bytes`) fit
in this: a graph keeps them reserved, out of the model's reach."""

GRAPHLESS: set[tuple] = set()
"""The call shapes without a graph, over :data:`GRAPH_LIMIT` or out of
memory capturing: these run without one (still without host waits)."""

_POOL: tuple | None = None
"""The memory pool every graph shares: they never run at once, and each
output is copied out of it."""


def tune(on: bool = True) -> None:
    """Make MMA-Sim the tuned baseline, or as released: tuned, its
    arithmetic never waits for the GPU (`mmasim_nosync`), and in decode
    (:func:`replaying`) each call shape is captured once as a CUDA graph and
    replayed, as LLM servers do.  Bit for bit the same."""
    global TUNED
    mmasim_nosync.install(on)
    TUNED = on
    GRAPHLESS.clear()
    release_graphs()


@contextmanager
def replaying() -> Iterator[None]:
    """Within, a tuned MMA-Sim replays each call shape's graph, captured on
    first use: for decode, whose shapes repeat every step.  A graph holds
    60-440 MiB beyond PyTorch's allocator, so prefill's one-off shapes run
    without one."""
    global _REPLAYING
    old, _REPLAYING = _REPLAYING, True
    try:
        yield
    finally:
        _REPLAYING = old


def release_graphs() -> None:
    """Drop every captured graph and its memory."""
    global _POOL
    _GRAPHS.clear()
    _POOL = None
    torch.cuda.empty_cache()

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
    dtypes = [DTYPES[scheme.x.elements], DTYPES[scheme.w.elements]]
    if scheme.applied == 'instruction':
        ins += [qa.scales, qw.scales]
        dtypes += [DTYPES[scheme.x.scaling.fmt]] * 2
    return ins, dtypes


def _key(design: str, ins: list[torch.Tensor], dtypes: list[torch.dtype]) -> tuple:
    return (design, *((tuple(x.shape), str(d)) for x, d in zip(ins, dtypes, strict=True)))


def graphed(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized) -> bool:
    """Whether this call's shape has a captured graph now."""
    return _key(design, *_operands(scheme, qa, qw)) in _GRAPHS


def _replay(design: str, ins: list[torch.Tensor], dtypes: list[torch.dtype]) -> torch.Tensor:
    """*design*'s call on *ins* (operands, then any scales) converted to
    *dtypes*, by its shape's graph, captured on first use: the operands
    copied in, the output copied out.  A shape whose graph does not fit runs
    without one (:data:`GRAPHLESS`)."""
    key = _key(design, ins, dtypes)
    (m, k), n = ins[0].shape, ins[1].shape[0]
    if m * n * _bytes(design, k) > GRAPH_LIMIT:
        GRAPHLESS.add(key)
    if key not in _GRAPHS and key not in GRAPHLESS:
        try:
            _GRAPHS[key] = _capture(design, ins, dtypes)
        except (torch.OutOfMemoryError, torch.AcceleratorError) as ex:
            if 'out of memory' not in str(ex):
                raise
            GRAPHLESS.add(key)   # the other graphs stay: dropping them re-captures each, mid-decode
            torch.cuda.empty_cache()
    if key in GRAPHLESS:
        a, b, *scales = (x.to(d) for x, d in zip(ins, dtypes, strict=True))
        return matmul(design, a, b, tuple(scales) or None)
    graph, static, out = _GRAPHS[key]
    for buf, x in zip(static, ins, strict=True):
        buf.copy_(x)
    graph.replay()
    return out.clone()


def _capture(design: str, ins: list[torch.Tensor],
             dtypes: list[torch.dtype]) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor], torch.Tensor]:
    """*design*'s call on buffers like *ins* in *dtypes*, run once on a side
    stream, then captured: in one block, as a graphed call fits in
    :data:`GRAPH_LIMIT`."""
    global _POOL
    static = [x.to(d) for x, d in zip(ins, dtypes, strict=True)]
    scales = (static[2], static[3]) if len(static) == 4 else None
    budget = _BUDGET or GRAPH_LIMIT
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _blocks(design, static[0], static[1], scales, budget)
    torch.cuda.current_stream().wait_stream(side)
    _POOL = _POOL or torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=_POOL):
        out = _blocks(design, static[0], static[1], scales, budget)
    return graph, static, out


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


def accumulate(design: str, scheme: quant.Scheme, qa: quant.Quantized,
               qw: quant.Quantized) -> torch.Tensor:
    """`qa @ qw.T` by *design*'s MMA-Sim operation, as :func:`swap.accumulate`
    by its kernel: block scales applied by its instructions, *scheme*'s others
    not yet.  The operands are converted to :data:`DTYPES` per call, not kept:
    tuned and :func:`replaying`, by the copy into the graph's buffers."""
    ins, dtypes = _operands(scheme, qa, qw)
    if TUNED and _REPLAYING and qa.elements.is_cuda:
        return _replay(design, ins, dtypes)
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
        storage) and the last input, its graphs, and return the GPU memory:
        between designs, so they do not pile up."""
        self._weights.clear()
        self._prepared.clear()
        self._input = None
        release_graphs()
        gc.collect()
        torch.cuda.empty_cache()

    def drop_storage(self) -> None:
        """Free the kernel's copies of the weights, for MMA-Sim's graphs; the
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
            z = swap.accumulate(design, self.scheme, qa, qw, qw.elements.to(storage))   # uncached: room for graphs
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
