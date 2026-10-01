"""
Local per-layer error: a linear layer's FP32 output `Ŷ` against the exact
product `Y` (FP64) of its quantized inputs `x` and weights `w`, elementwise
over every token.

- `normwise`: normwise relative error `||Ŷ - Y||_F / ||Y||_F`;
- `backward`: log2 of the componentwise backward error `|ŷ - y| / (|x|ᵀ|w|)`
  per element (Oettli-Prager; Higham, ch. 7), its mean over the elements it
  is not 0 for, its max, its 50th, 90th and 99th percentiles
  (`backward_p50`, ...), and the share it is 0 for (`backward_exact`); and
  normwise, `||ŷ - y||_F / |||x|ᵀ|w|||_F` (`backward_normwise`), in log2;
- `ulp`: `log2(1 + |ŷ - y| / ulp(y))` per element in FP32 ("bits of error"),
  mean and max;
- `rounded`: the fraction of elements equal to `fl(y)`, `y` rounded to FP32;
- `bias`: mean error `(ŷ - y) / (|x|ᵀ|w|)`, the drift about zero;
- `magnitude_bias`: mean error in magnitude, `sign(y) (ŷ - y) / (|x|ᵀ|w|)`,
  negative when errors lean toward zero;
- `flips`: the share of elements whose code on the scheme's activation grid
  differs between `ŷ` and `fl(y)`, each quantized as the next layer's input
  would be (its own scales, over the output block's columns: whole rows for
  these models; a scale a hair apart moves no code), over the blocks whose
  columns are whole scaling blocks (NaN if none);
- `quantization`: normwise relative error of `Y` itself against the exact
  product `Y_0` of the unquantized inputs, the quantization's own error.

With *s0*, :func:`local` also measures against `Y_0`.  Printed errors are in
log2, biases in units of u = 2^-24.
"""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, fields

import torch

from . import quant

METRICS = ('normwise', 'backward', 'ulp', 'rounded', 'bias', 'magnitude_bias', 'flips', 'quantization')
QUANTIZED_ONLY = ('rounded', 'flips', 'quantization')
"""Metrics taken against the quantized operands' product only."""
U = 2.0 ** -24
"""FP32's unit roundoff."""
_BINS, _LO, _HI = 1280, -160.0, 0.0
"""The histogram of log2 backward error: 1/8-bit bins over [-160, 0), outside
it clamped."""
PERCENTILES = (50, 90, 99)

_ELEMS = 1 << 24
"""Elements per block (weight columns, output rows) when comparing."""

@dataclass
class Stats:
    """One layer's sums over its output elements (or several layers')."""

    n: int = 0
    err: float = 0.0
    """`||Ŷ - Y||_F^2`"""
    ref: float = 0.0
    """`||Y||_F^2`"""
    log_backward_err: float = 0.0
    """The sum of the nonzero backward errors' log2"""
    log_backward_err_max: float = -math.inf
    backward_nonzero: int = 0
    log_backward_err_hist: torch.Tensor = field(default_factory=lambda: torch.zeros(_BINS, dtype=torch.long))
    backward_sq: float = 0.0
    """`||ŷ - y||_F^2` where `|x|ᵀ|w| > 0`"""
    scale_sq: float = 0.0
    """`|||x|ᵀ|w|||_F^2`"""
    flips: int = 0
    flips_n: int = 0
    """The elements `flips` measured"""
    ulp: float = 0.0
    ulp_max: float = 0.0
    rounded: int = 0
    bias: float = 0.0
    magnitude_bias: float = 0.0
    q_err: float = 0.0
    """`||Y - Y_0||_F^2`, `Y_0` the unquantized operands' exact product"""
    q_ref: float = 0.0
    """`||Y_0||_F^2`"""

    def __add__(self, other: 'Stats') -> 'Stats':
        """Pooled with *other*: sums added, maxima the larger."""
        return Stats(**{f.name: (max if f.name.endswith('_max') else sum)(
            (getattr(self, f.name), getattr(other, f.name))) for f in fields(self)})

    def report(self, metrics: Collection[str]) -> dict[str, float]:
        """Each of *metrics*' statistics, not log2 (but `backward`'s)."""
        n = max(self.n, 1)
        out = {
            'normwise': math.sqrt(self.err / self.ref) if self.ref else 0.0,
            'backward': self.log_backward_err / max(self.backward_nonzero, 1),
            'backward_max': self.log_backward_err_max,
            'backward_exact': 1 - self.backward_nonzero / n,
            **{f'backward_p{p}': _percentile(self.log_backward_err_hist, p) for p in PERCENTILES},
            'backward_normwise': 0.5 * math.log2(self.backward_sq / self.scale_sq)
            if self.backward_sq and self.scale_sq else -math.inf,
            'flips': self.flips / self.flips_n if self.flips_n else math.nan,
            'ulp': self.ulp / n, 'ulp_max': self.ulp_max,
            'rounded': self.rounded / n, 'bias': self.bias / n,
            'magnitude_bias': self.magnitude_bias / n,
            'quantization': math.sqrt(self.q_err / self.q_ref) if self.q_ref else 0.0,
        }
        return {k: v for k, v in out.items()
                if k.removesuffix('_max') in metrics or k.startswith('backward_') and 'backward' in metrics}


def _percentile(hist: torch.Tensor, p: float) -> float:
    """The *p*th percentile of the log2 values *hist* counts: its bin's middle."""
    total = int(hist.sum())
    if not total:
        return -math.inf
    i = int(torch.searchsorted(hist.cumsum(0), math.ceil(total * p / 100)))
    return _LO + (i + 0.5) * (_HI - _LO) / _BINS


def _rows(n: int) -> int:
    return max(1, _ELEMS // n)


@dataclass
class _Exact:
    """An output block's exact product `y` (FP64), and what the metrics
    derive from it alone."""

    y: torch.Tensor
    scale: torch.Tensor | None
    """`|x|ᵀ|w|`, if a metric needs it."""
    ulp: torch.Tensor | None
    """`y`'s FP32 ulp, if a metric needs it."""
    rounded: torch.Tensor | None
    """`fl(y)`, `y` rounded to FP32, if a metric needs it."""
    ref: float = 0.0
    """`||y||_F^2`, if `normwise` is measured."""
    sign: torch.Tensor | None = None
    """`sign(y)`, if `magnitude_bias` is measured."""
    positive: torch.Tensor | None = None
    """Where `scale > 0`, if `scale` is."""
    op: quant.Operand | None = None
    """The scheme's activation operand, if `flips` is measured."""
    grid: torch.Tensor | None = None
    """`fl(y)`'s codes on *op*'s grid, if `flips` is measured."""


def _exact(metrics: Collection[str], a: torch.Tensor, wt: torch.Tensor,
           wa: torch.Tensor | None, op: quant.Operand | None = None) -> _Exact:
    """`a @ wt`'s :class:`_Exact` for *metrics*; *wa* is `|wt|`, or `None`
    if unneeded; *op*, the scheme's activation operand, for `flips`."""
    y = a @ wt
    ulp = None
    if 'ulp' in metrics:
        # |y| in [2^(ex-1), 2^ex): its FP32 ulp is 2^(ex-24), at least 2^-149
        _, ex = torch.frexp(y)
        ex = torch.where(y == 0, -125, ex)
        ulp = torch.ldexp(torch.ones_like(y), (ex - 24).clamp(min=-149))
    scale = None if wa is None else a.abs() @ wa
    return _Exact(y, scale, ulp, y.float() if 'rounded' in metrics else None,
                  float((y * y).sum()) if 'normwise' in metrics else 0.0,
                  torch.sign(y) if 'magnitude_bias' in metrics else None,
                  None if scale is None else scale > 0,
                  *((op, quant.quantize(y.float(), op).elements) if 'flips' in metrics and op
                    and (op.scaling is None or y.shape[1] % (op.scaling.cols or 1) == 0) else ()))


def _against(s: Stats, metrics: Collection[str], r: _Exact, g: torch.Tensor) -> None:
    """Add output block *g*'s *metrics* against *r* to *s*."""
    e = g.double() - r.y
    s.n += e.numel()
    if 'normwise' in metrics:
        s.err += float((e * e).sum())
        s.ref += r.ref
    if r.scale is not None and r.positive is not None:
        eta = torch.where(r.positive, e / r.scale, 0.0)
        if 'backward' in metrics:
            s.backward_sq += float((torch.where(r.positive, e, 0.0) ** 2).sum())
            s.scale_sq += float((r.scale * r.scale).sum())
            size = eta.abs()
            log = torch.log2(size[size > 0])
            if log.numel():
                s.log_backward_err += float(log.sum())
                s.log_backward_err_max = max(s.log_backward_err_max, float(log.max()))
                s.backward_nonzero += log.numel()
                clamped = log.clamp(_LO, _HI - (_HI - _LO) / _BINS / 2).float()
                s.log_backward_err_hist += torch.histc(clamped, _BINS, _LO, _HI).long().cpu()
            del size, log
        if 'bias' in metrics:
            s.bias += float(eta.sum())
        if r.sign is not None:
            s.magnitude_bias += float((r.sign * eta).sum())
        del eta
    if r.ulp is not None:
        bits = torch.log2(1 + e.abs() / r.ulp)
        s.ulp += float(bits.sum())
        s.ulp_max = max(s.ulp_max, float(bits.max()))
        del bits
    if r.rounded is not None:
        s.rounded += int((g == r.rounded).sum())
    if r.grid is not None and r.op is not None:
        s.flips += int((quant.quantize(g, r.op).elements != r.grid).sum())
        s.flips_n += g.numel()


def local(outs: Sequence[tuple[Stats, torch.Tensor, Stats | None]], metrics: Collection[str],
          qa: quant.Quantized, qw: quant.Quantized,
          unquantized: tuple[torch.Tensor, torch.Tensor]) -> None:
    """For each of *outs*, `(s, got, s0)`, add *got* `[m, n]`, an output on
    *qa* and *qw*, to *s*'s *metrics*, in blocks, every output against the
    same exact products.  `quantization` compares their exact product with
    the *unquantized* operands'; *s0*, if given, takes the metrics (but
    :data:`QUANTIZED_ONLY`) against the latter."""
    x0, w0 = unquantized
    m, k = qa.elements.shape
    scaled = bool({'backward', 'bias', 'magnitude_bias'} & set(metrics))
    against0 = any(s0 is not None for _, _, s0 in outs)
    before = against0 or 'quantization' in metrics
    metrics0 = set(metrics) - set(QUANTIZED_ONLY)
    cols = _rows(k)
    for j in range(0, outs[0][1].shape[1], cols):
        wt = qw.dequantize(slice(j, j + cols)).T
        wa = wt.abs() if scaled else None
        w0t = w0[j:j + cols].double().T if before else None
        w0a = w0t.abs() if against0 and scaled else None
        rows = _rows(wt.shape[1])
        for i in range(0, m, rows):
            r = _exact(metrics, qa.dequantize(slice(i, i + rows)), wt, wa, qa.operand)
            r0 = None
            if w0t is not None:
                a0 = x0[i:i + rows].double()
                r0 = _exact(metrics0, a0, w0t, w0a) if against0 else _Exact(a0 @ w0t, None, None, None)
            q = None
            if r0 is not None and 'quantization' in metrics:
                q = float(((r.y - r0.y) ** 2).sum()), float((r0.y * r0.y).sum())
            for s, got, s0 in outs:
                g = got[i:i + rows, j:j + cols]
                _against(s, metrics, r, g)
                if s0 is not None and r0 is not None:
                    _against(s0, metrics0, r0, g)
                if q is not None:
                    s.q_err += q[0]
                    s.q_ref += q[1]


def fmt(key: str, v: float) -> str:
    """*v* as printed: errors in log2, `rounded` as a percentage, biases in u."""
    if key in ('rounded', 'backward_exact', 'flips'):
        return f'{v:.2%}'
    if key.endswith('bias'):
        return f'{v / U:+.3f}'
    if key.startswith(('ulp', 'backward')):
        return f'{v:.3f}'
    return f'{math.log2(v):.2f}' if v > 0 else '-inf'
