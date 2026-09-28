"""
Local per-layer error: a linear layer's FP32 output `Ŷ` against the exact
product `Y` (FP64) of its quantized inputs `x` and weights `w`, elementwise
over every token.

- `normwise`: normwise relative error `||Ŷ - Y||_F / ||Y||_F`;
- `backward`: componentwise backward error `|ŷ - y| / (|x|ᵀ|w|)` per
  element, mean and max (Oettli-Prager; Higham, ch. 7);
- `ulp`: `log2(1 + |ŷ - y| / ulp(y))` per element in FP32 ("bits of error"),
  mean and max;
- `rounded`: the fraction of elements equal to `fl(y)`, `y` rounded to FP32;
- `bias`: mean error `(ŷ - y) / (|x|ᵀ|w|)`, the drift about zero;
- `magnitude_bias`: mean error in magnitude, `sign(y) (ŷ - y) / (|x|ᵀ|w|)`,
  negative when errors lean toward zero;
- `quantization`: normwise relative error of `Y` itself against the exact
  product `Y_0` of the unquantized inputs, the quantization's own error.

With *s0*, :func:`local` also measures against `Y_0`.  Printed errors are in
log2, biases in units of u = 2^-24.
"""

import math
import random
import statistics
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, fields
from typing import TypeVar

import torch

from . import quant

METRICS = ('normwise', 'backward', 'ulp', 'rounded', 'bias', 'magnitude_bias', 'quantization')
QUANTIZED_ONLY = ('rounded', 'quantization')
"""Metrics taken against the quantized operands' product only."""
U = 2.0 ** -24
"""FP32's unit roundoff."""

_T = TypeVar('_T')

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
    backward: float = 0.0
    backward_max: float = 0.0
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
        """Each of *metrics*' statistics, not log2."""
        n = max(self.n, 1)
        out = {
            'normwise': math.sqrt(self.err / self.ref) if self.ref else 0.0,
            'backward': self.backward / n, 'backward_max': self.backward_max,
            'ulp': self.ulp / n, 'ulp_max': self.ulp_max,
            'rounded': self.rounded / n, 'bias': self.bias / n,
            'magnitude_bias': self.magnitude_bias / n,
            'quantization': math.sqrt(self.q_err / self.q_ref) if self.q_ref else 0.0,
        }
        return {k: v for k, v in out.items() if k.removesuffix('_max') in metrics}


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


def _exact(metrics: Collection[str], a: torch.Tensor, wt: torch.Tensor,
           wa: torch.Tensor | None) -> _Exact:
    """`a @ wt`'s :class:`_Exact` for *metrics*; *wa* is `|wt|`, or `None`
    if unneeded."""
    y = a @ wt
    ulp = None
    if 'ulp' in metrics:
        # |y| in [2^(ex-1), 2^ex): its FP32 ulp is 2^(ex-24), at least 2^-149
        _, ex = torch.frexp(y)
        ex = torch.where(y == 0, -125, ex)
        ulp = torch.ldexp(torch.ones_like(y), (ex - 24).clamp(min=-149))
    return _Exact(y, None if wa is None else a.abs() @ wa, ulp,
                  y.float() if 'rounded' in metrics else None)


def _against(s: Stats, metrics: Collection[str], r: _Exact, g: torch.Tensor) -> None:
    """Add output block *g*'s *metrics* against *r* to *s*."""
    e = g.double() - r.y
    s.n += e.numel()
    if 'normwise' in metrics:
        s.err += float((e * e).sum())
        s.ref += float((r.y * r.y).sum())
    if r.scale is not None:
        eta = torch.where(r.scale > 0, e / r.scale, 0.0)
        if 'backward' in metrics:
            s.backward += float(eta.abs().sum())
            s.backward_max = max(s.backward_max, float(eta.abs().max()))
        if 'bias' in metrics:
            s.bias += float(eta.sum())
        if 'magnitude_bias' in metrics:
            s.magnitude_bias += float((torch.sign(r.y) * eta).sum())
        del eta
    if r.ulp is not None:
        bits = torch.log2(1 + e.abs() / r.ulp)
        s.ulp += float(bits.sum())
        s.ulp_max = max(s.ulp_max, float(bits.max()))
        del bits
    if r.rounded is not None:
        s.rounded += int((g == r.rounded).sum())


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
            r = _exact(metrics, qa.dequantize(slice(i, i + rows)), wt, wa)
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
    if key == 'rounded':
        return f'{v:.2%}'
    if key.endswith('bias'):
        return f'{v / U:+.3f}'
    if key.startswith('ulp'):
        return f'{v:.3f}'
    return f'{math.log2(v):.2f}' if v > 0 else '-inf'


def paired(a: Sequence[float], b: Sequence[float] | None = None) -> tuple[float, float]:
    """The mean over units of `a - b` (or of *a*), and its standard error
    (NaN for one unit)."""
    d = a if b is None else [x - y for x, y in zip(a, b, strict=True)]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.nan
    return statistics.fmean(d), se


def p_value(mean: float, se: float) -> float:
    """Two-sided, for a mean of *mean* with standard error *se* against 0."""
    if se == 0:
        return 1.0 if mean == 0 else 0.0
    return math.erfc(abs(mean / se) / math.sqrt(2))


def holm(ps: Sequence[float]) -> list[float]:
    """*ps* Holm-adjusted, for the family they make; a NaN stays NaN and
    counts for nothing."""
    order = sorted((i for i, p in enumerate(ps) if not math.isnan(p)), key=lambda i: ps[i])
    out, top = list(ps), 0.0
    for rank, i in enumerate(order):
        top = max(top, min(1.0, (len(order) - rank) * ps[i]))
        out[i] = top
    return out


def bootstrap(units: Sequence[_T], stat: Callable[[Sequence[_T]], float],
                 iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile interval of *stat* over *units* resampled with
    replacement; a resample where it is NaN counts for nothing."""
    rng = random.Random(seed)
    vals = sorted(v for _ in range(iters)
                  if not math.isnan(v := stat(rng.choices(units, k=len(units)))))
    if not vals:
        return math.nan, math.nan
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]

