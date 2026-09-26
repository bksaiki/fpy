"""
Per-layer error: where each run's error enters, and how it grows with depth.

On a few WikiText-2 segments, each linear layer's FP32 output `Ŷ` is compared
elementwise with a reference, over every token and segment.  Local metrics
take as reference the exact product `Y` (FP64, of the same BF16-rounded
inputs `x` and weights `w`), so measure the layer's own error:

- `normwise`: normwise relative error `||Ŷ - Y||_F / ||Y||_F`;
- `backward`: componentwise backward error `|ŷ - y| / (|x|ᵀ|w|)` per
  element, mean and max (Oettli-Prager; Higham, ch. 7);
- `ulp`: `log2(1 + |ŷ - y| / ulp(y))` per element in FP32 ("bits of error"),
  mean and max;
- `rounded`: the fraction of elements equal to `fl(y)`, `y` rounded to FP32;
- `bias`: mean error `(ŷ - y) / (|x|ᵀ|w|)`, the drift about zero;
- `magnitude_bias`: mean error in magnitude, `sign(y) (ŷ - y) / (|x|ᵀ|w|)`,
  negative when errors lean toward zero.

`propagated` takes R0's output at the same layer, every layer before it run
the same way: its normwise relative error is the error the model has
gathered by then.  Only the selected metrics are computed (`propagated`
alone needs the R0 pass).  Printed per decoder block (its layers pooled),
errors in log2, biases in units of u = 2^-24; the JSON has every layer.

    python serve/layers.py                            # every run and metric
    python serve/layers.py -r amd.cdna2.bf16 -m backward magnitude_bias --segments 1
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, fields
from functools import partial

import perplexity
import swap
import torch

METRICS = ('normwise', 'propagated', 'backward', 'ulp', 'rounded', 'bias', 'magnitude_bias')
U = 2.0 ** -24
"""FP32's unit roundoff."""

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
    prop_err: float = 0.0
    prop_ref: float = 0.0
    backward: float = 0.0
    backward_max: float = 0.0
    ulp: float = 0.0
    ulp_max: float = 0.0
    rounded: int = 0
    bias: float = 0.0
    magnitude_bias: float = 0.0

    def __add__(self, other: 'Stats') -> 'Stats':
        """Pooled with *other*: sums added, maxima the larger."""
        return Stats(**{f.name: (max if f.name.endswith('_max') else sum)(
            (getattr(self, f.name), getattr(other, f.name))) for f in fields(self)})

    def report(self, metrics: Collection[str]) -> dict[str, float]:
        """Each of *metrics*' statistics, not log2."""
        n = max(self.n, 1)
        out = {
            'normwise': math.sqrt(self.err / self.ref) if self.ref else 0.0,
            'propagated': math.sqrt(self.prop_err / self.prop_ref) if self.prop_ref else 0.0,
            'backward': self.backward / n, 'backward_max': self.backward_max,
            'ulp': self.ulp / n, 'ulp_max': self.ulp_max,
            'rounded': self.rounded / n, 'bias': self.bias / n,
            'magnitude_bias': self.magnitude_bias / n,
        }
        return {k: v for k, v in out.items() if k.removesuffix('_max') in metrics}


def _rows(n: int) -> int:
    return max(1, _ELEMS // n)


def local(s: Stats, metrics: Collection[str], layer: torch.nn.Linear,
           x: torch.Tensor, got: torch.Tensor) -> None:
    """Add *got*, *layer*'s output on *x*, to *s*'s local metrics, in blocks
    of output columns and rows.  *layer* has no bias (`swap.patch`)."""
    xb = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    got = got.reshape(-1, got.shape[-1])
    scaled = bool({'backward', 'bias', 'magnitude_bias'} & set(metrics))
    cols = _rows(xb.shape[1])
    for j in range(0, got.shape[1], cols):
        wt = layer.weight[j:j + cols].to(torch.bfloat16).double().T
        wa = wt.abs() if scaled else None
        rows = _rows(wt.shape[1])
        for i in range(0, xb.shape[0], rows):
            a, g = xb[i:i + rows].double(), got[i:i + rows, j:j + cols]
            y = a @ wt
            e = g.double() - y
            s.n += e.numel()
            if 'normwise' in metrics:
                s.err += float((e * e).sum())
                s.ref += float((y * y).sum())
            if scaled:
                scale = a.abs() @ wa
                eta = torch.where(scale > 0, e / scale, 0.0)
                if 'backward' in metrics:
                    s.backward += float(eta.abs().sum())
                    s.backward_max = max(s.backward_max, float(eta.abs().max()))
                if 'bias' in metrics:
                    s.bias += float(eta.sum())
                if 'magnitude_bias' in metrics:
                    s.magnitude_bias += float((torch.sign(y) * eta).sum())
                del scale, eta
            if 'ulp' in metrics:
                # |y| in [2^(ex-1), 2^ex): its FP32 ulp is 2^(ex-24), at least 2^-149
                _, ex = torch.frexp(y)
                ex = torch.where(y == 0, -125, ex)
                bits = torch.log2(1 + e.abs() / torch.ldexp(torch.ones_like(y), (ex - 24).clamp(min=-149)))
                s.ulp += float(bits.sum())
                s.ulp_max = max(s.ulp_max, float(bits.max()))
                del ex, bits
            if 'rounded' in metrics:
                s.rounded += int((g == y.float()).sum())


def _propagated(s: Stats, got: torch.Tensor, ref: torch.Tensor) -> None:
    """Add *got* against *ref* (on the host) to *s*'s propagated sums."""
    got, ref = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
    rows = _rows(got.shape[1])
    for i in range(0, got.shape[0], rows):
        r = ref[i:i + rows].to(got.device).double()
        d = got[i:i + rows].double() - r
        s.prop_err += float((d * d).sum())
        s.prop_ref += float((r * r).sum())


def evaluate(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor], modes: Sequence[str],
    metrics: Collection[str] = METRICS,
) -> dict[str, dict[str, Stats]]:
    """Each of *modes*' :class:`Stats` over *segs* for *metrics*, every
    linear layer by name in model order.  *run* is `swap.patch(model)`'s."""
    layers = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    modes = [m for m in modes if m != 'fp32']
    stats = {mode: {n: Stats() for n in layers} for mode in modes}
    local_metrics = set(metrics) - {'propagated'}
    ref: dict[str, torch.Tensor] = {}

    def record(name: str, layer: torch.nn.Linear, inputs: tuple[torch.Tensor],
               y: torch.Tensor) -> None:
        if run.mode == 'fp32':
            ref[name] = y.cpu()
            return
        s = stats[run.mode][name]
        if local_metrics:
            local(s, local_metrics, layer, inputs[0], y)
        if 'propagated' in metrics:
            _propagated(s, y, ref[name])

    handles = [m.register_forward_hook(partial(record, n)) for n, m in layers.items()]
    try:
        with torch.no_grad():
            for seg in segs:
                for mode in ('fp32', *modes) if 'propagated' in metrics else modes:
                    run.mode = mode
                    model(seg)
    finally:
        run.mode = 'fp32'
        for h in handles:
            h.remove()
    return stats


def by_block(stats: dict[str, Stats]) -> dict[str, Stats]:
    """*stats* pooled per decoder block, any layer outside one as itself."""
    pooled: dict[str, Stats] = {}
    for name, s in stats.items():
        m = re.search(r'layers\.(\d+)\.', name)
        key = f'block {m.group(1)}' if m else name
        pooled[key] = pooled.get(key, Stats()) + s
    return pooled


def fmt(key: str, v: float) -> str:
    """*v* as printed: errors in log2, `rounded` as a percentage, biases in u."""
    if key == 'rounded':
        return f'{v:.2%}'
    if key.endswith('bias'):
        return f'{v / U:+.3f}'
    if key.startswith('ulp'):
        return f'{v:.3f}'
    return f'{math.log2(v):.2f}' if v > 0 else '-inf'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    swap.add_args(ap)
    ap.add_argument('-r', '--runs', nargs='*', choices=swap.RUNS, default=list(swap.RUNS),
                    help='runs besides fp32 (default: bf16-exact and every design)')
    ap.add_argument('-m', '--metrics', nargs='*', choices=METRICS, default=list(METRICS))
    ap.add_argument('--segments', type=int, default=4)
    ap.add_argument('-o', '--out', default=None, help='write every layer\'s metrics as JSON here')
    args = ap.parse_args(argv)

    segs = perplexity.segments(perplexity.wikitext(args.model))[:args.segments]
    model, run = swap.load(args.model, args.split_k, args.combine)

    stats = evaluate(model, run, segs, args.runs, args.metrics)
    blocks = {mode: {k: s.report(args.metrics) for k, s in by_block(layer).items()}
              for mode, layer in stats.items()}
    total = {mode: sum(layer.values(), Stats()).report(args.metrics)
             for mode, layer in stats.items()}
    keys = list(next(iter(total.values())))
    w = max(map(len, blocks))
    for key in keys:
        if key.endswith('_max'):
            continue
        print(f'\n{key}')
        print(f'{"":10} ' + ' '.join(f'{m:>{w}}' for m in blocks))
        for row in next(iter(blocks.values())):
            print(f'{row:10} ' + ' '.join(f'{fmt(key, b[row][key]):>{w}}' for b in blocks.values()))
    print(f'\nevery layer\n{"":{w}} ' + ' '.join(f'{k:>12}' for k in keys))
    for mode, t in total.items():
        print(f'{mode:{w}} ' + ' '.join(f'{fmt(k, t[k]):>12}' for k in keys))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({
                'model': args.model, 'segments': len(segs), 'metrics': args.metrics,
                'split_k': args.split_k, 'combine': args.combine,
                'runs': {mode: {n: s.report(args.metrics) for n, s in layer.items()}
                         for mode, layer in stats.items()},
            }, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
