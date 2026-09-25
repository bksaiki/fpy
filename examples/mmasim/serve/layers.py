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
- `bias`: mean signed error `sign(y) (ŷ - y) / (|x|ᵀ|w|)`, negative when
  errors lean toward zero.

`propagated` takes R0's output at the same layer, every layer before it run
the same way: its normwise relative error is the error the model has
gathered by then.  Only the selected metrics are computed (`propagated`
alone needs the R0 pass).  Printed per decoder block (its seven layers
pooled), errors in log2, bias in units of u = 2^-24; the JSON has every layer.

    python serve/layers.py                            # every run and metric
    python serve/layers.py -r amd.cdna2.bf16 -m backward bias --segments 1
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Self

import perplexity
import swap
import torch

METRICS = ('normwise', 'propagated', 'backward', 'ulp', 'rounded', 'bias')
U = 2.0 ** -24
"""FP32's unit roundoff."""

_ELEMS = 1 << 24
"""Output elements per block of rows when comparing: `lm_head`'s output is
311M."""


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

    def __iadd__(self, other: 'Stats') -> Self:
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            setattr(self, f.name, max(a, b) if f.name.endswith('_max') else a + b)
        return self

    def report(self, metrics: Collection[str]) -> dict[str, float]:
        """Each of *metrics*' statistics, linear."""
        n = max(self.n, 1)
        out = {
            'normwise': math.sqrt(self.err / self.ref) if self.ref else 0.0,
            'propagated': math.sqrt(self.prop_err / self.prop_ref) if self.prop_ref else 0.0,
            'backward': self.backward / n, 'backward_max': self.backward_max,
            'ulp': self.ulp / n, 'ulp_max': self.ulp_max,
            'rounded': self.rounded / n, 'bias': self.bias / n,
        }
        return {k: v for k, v in out.items() if k.removesuffix('_max') in metrics}


def _rows(n: int) -> int:
    return max(1, _ELEMS // n)


def _local(s: Stats, metrics: Collection[str], layer: torch.nn.Linear,
           x: torch.Tensor, got: torch.Tensor) -> None:
    """Add *got*, *layer*'s output on *x*, to *s*'s local metrics."""
    xb = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    wt = layer.weight.to(torch.bfloat16).double().T
    b = None if layer.bias is None else layer.bias.double()
    scaled = 'backward' in metrics or 'bias' in metrics
    wa = wt.abs() if scaled else None
    got = got.reshape(-1, got.shape[-1])
    rows = _rows(wt.shape[1])
    for i in range(0, xb.shape[0], rows):
        a, g = xb[i:i + rows].double(), got[i:i + rows]
        y = a @ wt if b is None else a @ wt + b
        e = g.double() - y
        s.n += e.numel()
        if 'normwise' in metrics:
            s.err += float((e * e).sum())
            s.ref += float((y * y).sum())
        if scaled:
            scale = a.abs() @ wa if b is None else a.abs() @ wa + b.abs()
            eta = torch.where(scale > 0, e / scale, 0.0)
            if 'backward' in metrics:
                s.backward += float(eta.abs().sum())
                s.backward_max = max(s.backward_max, float(eta.abs().max()))
            if 'bias' in metrics:
                s.bias += float((torch.sign(y) * eta).sum())
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
    local = set(metrics) - {'propagated'}
    ref: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def record(layer: torch.nn.Linear, inputs: tuple[torch.Tensor], y: torch.Tensor) -> None:
            if run.mode == 'fp32':
                ref[name] = y.cpu()
                return
            s = stats[run.mode][name]
            if local:
                _local(s, local, layer, inputs[0], y)
            if 'propagated' in metrics:
                _propagated(s, y, ref[name])
        return record

    handles = [m.register_forward_hook(hook(n)) for n, m in layers.items()]
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
        pooled.setdefault(f'block {m.group(1)}' if m else name, Stats()).__iadd__(s)
    return pooled


def _fmt(key: str, v: float) -> str:
    """*v* as printed: errors in log2, `rounded` as a percentage, `bias` in u."""
    if key == 'rounded':
        return f'{v:.2%}'
    if key == 'bias':
        return f'{v / U:+.3f}'
    if key.startswith('ulp'):
        return f'{v:.3f}'
    return f'{math.log2(v):.2f}' if v > 0 else '-inf'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('-r', '--runs', nargs='*', default=[m for m in swap.MODES if m != 'fp32'],
                    help='runs besides fp32 (default: bf16-exact and every design)')
    ap.add_argument('-m', '--metrics', nargs='*', choices=METRICS, default=list(METRICS))
    ap.add_argument('--segments', type=int, default=4)
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=['linear', 'tree'], default='linear')
    ap.add_argument('-o', '--out', default=None, help='write every layer\'s metrics as JSON here')
    args = ap.parse_args(argv)

    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = '\n\n'.join(datasets.load_dataset(
        'Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    ids = AutoTokenizer.from_pretrained(perplexity.MODEL)(text, return_tensors='pt').input_ids.cuda()
    segs = perplexity.segments(ids)[:args.segments]
    model = AutoModelForCausalLM.from_pretrained(perplexity.MODEL, dtype=torch.float32).cuda().eval()
    run = swap.patch(model)
    run.split_k, run.combine = args.split_k, args.combine

    stats = evaluate(model, run, segs, args.runs, args.metrics)
    blocks = {mode: {k: s.report(args.metrics) for k, s in by_block(layer).items()}
              for mode, layer in stats.items()}
    total = {}
    for mode, layer in stats.items():
        pooled = Stats()
        for s in layer.values():
            pooled += s
        total[mode] = pooled.report(args.metrics)
    w = max(map(len, blocks))
    for key in next(iter(total.values())):
        if key.endswith('_max'):
            continue
        print(f'\n{key}')
        print(f'{"":10} ' + ' '.join(f'{m:>{w}}' for m in blocks))
        for row in next(iter(blocks.values())):
            print(f'{row:10} ' + ' '.join(f'{_fmt(key, b[row][key]):>{w}}' for b in blocks.values()))
    keys = list(next(iter(total.values())))
    print(f'\nevery layer\n{"":{w}} ' + ' '.join(f'{k:>12}' for k in keys))
    for mode, t in total.items():
        print(f'{mode:{w}} ' + ' '.join(f'{_fmt(k, t[k]):>12}' for k in keys))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({
                'model': perplexity.MODEL, 'segments': len(segs), 'metrics': args.metrics,
                'split_k': args.split_k, 'combine': args.combine,
                'runs': {mode: {n: s.report(args.metrics) for n, s in layer.items()}
                         for mode, layer in stats.items()},
            }, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
