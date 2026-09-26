"""
Local per-layer error of each design on cached activations, without running
the model.

Every linear layer's BF16 input is captured once under the scheme's exact
run (`--scheme`, default `bf16`) on a workload (`workloads`), the same for
every design; each design the scheme applies to then runs only its kernel on
each cached input and weight, both quantized by the scheme, against their
exact product (`layers.local`).  `--by` splits the result by a tag.  A grid search
registers designs (`kernels.register`), captures once, and calls
:func:`evaluate` per design; the seconds reported per design include the
metrics' own FP64 work.

    python serve/local.py                               # every BF16 design
    python serve/local.py -d amd.cdna2.bf16 --tokens 512 --layers mlp -o s.json
    python serve/local.py --workload mtbench --by role
    python serve/local.py --scheme fp8-row
"""

import argparse
import bisect
import json
import random
import re
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import kernels
import layers
import perplexity
import quant
import swap
import torch
import workloads

METRICS = tuple(m for m in layers.METRICS if m != 'propagated')


@dataclass
class Activations:
    """Every linear layer's input, `[t, k]` BF16 on the host, one tensor per
    distinct input (`q/k/v_proj` share one, as do `gate/up_proj`)."""

    inputs: list[torch.Tensor]
    index: dict[str, int]
    """Each layer's input in :attr:`inputs`, by name."""
    tags: dict[str, list[str]]
    """Per tag, each row's label (`workloads.Sequence.tags`)."""

    def input(self, name: str) -> torch.Tensor:
        return self.inputs[self.index[name]]


def sample(seqs: list[workloads.Sequence], tokens: int | None) -> list[list[int]]:
    """A fixed random *tokens* positions over *seqs* (all, if `None` or no
    fewer), as each sequence's positions in order."""
    ends = [0]
    for seq in seqs:
        ends.append(ends[-1] + len(seq.ids))
    picks = range(ends[-1])
    if tokens is not None and tokens < ends[-1]:
        picks = sorted(random.Random(0).sample(picks, tokens))
    keep: list[list[int]] = [[] for _ in seqs]
    for p in picks:
        i = bisect.bisect_right(ends, p) - 1
        keep[i].append(p - ends[i])
    return keep


def capture(
    model: torch.nn.Module, run: swap.Run, seqs: list[workloads.Sequence],
    tokens: int | None = None, scheme: quant.Scheme = swap.BF16,
) -> Activations:
    """Each linear layer's input under *scheme*'s exact run, as BF16 (the
    values a deployment holds, before any quantizing), at :func:`sample`'s
    positions over *seqs*.  *run* is `swap.patch(model)`'s."""
    keep = sample(seqs, tokens)
    parts: list[list[torch.Tensor]] = []
    index: dict[str, int] = {}
    last, slot, rows = None, -1, None

    def record(name: str, layer: torch.nn.Linear, args: tuple[torch.Tensor]) -> None:
        nonlocal last, slot
        if args[0] is not last:
            last, slot = args[0], slot + 1
            if slot == len(parts):
                parts.append([])
            parts[slot].append(args[0][0, rows].to(torch.bfloat16).cpu())
        index[name] = slot

    handles = [m.register_forward_pre_hook(partial(record, n)) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    previous = run.scheme
    run.mode, run.scheme = f'{scheme.name}-exact', scheme
    try:
        with torch.no_grad():
            for seq, pos in zip(seqs, keep):
                if pos:
                    last, slot, rows = None, -1, torch.tensor(pos, device='cuda')
                    model(torch.tensor([seq.ids], device='cuda'))
    finally:
        run.mode, run.scheme = 'fp32', previous
        for h in handles:
            h.remove()
    tags = {k: [seq.tags[k][p] for seq, pos in zip(seqs, keep) for p in pos] for k in seqs[0].tags}
    return Activations([torch.cat(p) for p in parts], index, tags)


def evaluate(
    model: torch.nn.Module, acts: Activations, design: str,
    metrics: Collection[str] = METRICS, *, scheme: quant.Scheme = swap.BF16,
    by: str | None = None, split_k: int = 1, combine: kernels.Combine = 'linear',
    only: str | None = None,
) -> dict[str, dict[str, layers.Stats]]:
    """*design*'s local :class:`layers.Stats` for *metrics* under *scheme*,
    per group of rows (by the tag *by*; one group, `all`, without) and per
    linear layer of *model* (those whose name *only* matches, as a regex), on
    *acts*."""
    if not kernels.applicable(design, scheme):
        raise ValueError(f'{design} does not take {scheme.name}')
    groups: dict[str, torch.Tensor | slice] = {'all': slice(None)}
    if by is not None:
        labels = acts.tags[by]
        groups = {g: torch.tensor([i for i, x in enumerate(labels) if x == g], device='cuda')
                  for g in dict.fromkeys(labels)}
    stats: dict[str, dict[str, layers.Stats]] = {g: {} for g in groups}
    held = kernels.storage(design)[1]
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, torch.nn.Linear) or (only and not re.search(only, name)):
                continue
            a = acts.input(name).cuda().float()
            qa, qw = quant.quantize(a, scheme.x), quant.quantize(layer.weight, scheme.w)
            y = swap.gemm(design, scheme, qa, qw, kernels.prepare(qw.elements, held, split_k),
                          combine)
            for g, rows in groups.items():
                layers.local(stats[g].setdefault(name, layers.Stats()), metrics, qa.take(rows),
                             qw, y[rows], (a[rows], layer.weight))
    return stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    swap.add_args(ap)
    ap.add_argument('--scheme', type=quant.scheme, default=swap.BF16,
                    help=f'one of {", ".join(quant.SCHEMES)}, or fp8-row:fnuz, fp8-block:fnuz')
    ap.add_argument('-d', '--designs', nargs='*', choices=list(kernels.TILES),
                    help='designs the scheme applies to (default: all of them)')
    ap.add_argument('-m', '--metrics', nargs='*', choices=METRICS, default=list(METRICS))
    ap.add_argument('-w', '--workload', choices=['wikitext', 'mtbench'], default='wikitext')
    ap.add_argument('--tokens', type=int, default=perplexity.CONTEXT,
                    help='rows captured: WikiText-2\'s first, or sampled over MT-Bench')
    ap.add_argument('--by', choices=['role', 'category'], default=None,
                    help='split the result by this tag (MT-Bench)')
    ap.add_argument('--transcripts', type=Path, default=None,
                    help='MT-Bench conversations cache (default: results/mtbench-<model>.json)')
    ap.add_argument('--layers', default=None, help='only the linear layers this regex matches')
    ap.add_argument('-o', '--out', default=None, help='write every layer\'s metrics as JSON here')
    args = ap.parse_args(argv)
    designs = args.designs or kernels.designs(args.scheme)
    if refused := [d for d in designs if not kernels.applicable(d, args.scheme)]:
        ap.error(f'{args.scheme.name} does not apply to {", ".join(refused)}')

    model, run = swap.load(args.model)
    if args.workload == 'wikitext':
        acts = capture(model, run, workloads.wikitext(args.model, args.tokens),
                       scheme=args.scheme)
    else:
        from transformers import AutoTokenizer

        path = args.transcripts or (Path(__file__).parent / 'results'
                                    / f'mtbench-{args.model.replace("/", "--")}.json')
        seqs = workloads.mtbench(model, run, AutoTokenizer.from_pretrained(args.model), path)
        acts = capture(model, run, seqs, args.tokens, args.scheme)

    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    total: dict[str, dict[str, dict[str, float]]] = {}
    seconds: dict[str, float] = {}
    for design in designs:
        kernels.compiled(design)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stats = evaluate(model, acts, design, args.metrics, scheme=args.scheme, by=args.by,
                         split_k=args.split_k, combine=args.combine, only=args.layers)
        torch.cuda.synchronize()
        seconds[design] = time.perf_counter() - start
        total[design] = {g: sum(per.values(), layers.Stats()).report(args.metrics)
                         for g, per in stats.items()}
        results[design] = {g: {n: st.report(args.metrics) for n, st in per.items()}
                           for g, per in stats.items()}

    first = next(iter(total.values()))
    keys = list(next(iter(first.values())))
    w = max(map(len, total))
    for g in first:
        rows = acts.tags[args.by].count(g) if args.by else acts.inputs[0].shape[0]
        print(f'\n{g} ({rows} tokens)\n{"":{w}} ' + ' '.join(f'{k:>14}' for k in keys) + f' {"s":>6}')
        for design, t in total.items():
            print(f'{design:{w}} ' + ' '.join(f'{layers.fmt(k, t[g][k]):>14}' for k in keys)
                  + f' {seconds[design]:6.1f}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'model': args.model, 'scheme': args.scheme.name,
                       'workload': args.workload, 'tokens': args.tokens,
                       'by': args.by, 'layers': args.layers, 'metrics': args.metrics,
                       'split_k': args.split_k, 'combine': args.combine,
                       'seconds': seconds, 'total': total, 'runs': results}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
