"""
Local per-layer error of each design on cached activations, without running
the model.

Every linear layer's BF16 input is captured once under `--scheme`'s exact
run on a workload (`workloads`); each applicable design then runs only its
kernel on the cached, quantized operands, measured against the exact
product of the quantized and of the unquantized operands (`layers.local`).
`--by` splits by a tag.  `--models` takes masters (RTN, `lm_head`
unquantized unless `bf16`) and checkpoints (`checkpoints.weights_for`).
Seconds per design include the metrics' FP64 work.

    python serve/local.py                               # every BF16 design
    python serve/local.py -d amd.cdna2.bf16 --tokens 512 --layers mlp -o s.json
    python serve/local.py --workload mtbench --by role
    python serve/local.py --scheme fp8-row
    python serve/local.py --scheme nvfp4 --models Qwen/Qwen3-0.6B kaitchup/Qwen3-0.6B-NVFP4
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
from typing import Any

import checkpoints
import kernels
import layers
import perplexity
import quant
import swap
import torch
import workloads

METRICS = tuple(m for m in layers.METRICS if m != 'propagated')
REFERENCES = ('quantized', 'unquantized')
"""What a design's output is measured against: the exact product of the
quantized operands (its own effect), or of the unquantized ones (the
quantization's with its own)."""


@dataclass
class Activations:
    """Every linear layer's input, `[t, k]` BF16 on the host, one tensor per
    distinct input (`q/k/v_proj` share one, as do `gate/up_proj`)."""

    inputs: list[torch.Tensor]
    amax: list[torch.Tensor]
    """Per input, each row's sequence's `max |x|` `[t]`: a dynamic per-tensor
    scale's, as the model computes it per call."""
    index: dict[str, int]
    """Each layer's input in :attr:`inputs`, by name."""
    tags: dict[str, list[str]]
    """Per tag, each row's label (`workloads.Sequence.tags`)."""


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
    tokens: int | None = None,
) -> Activations:
    """Each linear layer's input under *run*'s scheme's exact run, as BF16
    (the values a deployment holds, before any quantizing), at
    :func:`sample`'s positions over *seqs*.  *run* is `swap.patch(model)`'s."""
    keep = sample(seqs, tokens)
    parts: list[list[torch.Tensor]] = []
    peaks: list[list[torch.Tensor]] = []
    index: dict[str, int] = {}
    last, slot, rows = None, -1, None

    def record(name: str, layer: torch.nn.Linear, args: tuple[torch.Tensor]) -> None:
        nonlocal last, slot
        if args[0] is not last:
            last, slot = args[0], slot + 1
            if slot == len(parts):
                parts.append([])
                peaks.append([])
            x = args[0].to(torch.bfloat16)
            parts[slot].append(x[0, rows].cpu())
            peaks[slot].append(x.abs().amax().float().expand(len(rows)).cpu())
        index[name] = slot

    handles = [m.register_forward_pre_hook(partial(record, n)) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    run.mode = f'{run.scheme.name}-exact'
    try:
        with torch.no_grad():
            for seq, pos in zip(seqs, keep):
                if pos:
                    last, slot, rows = None, -1, torch.tensor(pos, device='cuda')
                    model(torch.tensor([seq.ids], device='cuda'))
    finally:
        run.mode = 'fp32'
        for h in handles:
            h.remove()
    tags = {k: [seq.tags[k][p] for seq, pos in zip(seqs, keep) for p in pos] for k in seqs[0].tags}
    return Activations([torch.cat(p) for p in parts], [torch.cat(p) for p in peaks], index, tags)


def evaluate(
    model: torch.nn.Module, run: swap.Run, acts: Activations, design: str,
    metrics: Collection[str] = METRICS, *, by: str | None = None, only: str | None = None,
    masters: dict[str, torch.Tensor] | None = None, unquantized: bool = True,
) -> dict[str, dict[str, dict[str, layers.Stats]]]:
    """*design*'s :class:`layers.Stats` under *run*'s scheme, as `{reference:
    {group: {layer: Stats}}}`: references :data:`REFERENCES` (`unquantized`
    only if *unquantized*); groups by tag *by*, else `all`; layers matching
    regex *only*, minus *run*'s ignored ones.  *run* gives weights, static
    scales and `k` splits; unquantized weights are *masters*', else the
    layer's own."""
    scheme = run.scheme
    if not kernels.applicable(design, scheme):
        raise ValueError(f'{design} does not take {scheme.name}')
    groups: dict[str, torch.Tensor | slice] = {'all': slice(None)}
    if by is not None:
        labels = acts.tags[by]
        groups = {g: torch.tensor([i for i, x in enumerate(labels) if x == g], device='cuda')
                  for g in dict.fromkeys(labels)}
    stats: dict[str, dict[str, dict[str, layers.Stats]]] = {
        r: {g: {} for g in groups} for r in REFERENCES[:1 + unquantized]}
    held = kernels.storage(design)[1]
    with torch.no_grad():
        for name, layer in model.named_modules():
            if (not isinstance(layer, torch.nn.Linear) or id(layer.weight) in run.ignore
                    or (only and not re.search(only, name))):
                continue
            i = acts.index[name]
            a = acts.inputs[i].cuda().float()
            qa = run.quantize(a, layer.weight, acts.amax[i].cuda()[:, None])
            qw = run.weight(layer.weight)
            split = swap.slices(design, scheme, a.shape[1], run.split_k)
            y = swap.gemm(design, scheme, qa, qw, kernels.prepare(qw.elements, held, split),
                          run.combine)
            w0 = layer.weight if masters is None else masters[name].cuda()
            for g, rows in groups.items():
                s0 = None
                if unquantized:
                    s0 = stats['unquantized'][g][name] = layers.Stats()
                layers.local(stats['quantized'][g].setdefault(name, layers.Stats()), metrics,
                             qa.take(rows), qw, y[rows], (a[rows], w0), s0)
    return stats


def _load(name: str, scheme: quant.Scheme, *, requantize: bool, master: str | None,
          split_k: int, combine: kernels.Combine,
          ) -> tuple[torch.nn.Module, swap.Run, dict[str, Any], dict[str, torch.Tensor] | None]:
    """*name*'s model and run under *scheme*, its `about` (weights' source,
    unquantized layers, master or `None`), and the master's weights (`None`:
    the model's own)."""
    masters = None
    if checkpoints.is_checkpoint(name):
        model, ckpt = checkpoints.load(name)
        run = swap.patch(model)
        weights, source = checkpoints.weights_for(ckpt, scheme, requantize)
        ignore = ckpt.ignore
        swap.give(run, model, scheme, weights, ckpt.inputs if source == 'checkpoint' else None,
                  ignore)
        master = master or checkpoints.base_model(name)
        masters = None if master is None else checkpoints.master_weights(master)
    else:
        model, run = swap.load(name)
        source, ignore, master = 'rtn', [] if scheme.name == 'bf16' else ['lm_head'], name
        swap.give(run, model, scheme, {}, ignore=ignore)
    run.split_k, run.combine = split_k, combine
    return model, run, {'source': source, 'ignore': ignore, 'master': master}, masters


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    swap.add_args(ap)
    ap.add_argument('--models', nargs='*', default=None,
                    help='masters (RTN) and quantized checkpoints to evaluate (default: --model)')
    ap.add_argument('--scheme', type=quant.scheme, default=swap.BF16,
                    help=f'one of {", ".join(quant.SCHEMES)}, or fp8-row:fnuz, fp8-block:fnuz')
    ap.add_argument('--requantize', action='store_true',
                    help='allow a checkpoint in another scheme to be requantized, lossily')
    ap.add_argument('--master', default=None, help='the unquantized model a checkpoint is '
                    'measured against (default: its model card\'s base model)')
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

    out: dict[str, Any] = {
        'scheme': args.scheme.name, 'workload': args.workload, 'tokens': args.tokens,
        'by': args.by, 'layers': args.layers, 'split_k': args.split_k, 'combine': args.combine,
        'models': {}}
    for name in args.models or [args.model]:
        model, run, about, masters = _load(name, args.scheme, requantize=args.requantize,
                                           master=args.master, split_k=args.split_k,
                                           combine=args.combine)
        known = about['master'] is not None
        metrics = [m for m in args.metrics if m != 'quantization' or known]
        refs = {'quantized': metrics}
        if known:
            refs['unquantized'] = [m for m in metrics if m not in layers.QUANTIZED_ONLY]
        if args.workload == 'wikitext':
            seqs = workloads.wikitext(name, args.tokens)
        else:
            from transformers import AutoTokenizer

            path = args.transcripts or (Path(__file__).parent / 'results'
                                        / f'mtbench-{name.replace("/", "--")}.json')
            seqs = workloads.mtbench(model, run, AutoTokenizer.from_pretrained(name), path)
        acts = capture(model, run, seqs, args.tokens)

        results: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
        total: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        seconds: dict[str, float] = {}
        for design in designs:
            kernels.compiled(design)
            torch.cuda.synchronize()
            start = time.perf_counter()
            stats = evaluate(model, run, acts, design, metrics, by=args.by, only=args.layers,
                             masters=masters, unquantized=known)
            torch.cuda.synchronize()
            seconds[design] = time.perf_counter() - start
            for r, per_ref in stats.items():
                total.setdefault(r, {})[design] = {
                    g: sum(per.values(), layers.Stats()).report(refs[r]) for g, per in per_ref.items()}
                results.setdefault(r, {})[design] = {
                    g: {n: st.report(refs[r]) for n, st in per.items()} for g, per in per_ref.items()}

        print(f'\n== {name}: weights {about["source"]}, unquantized: '
              f'{", ".join(about["ignore"]) or "none"}; originals from '
              f'{about["master"] or "nothing (no master)"}')
        for r, by_design in total.items():
            print(f'\n-- against the {r} operands\' exact product: '
                  + ("the design's own effect" if r == 'quantized'
                     else "the quantization's and the design's"))
            first = next(iter(by_design.values()))
            keys = list(next(iter(first.values())))
            w = max(map(len, by_design))
            for g in first:
                rows = acts.tags[args.by].count(g) if args.by else acts.inputs[0].shape[0]
                print(f'\n{g} ({rows} tokens)\n{"":{w}} ' + ' '.join(f'{k:>14}' for k in keys)
                      + f' {"s":>6}')
                for design, t in by_design.items():
                    print(f'{design:{w}} ' + ' '.join(f'{layers.fmt(k, t[g][k]):>14}' for k in keys)
                          + f' {seconds[design]:6.1f}')
        out['models'][name] = {**about, 'metrics': refs, 'seconds': seconds,
                               'total': total, 'runs': results}
        del model, run, acts, masters
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
