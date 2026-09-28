"""
Local per-layer error of each design on cached activations, without running
the model.

Every linear layer's BF16 input is captured once under `--scheme`'s exact
run on a workload (`workloads`); each applicable design then runs only its
kernel on the cached, quantized operands, measured against the exact
product of the quantized and of the unquantized operands (`metrics.local`).
`--by` splits by a tag.  `--models` takes masters (RTN, `lm_head`
unquantized unless `bf16`) and checkpoints (`checkpoints.weights_for`).
The designs of a run share each layer's exact products, so the seconds
reported are each design's kernels and, once, the metrics.  `--acts`
caches the captured activations across runs.

    python serve/local.py                               # every BF16 design
    python serve/local.py -d amd.cdna2.bf16 --tokens 512 --layers mlp -o s.json
    python serve/local.py --workload mtbench --by role
    python serve/local.py --scheme fp8-row
    python serve/local.py --scheme nvfp4 --models Qwen/Qwen3-0.6B kaitchup/Qwen3-0.6B-NVFP4
"""

import argparse
import bisect
import hashlib
import json
import re
import sys
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from core import checkpoints, cli, kernels, metrics, swap, workloads

_OUT_ELEMS = 1 << 28
"""Design outputs (FP32 elements) held at once: past it, a layer's designs
run a few at a time, `lm_head` at 2048 tokens one at a time."""

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


def sample(seqs: list[workloads.Sequence], tokens: int | None, seed: int = 0) -> list[list[int]]:
    """*tokens* positions over *seqs* (`workloads.pick`), as each sequence's
    positions in order."""
    ends = [0]
    for seq in seqs:
        ends.append(ends[-1] + len(seq.ids))
    keep: list[list[int]] = [[] for _ in seqs]
    for p in workloads.pick(ends[-1], tokens, seed):
        i = bisect.bisect_right(ends, p) - 1
        keep[i].append(p - ends[i])
    return keep


def capture(
    model: torch.nn.Module, run: swap.Run, seqs: list[workloads.Sequence],
    tokens: int | None = None, seed: int = 0,
) -> Activations:
    """Each linear layer's input under *run*'s scheme's exact run, as BF16
    (the values a deployment holds, before any quantizing), at
    :func:`sample`'s positions over *seqs* by *seed*.  *run* is `swap.patch(model)`'s."""
    keep = sample(seqs, tokens, seed)
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
    model: torch.nn.Module, run: swap.Run, acts: Activations, designs: Sequence[str],
    names: Collection[str] = metrics.METRICS, *, by: str | None = None, only: str | None = None,
    masters: dict[str, torch.Tensor] | None = None, unquantized: bool = True,
    seconds: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, metrics.Stats]]]]:
    """Each of *designs*' :class:`metrics.Stats` for metrics *names* under
    *run*'s scheme, as `{design: {reference: {group: {layer: Stats}}}}`:
    references :data:`REFERENCES` (`unquantized` only if *unquantized*);
    groups by tag *by*, else `all`; layers matching regex *only*, minus
    *run*'s ignored ones.  *run* gives weights, static scales and `k`
    splits; unquantized weights are *masters*', else the layer's own.  The
    designs share each layer's exact products (`metrics.local`).  With
    *seconds*, each design's kernel time is added under its name and the
    metrics' under `metrics`."""
    scheme = run.scheme
    if refused := [d for d in designs if not kernels.applicable(d, scheme)]:
        raise ValueError(f'{", ".join(refused)} does not take {scheme.name}')
    groups: dict[str, torch.Tensor | slice] = {'all': slice(None)}
    if by is not None:
        labels = acts.tags[by]
        groups = {g: torch.tensor([i for i, x in enumerate(labels) if x == g], device='cuda')
                  for g in dict.fromkeys(labels)}
    stats = {d: {r: {g: {} for g in groups} for r in REFERENCES[:1 + unquantized]}
             for d in designs}

    def timed(key: str, start: float) -> None:
        if seconds is not None:
            torch.cuda.synchronize()
            seconds[key] = seconds.get(key, 0.0) + time.perf_counter() - start

    with torch.no_grad():
        for name, layer in model.named_modules():
            if (not isinstance(layer, torch.nn.Linear) or id(layer.weight) in run.ignore
                    or (only and not re.search(only, name))):
                continue
            i = acts.index[name]
            a = acts.inputs[i].cuda().float()
            qa = run.quantize(a, layer.weight, acts.amax[i].cuda()[:, None])
            qw = run.weight(layer.weight)
            w0 = layer.weight if masters is None else masters[name].cuda()
            per = max(1, _OUT_ELEMS // (a.shape[0] * layer.out_features))
            for c in range(0, len(designs), per):
                ys = {}
                for d in designs[c:c + per]:
                    start = time.perf_counter()
                    held = kernels.storage(d)[1]
                    split = swap.slices(d, scheme, a.shape[1], run.split_k)
                    ys[d] = swap.gemm(d, scheme, qa, qw, kernels.prepare(qw.elements, held, split),
                                      run.combine)
                    timed(d, start)
                start = time.perf_counter()
                for g, rows in groups.items():
                    outs = []
                    for d, y in ys.items():
                        s0 = None
                        if unquantized:
                            s0 = stats[d]['unquantized'][g][name] = metrics.Stats()
                        outs.append((stats[d]['quantized'][g].setdefault(name, metrics.Stats()),
                                     y[rows], s0))
                    metrics.local(outs, names, qa.take(rows), qw, (a[rows], w0))
                timed('metrics', start)
                del ys
    return stats


def _digest(key: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap)
    ap.add_argument('--models', nargs='*', default=None,
                    help='masters (RTN) and quantized checkpoints to evaluate (default: --model)')
    cli.add_scheme_args(ap)
    ap.add_argument('-d', '--designs', nargs='*', choices=list(kernels.TILES),
                    help='designs the scheme applies to (default: all of them)')
    ap.add_argument('-m', '--metrics', nargs='*', choices=metrics.METRICS, default=list(metrics.METRICS))
    ap.add_argument('-w', '--workload', choices=['wikitext', 'mtbench'], default='wikitext')
    ap.add_argument('--tokens', type=int, default=workloads.CONTEXT,
                    help='rows captured: WikiText-2\'s first, or sampled over MT-Bench')
    ap.add_argument('--by', choices=['role', 'category'], default=None,
                    help='split the result by this tag (MT-Bench)')
    ap.add_argument('--transcripts', type=Path, default=None,
                    help='MT-Bench conversations cache (default: results/mtbench-<model>.json)')
    ap.add_argument('--layers', default=None, help='only the linear layers this regex matches')
    ap.add_argument('--acts', type=Path, default=None,
                    help='cache captured activations here, reused for the same settings')
    ap.add_argument('-j', '--jobs', type=int, default=None,
                    help='processes compiling the designs (default: one per core)')
    ap.add_argument('-o', '--out', default=None, help='write every layer\'s metrics as JSON here')
    args = ap.parse_args(argv)
    designs = args.designs or kernels.designs(args.scheme)
    if refused := [d for d in designs if not kernels.applicable(d, args.scheme)]:
        ap.error(f'{args.scheme.name} does not apply to {", ".join(refused)}')
    kernels.precompile(designs, args.jobs)

    out: dict[str, Any] = {
        'scheme': args.scheme.name, 'workload': args.workload, 'tokens': args.tokens,
        'seed': args.seed,
        'by': args.by, 'layers': args.layers, 'split_k': args.split_k, 'combine': args.combine,
        'models': {}}
    for name in args.models or [args.model]:
        model, run, about = checkpoints.for_scheme(
            name, args.scheme, requantize=args.requantize, master=args.master,
            split_k=args.split_k, combine=args.combine)
        known = about['master'] is not None
        masters = (checkpoints.master_weights(about['master'])
                   if known and about['source'] != 'rtn' else None)
        names = [m for m in args.metrics if m != 'quantization' or known]
        refs = {'quantized': names}
        if known:
            refs['unquantized'] = [m for m in names if m not in metrics.QUANTIZED_ONLY]
        transcripts = args.transcripts or (Path(__file__).parent / 'results'
                                           / f'mtbench-{name.replace("/", "--")}.json')
        key = {'model': name, 'scheme': args.scheme.name, 'source': about['source'],
               'workload': args.workload, 'tokens': args.tokens, 'seed': args.seed,
               'transcripts': str(transcripts) if args.workload == 'mtbench' else None}
        cached = args.acts / f'acts-{_digest(key)}.pt' if args.acts else None
        if cached and cached.exists():
            acts = Activations(**torch.load(cached, weights_only=True))
        else:
            if args.workload == 'wikitext':
                seqs = workloads.wikitext(name, args.tokens)
            else:
                from transformers import AutoTokenizer

                seqs = workloads.mtbench(model, run, AutoTokenizer.from_pretrained(name),
                                         transcripts)
            acts = capture(model, run, seqs, args.tokens, args.seed)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                torch.save(vars(acts), cached)

        results: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
        total: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        seconds: dict[str, float] = {}
        every = evaluate(model, run, acts, designs, names, by=args.by, only=args.layers,
                         masters=masters, unquantized=known, seconds=seconds)
        for design, stats in every.items():
            for r, per_ref in stats.items():
                total.setdefault(r, {})[design] = {
                    g: sum(per.values(), metrics.Stats()).report(refs[r]) for g, per in per_ref.items()}
                results.setdefault(r, {})[design] = {
                    g: {n: st.report(refs[r]) for n, st in per.items()} for g, per in per_ref.items()}

        print(f'\n== {name}: weights {about["source"]}, unquantized: '
              f'{", ".join(about["ignore"]) or "none"}; originals from '
              f'{about["master"] or "nothing (no master)"}')
        for r, by_design in total.items():
            print('\n-- against the ' + (
                "post-quantization reference, the quantized operands' exact product: "
                "the design's own effect" if r == 'quantized' else
                "pre-quantization reference, the unquantized operands' exact product: "
                "the quantization's and the design's"))
            first = next(iter(by_design.values()))
            keys = list(next(iter(first.values())))
            w = max(map(len, by_design))
            for g in first:
                rows = acts.tags[args.by].count(g) if args.by else acts.inputs[0].shape[0]
                print(f'\n{g} ({rows} tokens)\n{"":{w}} ' + ' '.join(f'{k:>14}' for k in keys)
                      + f' {"s":>6}')
                for design, t in by_design.items():
                    print(f'{design:{w}} ' + ' '.join(f'{metrics.fmt(k, t[g][k]):>14}' for k in keys)
                          + f' {seconds[design]:6.1f}')
        print(f'\ns: each design\'s kernels; the metrics, shared by all {len(designs)}: '
              f'{seconds["metrics"]:.1f} s')
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
