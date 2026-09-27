"""
WikiText-2 perplexity, and each run's distance from the FP32 baseline.

The test split, joined and tokenized as the Hugging Face perplexity guide does,
is cut into 2048-token segments (the GPTQ convention; the remainder is
dropped).  Each segment runs through R0 (`fp32`) and then every other run on
the same model, and per predicted token this accumulates

- the negative log-likelihood of the next token, for perplexity;
- KL(R0 || run) of the next-token distributions;
- whether the top-1 token matches R0's;
- Δp, the change in the probability of the correct token (RMS reported),

as llama.cpp's `perplexity --kl-divergence` reports them.  Then, paired
over segments (:func:`against`): each run against R0, and each design against
the scheme's exact run, as Δ NLL (the log of the perplexity ratio), KL and
top-1 disagreement, each a mean over segments with its standard error, and
Δ NLL's p-value, Holm-adjusted over the runs sharing a reference.  Only one
segment's reference log-probabilities are held, on the host.

    python serve/perplexity.py                          # every run
    python serve/perplexity.py -r amd.cdna2.bf16 --segments 4
    python serve/perplexity.py --split-k 4 --combine tree -o tree.json
    python serve/perplexity.py --scheme fp8-row --segments 20
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import checkpoints
import metrics
import swap
import torch

CONTEXT = 2048
_ROWS = 512
"""Tokens per block of log-softmax and comparison."""


@dataclass
class Totals:
    """One run's sums over the predicted tokens, and what they report."""

    tokens: int = 0
    nll: float = 0.0
    nll_sq: float = 0.0
    kl: float = 0.0
    kl_sq: float = 0.0
    same_top1: int = 0
    dp_sq: float = 0.0
    seconds: float = 0.0
    nll_seg: list[float] = field(default_factory=list)
    """Each segment's mean NLL."""
    vs: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Per reference run, each segment's mean `kl` and top-1 `disagree`ment."""

    def add(self, nll: torch.Tensor, kl: torch.Tensor, same: torch.Tensor, dp: torch.Tensor) -> None:
        self.tokens += nll.numel()
        self.nll += nll.sum().item()
        self.nll_sq += (nll * nll).sum().item()
        self.kl += kl.sum().item()
        self.kl_sq += (kl * kl).sum().item()
        self.same_top1 += int(same.sum().item())
        self.dp_sq += (dp * dp).sum().item()

    def compare(self, refs: dict[str, torch.Tensor], lp: torch.Tensor,
                target: torch.Tensor) -> None:
        """Add one segment's next-token log-probabilities *lp* `[t, vocab]`
        against each of *refs* (on the host, R0's first, which the
        token-level sums take), *target* the tokens predicted."""
        nll, seg = 0.0, {name: [0.0, 0] for name in refs}
        for i in range(0, lp.shape[0], _ROWS):
            q = lp[i:i + _ROWS]
            tgt = target[i:i + _ROWS]
            rows = torch.arange(q.shape[0], device=q.device)
            nll += -q[rows, tgt].sum().item()
            for j, (name, ref) in enumerate(refs.items()):
                r = ref[i:i + _ROWS].to(q.device)
                kl, same = (r.exp() * (r - q)).sum(-1), r.argmax(-1) == q.argmax(-1)
                if j == 0:
                    self.add(nll=-q[rows, tgt], kl=kl, same=same,
                             dp=q[rows, tgt].exp() - r[rows, tgt].exp())
                seg[name][0] += kl.sum().item()
                seg[name][1] += int((~same).sum().item())
        t = lp.shape[0]
        self.nll_seg.append(nll / t)
        for name, (kl_sum, differ) in seg.items():
            v = self.vs.setdefault(name, {'kl': [], 'disagree': []})
            v['kl'].append(kl_sum / t)
            v['disagree'].append(differ / t)

    def report(self) -> dict[str, float]:
        """Means over tokens, each with its standard error."""
        n = self.tokens

        def mean_se(s: float, sq: float) -> tuple[float, float]:
            m = s / n
            return m, math.sqrt(max(sq / n - m * m, 0.0) / n)

        nll, nll_se = mean_se(self.nll, self.nll_sq)
        kl, kl_se = mean_se(self.kl, self.kl_sq)
        top1 = self.same_top1 / n
        return {
            'tokens': n, 'ppl': math.exp(nll), 'ppl_se': math.exp(nll) * nll_se,
            'kl': kl, 'kl_se': kl_se,
            'top1': top1, 'top1_se': math.sqrt(top1 * (1 - top1) / n),
            'rms_dp': math.sqrt(self.dp_sq / n), 'seconds': self.seconds,
        }


def wikitext(model: str) -> torch.Tensor:
    """WikiText-2's test split, joined as the Hugging Face perplexity guide
    does and tokenized for *model*: `[1, t]` on the GPU."""
    import datasets
    from transformers import AutoTokenizer

    text = '\n\n'.join(datasets.load_dataset(
        'Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    return AutoTokenizer.from_pretrained(model)(text, return_tensors='pt').input_ids.cuda()


def segments(ids: torch.Tensor, context: int = CONTEXT) -> list[torch.Tensor]:
    """*ids* `[1, t]` as whole segments of *context* tokens."""
    return [ids[:, s:s + context] for s in range(0, ids.shape[1] - context + 1, context)]


def _log_probs(model: torch.nn.Module, seg: torch.Tensor) -> torch.Tensor:
    """Next-token log-probabilities `[t - 1, vocab]` for the segment."""
    with torch.no_grad():
        logits = model(seg).logits[0, :-1].float()
    for block in logits.split(_ROWS):
        block.copy_(torch.log_softmax(block, -1))
    return logits


def evaluate(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor],
    modes: Sequence[str], *, progress: bool = False,
) -> dict[str, Totals]:
    """R0 and each of *modes* over *segs*: every mode's totals, `fp32`'s
    first, each against R0 and a design also against *run*'s scheme's exact
    run, if among *modes*.  *run* is `swap.patch(model)`'s; its `split_k` /
    `combine` apply to every design."""
    exact = f'{run.scheme.name}-exact'
    rest = sorted((m for m in modes if m != 'fp32'), key=lambda m: m != exact)
    totals = {mode: Totals() for mode in ('fp32', *rest)}
    try:
        for i, seg in enumerate(segs):
            target = seg[0, 1:]
            refs: dict[str, torch.Tensor] = {}
            for mode, t in totals.items():
                run.mode = mode
                start = time.perf_counter()
                lp = _log_probs(model, seg)
                torch.cuda.synchronize()
                t.seconds += time.perf_counter() - start
                if mode in ('fp32', exact):
                    refs[mode] = lp.cpu()
                t.compare({r: v for r, v in refs.items() if r != mode or mode == 'fp32'},
                          lp, target)
                del lp
            if progress:
                print(f'segment {i + 1}', file=sys.stderr, flush=True)
    finally:
        run.mode = 'fp32'
    return totals


def against(totals: dict[str, Totals]) -> dict[str, dict[str, dict[str, float]]]:
    """Paired over segments, per reference and run (:func:`evaluate`'s
    comparisons, a run against itself left out): the mean and standard error
    of Δ NLL, KL and top-1 disagreement, and Δ NLL's p-value, Holm-adjusted
    over the runs sharing the reference."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ref, base in totals.items():
        rows: dict[str, dict[str, float]] = {}
        for mode, t in totals.items():
            if mode == ref or ref not in t.vs:
                continue
            r = rows[mode] = {}
            r['dnll'], r['dnll_se'] = metrics.paired(t.nll_seg, base.nll_seg)
            for k in ('kl', 'disagree'):
                r[k], r[f'{k}_se'] = metrics.paired(t.vs[ref][k])
            r['p'] = metrics.p_value(r['dnll'], r['dnll_se'])
        for r, p in zip(rows.values(), metrics.holm([r['p'] for r in rows.values()])):
            r['p_holm'] = p
        if rows:
            out[ref] = rows
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    swap.add_args(ap)
    checkpoints.add_args(ap)
    ap.add_argument('-r', '--runs', nargs='*',
                    help="runs besides fp32 (default: the scheme's exact run and every design)")
    ap.add_argument('--segments', type=int, default=None,
                    help='this many segments at random (default: all)')
    ap.add_argument('-o', '--out', default=None, help='write the results as JSON here')
    args = ap.parse_args(argv)

    runs = checkpoints.runs(ap, args)
    segs = segments(wikitext(args.model))
    segs = [segs[i] for i in swap.pick(len(segs), args.segments, args.seed)]
    model, run, about = checkpoints.for_scheme(
        args.model, args.scheme, requantize=args.requantize, master=args.master,
        split_k=args.split_k, combine=args.combine)

    totals = evaluate(model, run, segs, runs, progress=True)
    results = {
        'model': args.model, 'scheme': args.scheme.name, **about,
        'context': CONTEXT, 'segments': len(segs), 'seed': args.seed,
        'split_k': args.split_k, 'combine': args.combine,
        'runs': {mode: t.report() for mode, t in totals.items()},
        'paired': against(totals),
        'per_segment': {mode: {'nll': t.nll_seg, **t.vs} for mode, t in totals.items()},
    }
    print(f'{"run":20} {"ppl":>14} {"KL":>20} {"top-1":>16} {"RMS dp":>8} {"s":>7}')
    for mode, r in results['runs'].items():
        print(f'{mode:20} {r["ppl"]:8.4f} ±{r["ppl_se"]:.3f} '
              f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["top1"]:8.4%} ±{r["top1_se"]:.3%} '
              f'{r["rms_dp"]:8.3%} {r["seconds"]:7.0f}')
    for ref, rows in results['paired'].items():
        print(f'\npaired over {len(segs)} segments, against {ref}\n'
              f'{"run":20} {"Δ NLL":>22} {"Holm p":>7} {"KL":>20} {"top-1 disagree":>18}')
        for mode, r in rows.items():
            print(f'{mode:20} {r["dnll"]:+10.3e} ±{r["dnll_se"]:.1e} {r["p_holm"]:7.3f} '
                  f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["disagree"]:8.4%} ±{r["disagree_se"]:.3%}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
