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

as llama.cpp's `perplexity --kl-divergence` reports them.  Nothing is stored
per token beyond one segment, so a run is a single pass.

    python serve/perplexity.py                          # every run
    python serve/perplexity.py -r amd.cdna2.bf16 --segments 4
    python serve/perplexity.py --split-k 4 --combine tree -o tree.json
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import swap
import torch

CONTEXT = 2048
_ROWS = 512
"""Tokens per block when comparing distributions: a segment's full-vocabulary
log-probabilities are 1.2-2 GB in FP32, so R0's are held on the host."""


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

    def add(self, nll: torch.Tensor, kl: torch.Tensor, same: torch.Tensor, dp: torch.Tensor) -> None:
        self.tokens += nll.numel()
        self.nll += nll.sum().item()
        self.nll_sq += (nll * nll).sum().item()
        self.kl += kl.sum().item()
        self.kl_sq += (kl * kl).sum().item()
        self.same_top1 += int(same.sum().item())
        self.dp_sq += (dp * dp).sum().item()

    def compare(self, ref: torch.Tensor, lp: torch.Tensor, target: torch.Tensor) -> None:
        """Add one segment's next-token log-probabilities *lp* `[t, vocab]`
        against R0's *ref* (on the host), *target* the tokens predicted."""
        for i in range(0, lp.shape[0], _ROWS):
            q = lp[i:i + _ROWS]
            r = ref[i:i + _ROWS].to(q.device)
            tgt = target[i:i + _ROWS]
            rows = torch.arange(q.shape[0], device=q.device)
            self.add(nll=-q[rows, tgt],
                     kl=(r.exp() * (r - q)).sum(-1),
                     same=r.argmax(-1) == q.argmax(-1),
                     dp=q[rows, tgt].exp() - r[rows, tgt].exp())

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
    first.  *run* is `swap.patch(model)`'s; its `split_k` / `combine` apply to
    every design."""
    totals = {mode: Totals() for mode in ('fp32', *[m for m in modes if m != 'fp32'])}
    for i, seg in enumerate(segs):
        target = seg[0, 1:]
        ref = None
        for mode, t in totals.items():
            run.mode = mode
            start = time.perf_counter()
            lp = _log_probs(model, seg)
            torch.cuda.synchronize()
            t.seconds += time.perf_counter() - start
            if ref is None:
                ref = lp.cpu()
            t.compare(ref, lp, target)
            del lp
        if progress:
            print(f'segment {i + 1}', file=sys.stderr, flush=True)
    run.mode = 'fp32'
    return totals


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--model', default=swap.MODEL)
    ap.add_argument('-r', '--runs', nargs='*', default=[m for m in swap.MODES if m != 'fp32'],
                    help='runs besides fp32 (default: bf16-exact and every design)')
    ap.add_argument('--segments', type=int, default=None,
                    help='only the first this many segments (default: all)')
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=['linear', 'tree'], default='linear')
    ap.add_argument('-o', '--out', default=None, help='write the results as JSON here')
    args = ap.parse_args(argv)

    import datasets
    from transformers import AutoTokenizer

    text = '\n\n'.join(datasets.load_dataset(
        'Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    ids = AutoTokenizer.from_pretrained(args.model)(text, return_tensors='pt').input_ids.cuda()
    segs = segments(ids)[:args.segments]
    model, run = swap.load(args.model)
    run.split_k, run.combine = args.split_k, args.combine

    totals = evaluate(model, run, segs, args.runs, progress=True)
    results = {
        'model': args.model, 'context': CONTEXT, 'segments': len(segs),
        'split_k': args.split_k, 'combine': args.combine,
        'runs': {mode: t.report() for mode, t in totals.items()},
    }
    print(f'{"run":20} {"ppl":>14} {"KL":>20} {"top-1":>16} {"RMS dp":>8} {"s":>7}')
    for mode, r in results['runs'].items():
        print(f'{mode:20} {r["ppl"]:8.4f} ±{r["ppl_se"]:.3f} '
              f'{r["kl"]:10.3e} ±{r["kl_se"]:.1e} {r["top1"]:8.4%} ±{r["top1_se"]:.3%} '
              f'{r["rms_dp"]:8.3%} {r["seconds"]:7.0f}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
