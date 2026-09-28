"""
Runs scored against references on the same token sequences, teacher-forced.

Each sequence runs through R0 (`fp32`) and then every other run, and per
predicted token :class:`Totals` accumulates the next token's NLL, KL(ref ||
run), top-1 agreement and Δp against R0 (llama.cpp's statistics), and per
sequence the mean NLL, KL and top-1 disagreement against each reference (R0,
and for a design the scheme's exact run) with its first disagreement.
:func:`against` pairs them over sequences.  Log-probabilities go through a
block of positions at a time; a reference's stay on the GPU, block by block.
"""

import itertools
import math
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from . import metrics, swap

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
    first: dict[str, list[int | None]] = field(default_factory=dict)
    """Per reference run, each segment's first top-1 disagreement, if any."""
    first_miss: list[int | None] = field(default_factory=list)
    """Each segment's first position whose top-1 is not the target, if any."""

    def add(self, nll: torch.Tensor, kl: torch.Tensor, same: torch.Tensor, dp: torch.Tensor) -> None:
        self.tokens += nll.numel()
        self.nll += nll.sum().item()
        self.nll_sq += (nll * nll).sum().item()
        self.kl += kl.sum().item()
        self.kl_sq += (kl * kl).sum().item()
        self.same_top1 += int(same.sum().item())
        self.dp_sq += (dp * dp).sum().item()

    def compare(self, refs: dict[str, list[torch.Tensor] | None], lps: Iterable[torch.Tensor],
                target: torch.Tensor) -> None:
        """Add one segment's next-token log-probabilities, *lps* in blocks of
        positions `[b, vocab]`, against each of *refs* (its log-probabilities
        in the same blocks, or `None` for these same ones; R0's first, which
        the token-level sums take), *target* the tokens predicted."""
        nll, seg = 0.0, {name: [0.0, 0] for name in refs}
        first: dict[str, int | None] = dict.fromkeys(refs)
        miss, i = None, 0
        for b, q in enumerate(lps):
            tgt = target[i:i + q.shape[0]]
            rows = torch.arange(q.shape[0], device=q.device)
            nll += -q[rows, tgt].sum().item()
            if miss is None and (hit := q.argmax(-1) == tgt).logical_not().any():
                miss = i + int(hit.logical_not().nonzero()[0])
            for j, (name, ref) in enumerate(refs.items()):
                r = q if ref is None else ref[b]
                kl, same = (r.exp() * (r - q)).sum(-1), r.argmax(-1) == q.argmax(-1)
                if j == 0:
                    self.add(nll=-q[rows, tgt], kl=kl, same=same,
                             dp=q[rows, tgt].exp() - r[rows, tgt].exp())
                seg[name][0] += kl.sum().item()
                seg[name][1] += int((~same).sum().item())
                if first[name] is None and not same.all():
                    first[name] = i + int((~same).nonzero()[0])
            i += q.shape[0]
        t = i
        self.nll_seg.append(nll / t)
        self.first_miss.append(miss)
        for name, (kl_sum, differ) in seg.items():
            v = self.vs.setdefault(name, {'kl': [], 'disagree': []})
            v['kl'].append(kl_sum / t)
            v['disagree'].append(differ / t)
            self.first.setdefault(name, []).append(first[name])

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


@torch.no_grad()
def _log_probs(model: torch.nn.Module, seg: torch.Tensor, first: int = 1) -> Iterator[torch.Tensor]:
    """Next-token log-probabilities of *seg*'s tokens from *first* on, in
    blocks of :data:`_ROWS` positions, `lm_head` applied a block at a time so
    the whole `[t, vocab]` is never held."""
    h = model.model(seg).last_hidden_state[0, first - 1:-1]
    for block in h.split(_ROWS):
        yield torch.log_softmax(model.lm_head(block).float(), -1)


def _keep(blocks: Iterable[torch.Tensor], kept: list[torch.Tensor]) -> Iterator[torch.Tensor]:
    """*blocks*, each also kept on *kept*."""
    for b in blocks:
        kept.append(b)
        yield b


def evaluate(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor],
    modes: Sequence[str], *, starts: Iterable[int] | None = None, progress: bool = False,
) -> dict[str, Totals]:
    """R0 and each of *modes* over *segs*: every mode's totals, `fp32`'s
    first, each against R0 and a design also against *run*'s scheme's exact
    run, if among *modes*.  Each segment is predicted from its position in
    *starts* on (1: every token but the first).  *run* is
    `swap.patch(model)`'s; its `split_k` / `combine` apply to every design."""
    exact = f'{run.scheme.name}-exact'
    rest = sorted((m for m in modes if m != 'fp32'), key=lambda m: m != exact)
    totals = {mode: Totals() for mode in ('fp32', *rest)}
    try:
        for i, (seg, first) in enumerate(zip(segs, starts or itertools.repeat(1))):
            target = seg[0, first:]
            refs: dict[str, list[torch.Tensor] | None] = {}
            for mode, t in totals.items():
                run.mode = mode
                start = time.perf_counter()
                kept: list[torch.Tensor] = []
                lps = _log_probs(model, seg, first)
                if mode in ('fp32', exact):
                    lps = _keep(lps, kept)
                t.compare({'fp32': None} if mode == 'fp32' else refs, lps, target)
                torch.cuda.synchronize()
                t.seconds += time.perf_counter() - start
                if kept:
                    refs[mode] = kept
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
