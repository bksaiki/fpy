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
import statistics
import sys
import time
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from . import stats, swap

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
    """Each sequence's mean NLL."""
    vs: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Per reference run, each sequence's mean `kl` and top-1 `disagree`ment."""
    first: dict[str, list[int | None]] = field(default_factory=dict)
    """Per reference run, each sequence's first top-1 disagreement, if any."""
    first_miss: list[int | None] = field(default_factory=list)
    """Each sequence's first position whose top-1 is not the target, if any."""

    def per_sequence(self) -> dict[str, Any]:
        return {'nll': self.nll_seg, **self.vs, 'first': self.first}

    def compare(self, refs: dict[str, list[torch.Tensor] | None], lps: Iterable[torch.Tensor],
                target: torch.Tensor) -> None:
        """Add one sequence's next-token log-probabilities *lps* (blocks
        `[b, vocab]`) against each of *refs* (its blocks, or `None` for *lps*
        itself); the token-level sums take the first (R0).  *target*: the
        tokens predicted."""
        nll, seg = 0.0, {name: [0.0, 0] for name in refs}
        first: dict[str, int | None] = dict.fromkeys(refs)
        miss, i = None, 0
        for b, q in enumerate(lps):
            tgt = target[i:i + q.shape[0]]
            rows = torch.arange(q.shape[0], device=q.device)
            lq = q[rows, tgt]
            nll += -lq.sum().item()
            if miss is None and (hit := q.argmax(-1) == tgt).logical_not().any():
                miss = i + int(hit.logical_not().nonzero()[0])
            for j, (name, ref) in enumerate(refs.items()):
                r = q if ref is None else ref[b].to(q.device)
                kl, same = (r.exp() * (r - q)).sum(-1), r.argmax(-1) == q.argmax(-1)
                if j == 0:
                    dp = lq.exp() - r[rows, tgt].exp()
                    self.tokens += lq.numel()
                    self.nll -= lq.sum().item()
                    self.nll_sq += (lq * lq).sum().item()
                    self.kl += kl.sum().item()
                    self.kl_sq += (kl * kl).sum().item()
                    self.same_top1 += int(same.sum().item())
                    self.dp_sq += (dp * dp).sum().item()
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
def _log_probs(model: torch.nn.Module, seg: torch.Tensor, first: int = 1,
               dtype: torch.dtype = torch.float32) -> Iterator[torch.Tensor]:
    """Next-token log-probabilities of *seg*'s tokens from *first* on, in
    *dtype*, in blocks of :data:`_ROWS` positions (a quarter in FP64),
    `lm_head` applied a block at a time so the whole `[t, vocab]` is never
    held."""
    h = model.model(seg).last_hidden_state[0, first - 1:-1]
    for block in h.split(_ROWS // 4 if dtype == torch.float64 else _ROWS):
        yield torch.log_softmax(model.lm_head(block).to(dtype), -1)


def _keep(blocks: Iterable[torch.Tensor], kept: list[torch.Tensor]) -> Iterator[torch.Tensor]:
    """*blocks*, each also kept on *kept*, in host memory (a large vocabulary's
    references would not fit beside the model)."""
    for b in blocks:
        kept.append(b.cpu())
        yield b


def evaluate(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor],
    modes: Sequence[str], *, starts: Iterable[int] | None = None, seed: int = 0,
    progress: bool = False,
) -> dict[str, Totals]:
    """R0 and each of *modes* over *segs*: every mode's totals, `fp32`'s
    first, each against R0 and a design also against *run*'s scheme's exact
    run, if among *modes*.  Each segment is predicted from its position in
    *starts* on (1: every token but the first).  *run* is
    `swap.patch(model)`'s; its `split_k` / `combine` apply to every design.
    A random-sign run's signs are *seed*'s and the segment's (`Run.reseed`)."""
    exact = f'{run.scheme.name}-exact'
    rest = sorted((m for m in modes if m != 'fp32'), key=lambda m: m != exact)
    totals = {mode: Totals() for mode in ('fp32', *rest)}
    try:
        for i, (seg, first) in enumerate(zip(segs, starts or itertools.repeat(1))):
            target = seg[0, first:]
            refs: dict[str, list[torch.Tensor] | None] = {}
            for mode, t in totals.items():
                run.mode = mode
                run.reseed(seed, i)
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


@dataclass
class Divergence:
    """One run's distance from the exact run, per sequence: mean KL(R1 ||
    run) and top-1 disagreement, and, frozen, the share of its activation
    elements (and scales) that would round differently; its seconds over
    them all."""

    kl: list[float] = field(default_factory=list)
    disagree: list[float] = field(default_factory=list)
    rerounded: list[float] = field(default_factory=list)
    rescaled: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, ref: list[torch.Tensor], lps: Iterable[torch.Tensor],
            scored: torch.Tensor | None = None) -> None:
        """One sequence: *lps* against *ref*'s blocks (pinned, on the host),
        summed on the GPU and read once, over the positions *scored* marks
        (every one if `None`)."""
        kl = differ = t = 0
        for r, q in zip(ref, lps, strict=True):
            r = r.to(q.device, non_blocking=True)
            k, d = (r.exp() * (r - q)).sum(-1), r.argmax(-1) != q.argmax(-1)
            if scored is not None:
                m = scored[t:t + q.shape[0]].to(q.device)
                k, d = k[m], d[m]
            kl = kl + k.sum()
            differ = differ + d.sum()
            t += q.shape[0]
        n = t if scored is None else int(scored.sum())
        self.kl.append(float(kl) / n)
        self.disagree.append(float(differ) / n)

    def report(self) -> dict[str, float]:
        """Means over sequences, each with its standard error, and the
        total seconds."""
        out: dict[str, float] = {}
        for k in ('kl', 'disagree', 'rerounded', 'rescaled'):
            if v := getattr(self, k):
                out[k], out[f'{k}_se'] = stats.paired(v)
        out['seconds'] = self.seconds
        return out


def frozen(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor], modes: Sequence[str],
    free: Collection[str] = (), *, scored: Sequence[torch.Tensor] | None = None, seed: int = 0,
    progress: bool = False,
) -> dict[str, dict[str, Divergence]]:
    """Each of *modes* against *run*'s scheme's exact run over *segs*,
    frozen to it (`swap.Frozen`), and each of *free* as it runs:
    `{'frozen': ..., 'free': ...}`, by mode; per segment, over the positions
    its *scored* marks (each predicting the next token), if given.  Log-probabilities in FP64: in
    FP32, their rounding alone is a KL near 1e-9, above what a frozen run
    can be.  A random-sign run's signs are *seed*'s and the segment's
    (`Run.reseed`), frozen or free alike."""
    exact = f'{run.scheme.name}-exact'
    out: dict[str, dict[str, Divergence]] = {'frozen': {}, 'free': {}}
    pinned: list[torch.Tensor] = []
    try:
        for i, seg in enumerate(segs):
            run.mode, run.record, run.frozen = exact, swap.Frozen(), None
            ref = [_pin(pinned, j, b) for j, b in enumerate(_log_probs(model, seg, dtype=torch.float64))]
            rec, run.record = run.record, None
            for mode in dict.fromkeys([*modes, *free]):
                run.mode = mode
                for kind in [k for k, of in (('frozen', modes), ('free', free)) if mode in of]:
                    start = time.perf_counter()
                    run.frozen = rec.replay() if kind == 'frozen' else None
                    run.reseed(seed, i)
                    d = out[kind].setdefault(mode, Divergence())
                    d.add(ref, _log_probs(model, seg, dtype=torch.float64), scored and scored[i])
                    torch.cuda.synchronize()
                    d.seconds += time.perf_counter() - start
                    if run.frozen is not None:
                        a, n, b, m = run.frozen.rerounded
                        d.rerounded.append(float(a) / n)
                        if m:
                            d.rescaled.append(float(b) / m)
            del rec
            torch.cuda.empty_cache()
            if progress:
                print(f'segment {i + 1}', file=sys.stderr, flush=True)
    finally:
        run.mode, run.record, run.frozen = 'fp32', None, None
    return out


def _pin(pinned: list[torch.Tensor], i: int, b: torch.Tensor) -> torch.Tensor:
    """*b* copied to the *i*-th of *pinned*'s host buffers, made or remade
    to its shape: the copies back to the GPU then run without blocking."""
    if i >= len(pinned) or pinned[i].shape != b.shape:
        buf = torch.empty(b.shape, dtype=b.dtype, pin_memory=True)
        pinned[i:i + 1] = [buf]
    return pinned[i].copy_(b)


def median_first(firsts: Sequence[int | None]) -> float:
    """The median of the first disagreements that happen (NaN if none do)."""
    hit = [f for f in firsts if f is not None]
    return statistics.median(hit) if hit else math.nan


def against(totals: dict[str, Totals]) -> dict[str, dict[str, dict[str, float]]]:
    """Paired over sequences, per reference and run (:func:`evaluate`'s
    comparisons, a run against itself left out): the mean and standard error
    of Δ NLL, KL, top-1 disagreement, and the fraction of sequences that
    disagree at all; Δ NLL's p-value, Holm-adjusted over the runs sharing
    the reference; and the median first disagreement, with a bootstrap 95%
    interval."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ref, base in totals.items():
        rows: dict[str, dict[str, float]] = {}
        for mode, t in totals.items():
            if mode == ref or ref not in t.vs:
                continue
            r = rows[mode] = {}
            r['dnll'], r['dnll_se'] = stats.paired(t.nll_seg, base.nll_seg)
            for k in ('kl', 'disagree'):
                r[k], r[f'{k}_se'] = stats.paired(t.vs[ref][k])
            firsts = t.first[ref]
            r['diverged'], r['diverged_se'] = stats.paired([float(f is not None) for f in firsts])
            r['median'] = median_first(firsts)
            r['median_lo'], r['median_hi'] = stats.bootstrap(firsts, median_first)
            r['p'] = stats.p_value(r['dnll'], r['dnll_se'])
        for r, p in zip(rows.values(), stats.holm([r['p'] for r in rows.values()])):
            r['p_holm'] = p
        if rows:
            out[ref] = rows
    return out
