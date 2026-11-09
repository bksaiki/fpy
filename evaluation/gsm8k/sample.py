"""
MMA-Sim's time on GSM8K, predicted, as a full run takes too long for every
design (`run.py --mmasim full` measures one).

MMA-Sim's run is FPy's, bit for bit, so it decodes as many steps.  It
costs a prefill of the shared prefix, one of each question, and a decode
step per token:

    T = prefill(prefix) + Σᵢ prefill(rowsᵢ) + steps × step

Both ways below start from FPy's prefill of the first batch, recording the
linear calls of 3 questions' prefills (the shortest, median and longest)
and of one decode step:

- :func:`layers`: MMA-Sim runs each distinct layer shape of those passes
  once, on the recorded operands; a pass's time is each shape's times its
  count.  `step` is the decode pass; `prefill` is the line through the 3
  questions' passes, which also predicts the prefix.
- :func:`in_model`: MMA-Sim runs the model itself, timing the 3 questions'
  prefills and decode steps until their mean is known to
  `config.GSM8K_SAMPLE_CI`; it checks :func:`layers` per kind of design.

Every call MMA-Sim runs is also compared, accumulator to accumulator, with
FPy's kernel on the same operands (`mmasim_matmul.Run.check`).
"""

import contextlib
import math
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import batched
import config
import mmasim_matmul
import torch
from transformers import DynamicCache

_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131)
"""Student's t, two-sided 95%, for 1 to 15 degrees of freedom."""


def half_width(xs: list[float]) -> float:
    """The 95% confidence interval of *xs*' mean, half its width over the
    mean."""
    df = len(xs) - 1
    t = _T95[df - 1] if df <= len(_T95) else 1.96
    return t * statistics.stdev(xs) / math.sqrt(len(xs)) / statistics.fmean(xs)


def enough(steps: list[float], least: int = config.GSM8K_SAMPLE_STEPS[0],
           most: int = config.GSM8K_SAMPLE_STEPS[1], ci: float = config.GSM8K_SAMPLE_CI) -> bool:
    """Whether *steps* time enough decode steps."""
    return len(steps) >= most or (len(steps) >= max(least, 2) and half_width(steps) <= ci)


def line(rows: list[int], seconds: list[float]) -> tuple[float, float]:
    """`(a, b)` of `seconds ≈ a + b · rows`, by least squares; `b = 0` if
    every row count is the same."""
    if len(set(rows)) < 2:
        return statistics.fmean(seconds), 0.0
    b, a = statistics.linear_regression(rows, seconds)
    return a, b


def predict(fpy: dict[str, Any], fit: tuple[float, float], step: float) -> dict[str, float]:
    """MMA-Sim's prefix, prefill and decode seconds over *fpy*'s record's
    batches: each prefill by *fit*, each of its decode steps *step*."""
    a, b = fit
    prefix = prefill = decode = 0.0
    for bt in fpy['batches']:
        n = bt['shared']
        own = fpy['prompt_tokens'][bt['first']:bt['first'] + len(bt['questions'])]
        prefix += a + b * n if n else 0.0
        prefill += sum(a + b * (k - n) for k in own)
        decode += len(bt['steps']) * step
    return {'prefix': prefix, 'prefill': prefix + prefill, 'decode': decode}


def picks(rows: list[int]) -> list[int]:
    """The shortest, median and longest of *rows*, as indices."""
    order = sorted(range(len(rows)), key=rows.__getitem__)
    return sorted({order[0], order[len(order) // 2], order[-1]})


Call = tuple[Any, Any, torch.Tensor, Any]
"""A recorded linear call: `qa`, `qw`, `w`, and (once run) an accumulator."""


@dataclass
class Prefilled:
    """FPy's prefill of a run's first batch, its sampled questions' calls."""

    prompts: list[list[int]]
    prefix: DynamicCache
    n: int
    """The shared prefix's length."""
    state: batched.State
    rows: list[int]
    """Each question's own tokens."""
    sampled: list[int]
    """The questions sampled (:func:`picks`)."""
    passes: dict[int, list[Call]]
    """Each sampled question's prefill's calls."""


@torch.no_grad()
def prefilled(model: torch.nn.Module, run: mmasim_matmul.Run, prompts: list[list[int]],
              design: str, fpy: dict[str, Any]) -> Prefilled:
    """*design*'s FPy prefill of *prompts*' first batch, as *fpy* (FPy's
    record of the run) batched it, recording the sampled questions' calls."""
    first = fpy['batches'][0]
    prompts = prompts[first['first']:first['first'] + len(first['questions'])]
    run.mode = design
    pre, n = batched.prefix(model, prompts)
    rows = [len(p) - n for p in prompts]
    sampled = picks(rows)
    caches, firsts, passes = [], [], {}
    try:
        for i, p in enumerate(prompts):
            run.calls = [] if i in sampled else None
            c, x = batched.question(model, pre, n, p)
            caches.append(c)
            firsts.append(x)
            if i in sampled:
                passes[i] = run.calls
    finally:
        run.calls = None
    return Prefilled(prompts, pre, n, batched.batch(prompts, caches, firsts), rows, sampled, passes)


def graphless(design: str) -> list[list]:
    """*design*'s call shapes whose graph did not fit: their operands'
    shapes."""
    return sorted([[s for s, _ in k[1:]] for k in mmasim_matmul.GRAPHLESS if k[0] == design])


def _shape(call: Call) -> tuple[int, int, int]:
    qa, qw = call[0], call[1]
    return qa.elements.shape[0], qw.elements.shape[0], qa.elements.shape[1]


def _pass(run: mmasim_matmul.Run, design: str, calls: list[Call], check: Counter[str],
          decode: bool = False) -> tuple[float, list[dict[str, Any]]]:
    """*calls*' MMA-Sim time: each distinct `(m, n, k)`'s first call run
    once untimed (warming the allocator; capturing a decode pass's graphs,
    all resident then, as in the model's decode), then each once timed,
    times its count; and per shape, its count, seconds, and whether a graph
    replayed it.  The timed calls are checked into *check*."""
    counts = Counter(map(_shape, calls))
    firsts = {}
    for c in calls:
        firsts.setdefault(_shape(c), c)
    shapes, timed = [], []
    with mmasim_matmul.replaying() if decode else contextlib.nullcontext():
        for qa, qw, _, _ in firsts.values():
            mmasim_matmul.accumulate(design, run.scheme, qa, qw)
        for shape, (qa, qw, w, _) in firsts.items():
            t = batched.clock()
            acc = mmasim_matmul.accumulate(design, run.scheme, qa, qw)
            t = batched.clock() - t
            timed.append((qa, qw, w, acc))
            shapes.append({'m': shape[0], 'n': shape[1], 'k': shape[2], 'count': counts[shape],
                           'seconds': t, 'graph': mmasim_matmul.graphed(design, run.scheme, qa, qw)})
    mmasim_matmul.release_graphs()
    run.calls = timed
    try:
        check.update(run.check())
    finally:
        run.calls = None
    return sum(s['count'] * s['seconds'] for s in shapes), shapes


@torch.no_grad()
def layers(model: torch.nn.Module, run: mmasim_matmul.Run, design: str, fpy: dict[str, Any],
           p: Prefilled) -> dict[str, Any]:
    """MMA-Sim's time for *design*, predicted per layer from *p*'s passes and
    one FPy decode step from *p*'s state (which it advances)."""
    run.mode, run.calls = design, []
    try:
        p.state.step(model)
        step_calls = run.calls
    finally:
        run.calls = None
    run.mode = mmasim_matmul.PREFIX + design
    run.drop_storage()
    check: Counter[str] = Counter()
    step, step_shapes = _pass(run, design, step_calls, check, decode=True)
    questions = [_pass(run, design, p.passes[i], check) for i in p.sampled]
    seconds = [t for t, _ in questions]
    fit = line([p.rows[i] for i in p.sampled], seconds)
    return {
        **predict(fpy, fit, step), 'step': step, 'fit': fit,
        'graphless': graphless(design),
        'passes': {'step': step_shapes,
                   'questions': [{'question': i, 'rows': p.rows[i], 'seconds': t, 'shapes': s}
                                 for i, (t, s) in zip(p.sampled, questions, strict=True)]},
        'check': dict(check),
    }


@torch.no_grad()
def in_model(model: torch.nn.Module, run: mmasim_matmul.Run, design: str, fpy: dict[str, Any],
             p: Prefilled) -> dict[str, Any]:
    """MMA-Sim's time for *design*, predicted from its own run of the model:
    *p*'s sampled questions' prefills, then decode steps from *p*'s state
    (which it advances) until :func:`enough`."""
    check: Counter[str] = Counter()

    def timed(f: Callable[[], object]) -> float:
        t = batched.clock()
        f()
        t = batched.clock() - t
        check.update(run.check())
        return t

    run.mode = mmasim_matmul.PREFIX + design
    run.drop_storage()
    try:
        run.calls = []   # prefill before any graph, as in a run
        questions = [timed(lambda q=p.prompts[i]: batched.question(model, p.prefix, p.n, q))
                     for i in p.sampled]
        run.calls = None
        if mmasim_matmul.TUNED:   # capture decode's graphs, outside the timing
            with mmasim_matmul.replaying():
                p.state.step(model)
        run.calls = []
        steps: list[float] = []
        with mmasim_matmul.replaying():
            while not enough(steps):
                steps.append(timed(lambda: p.state.step(model)))
    finally:
        run.calls = None
        mmasim_matmul.release_graphs()
    fit = line([p.rows[i] for i in p.sampled], questions)
    step = statistics.fmean(steps)
    return {
        **predict(fpy, fit, step), 'step': step, 'fit': fit,
        'graphless': graphless(design),
        'sample': {'questions': p.sampled, 'rows': [p.rows[i] for i in p.sampled],
                   'question_seconds': questions, 'steps': steps,
                   'half_width': half_width(steps) if len(steps) > 1 else None},
        'check': dict(check),
    }
