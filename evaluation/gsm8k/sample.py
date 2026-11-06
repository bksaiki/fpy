"""
MMA-Sim's time on GSM8K, predicted from a sample, as a full run takes too
long for every design.

MMA-Sim's run is FPy's, bit for bit, so it decodes as many steps.  It
costs a prefill of the shared prefix, one of each question, and a decode
step per token.  The sample times:

- 3 questions' prefills (the shortest, median and longest), fit to a line
  in their tokens, which predicts every prefill, the prefix's included;
- decode steps from the batch FPy's kernel has prefilled, until their mean
  is known to `config.GSM8K_SAMPLE_CI`.

Every sampled call is also recomputed by FPy's kernel and compared bit for
bit.
"""

import math
import statistics
from collections import Counter
from collections.abc import Callable
from typing import Any

import batched
import config
import mmasim_matmul
import torch

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


@torch.no_grad()
def measure(model: torch.nn.Module, run: mmasim_matmul.Run, prompts: list[list[int]],
            design: str, fpy: dict[str, Any]) -> dict[str, Any]:
    """MMA-Sim's record for *design*, predicted from a sample of *prompts*'
    first batch; *fpy* is FPy's record of the same run."""
    first = fpy['batches'][0]
    prompts = prompts[first['first']:first['first'] + len(first['questions'])]
    run.mode = design
    pre, n = batched.prefix(model, prompts)
    caches, firsts = zip(*(batched.question(model, pre, n, p) for p in prompts), strict=True)
    state = batched.batch(prompts, list(caches), list(firsts))

    check: Counter[str] = Counter()

    def timed(f: Callable[[], object]) -> float:
        t = batched.clock()
        f()
        t = batched.clock() - t
        check.update(run.check())
        return t

    run.mode, run.calls = mmasim_matmul.PREFIX + design, []
    try:
        rows = [len(p) - n for p in prompts]
        sampled = picks(rows)
        questions = [timed(lambda p=prompts[i]: batched.question(model, pre, n, p)) for i in sampled]
        steps: list[float] = []
        while not enough(steps):
            steps.append(timed(lambda: state.step(model)))
    finally:
        run.calls = None
    fit = line([rows[i] for i in sampled], questions)
    step = statistics.fmean(steps)
    return {
        'backend': 'mmasim', 'design': design, 'model': model.config.name_or_path,
        'problems': fpy['problems'], 'max_new': fpy['max_new'], 'batch': fpy['batch'],
        **predict(fpy, fit, step), 'sampled': True,
        'sample': {'questions': sampled, 'rows': [rows[i] for i in sampled], 'question_seconds': questions,
                   'fit': fit, 'steps': steps, 'step': step,
                   'half_width': half_width(steps) if len(steps) > 1 else None},
        'check': dict(check),
    }
