"""
MMA-Sim's prediction from a sample, on hand-written samples.

    pytest -n 8 -q evaluation/gsm8k/tests/test_sample.py
"""

from types import SimpleNamespace

import mmasim_matmul
import pytest
import sample
import torch


def test_line():
    assert sample.line([10, 20, 40], [3.0, 5.0, 9.0]) == pytest.approx((1.0, 0.2))
    assert sample.line([7, 7], [2.0, 4.0]) == (3.0, 0.0)


def test_picks():
    assert sample.picks([50, 10, 30, 90, 70]) == [0, 1, 3]
    assert sample.picks([5, 6]) == [0, 1]


def test_enough():
    assert not sample.enough([1.0, 1.0, 1.0])                    # fewer than the least
    assert sample.enough([1.0] * 4)                              # no spread
    assert not sample.enough([1.0, 1.1, 0.9, 1.0])               # CI ±10%
    assert sample.enough([1.0, 1.1, 0.9, 1.0] * 4)               # the most
    assert sample.half_width([1.0, 1.1, 0.9, 1.0]) == pytest.approx(3.182 * 0.0816497 / 2, rel=1e-4)


def test_predict():
    fpy = {'prompt_tokens': [100, 120, 110, 90],
           'batches': [{'first': 0, 'shared': 80, 'questions': [0, 0], 'steps': [0] * 5},
                       {'first': 2, 'shared': 85, 'questions': [0, 0], 'steps': [0] * 3}]}
    got = sample.predict(fpy, (2.0, 0.5), 10.0)
    prefix = (2 + 0.5 * 80) + (2 + 0.5 * 85)
    questions = (2 + 0.5 * 20) + (2 + 0.5 * 40) + (2 + 0.5 * 25) + (2 + 0.5 * 5)
    assert got == pytest.approx({'prefix': prefix, 'prefill': prefix + questions, 'decode': 80.0})


def _call(m: int, n: int, k: int) -> tuple:
    return (SimpleNamespace(elements=torch.empty(m, k)), SimpleNamespace(elements=torch.empty(n, k)),
            torch.empty(n, k), None)


def test_a_pass_replays_every_call(monkeypatch: pytest.MonkeyPatch):
    """Every call runs once untimed, then once timed in order; the pass is
    the timed span; shapes are counted; the timed calls are checked."""
    ran, ticks = [], iter([1.0, 3.5])
    monkeypatch.setattr(mmasim_matmul, 'accumulate', lambda d, s, qa, qw: ran.append(qa) or qa.elements)
    monkeypatch.setattr(sample.batched, 'clock', lambda: next(ticks))
    checked = []
    run = SimpleNamespace(scheme=None, calls=None,
                          check=lambda: checked.append(len(run.calls)) or {'calls': len(run.calls)})
    calls = [_call(32, 8, 16), _call(32, 8, 16), _call(32, 4, 16), _call(1, 8, 16), _call(32, 8, 16)]
    check = sample.Counter()
    seconds, shapes = sample._pass(run, 'd', calls, check)
    assert [(s['m'], s['n'], s['k'], s['count']) for s in shapes] == \
        [(32, 8, 16, 3), (32, 4, 16, 1), (1, 8, 16, 1)]
    assert seconds == 2.5 and ran == [c[0] for c in calls] * 2
    assert checked == [5] and check == {'calls': 5} and run.calls is None
