"""
MMA-Sim's prediction from a sample, on hand-written samples.

    pytest -n 8 -q evaluation/gsm8k/tests/test_sample.py
"""

import pytest
import sample


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
