"""
Input sources draw well-formed, reproducible arguments.

    pytest -n 8 evaluation/analysis_precision/tests/test_inputs.py
"""

import math

from compile import DESIGNS, _fmt
from inputs import SOURCES, AdversarialSource

import fpy2 as fp

# scalar and list arguments, an MX scale, and formats without infinities
_NAMES = ['nv.volta.f16.f32', 'nv.blackwell.mx.e4m3', 'amd.cdna3.fp8']


def _leaves(args):
    for a in args:
        yield from a if isinstance(a, list) else [a]


def test_each_draw_fits_its_argument():
    for source in SOURCES.values():
        for name in _NAMES:
            _, ts = dict(DESIGNS)[name]()
            for args in source.inputs(ts, 64, seed=0):
                assert len(args) == len(ts)
                for a, t in zip(args, ts, strict=True):
                    xs = a if isinstance(a, list) else [a]
                    assert not isinstance(a, list) or len(a) == t.length
                    assert all(_fmt(t).representable_in(fp.Float.from_float(x)) for x in xs)


def test_a_seed_reproduces_its_draws():
    _, ts = dict(DESIGNS)['nv.volta.f16.f32']()
    for source in SOURCES.values():
        def draw(seed, source=source):
            return [[repr(x) for x in _leaves(a)] for a in source.inputs(ts, 16, seed)]
        assert draw(1) == draw(1) != draw(2)


def test_random_reaches_hard_cases():
    _, ts = dict(DESIGNS)['nv.volta.f16.f32']()
    xs = [x for a in SOURCES['random'].inputs(ts, 64, seed=0) for x in _leaves(a)]
    assert any(x == 0 for x in xs) and any(math.isinf(x) or math.isnan(x) for x in xs)


def test_adversarial_draws_sit_at_one_end():
    _, ts = dict(DESIGNS)['nv.volta.f16.f32']()
    fmt = _fmt(ts[0])
    hi = fmt.to_ordinal(fmt.maxval())
    w = (hi + 1) // AdversarialSource.WINDOW
    edge_lo, edge_hi = float(fmt.from_ordinal(w)), float(fmt.from_ordinal(hi - w + 1))
    tiny, huge, mixed = SOURCES['adversarial'].inputs(ts, 3, seed=0)
    assert all(0 < abs(x) <= edge_lo for x in tiny[0])
    assert all(abs(x) >= edge_hi for x in huge[0])
    assert all(abs(x) <= edge_lo or abs(x) >= edge_hi for x in mixed[0])
