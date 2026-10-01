"""
The generator (`generator/`) against the paper's models: at every catalog
point it computes as `compile.DESIGNS`' builder does, bit for bit, under the
interpreter; a design it cannot form is refused; and two structures agree
where they must.

    pytest -n 8 tests/test_generator.py
"""

import os
import random
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import DESIGNS, _fmt, _length, _row, _same
from generator import Design, build, unformable
from generator.catalog import CATALOG

import fpy2 as fp

DRAWS = 100
"""Draws per mix of inputs, per design."""

MIXES = (0.0, 0.25, 1.0, 'subnormal')
"""Each draw's inputs: over each format's range and near one, a quarter hard
cases (zeros, extremes, specials), all hard cases, or subnormal-centred."""


def _draw(t: fp.types.Type, rng: random.Random, mix: float | str) -> float | list[float]:
    if not isinstance(mix, str):
        return _row(t, rng, mix)
    fmt = _fmt(t)
    normal = getattr(fmt, 'min_normal', None)
    if normal is None:
        return _row(t, rng, 0.0)
    hi = fmt.to_ordinal(normal())

    def one() -> float:
        o = rng.randint(0, hi)
        return float(fmt.from_ordinal(-o if rng.random() < 0.5 else o))
    return [one() for _ in range(_length(t))] if isinstance(t, fp.types.ListType) else one()


def _agree(f0: fp.Function, f1: fp.Function, types: list, seed: int = 0) -> list[str]:
    """The draws on which *f0* and *f1* differ, as text."""
    rng, bad = random.Random(seed), []
    for mix in MIXES:
        for _ in range(DRAWS):
            args = [_draw(t, rng, mix) for t in types]
            got, want = float(f1(*args)), float(f0(*args))
            if not _same(got, want):
                bad.append(f'{mix}: {got!r} != {want!r} on {args}')
    return bad


def test_the_catalog_is_every_design() -> None:
    assert list(CATALOG) == [name for name, _ in DESIGNS]


@pytest.mark.parametrize('name', [name for name, _ in DESIGNS])
def test_a_catalog_point_computes_as_its_model(name: str) -> None:
    f0, types = dict(DESIGNS)[name]()
    f1, got = build(CATALOG[name])
    assert repr(got) == repr(types)
    assert not (bad := _agree(f0, f1, types)), f'{len(bad)} differ, first: {bad[0]}'


def test_a_fused_sum_wide_enough_is_exact() -> None:
    """Keeping every bit, a T-FDPA chain rounded to nearest is the exact sum
    rounded once (E-FDPA)."""
    wide = Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 16, F=600)
    (f0, types), (f1, _) = build(replace(wide, sum='exact')), build(wide)
    assert not (bad := _agree(f0, f1, types)), f'{len(bad)} differ, first: {bad[0]}'


@pytest.mark.parametrize('design, why', [
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 12), 'whole instructions'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, products='rounded'), 'rounded products'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 6, fp.FP32, 6, sum='tree'), 'power of two'),
    (Design(fp.FP32, fp.FP32, fp.FP32, 2, fp.FP32, 4, sum='fma'), '`L` = 1'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, sum='exact', join='separate'), 'separate join'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, groups='even-odd'), 'even-odd'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 16, scales=fp.MX_E8M0), 'one instruction'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, scales=fp.MX_E8M0, G=3), 'does not divide'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, G=4), 'needs scales'),
])
def test_an_unformable_design_is_refused(design: Design, why: str) -> None:
    assert why in (unformable(design) or '')
    with pytest.raises(ValueError, match=why.replace('`', '.')):
        build(design)


@pytest.mark.parametrize('remix', [
    Design(fp.BF16, fp.BF16, fp.FP32, 16, fp.FP32, 32, join='separate', F=13),
    Design(fp.FP16, fp.FP16, fp.FP32, 8, nv_rz := fp.IEEEContext(8, 32, fp.RM.RTZ), 16, ftz=True),
    Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 16, sum='tree'),
    Design(fp.BF16, fp.BF16, fp.FP32, 4, nv_rz, 8, sum='exact', ftz=True),
])
def test_a_remix_builds_and_runs(remix: Design) -> None:
    """Mixes no model has (NVIDIA's chain joined as CDNA3 joins, FTZ on a
    fused sum, a wider tree of exact products) build and run on every mix."""
    f, types = build(remix)
    rng = random.Random(0)
    for mix in MIXES:
        float(f(*[_draw(t, rng, mix) for t in types]))
