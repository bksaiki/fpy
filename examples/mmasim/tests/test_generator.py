"""
The generator (`generator/`) against the paper's models: at every catalog
point it computes as `compile.DESIGNS`' builder does, bit for bit, under the
interpreter; every combination of its stages is a design, only sizes that
disagree refused; and structures agree where they must.

    pytest -n 8 tests/test_generator.py
"""

import os
import random
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import DESIGNS, _fmt, _length, _row, _same
from generator import Design, build, invalid
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
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 16, scales=fp.MX_E8M0), 'one instruction'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, scales=fp.MX_E8M0, G=3), 'does not divide'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 8, fp.FP32, 8, G=4), 'needs scales'),
    (Design(fp.BF16, fp.BF16, fp.FP32, 1, fp.FP32, 4, groups='even-odd'), 'two terms'),
])
def test_sizes_that_disagree_are_refused(design: Design, why: str) -> None:
    assert why in (invalid(design) or '')
    with pytest.raises(ValueError, match=why.replace('`', '.')):
        build(design)


def _combinations() -> list[Design]:
    """Every combination of the stages, unscaled (BF16) and scaled (MXFP8, per
    instruction and per group of 16), at `L` = 32."""
    out = []
    for sum_ in ('fused', 'tree', 'exact', 'fma'):
        for join_ in ('inside', 'separate'):
            for groups in ('none', 'even-odd'):
                for products in ('exact', 'rounded'):
                    for ftz in (False, True):
                        stages = {'sum': sum_, 'join': join_, 'groups': groups,
                                  'products': products, 'ftz': ftz}
                        out.append(Design(fp.BF16, fp.BF16, fp.FP32, 32, fp.FP32, 32, **stages))
                        for G in (None, 16):
                            out.append(Design(fp.MX_E4M3, fp.MX_E4M3, fp.FP32, 32, RZ, 32, F=25,
                                              scales=fp.MX_E8M0, G=G, **stages))
    return out


RZ = fp.IEEEContext(8, 32, fp.RM.RTZ)


@pytest.mark.parametrize('design', _combinations(), ids=lambda d: (
    f'{d.sum}-{d.join}-{d.groups}-{d.products}-{"ftz" if d.ftz else "noftz"}-'
    f'{"unscaled" if d.scales is None else "G" + str(d.G)}'))
def test_every_combination_is_a_design(design: Design) -> None:
    """None is refused; each builds and runs on every mix of inputs."""
    assert invalid(design) is None
    f, types = build(design)
    rng = random.Random(0)
    for mix in MIXES:
        for _ in range(3):
            float(f(*[_draw(t, rng, mix) for t in types]))


def test_a_chain_of_fmas_does_not_read_l() -> None:
    """An FMA rounds after every element, whatever the instruction's length."""
    one = CATALOG['fp32 (fma)']
    (f0, types), (f1, _) = build(one), build(replace(one, L=4))
    assert not (bad := _agree(f0, f1, types)), f'{len(bad)} differ, first: {bad[0]}'


def test_an_exact_sum_split_even_and_odd_is_the_exact_sum() -> None:
    """Exact sums of halves, added exactly: the same exact sum, rounded once."""
    cdna1 = CATALOG['amd.cdna1.f16']
    (f0, types), (f1, _) = build(cdna1), build(replace(cdna1, groups='even-odd'))
    assert not (bad := _agree(f0, f1, types)), f'{len(bad)} differ, first: {bad[0]}'
