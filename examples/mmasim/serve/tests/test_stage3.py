"""Stage 3's sweep (`serve/stage3/sweep.py`): its points, without running them."""

import importlib.util
from pathlib import Path

import generator
import pytest
from core import kernels

_SPEC = importlib.util.spec_from_file_location(
    'stage3_sweep', Path(__file__).resolve().parents[1] / 'stage3' / 'sweep.py')
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

FIELDS = {'rho_bits': {'rho'}, 'rho_mode': {'rho'}, 'L': {'L', 'k'}}
"""The design fields a parameter moves, where not the one of its name."""

ALL = sweep.points(list(sweep.ANCHORS), list(sweep.PARAMS))


def _about(d: generator.Design) -> dict:
    out = generator.about(d)
    out.pop('exact')
    return out


@pytest.mark.parametrize('p', ALL, ids=lambda p: f'{p.anchor}-{p.param}-{p.value}')
def test_a_point_is_its_anchor_but_for_its_parameter(p: object) -> None:
    anchor, mine = _about(sweep.CATALOG[p.anchor]), _about(p.design)
    moved = {k for k in anchor if anchor[k] != mine[k]}
    assert moved <= FIELDS.get(p.param, {p.param})
    assert sweep.PARAMS[p.param][0](p.design) == p.value


def test_each_anchor_has_its_own_value_once_a_parameter() -> None:
    for a in sweep.ANCHORS:
        for param in sweep.PARAMS:
            mine = [p for p in ALL if (p.anchor, p.param) == (a, param)]
            if mine:
                assert sum(p.is_anchor for p in mine) == 1, (a, param)


def test_a_parameter_the_anchor_does_not_read_is_not_swept() -> None:
    assert not [p for p in ALL if p.param == 'F2' and sweep.CATALOG[p.anchor].join != 'separate']


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('TILES', '_BUILDS', 'REGISTERED', '_PICKLED', '_KERNELS'):
        monkeypatch.setattr(kernels, name, dict(getattr(kernels, name)))


def test_register_adds_each_formable_design_once(registry: None) -> None:
    sweep.register()
    formable = {generator.name(p.design) for p in ALL if generator.unformable(p.design) is None}
    assert formable <= set(kernels.TILES)
    sweep.register()    # again, as a second `local.py --register` would: no clash
