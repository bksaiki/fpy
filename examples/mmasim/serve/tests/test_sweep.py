"""The design-parameter sweep (`serve/sweep.py`, `serve/plot_sweep.py`), without running it."""

import importlib.util
from pathlib import Path

import generator
import pytest
from core import kernels

_SPEC = importlib.util.spec_from_file_location(
    'sweep', Path(__file__).resolve().parents[1] / 'sweep.py')
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

FIELDS = {'rho_bits': {'rho'}, 'rho_mode': {'rho'}, 'L': {'L', 'k'}, 'F': {'F', 'F2'}, 'guard': {'F2'}}
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
    assert not [p for p in ALL if p.param == 'guard' and sweep.CATALOG[p.anchor].join != 'separate']


def test_a_separate_join_widens_with_its_sum() -> None:
    """Along `F`, CDNA3's join keeps its 7 guard bits; another anchor's
    `F2`, unread, is left as it is."""
    for p in ALL:
        if p.param == 'F':
            base = sweep.CATALOG[p.anchor]
            if base.join == 'separate':
                assert p.design.F2 - p.design.F == base.F2 - base.F == 7
            else:
                assert p.design.F2 == base.F2


def test_every_point_is_a_design(registry: None) -> None:
    """None invalid; `register()` adds each, and again without a clash."""
    assert not [p for p in ALL if generator.invalid(p.design)]
    sweep.register()
    assert {generator.name(p.design) for p in ALL} <= set(kernels.TILES)
    sweep.register()


def test_a_figure_draws_from_the_table_alone(tmp_path: Path) -> None:
    """A numeric parameter and a categorical one, a point refused: a PNG."""
    spec = importlib.util.spec_from_file_location(
        'plot_sweep', Path(__file__).resolve().parents[1] / 'plot_sweep.py')
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    row = {'anchor': 'a', 'scheme': 'bf16', 'is_anchor': 'False', 'status': 'ok'}
    rows = [{**row, 'param': 'F', 'value': str(v), 'backward_normwise': str(-v),
             'is_anchor': str(v == 13)} for v in (8, 13, 25)]
    rows += [{**row, 'param': 'sum', 'value': 'fused', 'backward_normwise': '-13'},
             {**row, 'param': 'sum', 'value': 'exact', 'status': 'refused', 'backward_normwise': ''}]
    out = tmp_path / 'f.png'
    plot.figure(rows, 'backward_normwise', out)
    assert out.stat().st_size > 0


def test_the_grid_draws_from_its_table_alone(tmp_path: Path) -> None:
    """Two formats, an anchor each over `F`, its join own and swapped, frozen
    and free, and one that does not read `F`: a PNG."""
    spec = importlib.util.spec_from_file_location(
        'plot_grid', Path(__file__).resolve().parents[1] / 'plot_grid.py')
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    row = {'status': 'ok', 'floor_kl': '1e-12', 'k0_kl': '1e-2'}
    rows = [{**row, 'scheme': sch, 'anchor': a, 'value': f'F={f}{sw}', 'F': str(f),
             'is_anchor': str(f == 25 and not sw), 'kl': str(2.0 ** -f), 'free_kl': str(1e-2 + 2.0 ** -f)}
            for sch, a in (('fp8-row', 'nv.blackwell.e4m3.f32'), ('bf16', 'nv.hopper.bf16.f32'))
            for f in (10, 16, 25) for sw in ('', ' join=swapped')]
    rows.append({**row, 'scheme': 'bf16', 'anchor': 'amd.cdna2.bf16', 'value': 'F=24', 'F': '24',
                 'is_anchor': 'True', 'kl': '1e-9', 'free_kl': '1e-2'})
    out = tmp_path / 'g.png'
    plot.figure(rows, out)
    assert out.stat().st_size > 0


def test_the_grid_is_113_designs() -> None:
    """Eight anchors over `F` (7 widths, an anchor's own added), CDNA2's tree
    at its own alone, each also with its join swapped: 113, each distinct,
    each anchor's own among them, its accumulator its own; a swapped
    separate join keeps 7 guard bits."""
    pts = sweep.grid()
    assert len(pts) == len({p.design for p in pts}) == 113
    swapped = [p for p in pts if p.value.endswith('join=swapped')]
    assert len(swapped) == 60 and all(p.design.join != sweep.CATALOG[p.anchor].join for p in swapped)
    assert all(p.design.F2 - p.design.F == 7 for p in swapped if p.design.join == 'separate')
    assert all(p.design.rho == sweep.CATALOG[p.anchor].rho for p in pts)
    assert {p.anchor for p in pts if p.is_anchor} == set(sweep.GRID[0])
