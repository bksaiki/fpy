"""The grid (`serve/sweep.py`) and its plot (`serve/plot_grid.py`), without running them."""

import importlib.util
from pathlib import Path

import generator
from core import kernels

_SPEC = importlib.util.spec_from_file_location(
    'sweep', Path(__file__).resolve().parents[1] / 'sweep.py')
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_every_point_is_a_design(registry: None) -> None:
    """None invalid; `register()` adds each, and again without a clash."""
    pts = sweep.grid()
    assert not [p for p in pts if generator.invalid(p.design)]
    sweep.register()
    assert {generator.name(p.design) for p in pts} <= set(kernels.TILES)
    sweep.register()


def test_the_grid_draws_from_its_table_alone(tmp_path: Path) -> None:
    """Two formats, an anchor each over `F`, c added with the products and after them, frozen
    and free, and one that does not read `F`: a PNG."""
    spec = importlib.util.spec_from_file_location(
        'plot_grid', Path(__file__).resolve().parents[1] / 'plot_grid.py')
    assert spec is not None and spec.loader is not None
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    row = {'status': 'ok', 'floor_kl': '1e-12', 'k0_kl': '1e-2'}
    rows = [{**row, 'scheme': sch, 'anchor': a, 'value': f'F={f}{sw}', 'F': str(f), 'join': j,
             'is_anchor': str(f == 25 and not sw), 'kl': str(2.0 ** -f), 'free_kl': str(1e-2 + 2.0 ** -f)}
            for sch, a in (('fp8-row', 'nv.blackwell.mx.e4m3'), ('bf16', 'nv.hopper.bf16.f32'))
            for f in (10, 16, 25) for sw, j in (('', 'inside'), (' join=swapped', 'separate'))]
    rows.append({**row, 'scheme': 'bf16', 'anchor': 'amd.cdna2.bf16', 'value': 'F=24', 'F': '24', 'join': 'inside',
                 'is_anchor': 'True', 'kl': '1e-9', 'free_kl': '1e-2'})
    out = tmp_path / 'g.png'
    plot.figure(rows, out)
    assert out.stat().st_size > 0


def test_the_grid_is_113_designs() -> None:
    """Eight anchors over `F` (7 widths, an anchor's own added), CDNA2's tree
    at its own alone, each also with its join swapped: 113, each distinct,
    each anchor's own among them, its accumulator its own; every separate
    join keeps CDNA3's 7 guard bits."""
    pts = sweep.grid()
    assert len(pts) == len({p.design for p in pts}) == 113
    swapped = [p for p in pts if p.value.endswith('join=swapped')]
    assert len(swapped) == 60 and all(p.design.join != sweep.CATALOG[p.anchor].join for p in swapped)
    assert all(p.design.F2 - p.design.F == 7 for p in pts if p.design.join == 'separate')
    assert all(p.design.rho == sweep.CATALOG[p.anchor].rho for p in pts)
    assert {p.anchor for p in pts if p.is_anchor} == set(sweep.GRID_ANCHORS)
