"""
The designs' Triton matmuls, as a linear layer's `x @ W.T`.

Each design compiles once with `m`, `n` and `k` symbolic; `k` must be a
multiple of the design's length.  A scaled product also compiles once per
static `k`, in one kernel (:func:`fused`).  Its inputs hold values of its input
formats in the kernel's storage dtype; the result is its FP32 accumulator.
:func:`linear` does it all per call; `swap` calls :func:`prepare` once per
weight and :func:`matmul` per call.
"""

import hashlib
import json
import os
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compile_triton as ct
import generator
from compile import DESIGNS, in_processes
from compile import _length as _arg_length

import fpy2
from fpy2.backend.triton import KernelSource, launch
from fpy2.backend.triton.launcher import _torch_dtype

from . import outputs, quant

TILES: dict[str, tuple[int, int]] = {
    'nv.ampere.bf16.f32': (32, 1),
    'nv.hopper.bf16.f32': (32, 1),
    'amd.cdna2.bf16': (16, 64),
    'amd.cdna2.bf16_1k': (16, 64),
    'amd.cdna3.bf16': (64, 1),
    'nv.ada.e4m3.f32': (128, 1),
    'nv.hopper.e4m3.f32': (32, 1),
    'nv.blackwell.e4m3.f32': (32, 1),
    'amd.cdna3.fp8': (64, 1),
    'nv.blackwell.mx.e4m3': (32, 1),
    'nv.blackwell.mx.e2m1': (32, 1),
    'nv.blackwell.mxfp4': (16, 1),
    'nv.blackwell.nvfp4': (16, 1),
}
"""Each design and its fixed tile `(block, block_m)`, the fastest per
`bench/speed.py --best`."""

FUSED_ROWS: dict[str, int] = {'nv.blackwell.mxfp4': 32, 'nv.blackwell.nvfp4': 32}
"""Past this many rows a scaled design's launches beat its one kernel
(:func:`fused`); an unlisted design always runs in one kernel."""

Combine = Literal['linear', 'tree']

_BUILDS: dict[str, Callable] = dict(DESIGNS)

_KERNELS: dict[tuple[str, int | None], KernelSource] = {}
"""Each design compiled so far: over any `k` (`None`), or fused over one."""

_SECONDS: dict[tuple[str, int | None], float] = {}
"""Each of :data:`_KERNELS`' compile time, where known (a kernel cached
before its time was kept has none)."""

REGISTERED: dict[str, dict[str, Any] | None] = {}
"""Each :func:`register`ed design's description (its parameters), by name."""

_PICKLED: dict[str, Callable] = {}
"""The builds that pickle, by design: compiled in worker processes."""

CACHE: Path | None = None
"""Where compiled kernels persist across runs (`cli.parse` sets it), keyed
by the design and every source a kernel depends on; `None`: they do not."""

_ROWS = 65535
"""Rows per launch at most: CUDA's grid limit on its second axis."""

_ELEMS = 1 << 26
"""Output elements per block of rows: bounds the partials' memory under
`split_k > 1`, and a launch's size."""


def register(name: str, build: Callable, block: int, block_m: int,
             about: dict[str, Any] | None = None) -> None:
    """Add a design, as `compile.DESIGNS` defines them (*build* returns the
    FPy function and its argument types), run at the tile `(block, block_m)`;
    *about* (its parameters) describes it in results and keys its disk
    cache, without which it is not cached on disk."""
    if name in TILES:
        raise ValueError(f'{name!r} is already a design')
    _BUILDS[name] = build
    TILES[name] = (block, block_m)
    REGISTERED[name] = about


def tile(d: generator.Design) -> tuple[int, int]:
    """*d*'s tile `(block, block_m)`, the fastest of its kind at the model's
    layer shapes: wider the more often `c` is added (short `L`, `c` after the
    products), narrower for `L` ≥ 64."""
    if d.sum == 'tree':
        return (128 if d.join == 'separate' else 32), 1
    if d.L >= 64:
        return 16, 1
    if d.L >= 32:
        return 32, 1
    return (64 if d.L >= 16 or d.join == 'separate' else 32), 1


def register_design(d: generator.Design, name: str | None = None) -> str:
    """Register generator design *d* under *name* (default
    `generator.name`), at its :func:`tile`, described by `generator.about`;
    its name."""
    name = name or generator.name(d)
    build = partial(generator.build, d)
    register(name, build, *tile(d), about=generator.about(d))
    _PICKLED[name] = build
    return name


@cache
def _args(design: str) -> list[Any]:
    """*design*'s argument types."""
    if design not in TILES:
        raise ValueError(f'{design!r} is not one of {sorted(TILES)}')
    return _BUILDS[design]()[1]


def formats(design: str) -> tuple[Any, Any, Any]:
    """*design*'s activation, weight and accumulator formats."""
    a, b, c = _args(design)[:3]
    return a.elt.fmt, b.elt.fmt, c.fmt


def block_scaled(design: str) -> bool:
    """Whether *design* takes scales, as an instruction's own arguments."""
    return len(_args(design)) == 5


def group(design: str) -> int:
    """How many elements of `k` one of a block-scaled *design*'s scales
    covers in a call."""
    k0, scale = _arg_length(_args(design)[0]), _args(design)[3]
    return k0 // scale.length if hasattr(scale, 'length') else k0


def applicable(design: str, scheme: quant.Scheme) -> bool:
    """Whether *design* takes *scheme*'s elements, and its scales: applied
    outside an unscaled design, or taken by a block-scaled one in its format,
    in groups dividing the scheme's blocks."""
    x, w, _ = formats(design)
    if (x, w) != (scheme.x.elements.format(), scheme.w.elements.format()):
        return False
    if not block_scaled(design):
        return scheme.applied in ('none', 'epilogue', 'k-blocks')
    s, arg = scheme.x.scaling, _args(design)[3]
    scale = getattr(arg, 'elt', arg)
    return (scheme.applied == 'instruction' and scale.fmt == s.fmt.format()
            and s.cols % group(design) == 0)


def designs(scheme: quant.Scheme) -> list[str]:
    return [d for d in TILES if applicable(d, scheme)]


def compiled(design: str) -> tuple[KernelSource, int]:
    """*design*'s kernel and the length its `k` must be a multiple of (a
    block-scaled design's, one instruction)."""
    return _kernel(design), _arg_length(_args(design)[0])


def fused(design: str, k: int) -> KernelSource:
    """*design* over a static *k* in one kernel (`compile_triton.fuse`),
    launched on `(a [m, k], w [n, k], C, xs, ys, out)`, a scale list per row."""
    return _kernel(design, k)


def _kernel(design: str, k: int | None = None) -> KernelSource:
    """*design*'s kernel over any `k` (`None`), or fused over *k*; cached, in
    memory and in :data:`CACHE`."""
    if (design, k) not in _KERNELS and not _load(design, k):
        start = time.perf_counter()
        _KERNELS[design, k] = _compile(_BUILDS[design], k)
        _SECONDS[design, k] = time.perf_counter() - start
        _store(design, k)
    return _KERNELS[design, k]


def _compile(build: Callable, k: int | None) -> KernelSource:
    return ct.compile_matmul(build, None)[0] if k is None else ct.compile_fused(build, k)


def compile_seconds(design: str) -> float | None:
    """The seconds *design*'s kernels compiled in so far took to compile (`None`
    if any's time is unknown, or none has)."""
    ks = [key for key in _KERNELS if key[0] == design]
    if not ks or any(key not in _SECONDS for key in ks):
        return None
    return sum(_SECONDS[key] for key in ks)


@cache
def _sources() -> str:
    """A digest of every source a kernel depends on: FPy, the models, the
    generator, the compilers."""
    here = Path(__file__).resolve().parents[2]
    return outputs.sources(Path(fpy2.__file__).parent, here / 'models', here / 'generator',
                           here / 'compile.py', here / 'compile_triton.py')


def _path(design: str, k: int | None) -> Path | None:
    """Where :data:`CACHE` keeps *design*'s kernel over *k*, if it does."""
    if CACHE is None or (design in REGISTERED and REGISTERED[design] is None):
        return None
    key = json.dumps([design, k, REGISTERED.get(design), _sources()], sort_keys=True, default=str)
    return CACHE / f'{hashlib.sha256(key.encode()).hexdigest()[:24]}.pkl'


def _load(design: str, k: int | None) -> bool:
    """Whether *design*'s kernel over *k* was in :data:`CACHE` (now in memory,
    with its compile time if kept beside it)."""
    path = _path(design, k)
    if path is None or not path.exists():
        return False
    _KERNELS[design, k] = pickle.loads(path.read_bytes())
    if (t := path.with_suffix('.json')).exists():
        _SECONDS[design, k] = json.loads(t.read_text())['seconds']
    return True


def _store(design: str, k: int | None) -> None:
    """*design*'s kernel over *k* into :data:`CACHE`, whole or not at all, and
    its compile time beside it."""
    path = _path(design, k)
    if path is not None:
        outputs.replace(path, lambda f: pickle.dump(_KERNELS[design, k], f), binary=True)
        if (design, k) in _SECONDS:
            outputs.write_json(path.with_suffix('.json'), {'seconds': _SECONDS[design, k]})


def precompile(designs: Sequence[str], jobs: int | None = None, ks: Sequence[int] = ()) -> None:
    """Compile *designs* not yet compiled, and each fused over each of *ks*
    (:func:`fused`), in *jobs* processes (one per core by default): MMA-Sim's
    and :func:`register_design`'s; another :func:`register`ed one compiles
    here."""
    todo = [(d, k) for d in designs for k in (None, *ks) if (d, k) not in _KERNELS and not _load(d, k)]
    _compile_all(todo, jobs)
    for t in todo:
        _kernel(*t)     # a refused one again, to raise its error


def compile_designs(designs: Sequence[str], jobs: int | None = None) -> dict[str, str]:
    """Compile each of *designs* (its kernel over any `k`) not yet compiled,
    as :func:`precompile` does; the refused ones' errors, by design."""
    todo = [(d, None) for d in designs if (d, None) not in _KERNELS and not _load(d, None)]
    return {d: why for (d, _), why in _compile_all(todo, jobs).items()}


def _compile_all(todo: Sequence[tuple[str, int | None]], jobs: int | None) -> dict[tuple[str, int | None], str]:
    """Compile *todo*, in *jobs* processes where a worker can rebuild it (from
    its pickled build or MMA-Sim name), else here, keeping the compiled in
    :data:`CACHE`; the refused ones' errors."""
    named = [(d, k, _PICKLED.get(d)) for d, k in todo
             if d in _PICKLED or _BUILDS[d] is dict(DESIGNS).get(d)]
    jobs = min(len(named), jobs or os.cpu_count() or 1)
    done = dict(zip([t[:2] for t in named], in_processes(_compile_or_refuse, named, jobs)))
    done.update({t: _compile_or_refuse((*t, _BUILDS[t[0]])) for t in todo if t not in done})
    refused = {}
    for (d, k), (kernel, why, seconds) in done.items():
        if kernel is None:
            refused[d, k] = why
        else:
            _KERNELS[d, k], _SECONDS[d, k] = kernel, seconds
            _store(d, k)
    return refused


def _compile_or_refuse(
    item: tuple[str, int | None, Callable | None],
) -> tuple[KernelSource | None, str, float]:
    """A design's kernel, from its build, else (in a worker) by its MMA-Sim
    name, and the seconds it took; or `None` and why it was refused."""
    design, k, build = item
    start = time.perf_counter()
    try:
        return _compile(build or _BUILDS[design], k), '', time.perf_counter() - start
    except Exception as ex:  # noqa: BLE001 -- a refusal is a result
        return None, f'{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ""}', 0.0


@cache
def storage(design: str) -> tuple[torch.dtype, ...]:
    """The torch dtypes *design*'s kernel holds its activations and weights
    in, and a block-scaled one's scales."""
    held = dict(compiled(design)[0].dtypes)
    return tuple(_torch_dtype(held[i]) for i in ((0, 1, 3) if block_scaled(design) else (0, 1)))


def prepare(w: torch.Tensor, dtype: torch.dtype, split_k: int = 1) -> torch.Tensor:
    """*w* `[n, k]`, holding a design's weight values, in *dtype* as
    `[split_k, n, k / split_k]` contiguous slices of `k`: a view of *w* at
    `split_k = 1` when it is in *dtype* already."""
    n, k = w.shape
    if split_k < 1 or k % split_k:
        raise ValueError(f'`k` = {k} does not split into {split_k} slices')
    b = w.to(dtype)
    if split_k == 1:
        return b.unsqueeze(0)
    return b.view(n, split_k, k // split_k).transpose(0, 1).contiguous()


def _launch(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, design: str,
            *scales: torch.Tensor, kernel: KernelSource | None = None) -> None:
    """Accumulate `a @ b.T` by *design* (block-scaled: with *scales*), or by
    its *kernel*, onto *y* `[m, n]`, which is both `C` and the output.
    `block_m` is capped at the power of two `>= m`."""
    block, block_m = TILES[design]
    m = a.shape[0]
    launch(kernel or compiled(design)[0], [a, b, y, *scales, y],
           block=block, block_m=min(block_m, 1 << (m - 1).bit_length()))


def _part(a: torch.Tensor, w: torch.Tensor, design: str, j: int) -> torch.Tensor:
    """Slice *j*'s partial, `a[j] @ w[j].T` from a zero accumulator."""
    out = torch.zeros(a.shape[1], w.shape[1], dtype=torch.float32, device=a.device)
    _launch(a[j], w[j], out, design)
    return out


def _tree(part: Callable[[int], torch.Tensor], s: int, n: int) -> torch.Tensor:
    """Partials `[s, s + n)` summed pairwise, the left half the largest power
    of two below `n`."""
    if n == 1:
        return part(s)
    h = 1 << ((n - 1).bit_length() - 1)
    return _tree(part, s, h).add_(_tree(part, s + h, n - h))


def matmul(
    a: torch.Tensor, w: torch.Tensor, design: str, combine: Combine = 'linear',
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """`a @ w.T` by *design*, FP32 `[m, n]`, for `a` holding its activation values
    and `w` from :func:`prepare` in *design*'s :func:`storage`.

    With `w` in `s > 1` slices, each slice goes through the kernel from a zero
    accumulator and the FP32 partials are summed left to right (`linear`) or
    pairwise (`tree`); one slice accumulates `k` in order through the design
    alone.  With *scales*, `w` is whole (:func:`_scaled`)."""
    if scales is not None:
        return _scaled(a, w, design, *scales)
    s, n, step = w.shape
    m, k = a.shape
    k0 = compiled(design)[1]
    if k != s * step or step % k0:
        raise ValueError(
            f"`k` = {k} does not split into {s} slices of a multiple of {design}'s length {k0}")
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    parts = a.view(m, s, step).transpose(0, 1).contiguous() if s > 1 else a.unsqueeze(0)
    rows = min(max(1, _ELEMS // n), _ROWS)
    for i in range(0, m, rows):
        blk, ab = y[i:i + rows], parts[:, i:i + rows]
        if combine == 'tree' and s > 1:
            blk.copy_(_tree(partial(_part, ab, w, design), 0, s))
        else:
            _launch(ab[0], w[0], blk, design)
            for j in range(1, s):
                blk += _part(ab, w, design, j)
    return y


def _scaled(a: torch.Tensor, w: torch.Tensor, design: str,
            xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """:func:`matmul` with scales, `w` whole (`[1, n, k]`) and *xs* `[m, s]`,
    *ys* `[n, s]`, each row's scales in order along `k`.  At most
    :data:`FUSED_ROWS` rows, one kernel (:func:`fused`); past them a launch per
    instruction, each onto the last's result, or per `compile_triton.PROMOTE`
    of `k` from zero, each partial scaled as it is summed, `y + p * (s_x *
    s_w)` in FP32."""
    if w.shape[0] != 1:
        raise ValueError(f'{design} takes a scaled product\'s weights whole, not in slices')
    (_, n, k), m = w.shape, a.shape[0]
    chained = block_scaled(design)
    step = _arg_length(_args(design)[0]) if chained else ct.PROMOTE
    if a.shape[1] != k or k % step:
        raise ValueError(f"`k` = {a.shape[1]} is not whole blocks of {step} for {design}")
    if not chained and xs.shape[1] * step != k:
        raise ValueError(f'scales per {k // xs.shape[1]} of `k`, where {design} promotes every {step}')
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    rows = min(max(1, _ELEMS // n), _ROWS)
    if m <= FUSED_ROWS.get(design, m):
        kernel = fused(design, k)
        for i in range(0, m, rows):
            _launch(a[i:i + rows], w[0], y[i:i + rows], design, xs[i:i + rows], ys, kernel=kernel)
        return y
    s = k // step
    per = xs.shape[1] // s
    parts, ws = (t.view(t.shape[0], s, step).transpose(0, 1).contiguous() for t in (a, w[0]))
    for i in range(0, m, rows):
        blk, ab, sx = y[i:i + rows], parts[:, i:i + rows], xs[i:i + rows]
        for j in range(s):
            if chained:
                cut = j if per == 1 else slice(j * per, (j + 1) * per)
                _launch(ab[j], ws[j], blk, design, sx[:, cut].contiguous(), ys[:, cut].contiguous())
            else:
                blk += _part(ab, ws, design, j) * (sx[:, j, None] * ys[:, j])
    return y


def linear(
    x: torch.Tensor, w: torch.Tensor, design: str, *,
    split_k: int = 1, combine: Combine = 'linear',
) -> torch.Tensor:
    """`x @ w.T` by an unscaled *design*, for `x` `[..., k]` and `w` `[n, k]` (as
    `nn.Linear` stores it): both rounded to its input formats (unscaled, as
    torch casts), the FP32 result.  *split_k* splits `k` into that many
    contiguous slices (:func:`matmul`)."""
    fx, fw = (quant.DTYPES[quant.context(f)] for f in formats(design)[:2])
    held_x, held_w = storage(design)
    a = x.reshape(-1, x.shape[-1]).to(fx).to(held_x)
    y = matmul(a, prepare(w.to(fw), held_w, split_k), design, combine)
    return y.reshape(*x.shape[:-1], w.shape[0])
