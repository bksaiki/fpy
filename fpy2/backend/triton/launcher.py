"""
Triton backend: running an emitted kernel.

Everything here needs a GPU: Triton has no CPU target, so importing `triton`
is not enough to run anything.  Every entry point checks :func:`unavailable`.
"""

from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from ..backend import CompileError
from .emitter import KernelSource

if TYPE_CHECKING:
    import torch

__all__ = ['MODULE_PREAMBLE', 'launch', 'load_kernel', 'unavailable']


def unavailable() -> str | None:
    """Why an emitted kernel cannot be run here, or ``None`` if it can."""
    # broad: a broken install raises `OSError`, `RuntimeError`, ...
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f'torch does not import: {type(e).__name__}: {e}'
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        return f'triton does not import: {type(e).__name__}: {e}'
    if not torch.cuda.is_available():
        return 'no GPU: triton has no CPU target, so a kernel cannot run'
    return None


MODULE_PREAMBLE = (
    'import triton\n'
    'import triton.language as tl\n'
    # for the exact operations only libdevice spells, such as `ldexp`
    'from triton.language.extra import libdevice\n\n\n'
)
"""The imports an emitted kernel's module needs."""

_LOADED: dict[str, Any] = {}
"""Kernels already written and imported, keyed by source.

Triton compiles on first launch and caches against the file, so loading the
same source twice would pay for it twice.  Keyed by text rather than by
`KernelSource`, which is a mutable dataclass and so unhashable.
"""


def load_kernel(src: KernelSource) -> Any:
    """*src* compiled to a callable ``triton.jit`` kernel.

    Written to a file, not `exec`ed: `@triton.jit` reads its function's
    source back from one.  The file outlives this call, since Triton re-reads
    it while compiling.
    """
    why = unavailable()
    if why is not None:
        raise CompileError(f'cannot load a Triton kernel: {why}')
    cached = _LOADED.get(src.source)
    if cached is not None:
        return cached

    path = Path(mkdtemp(prefix='fpy-triton-')) / f'{src.name}.py'
    path.write_text(MODULE_PREAMBLE + src.source + '\n')

    spec = spec_from_file_location(f'fpy_triton_{src.name}', path)
    if spec is None or spec.loader is None:
        raise CompileError(f'cannot load the emitted kernel at `{path}`')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, src.name)
    _LOADED[src.source] = kernel
    return kernel


TUNING: tuple[tuple[int, int], ...] = (
    (16, 4), (32, 2), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8),
)
"""The (block, warps) pairs :func:`launch` tries when it picks the block."""

_TUNED: dict[str, Any] = {}
"""Kernels wrapped in `triton.autotune`, keyed by source as `_LOADED` is."""


def _tuned(src: KernelSource) -> Any:
    """*src* under `triton.autotune`: each of `TUNING` timed on the first
    launch at each shape, keyed on the sizes the kernel takes, and every
    argument it stores through restored between configs, since each one is
    run more than once."""
    import triton
    cached = _TUNED.get(src.source)
    if cached is None:
        configs = [triton.Config({src.block: b}, num_warps=w) for b, w in TUNING]
        cached = _TUNED[src.source] = triton.autotune(
            configs, key=[name for name, _, _ in src.sizes],
            restore_value=list(src.writes),
        )(load_kernel(src))
    return cached


def launch(
    src: KernelSource,
    args: Sequence[Any],
    *,
    block: int | None = None,
    grid: int | None = None,
) -> None:
    """Run *src* over *args*, which are `torch.Tensor`s and scalars.

    *block* is the tile width; left out, the launch picks it, and the number
    of warps with it, from `TUNING` by timing each on the first launch at a
    shape.

    *grid* defaults to covering :attr:`KernelSource.grid_extent` in tiles of
    *block*: the last instance runs a full tile, its excess rows masked off.
    A second axis, where there is one, runs :attr:`KernelSource.grid_outer`
    programs.

    ``enable_fp_fusion`` is taken from *src*, which knows whether contracting
    a multiply-add is observable.  ``enable_reflect_ftz`` is off: libdevice
    would flush subnormals, which FPy does not.

    Every argument is passed by name, so the block may sit anywhere among the
    parameters.
    """
    why = unavailable()
    if why is not None:
        raise CompileError(f'cannot launch a Triton kernel: {why}')
    import triton

    named = _named(src, args)
    _check_layout(src, named, block)
    # an unproven length is read off the tensor that has it
    sizes = {name: named[src.params[pos]].shape[depth] for name, pos, depth in src.sizes}
    # before compiling, which is costly
    if grid is None and src.grid_extent is None:
        raise CompileError(
            'this kernel tiled nothing, so its grid has no extent to derive; '
            'pass `grid` explicitly'
        )
    outer = () if src.grid_outer is None else (_extent(src.grid_outer, sizes),)
    if 0 in outer or (grid is None and _extent(src.grid_extent, sizes) == 0):
        return  # nothing to compute, and no grid to launch it on

    def dims(width: int) -> tuple[int, ...]:
        n = triton.cdiv(_extent(src.grid_extent, sizes), width) if grid is None else grid
        return (n, *outer)

    options = {'enable_fp_fusion': src.enable_fp_fusion, 'enable_reflect_ftz': False}
    if block is not None:
        tile = {} if src.block is None else {src.block: block}
        load_kernel(src)[dims(block)](**named, **tile, **sizes, **options)
        return
    if src.block is None:
        raise CompileError('there is nothing to tune over; pass `block`')
    _tuned(src)[lambda meta: dims(meta[src.block])](**named, **sizes, **options)


def _named(src: KernelSource, args: Sequence[Any]) -> dict[str, Any]:
    """*args* keyed by the parameter each binds: every one but the block and
    the sizes, in order."""
    names = [
        p for p in src.params[:len(src.params) - len(src.sizes)]
        if p != f'{src.block}: tl.constexpr'
    ]
    if len(args) != len(names):
        raise TypeError(f'the kernel takes {len(names)} arguments, not {len(args)}')
    return dict(zip(names, args))


_MAX_OFFSET = 2 ** 31 - 1
"""The kernel's offsets are `int32`."""


def _check_layout(src: KernelSource, named: dict[str, Any], block: int | None) -> None:
    """Refuse a tensor the kernel's offsets do not describe.

    The kernel computes row-major offsets from the shape it was compiled for,
    so a strided view, a different rank or length, or two tensors disagreeing
    on a shared length would be addressed wrongly, silently; so would one
    whose offsets, counting the rows a tile runs past its end, overflow
    `int32`.  It does not copy: a copy of an output would take the writes.
    """
    import torch
    reach = block if block is not None else max(b for b, _ in TUNING)
    dtypes = dict(src.dtypes)
    lengths: dict[str, tuple[str, int]] = {}
    for pos, dims in src.shapes:
        name = src.params[pos]
        t = named[name]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f'`{name}` is a list, so it takes a tensor, not {type(t).__name__}')
        if not t.is_contiguous():
            raise ValueError(
                f'`{name}` is not contiguous; the kernel addresses it row '
                'major from its shape')
        if t.dim() != len(dims):
            raise ValueError(f'`{name}` has {t.dim()} dimensions, not {len(dims)}')
        want_dtype = _torch_dtype(dtypes[pos]) if pos in dtypes else t.dtype
        if t.dtype != want_dtype:
            raise ValueError(f'`{name}` holds {t.dtype}; the kernel was compiled for {want_dtype}')
        if t.numel() + reach - 1 > _MAX_OFFSET:
            raise ValueError(
                f'`{name}` has {t.numel()} elements, too many for the '
                "kernel's int32 offsets")
        for depth, (want, have) in enumerate(zip(dims, t.shape)):
            if isinstance(want, int) and want != have:
                raise ValueError(
                    f'`{name}` is {have} long at dimension {depth}; the kernel '
                    f'was compiled for {want}')
            if isinstance(want, str):
                first, n = lengths.setdefault(want, (name, have))
                if n != have:
                    raise ValueError(
                        f'`{name}` is {have} long at dimension {depth} and '
                        f'`{first}` {n}, where the kernel has one length')


def _torch_dtype(tl_dtype: str) -> 'torch.dtype':
    """The torch dtype of Triton dtype *tl_dtype*, spelled `tl.float16`."""
    import torch
    name = tl_dtype.removeprefix('tl.')
    return torch.bool if name == 'int1' else getattr(torch, name)


def _extent(extent: int | str | None, sizes: dict[str, int]) -> int:
    """A grid extent: a proven length, or the size parameter holding one."""
    assert extent is not None
    return sizes[extent] if isinstance(extent, str) else extent
