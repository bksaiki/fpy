"""
Triton backend: running an emitted kernel.

Everything here needs hardware.  Triton compiles for `amd` and `nvidia` only
-- there is no CPU target in mainline, and handing a kernel a CPU tensor fails
with *"Pointer argument cannot be accessed from Triton"* -- so importing
`triton` successfully is not enough to run anything.  :func:`unavailable`
answers all three questions at once, and every entry point here checks it.
"""

from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from ..backend import CompileError
from .emitter import KernelSource

__all__ = ['launch', 'load_kernel', 'unavailable']


def unavailable() -> str | None:
    """Why an emitted kernel cannot be run here, or ``None`` if it can.

    Three separate requirements, and the third is the one that surprises:
    `triton` imports fine on a machine with no GPU, and only fails at launch.
    """
    # Broad on purpose: the question is whether it *works*, and a broken
    # install raises well outside `ImportError` -- `OSError` for a missing
    # shared object, `RuntimeError` for a driver mismatch.  Narrowing here
    # would turn "unavailable" into a crash.
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


_MODULE_PREAMBLE = (
    'import triton\n'
    'import triton.language as tl\n'
    # `ldexp` is IEEE 754's `scaleB`, which only libdevice exposes.  It is
    # not a transcendental, so the op table's exclusion does not reach it.
    'from triton.language.extra import libdevice\n\n\n'
)

_LOADED: dict[str, Any] = {}
"""Kernels already written and imported, keyed by source.

Triton compiles on first launch and caches against the file, so loading the
same source twice would pay for it twice.  Keyed by text rather than by
`KernelSource`, which is a mutable dataclass and so unhashable.
"""


def load_kernel(src: KernelSource) -> Any:
    """*src* compiled to a callable ``triton.jit`` kernel.

    **Written to a file, not `exec`ed.**  `@triton.jit` reads its function's
    source back with `inspect.getsourcelines` to compile it, so a function
    defined in a bare namespace fails with *"@jit functions should be defined
    in a Python file"*.  The same constraint `@fp.fpy` has, for the same
    reason.

    The file outlives this call deliberately: Triton re-reads it while
    compiling, and cached results key on it.
    """
    why = unavailable()
    if why is not None:
        raise CompileError(f'cannot load a Triton kernel: {why}')
    cached = _LOADED.get(src.source)
    if cached is not None:
        return cached

    path = Path(mkdtemp(prefix='fpy-triton-')) / f'{src.name}.py'
    path.write_text(_MODULE_PREAMBLE + src.source + '\n')

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
"""The (block, warps) pairs :func:`launch` tries when it picks the block.

Measured over the mmasim matmuls at 1024 x 1024 (`examples/mmasim/bench`):
no one pair is best for every design, and these seven hold every design's
best within about 2%.  The best moves with the design -- 32 by 1 for volta,
64 by 8 for mxfp8 -- and a fixed 16 by 4 was up to 7x slower."""

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
    *block*, which is what the emitted mask expects: the last instance runs a
    full tile and the over-run lanes are masked off.  A second axis, where
    there is one, runs :attr:`KernelSource.grid_outer` programs.

    ``enable_fp_fusion`` is taken from *src* rather than from the caller.  It
    is a property of the program -- whether contracting a multiply-add is
    observable -- and the compiler derived it; letting a launcher override it
    would make the answer depend on who ran the kernel.
    """
    why = unavailable()
    if why is not None:
        raise CompileError(f'cannot launch a Triton kernel: {why}')
    import triton

    # an unproven length is read off the tensor that has it
    sizes = {name: args[pos].shape[depth] for name, pos, depth in src.sizes}
    # the grid before the kernel: deriving it is cheap and compiling is not,
    # so a missing extent should not cost a compile to discover
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

    if block is not None:
        load_kernel(src)[dims(block)](
            *args, block, *sizes.values(), enable_fp_fusion=src.enable_fp_fusion,
        )
        return
    if not TUNING or src.block is None:
        raise CompileError('there is nothing to tune over; pass `block`')
    # the sizes by name: the block they follow is the config's to pass
    _tuned(src)[lambda meta: dims(meta[src.block])](
        *args, **sizes, enable_fp_fusion=src.enable_fp_fusion,
    )


def _extent(extent: int | str | None, sizes: dict[str, int]) -> int:
    """A grid extent: a proven length, or the size parameter holding one."""
    assert extent is not None
    return sizes[extent] if isinstance(extent, str) else extent
