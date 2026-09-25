"""
Triton backend: the compiler driver.

The pipeline, in order, and why:

1. ``FreeVarElim`` and ``ZipElim``: a kernel runs from a generated file and
   cannot reference a closure, and a ``zip`` binds a tuple, which has no
   storage.
2. ``Specialize``: a kernel argument is a bare pointer, so its offsets need
   proven lengths.
3. ``Simplify``: folds the constants ``tl.static_range`` needs as trip counts,
   before inlining copies what it would clear.
4. The normal form, then ``unfold_round`` and the normal form again: the
   lowering sees inlined callees and emits branches of its own.
5. ``Simplify``: clears what the lowerings leave.
6. Tiling, after the normal form so it sees inlined loops; then ``Simplify``.

The ``Simplify`` steps run only under ``optimize``.  A kernel writes through
pointers and returns nothing, and its tile width is a compile-time parameter,
so a compiled function takes both its output and its tile width as arguments.
"""

from ...function import Function
from ...module import Module
from ...number import Context
from ...transform import (
    AssertElim,
    FreeVarElim,
    Simplify,
    Specialize,
    ZipElim,
)
from ...types import Type
from ..backend import Backend, CompileError
from .emitter import KernelSource, emit_kernel
from .normalize import normalize, normalize_module
from .unfold_round import UnfoldMode, unfold
from .vectorize import tile_loops

__all__ = ['TritonCompiler']

_UnfoldMode = UnfoldMode


class TritonCompiler(Backend):
    """Compiles an FPy function to a ``@triton.jit`` kernel.

    Args:
        block:
            The parameter holding the tile width, which becomes the kernel's
            ``tl.constexpr`` for the launcher to pick.  Default ``'BLOCK'``.
        drop_asserts:
            Drop an ``assert`` rather than refusing the program: a kernel
            cannot raise.  A semantic change, so opt-in.  Default ``False``.
        optimize:
            Run ``Simplify`` around the normal form and tiling.  Besides
            cleaning up, this widens what compiles: folding a constant can
            make a trip count provable or a literal representable.  Default
            ``True``.
        unfold:
            An :class:`~fpy2.backend.unfold_round.UnfoldMode`, as for
            the cpp backend.  ``ROUNDINGS`` lowers a rounding Triton cannot
            spell into integer arithmetic instead of refusing it;
            ``DOUBLE_ROUND`` also computes arithmetic under such a context at
            a native intermediate and re-rounds.  Default ``NONE``.
    """

    UnfoldMode = _UnfoldMode

    block: str
    drop_asserts: bool
    optimize: bool
    unfold: _UnfoldMode

    def __init__(
        self,
        *,
        block: str = 'BLOCK',
        drop_asserts: bool = False,
        optimize: bool = True,
        unfold: _UnfoldMode = _UnfoldMode.NONE,
    ) -> None:
        self.block = block
        self.drop_asserts = drop_asserts
        self.optimize = optimize
        self.unfold = unfold

    def compile(
        self,
        func: Function,
        *,
        ctx: Context | None = None,
        arg_types: 'list[Type | None] | None' = None,
    ) -> KernelSource:
        """*func* as a Triton kernel.

        *ctx* and *arg_types* are the monomorphization spec, as for any
        backend; the lengths in *arg_types* are what make the offsets
        computable.
        """
        if not isinstance(func, Function):
            raise TypeError(f"Expected a 'Function', got {func}")
        if not any(str(a.name) == self.block for a in func.ast.args):
            raise CompileError(
                f'`{func.name}` has no `{self.block}` parameter; a kernel '
                'takes its tile width as an argument, since the launcher '
                'picks the value'
            )
        module = Module()
        module.add(func, ctx=ctx, arg_types=arg_types)
        return self._compile_one(self._specialize(module), func.name)

    def compile_module(self, module: Module) -> list[KernelSource]:
        """Every public entry of *module*, each as its own kernel."""
        if not isinstance(module, Module):
            raise TypeError(f"Expected a 'Module', got {module}")
        spec = self._specialize(module)
        return [self._compile_one(spec, entry.name) for entry in spec]

    def _specialize(self, module: Module) -> Module:
        """`AssertElim` where asked, then steps 1 and 2 of the pipeline."""
        if self.drop_asserts:
            # module-wide: the assert is often in a callee
            module = module.map(lambda _m, fd: AssertElim.apply(fd))
        module = module.map(lambda _m, fd: ZipElim.apply(FreeVarElim.apply(fd)))
        return Specialize.apply(module, size_key=True)

    def _compile_one(self, spec: Module, name: str) -> KernelSource:
        """One specialized entry, steps 3 to 6 of the pipeline."""
        if self.optimize:
            # over the callees too, before inlining copies them
            spec = spec.map(lambda _m, fd: Simplify.apply(fd))
        folded = spec.get(name).func

        normalized = Module()
        normalized.add(folded)
        ready = normalize_module(normalized).get(folded.name).func

        if self.unfold is not _UnfoldMode.NONE:
            ready = ready.with_ast(normalize(unfold(ready.ast, self.unfold)))

        if self.optimize:
            ready = ready.with_ast(Simplify.apply(ready.ast))

        tiles = tile_loops(ready.ast, self.block)
        if self.optimize:
            tiles = tiles.rewritten(Simplify.apply(tiles.func))

        return emit_kernel(
            tiles.func,
            tiles.tiled,
            guards=tiles.guards,
            lanes=tiles.lanes,
            grid=tiles.grid,
            block=self.block,
            drop_asserts=self.drop_asserts,
        )
