"""
Triton backend: the compiler driver.

Runs the pipeline in the one order that works.  Three of the steps' orderings
are not obvious and were each found the hard way, so they are stated here
rather than left for a caller to rediscover:

- **`Specialize` first.**  Nothing downstream can proceed without concrete
  contexts and proven lengths: a kernel argument is a bare pointer, so the
  only length available for offset arithmetic is the one specialization
  proved.
- **`FreeVarElim` before everything.**  A kernel runs from a generated file
  and cannot reference a closure, so a captured value has to become a binding
  first.  The cpp backend runs it unconditionally for the same reason.
- **`Simplify` first**, under ``optimize``, over every function: it folds the
  constants `tl.static_range` needs as trip counts -- a `range(K)` naming a
  *foreign* constant arrives as a free variable, `Specialize` monomorphizing
  contexts and types, not closure values -- and clears what inlining would
  otherwise copy.
- **`Simplify` last**, under ``optimize``.  The lowerings above leave debris
  only a later pass can see, which the cpp backend says of its own pipeline
  too.
- **Tiling after the normal form, never before.**  The loops to tile include
  the ones inlining brings in.

**What the ABI asks of the program.**  A Triton kernel writes through pointers
its launcher owns and returns nothing, and its tile width is a compile-time
parameter.  So a function compiled here takes its output as an argument and
takes the tile width as one too.  That is the same principle the design uses
for the batch dimension: say it in the program rather than invent a
convention around it.
"""

from ...ast import FuncDef
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

_UnfoldMode = UnfoldMode
from .vectorize import tile_loops

__all__ = ['TritonCompiler']


class TritonCompiler(Backend):
    """Compiles an FPy function to a ``@triton.jit`` kernel.

    Args:
        block:
            The name of the function parameter holding the tile width.  It
            becomes the kernel's ``tl.constexpr``, so the launcher picks the
            value -- host-side from the data, as a fused softmax does, or by
            ``triton.autotune``, as a matmul does.  Default ``'BLOCK'``.
        drop_asserts:
            Skip an ``assert`` rather than refusing the program.  A kernel
            cannot raise, so an assert has no spelling either way; this picks
            which answer.  Dropping one is a *semantic* change, so it is
            opt-in and a launcher wanting the check runs it host-side.
            Default ``False``.
        optimize:
            Run ``Simplify`` before the normal form and after it.  Sound
            either way, and like
            the cpp backend's flag of the same name it does *not* mean the
            surface AST reaches the emitter untouched -- ``FreeVarElim``,
            ``Specialize``, the normal form and tiling run regardless.

            Unlike cleanup, these **widen what compiles**: ``False`` emits
            strictly fewer programs, because folding a constant is sometimes
            what makes a trip count provable or a literal representable.
            Measured over the library corpus: 30 emit with both, 27 with
            neither.  Default ``True``.
        unfold:
            An :class:`~fpy2.backend.triton.unfold_round.UnfoldMode`, as for
            the cpp backend.  ``ROUNDINGS`` lowers a rounding Triton cannot
            spell -- round-toward-zero to FP32, say -- into integer
            arithmetic instead of refusing it; ``DOUBLE_ROUND`` also computes
            arithmetic under such a context at a native intermediate and
            re-rounds.  Default ``NONE``.
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
    ):
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
        """`AssertElim` where asked, then `FreeVarElim` and `ZipElim`, then
        `Specialize`.

        A kernel is executed from a generated file whose namespace holds only
        `triton` and `tl`, so it cannot reference a closure at all -- which is
        the case `FreeVarElim` exists for, and why the cpp backend runs it
        unconditionally too.  Before `Specialize`, so a captured value is a
        binding the analyses can see rather than a free name.

        `ZipElim` because a `zip` has no Triton spelling either way: it binds
        a tuple, and the emitter has no tuple storage.  Rewritten to an
        indexed loop it becomes ordinary subscripts, which a lane loop reads
        at its own index.
        """
        if self.drop_asserts:
            # module-wide, since the one that matters is usually in a callee
            # waiting to be inlined
            module = module.map(lambda _m, fd: AssertElim.apply(fd))
        module = module.map(lambda _m, fd: ZipElim.apply(FreeVarElim.apply(fd)))
        return Specialize.apply(module, size_key=True)

    def _compile_one(self, spec: Module, name: str) -> KernelSource:
        """One specialized entry, from the normal form through to source."""
        if self.optimize:
            # first too, and over the callees: a wrapper's temporaries and a
            # callee's copies go before inlining and unrolling multiply them
            spec = spec.map(lambda _m, fd: Simplify.apply(fd))
        folded = spec.get(name).func

        normalized = Module()
        normalized.add(folded)
        ready = normalize_module(normalized).get(folded.name).func

        if self.unfold is not _UnfoldMode.NONE:
            # after the normal form, so the callees' roundings are inlined
            # where it can see them, and re-normalized after: the lowering
            # emits branches of its own
            ready = ready.with_ast(normalize(unfold(ready.ast, self.unfold)))

        if self.optimize:
            # before tiling, as the cpp backend does last: the lowerings above leave debris
            # only a later pass can see -- a captured value materialized and
            # then inlined, a copy of a bound nothing reads again
            ready = ready.with_ast(Simplify.apply(ready.ast))

        # the emitter reduces across a tile's lanes, not its rows
        tiles = tile_loops(ready.ast, self.block, reductions=False, lanes=True)
        if self.optimize:
            # and after tiling, which leaves bounds and copies of its own
            tiles = tiles.rewritten(Simplify.apply(tiles.func))

        # compile the final function
        return emit_kernel(
            tiles.func,
            tiles.tiled,
            guards=tiles.guards,
            lanes=tiles.lanes or (),
            grid=tiles.grid,
            block=self.block,
            drop_asserts=self.drop_asserts,
        )
