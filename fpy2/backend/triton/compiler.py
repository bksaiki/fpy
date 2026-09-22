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
- **`ConstFold` before emitting.**  `tl.static_range` needs its trip count as
  a compile-time constant, and a `range(K)` naming a *foreign* constant
  arrives as a free variable -- `Specialize` monomorphizes contexts and types,
  not closure values.
- **`Simplify` last**, under ``optimize``.  The lowerings above leave debris
  only a later pass can see, which the cpp backend says of its own pipeline
  too.
- **Tiling after the normal form, never before.**  A masked body is a guarded
  element write, which `SimplifyIf` refuses to hoist -- correctly, since
  hoisting would make the out-of-range store unconditional.  Normalizing after
  tiling would therefore reject this pipeline's own output.

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
from ...transform import ConstFold, FreeVarElim, Simplify, Specialize
from ...types import Type
from ..backend import Backend, CompileError
from .emitter import KernelSource, emit_kernel
from .normalize import normalize_module
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
            Run ``ConstFold`` and ``Simplify``.  Sound either way, and like
            the cpp backend's flag of the same name it does *not* mean the
            surface AST reaches the emitter untouched -- ``FreeVarElim``,
            ``Specialize``, the normal form and tiling run regardless.

            Unlike cleanup, these **widen what compiles**: ``False`` emits
            strictly fewer programs, because folding a constant is sometimes
            what makes a trip count provable or a literal representable.
            Measured over the library corpus: 30 emit with both, 27 with
            neither.  Default ``True``.
    """

    block: str
    drop_asserts: bool
    optimize: bool

    def __init__(
        self,
        *,
        block: str = 'BLOCK',
        drop_asserts: bool = False,
        optimize: bool = True,
    ):
        self.block = block
        self.drop_asserts = drop_asserts
        self.optimize = optimize

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

    @staticmethod
    def _specialize(module: Module) -> Module:
        """`FreeVarElim` first, then `Specialize`.

        A kernel is executed from a generated file whose namespace holds only
        `triton` and `tl`, so it cannot reference a closure at all -- which is
        the case `FreeVarElim` exists for, and why the cpp backend runs it
        unconditionally too.  Before `Specialize`, so a captured value is a
        binding the analyses can see rather than a free name.
        """
        module = module.map(lambda _m, fd: FreeVarElim.apply(fd))
        return Specialize.apply(module, size_key=True)

    def _compile_one(self, spec: Module, name: str) -> KernelSource:
        """One specialized entry, from the normal form through to source."""
        func = spec.get(name).func
        folded = (
            func.with_ast(ConstFold.apply(func.ast)) if self.optimize else func
        )

        normalized = Module()
        normalized.add(folded)
        ready = normalize_module(normalized).get(folded.name).func

        if self.optimize:
            # last, as the cpp backend does: the lowerings above leave debris
            # only a later pass can see -- a captured value materialized and
            # then inlined, a copy of a bound nothing reads again
            ready = ready.with_ast(Simplify.apply(ready.ast))

        tiles = tile_loops(ready.ast, self.block)
        return emit_kernel(
            tiles.func,
            tiles.tiled,
            block=self.block,
            drop_asserts=self.drop_asserts,
        )
