"""
Triton backend: the compiler driver.

Runs the pipeline in the one order that works.  Three of the steps' orderings
are not obvious and were each found the hard way, so they are stated here
rather than left for a caller to rediscover:

- **`Specialize` first.**  Nothing downstream can proceed without concrete
  contexts and proven lengths: a kernel argument is a bare pointer, so the
  only length available for offset arithmetic is the one specialization
  proved.
- **`ConstFold` before emitting.**  `tl.static_range` needs its trip count as
  a compile-time constant, and a `range(K)` naming a *foreign* constant
  arrives as a free variable -- `Specialize` monomorphizes contexts and types,
  not closure values.
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
from ...transform import ConstFold, Specialize
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
    """

    block: str
    drop_asserts: bool

    def __init__(self, *, block: str = 'BLOCK', drop_asserts: bool = False):
        self.block = block
        self.drop_asserts = drop_asserts

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
        return self._compile_one(
            Specialize.apply(module, size_key=True), func.name,
        )

    def compile_module(self, module: Module) -> list[KernelSource]:
        """Every public entry of *module*, each as its own kernel."""
        if not isinstance(module, Module):
            raise TypeError(f"Expected a 'Module', got {module}")
        spec = Specialize.apply(module, size_key=True)
        return [self._compile_one(spec, entry.name) for entry in spec]

    def _compile_one(self, spec: Module, name: str) -> KernelSource:
        """One specialized entry, from the normal form through to source."""
        func = spec.get(name).func
        folded = func.with_ast(ConstFold.apply(func.ast))

        normalized = Module()
        normalized.add(folded)
        ready = normalize_module(normalized).get(folded.name).func

        tiles = tile_loops(ready.ast, self.block)
        return emit_kernel(
            tiles.func,
            tiles.tiled,
            block=self.block,
            drop_asserts=self.drop_asserts,
        )
