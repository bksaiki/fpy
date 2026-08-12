"""
Scheduling language: monomorphization
"""

from collections.abc import Collection

from ..function import Function
from ..number import Context
from ..transform import Monomorphize
from ..types import Type


def monomorphize(
    func: Function,
    ctx: Context | None = None,
    args: Collection[Type | None] | None = None
) -> Function:
    """
    Specialize `func` to a rounding context and/or argument types.

    Pinning a context or concrete argument formats is what lets later
    passes act on them, e.g. :func:`fpy2.strategies.simplify` folding
    the context and eliminating rounding operations that the
    pinned formats make identities.

    Parameters
    ----------
    func : Function
        The function to transform.
    ctx : Context | None
        The rounding context to pin as the function's caller context.
        If the function already pins a context (``@fp.fpy(ctx=...)``),
        the pin wins, so requesting a non-equivalent context raises
        rather than silently doing nothing. If `None`, the context is
        unchanged.
    args : Collection[Type | None] | None
        One entry per parameter (e.g. ``fpy2.types.RealType(fp.FP32)``);
        a `None` entry leaves that parameter's annotation unchanged.
        Each entry must be consistent with the inferred parameter type.
        If `None`, all annotations are unchanged.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    ValueError
        If `args` has the wrong arity, or an entry (or `ctx`) conflicts
        with the function's inferred types.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    # `Monomorphize` silently keeps a pinned function context; fail
    # rather than silently no-op (an `FPCoreContext` pin is replaced,
    # so only a `Context` pin can conflict).
    if (
        ctx is not None
        and isinstance(func.ast.ctx, Context)
        and not func.ast.ctx.is_equiv(ctx)
    ):
        raise ValueError(
            f'function `{func.name}` already pins context {func.ast.ctx}, '
            f'which is not equivalent to {ctx}'
        )

    ast = Monomorphize.apply(func.ast, ctx, args)
    return func.with_ast(ast)
