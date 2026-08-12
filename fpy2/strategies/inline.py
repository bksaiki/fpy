"""
Scheduling language: function inlining
"""

from collections.abc import Iterable

from ..function import Function
from ..transform import FuncInline


def inline(
    func: Function,
    where: int | None = None,
    *,
    funcs: Iterable[Function] | None = None,
    recursive: bool = True
) -> Function:
    """
    Inline calls to FPy functions in `func`.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the call site to inline, counting candidate sites
        (calls to FPy functions that pass the `funcs` filter) in visit
        order, outermost-first. If `None`, inline every candidate site.
    funcs : Iterable[Function] | None
        Restrict inlining to calls to these functions.
        If `None`, every call to an FPy function is a candidate.
    recursive : bool
        If `True` (the default), inlined callee bodies are themselves
        fully inlined before being spliced into `func`, so an inlined
        site leaves no user-function calls behind. If `False`, only one
        level of calls is inlined.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    CallGraphError
        If the call graph reachable from `func` contains a cycle
        (FPy forbids recursion; inlining a recursive call would not
        terminate).
    ValueError
        If `where` does not correspond to a candidate call site.
    RuntimeError
        If a callee cannot be inlined: its body must end in exactly
        one trailing `return` statement, and its free variables must
        not conflict with the caller's.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    if funcs is not None:
        funcs = tuple(funcs)
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError(f"Expected a \'Function\', got {f}")

    ast = FuncInline.apply(func.ast, funcs=funcs, recursive=recursive, where=where)
    return func.with_ast(ast)
