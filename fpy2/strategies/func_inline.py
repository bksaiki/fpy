"""
Scheduling language: function inlining
"""

from collections.abc import Iterable

from ..function import Function
from ..transform import FuncInline
from ..transform.utils.cursor import Cursor


def inline(
    func: Function,
    where: int | Cursor | None = None,
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
    where : int | Cursor | None
        Which call site to inline, among the candidates -- calls to FPy
        functions that pass the `funcs` filter. An index counts them in visit
        order, outermost-first; a :class:`fpy2.strategies.StmtCursor` /
        :class:`fpy2.strategies.BlockCursor` names a program point and takes every
        candidate call at or beneath it, which is coarser than the index: a
        statement holding two candidate calls names both. A cursor or region
        from an earlier program is forwarded to this one first. If `None`,
        inline every candidate site.
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
    TransformReferenceError
        If `where` does not correspond to a candidate call site.
    RuntimeError
        If a callee cannot be inlined: its body must end in exactly
        one trailing `return` statement, its free variables must not
        conflict with the caller's, and the call must not be in a
        `while` condition (the spliced body would be evaluated only
        once, before the loop).

    Examples
    --------
    ::

        @fp.fpy
        def sq(x: fp.Real) -> fp.Real:
            return x * x

        @fp.fpy
        def hypot2(x: fp.Real, y: fp.Real) -> fp.Real:
            return sq(x) + sq(y)

    ``inline(hypot2)`` yields::

        @fp.fpy
        def hypot2(x, y):
            x3 = x
            t = (x3 * x3)
            x4 = y
            t5 = (x4 * x4)
            return (t + t5)

    while ``inline(hypot2, 0)`` inlines only the ``sq(x)`` site and
    keeps the ``sq(y)`` call.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")
    if funcs is not None:
        funcs = tuple(funcs)
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError(f"Expected a \'Function\', got {f}")

    return func.with_edits(FuncInline.apply_with_edits(
        func.ast, funcs=funcs, recursive=recursive, where=func.rebase(where)
    ))
