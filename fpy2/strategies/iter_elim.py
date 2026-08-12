"""
Scheduling language: derived-iterable elimination
"""

from ..function import Function
from ..transform import EnumerateElim, ZipElim


def elim_iter(
    func: Function, *,
    enable_enumerate: bool = True,
    enable_zip: bool = True
) -> Function:
    """
    Rewrite ``enumerate(...)`` and ``zip(...)`` iterables in `func` into
    indexed loops over their sources, so no intermediate list of tuples
    is ever materialized.

    Run this *before* loop operators such as
    :func:`fpy2.strategies.split` and :func:`fpy2.strategies.unroll_for`:
    they materialize the loop's iterable into a temporary, which for a
    derived iterable builds exactly the tuple list this pass avoids —
    and their target rewrites defeat this pass's patterns, so the
    opportunity does not survive them.

    Patterns that do not match are left unchanged; see
    :class:`fpy2.transform.EnumerateElim` and
    :class:`fpy2.transform.ZipElim` for what is recognized
    (``enumerate(zip(...))`` collapses both intermediates at once).

    Parameters
    ----------
    func : Function
        The function to transform.
    enable_enumerate : bool
        Rewrite ``enumerate(...)`` iterables (default `True`).
    enable_zip : bool
        Rewrite ``zip(...)`` iterables (default `True`).

    Returns
    -------
    Function
        The transformed function.

    .. note::
        The rewrites take the loop bound from the first source. Iterating
        over a mismatched-length ``zip`` is undefined behavior in FPy, so
        for such programs the rewritten function may observably differ
        from the original (silently truncating, or failing with a
        different error).
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    # EnumerateElim first: it recognizes ``enumerate(zip(...))`` itself,
    # and after its rewrite a ``zip`` sits on an assignment RHS, out of
    # ZipElim's reach.
    ast = func.ast
    if enable_enumerate:
        ast = EnumerateElim.apply(ast)
    if enable_zip:
        ast = ZipElim.apply(ast)

    return func.with_ast(ast)
