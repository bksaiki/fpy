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
    and their target rewrites defeat this pass's patterns.

    Patterns that do not match are left unchanged; see
    :class:`fpy2.transform.EnumerateElim` and
    :class:`fpy2.transform.ZipElim` for what is recognized
    (``enumerate(zip(...))`` collapses both intermediates at once).

    A cursor does not cross this pass: the loop path emits a preamble binding
    each source and replaces the loop, and the comprehension path rewrites
    expressions in place, neither of which the pass reports. Aim what you need
    before it, or re-list the sites after.

    Parameters
    ----------
    func : Function
        The function to transform.
    enable_enumerate : bool
        Rewrite ``enumerate(...)`` iterables (default `True`). This
        handles ``enumerate(zip(...))`` as a unit, so the ``zip`` inside
        is eliminated regardless of `enable_zip`.
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

    Examples
    --------
    ::

        @fp.fpy
        def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x, y in zip(xs, ys):
                acc = acc + x * y
            return acc

    ``elim_iter(dot)`` yields::

        @fp.fpy
        def dot(xs, ys):
            acc = 0
            _src = xs
            _src5 = ys
            for _i in range(len(_src)):
                x = _src[_i]
                y = _src5[_i]
                acc = (acc + (x * y))
            return acc
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
