"""
Scheduling language: derived-iterable unfolding
"""

from ..ast import NamedId
from ..function import Function
from ..transform import Cursor, UnfoldEnumerate, UnfoldZip


def _unfold(cls, func: Function, where, temp_id) -> Function:
    if not isinstance(func, Function):
        raise TypeError(f'Expected a \'Function\', got {func}')
    return func.with_ast(cls.apply(
        func.ast, where=func.rebase(where), temp_id=temp_id,
    ))


def unfold_zip(
    func: Function,
    where: int | Cursor | None = None,
    *,
    temp_id: NamedId | None = None,
) -> Function:
    """
    State a ``zip`` in `func` as the comprehension it stands for.

    ``zip(xs1, ..., xsk)`` takes ``len(xs1)`` elements and asserts that every
    other iterable is that long, so the unfolding is that assertion followed by
    ``[(xs1[i], ..., xsk[i]) for i in range(len(xs1))]``.  The assertion is a
    *claim*: unequal lengths are undefined in FPy, so it says what a
    well-defined program already guarantees rather than adding a check.  It is
    also what carries the node's strictness —
    :class:`fpy2.analysis.ArraySizeInfer` pins an unconditional ``zip``'s
    iterables to one length and reads an assert the same way, so without it a
    proven length on one iterable stops reaching the others.

    :func:`fpy2.strategies.elim_iter` is the opposite trade: it fuses the
    ``zip`` into an indexed loop so no list of tuples is built at all. Unfold
    where you want one form to reason about; fuse where you want the loop.

    An argument that is not already a name is bound above the comprehension
    first, since the rewrite reads it twice. That needs a statement slot, so a
    ``zip`` in a comprehension element or a ``while`` condition is refused;
    :func:`fpy2.strategies.to_hoistable` gives every position one.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which ``zip`` to unfold: an index counting them in visit order,
        outermost-first, or a cursor naming one exactly, or a statement cursor
        or region taking every one at or beneath it. If `None`, unfold them
        all. :func:`fpy2.strategies.sites` lists them.
    temp_id : NamedId | None
        The name every temporary this rewrite mints is a refresh of.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a ``zip`` this rewrite refuses, or a
        region whose every candidate it refuses.
    TransformReferenceError
        If an explicit `where` names no ``zip``, or a cursor of a program this
        one was not derived from.

    Examples
    --------
    ::

        @fp.fpy
        def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x, y in zip(xs, ys):
                acc = acc + x * y
            return acc

    ``unfold_zip(dot)`` yields::

        @fp.fpy
        def dot(xs, ys):
            acc = 0
            assert len(ys) == len(xs)
            for x, y in [(xs[t], ys[t]) for t in range(len(xs))]:
                acc = (acc + (x * y))
            return acc
    """
    return _unfold(UnfoldZip, func, where, temp_id)


def unfold_enumerate(
    func: Function,
    where: int | Cursor | None = None,
    *,
    temp_id: NamedId | None = None,
) -> Function:
    """
    State an ``enumerate`` in `func` as the comprehension it stands for.

    ``enumerate(xs)`` is ``[(i, xs[i]) for i in range(len(xs))]``. Unlike
    :func:`fpy2.strategies.unfold_zip` this has nothing to carry: the index's
    format is ``INTEGER`` either way and the length comes from the same list.

    The argument is bound above the comprehension where it is not already a
    name, so it is built once; that needs a statement slot, which
    :func:`fpy2.strategies.to_hoistable` gives every position.
    ``enumerate(zip(...))`` therefore unfolds in either order — the inner form
    ends up on an assignment of its own.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which ``enumerate`` to unfold: an index counting them in visit order,
        outermost-first, or a cursor naming one exactly, or a statement cursor
        or region taking every one at or beneath it. If `None`, unfold them
        all. :func:`fpy2.strategies.sites` lists them.
    temp_id : NamedId | None
        The name every temporary this rewrite mints is a refresh of.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names an ``enumerate`` this rewrite refuses, or
        a region whose every candidate it refuses.
    TransformReferenceError
        If an explicit `where` names no ``enumerate``, or a cursor of a program
        this one was not derived from.

    Examples
    --------
    ::

        @fp.fpy
        def total(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for i, x in enumerate(xs):
                acc = acc + x
            return acc

    ``unfold_enumerate(total)`` yields::

        @fp.fpy
        def total(xs):
            acc = 0
            for i, x in [(t, xs[t]) for t in range(len(xs))]:
                acc = (acc + x)
            return acc
    """
    return _unfold(UnfoldEnumerate, func, where, temp_id)
