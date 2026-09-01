"""
Scheduling language: comprehension lowering
"""

from ..function import Function
from ..transform import CompToLoop, Cursor
from ..utils import NamedId


def comp_to_loop(
    func: Function,
    where: int | Cursor | None = None,
    *,
    temp_id: str = 't',
    dependent: bool = True,
) -> Function:
    """
    Lower a list comprehension in `func` into an allocation and a loop.

    A comprehension is the only *expression* in FPy that binds names, and that
    binding is why no other rewrite can reach inside one: a pass that needs to
    emit a statement has nowhere to put it, so
    :func:`fpy2.strategies.elim_round` and :func:`fpy2.strategies.insert_round`
    both leave a comprehension's element alone.  Lowering it to a loop is what
    makes that code schedulable.

    Several clauses are a cartesian product, so they become nested loops over
    the original targets with a running write index.  The allocation's length is
    the product of the clause lengths -- except for a *dependent* clause list,
    one whose later iterable mentions an earlier target, as in
    ``[b for a in xs for b in a]``.  Its length is a sum, so that one is built a
    row at a time and flattened, which costs a materialised row per outer
    element.  Pass ``dependent=False`` to leave it alone instead.

    This lowers what it can and leaves the rest as it was; it never raises over a
    comprehension it cannot lower.  :func:`fpy2.strategies.refusals` names each
    one and why.  Nothing guarantees the result is comprehension-free, and a
    caller who needs that checks: a comprehension nested in another gets its
    statement slot once the outer one is lowered, so applying this to a fixpoint
    clears those.

    Run :func:`fpy2.strategies.simplify` afterwards to fold away the temporaries
    the rewrite introduces.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which comprehension to lower: an index counting the comprehensions this
        rewrite acts on, in visit order, outermost-first, or a cursor, which
        names one exactly, or a statement cursor or region, which takes every one
        at or beneath it. If `None`, lower them all. A comprehension this rewrite
        leaves alone is not one of them and takes no index; naming it with a
        cursor says why it was left.
    temp_id : str
        The prefix for every name the rewrite mints, `t` by default. Freshened
        against the program, so it never shadows.
    dependent : bool
        Whether to lower a dependent clause list, whose length is a sum rather
        than a product. `True` by default: a rewrite that leaves one
        comprehension behind leaves its caller the whole comprehension problem.
        `False` declines it, for a consumer with a better lowering of its own.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a comprehension this rewrite leaves alone,
        or a region whose every candidate it leaves; the message says why.
    TransformReferenceError
        If an explicit `where` names no candidate comprehension, or a cursor of a
        program this one was not derived from.

    Examples
    --------
    A scaled copy of a list::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def scale(xs, k):
            return [(k * x) for x in xs]

    ``comp_to_loop(scale)`` allocates the result and fills it::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def scale(xs, k):
            t = xs
            acc = fp.empty(len(t))
            for i in range(len(t)):
                x = t[i]
                acc[i] = (k * x)
            return acc

    The element is now an ordinary statement, so a rounding rewrite can reach it.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(CompToLoop.apply_with_edits(
        func.ast, where=func.rebase(where), temp_id=NamedId(temp_id),
        dependent=dependent,
    ))
