"""
Scheduling language: unrolling proven-length sequences
"""

from ..function import Function
from ..transform import Scalarize


def unroll_seqs(func: Function, *, cap: int = 256) -> Function:
    """
    Unroll each proven-length comprehension in `func` into one value per
    element.

    The list is **not removed**: it is rebound to a literal of the element
    names, so every use keeps working — ``len``, a subscript, a fold, a call
    taking the whole list. What moves is where the element expressions are
    *evaluated*, from inside the comprehension into statements of their own.

    That is the point. A pass which splices statements into the enclosing
    block cannot reach inside a comprehension, because the comprehension's
    targets are not bound there — :func:`inline` refuses a call in that
    position for exactly this reason. Unrolling first puts each call in a
    statement where it can be reached.

    A sequence is left alone, not refused, when its length is not proven,
    when it is longer than `cap`, or when it sits in a lazily evaluated
    position such as an ``if`` expression's arm — hoisting out of one would
    make it unconditional. Declining to unroll costs an unrolling, never the
    compile: the sequence simply stays a sequence, and whatever handles one
    of unknown length handles it.

    Sizes come from the array size analysis, so this is most useful after
    :func:`monomorphize`, which is what makes a length provable at all.

    Parameters
    ----------
    func : Function
        The function to transform.
    cap : int
        How long a sequence may be and still unroll. The default of 256 is
        four times the widest MMA instruction in ``examples/mmasim``; a
        sequence of thousands is better left as a loop than turned into
        thousands of statements every later pass must walk.

    Returns
    -------
    Function
        The transformed function.

    Examples
    --------
    ::

        @fp.fpy
        def weighted(xs: list[fp.Real]) -> fp.Real:
            ys = [xs[i] * 2 for i in range(3)]
            return sum(ys)

    ``unroll_seqs(weighted)`` yields::

        @fp.fpy
        def weighted(xs):
            e = (xs[0] * 2)
            e3 = (xs[1] * 2)
            e4 = (xs[2] * 2)
            ys = [e, e3, e4]
            return sum(ys)

    where ``sum(ys)`` still has a ``ys`` to fold.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a 'Function', got {func}")

    return func.with_ast(Scalarize.apply(func.ast, cap=cap))
