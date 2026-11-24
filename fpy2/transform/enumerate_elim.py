"""
Rewrite ``enumerate(...)`` iterables into indexed loops over the source
vector, so no backend has to materialize an intermediate list of
``(index, element)`` tuples.

.. code-block:: python

   for i, x in enumerate(xs):   ->   _src0 = xs
       BODY                          for i in range(len(_src0)):
                                         x = _src0[i]
                                         BODY

Ordering: run this *before* :class:`fpy2.transform.ForUnpack`, which rewrites
``for (i, x) in iter:`` into ``for t in iter: i, x = t``, whose ``NamedId``
target defeats the guard below.  See :mod:`fpy2.transform.iter_elim` for the
machinery shared with :class:`fpy2.transform.ZipElim`.
"""

from ..analysis import DefineUse, SyntaxCheck
from ..ast.fpyast import (
    Enumerate,
    Expr,
    FuncDef,
    NamedId,
    TupleBinding,
    UnderscoreId,
    Zip,
)
from ..utils import Id
from .iter_elim import IterElimInstance, Plan, Slot, plan_for_zip


def _split_target(
    target: Id | TupleBinding, iterable: Expr,
) -> tuple[Id, Slot] | None:
    """Split ``for i, x in enumerate(src)``'s target into its index and
    element slots, or ``None`` if *target* / *iterable* aren't that pattern.

    The index slot must be a plain name or a discard: ``enumerate`` yields an
    integer there, so a destructuring slot would be ill-typed.
    """
    if not isinstance(iterable, Enumerate):
        return None
    if not isinstance(target, TupleBinding) or len(target.elts) != 2:
        return None
    idx, elt = target.elts
    if not isinstance(idx, (NamedId, UnderscoreId)):
        return None
    if not isinstance(elt, (NamedId, UnderscoreId, TupleBinding)):
        return None
    return idx, elt


def _plan(src: Expr, elt: Slot) -> Plan:
    """How to reach the element bound by *elt* when iterating ``enumerate(src)``.

    A ``zip`` source decomposes into its own arguments, so neither intermediate
    list is built.  Anything else — including a ``zip`` whose slot arity
    disagrees — is indexed whole.
    """
    if isinstance(src, Zip):
        plan = plan_for_zip(list(src.args), elt)
        if plan is not None:
            return plan
    return Plan([src], [elt], tupled=False)


class _EnumerateElimInstance(IterElimInstance):
    def _match(self, target: Slot, iterable: Expr) -> tuple[Id, Plan] | None:
        split = _split_target(target, iterable)
        if split is None:
            return None
        assert isinstance(iterable, Enumerate)
        idx, elt = split
        return idx, _plan(iterable.arg, elt)


class EnumerateElim:
    """Rewrite ``enumerate(...)`` iterables into indexed loops over the
    source vector, including the ``enumerate(zip(...))`` case where both
    intermediate lists collapse into direct indexing.  See the module
    docstring for the patterns recognized and the ordering constraint with
    :class:`fpy2.transform.ForUnpack`."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a new
        ``FuncDef``; the input is not mutated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        out = _EnumerateElimInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
