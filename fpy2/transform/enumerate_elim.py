"""
Rewrite ``enumerate(...)`` iterables into indexed loops over the source
vector, so no backend has to materialize an intermediate list of
``(index, element)`` tuples.

1. **For-loop over enumerate.**

   .. code-block:: python

      for i, x in enumerate(xs):   ->   _src0 = xs
          BODY                          for i in range(len(_src0)):
                                            x = _src0[i]
                                            BODY

   The index slot gets no assignment: ``enumerate``'s index and
   ``range(len(...))``'s counter are the same value, so the user's own name
   becomes the loop variable.  The ``_src0`` temp evaluates the iterable
   exactly once and keeps the loop bound out of reach of a body that rebinds
   the source name.  A discarded element slot emits no assignment, but its
   source is still bound so any side-effect fires once.

2. **List comp over enumerate**, when the source is an access path.

   .. code-block:: python

      [elt for i, x in enumerate(xs)]   ->   [elt[x -> xs[i]] for i in range(len(xs))]

   The substitution is scope-aware: a nested comprehension that re-binds
   ``x`` shadows it, and those uses are left alone.

   A comprehension is an expression with no statement slot to host the
   ``_src0 = ...`` binding, so the source is inlined and re-evaluated per
   iteration.  That is only sound for an ``is_access_path`` — pure and O(1);
   anything else is left for the backend to materialize.

Composing with ``zip``
----------------------

``enumerate(zip(...))`` lowered naively builds *two* lists: the ``zip``'s
tuples and the ``enumerate``'s ``(index, tuple)`` pairs.  Indexing the
``zip``'s own arguments removes both at once:

.. code-block:: python

   for i, (a, b) in enumerate(zip(xs, ys)):   ->   _src0 = xs
       BODY                                        _src1 = ys
                                                   for i in range(len(_src0)):
                                                       a = _src0[i]
                                                       b = _src1[i]
                                                       BODY

This has to happen here rather than in :class:`fpy2.transform.ZipElim`, which
matches a ``zip`` in *iterable* position: after a plain ``enumerate`` rewrite
the ``zip`` sits on the right-hand side of ``_src0 = zip(xs, ys)``, out of its
reach.  Taking the bound from the first argument matches ``ZipElim`` and is
faithful because FPy's ``zip`` is strict.

A slot bound to the *whole* tuple gets it rebuilt per iteration instead —
``p = (_src0[i], _src1[i])``, one stack tuple rather than a list of them — and
the comp path substitutes ``p -> (xs[i], ys[i])``.  Neither inlines the
``zip`` itself, only one access path per source plus a tuple construction, so
both stay pure and O(1) per element.

The one shape left materialized is an element slot whose arity *disagrees*
with the ``zip``'s.  That program is already ill-typed, and keeping the
``zip`` keeps its diagnostic rather than trading it for an arity-mismatched
destructure.

A slot may also be a nested ``TupleBinding`` (``for i, (a, b) in
enumerate(pairs)``).  The for-loop path lowers it to a destructuring assign of
any arity; the comp path reaches its leaves by ``fst``/``snd``, which is
pair-only — see
:func:`fpy2.transform.iter_elim.comp_binding_is_pairs`.

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
