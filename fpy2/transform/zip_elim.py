"""
Rewrite ``zip(...)`` iterables into indexed loops over the source
vectors, removing the need for any backend to materialize an
intermediate list of tuples.

Two positions are recognized — a ``for``-loop iterable and a comprehension
iterable:

1. **For-loop over zip.**

   .. code-block:: python

      for a, b in zip(xs, ys):
          BODY

   is rewritten to::

      _src0 = xs
      _src1 = ys
      for _i in range(len(_src0)):
          a = _src0[_i]
          b = _src1[_i]
          BODY

   The per-source temporaries preserve "evaluate each iterable
   exactly once" semantics, identical to a faithful ``zip``
   implementation.  Underscore-binding targets emit no per-iteration
   assignment but their source is still bound to a temp so its
   side-effects fire exactly once.

2. **List comp over zip**, when every argument is an access path.

   .. code-block:: python

      [elt for a, b in zip(xs, ys)]

   is rewritten to::

      [elt[a -> xs[_i], b -> ys[_i]] for _i in range(len(xs))]

   The substitution is scope-aware: if ``elt`` contains a nested
   comprehension or other construct that re-binds ``a`` or ``b``,
   the shadowed uses are not rewritten.

   Restriction: every ``zip`` argument must be an
   :func:`fpy2.transform.iter_elim.is_access_path` — a ``Var`` or a pure,
   O(1) index/projection chain over one.  A list comprehension is an
   expression with no statement-level "preamble" to host the
   ``_srcK = ...`` bindings, so arguments are inlined and re-evaluated
   per iteration, which is only sound when they are pure and cheap.
   Comps with any other argument are left alone.

Whole-tuple targets
-------------------

A target need not destructure.  Bound to one name, the tuple is rebuilt per
iteration from the indexed sources — one stack tuple rather than an n-element
list of them::

   for p in zip(xs, ys):   ->   _src0 = xs
       BODY                     _src1 = ys
                                for _i in range(len(_src0)):
                                    p = (_src0[_i], _src1[_i])
                                    BODY

and the comp path substitutes ``p -> (xs[_i], ys[_i])``.  A discarded target
(``for _ in zip(xs, ys)``) builds nothing at all — its sources stay bound only
for their side-effects and their length.

Note ``zip(xs)`` yields *1-tuples*, so the decision to build a tuple keys on
the plan rather than on the number of sources.

Nested binding slots
--------------------

A binding slot may itself be a nested ``TupleBinding`` (e.g.
``for (a, b), c in zip(pairs, xs)``).  In the for-loop path the nested
slot lowers to a destructuring assignment ``(a, b) = _srcK[_i]`` (any arity;
no ``fst``/``snd``).  In the comp path — which has no statement context —
each leaf name is reached by an ``fst``/``snd`` chain over ``argK[_i]``
(for a pair ``t``, ``fst(t)`` / ``snd(t)``).  Because ``fst``/``snd`` are
pair-only, the comp path rewrites a nested slot only when it (and any
binding nested within it) is a pair; a comp with a nested slot of arity != 2
is left unchanged for the backend to materialize (see
:func:`fpy2.transform.iter_elim.comp_binding_is_pairs`).

Patterns that don't match the guards are left unchanged: a range iterable, a
destructuring target whose arity disagrees with the ``zip``'s (already
ill-typed, so keeping the ``zip`` keeps its diagnostic), and non-access-path
arguments in the comp path.

Ordering note: run :class:`ZipElim` *before*
:class:`fpy2.transform.ForUnpack`.  ``ForUnpack`` rewrites
``for (a, b) in iter:`` into ``for t in iter: a, b = t``, which
turns the ``ForStmt``'s target into a :class:`NamedId` and thereby
defeats this transform's guard.

See :mod:`fpy2.transform.iter_elim` for the machinery this shares with
:class:`fpy2.transform.EnumerateElim`, which handles the analogous
``enumerate(...)`` patterns.
"""

from ..analysis import DefineUse, SyntaxCheck
from ..ast.fpyast import Expr, FuncDef, UnderscoreId, Zip
from ..utils import Id
from .iter_elim import IterElimInstance, Plan, Slot, plan_for_zip


class _ZipElimInstance(IterElimInstance):
    def _match(self, target: Slot, iterable: Expr) -> tuple[Id, Plan] | None:
        """Fires iff *iterable* is a :class:`Zip` whose element *target* can
        read by indexing the ``zip``'s own arguments: see
        :func:`fpy2.transform.iter_elim.plan_for_zip`."""
        if not isinstance(iterable, Zip):
            return None
        plan = plan_for_zip(list(iterable.args), target)
        return None if plan is None else (UnderscoreId(), plan)


class ZipElim:
    """Rewrite ``zip(...)`` iterables into indexed loops over the
    source vectors.  See the module docstring for the patterns
    recognized and the ordering constraint with
    :class:`fpy2.transform.ForUnpack`."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns
        a new ``FuncDef``; the input is not mutated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        out = _ZipElimInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
