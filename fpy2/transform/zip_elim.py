"""
Rewrite ``zip(...)`` iterables into indexed loops over the source
vectors, removing the need for any backend to materialize an
intermediate list of tuples.

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
