"""
Statement form: hoistable, with every comprehension and derived iterable lowered.
"""

from ..ast.fpyast import FuncDef
from .comp_to_loop import CompToLoop
from .hoistable import Hoistable
from .simplify import Simplify
from .unfold_iter import UnfoldEnumerate, UnfoldZip


class StatementForm:
    """Hoistable form, with every comprehension and derived iterable lowered.

    No pass here is a fixpoint alone and each supplies what the others lack.
    ``Hoistable`` seals a comprehension's element -- it runs once per iteration,
    so the slot before the enclosing statement is no place for its temporaries --
    and ``CompToLoop`` makes the loop that *is* that slot; ``CompToLoop``
    declines a comprehension in a ternary arm or a ``while`` condition for want
    of a slot, and ``Hoistable`` gives it one.  The unfolds need a slot too, and
    a ``zip`` inside a comprehension only gets one once ``CompToLoop`` has
    opened it.

    Iterating terminates without a cap, though not because the comprehension
    count falls -- lowering a dependent clause list *raises* it, peeling one
    comprehension into a row comprehension plus a nested one.  What falls is the
    clause count of the dependent one, by one per peel, and a single-clause
    comprehension cannot be dependent.  The unfolds lower the count of `Zip` and
    `Enumerate` nodes, which nothing here creates.  Everything else lowers
    outright, and ``Hoistable`` is idempotent over its own output.

    With *simplify*, ``Simplify`` runs on the result, clearing the temporaries
    the lowering binds.  *index_ranges* is ``CompToLoop``'s.
    """

    @staticmethod
    def apply(func: FuncDef, *, simplify: bool = False, index_ranges: bool = False) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        while True:
            func = Hoistable.apply(func)
            # after `Hoistable`, which gives each a slot for the binding it
            # needs, and before `CompToLoop`, which lowers the comprehension it
            # leaves
            func = UnfoldEnumerate.apply(func)
            func = UnfoldZip.apply(func)
            log = CompToLoop.apply_with_edits(func, index_ranges=index_ranges)
            if not log.edits:
                return Simplify.apply(func) if simplify else func
            func = log.result
