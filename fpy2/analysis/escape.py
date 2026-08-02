"""
Escape summaries: which of a function's list parameters outlive the call.

A caller that hands a list to a callee has to assume the worst — the callee might
keep it — so :mod:`fpy2.analysis.alias` marks any call argument *shared outward*.
That is what makes a compiled-to-compiled boundary keep its handles, and it is
the whole cost of decomposing a program into several functions.

A summary answers the question that assumption is standing in for: **after this
call returns, can anything outside still reach the argument's list?**  A caller
holding a summary that says no can stop marking the argument shared.

Retention is not mutation.  ``zs[0] = 99`` writes the caller's list *during* the
call and keeps nothing afterwards, so it does not retain — which is the case most
worth getting right, since a conservative reading of "the callee touches it"
would give up exactly the kernels worth unboxing.

A parameter is retained when its alias region:

- is returned — the caller ends up with a second reference to it;
- is reachable from another parameter, e.g. ``yss[0] = xs``, so the caller can
  find it through something it already holds;
- is handed to a callee that retains it, or to one this analysis cannot see.

Because the routes are read off alias regions rather than off syntax, aliasing
inside the callee is handled for free: ``ys = xs; return ys`` retains ``xs``
because both names are one region.

Computed leaves-first over :attr:`CallGraphAnalysis.order`, so a callee's summary
exists before its callers need it.  No fixpoint is needed and none would help:
the parser rejects forward references, so recursion cannot be written and the
order is always a DAG.  A missing summary — a foreign callee, or the cycle
:mod:`.call_graph` guards against in a programmatically-built module — reads as
*retains everything*, which is the only case a test can reach: a recursive FPy
function cannot be written at all.
"""

from dataclasses import dataclass

from ..ast import Call, DefaultVisitor, Expr, FuncDef, NamedId
from ..function import Function
from .alias import Alias, AliasAnalysis, Region
from .define_use import DefineUse, DefineUseAnalysis


@dataclass(frozen=True)
class EscapeSummary:
    """Which of a function's parameters outlive a call to it, by index.

    A parameter that carries no list is never retained; there is nothing to hold.
    """

    retained: frozenset[int]

    def retains(self, index: int) -> bool:
        return index in self.retained


class Escape:
    """Escape summaries for FPy programs."""

    @staticmethod
    def analyze(
        func: FuncDef,
        summaries: dict[FuncDef, EscapeSummary] | None = None,
        def_use: DefineUseAnalysis | None = None,
        alias: AliasAnalysis | None = None,
    ) -> EscapeSummary:
        """Which of *func*'s parameters outlive a call to it.

        Args:
            func: the function to summarize.
            summaries: summaries of its callees, keyed by ``FuncDef``.  A callee
                absent from it is assumed to retain everything.
            def_use: reuse an existing def-use analysis.
            alias: reuse an existing alias analysis.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if alias is None:
            alias = Alias.analyze(func, def_use=def_use)

        held = _RetainedRegions(alias, summaries or {})
        held._visit_function(func, None)

        retained: set[int] = set()
        for i, arg in enumerate(func.args):
            if not isinstance(arg.name, NamedId):
                continue
            d = def_use.find_def_from_site(arg.name, arg)
            depth = 0
            while (region := alias.region_of(d, depth)) is not None:
                if held.retains(region, arg):
                    retained.add(i)
                    break
                depth += 1
        return EscapeSummary(frozenset(retained))


class _RetainedRegions(DefaultVisitor):
    """Collects the regions a call leaves reachable from outside."""

    def __init__(
        self,
        alias: AliasAnalysis,
        summaries: dict[FuncDef, EscapeSummary],
    ):
        self.alias = alias
        self.summaries = summaries
        self.by_call: set[Region] = set()

    def retains(self, region: Region, arg) -> bool:
        if region in self.by_call:
            return True
        sites = self.alias.sites_at(region)
        if any(self.alias.is_returned(s) for s in sites):
            return True
        # reachable from a *different* parameter: the caller can find it through
        # something it already holds
        return any(
            s.kind == 'param' and s.node is not arg for s in sites
        )

    def _visit_call(self, e: Call, ctx):
        summary = None
        if isinstance(e.fn, Function):
            summary = self.summaries.get(e.fn.ast)
        for i, a in enumerate(e.args):
            if self.alias.region_of_expr(a) is None:
                continue
            if summary is None or summary.retains(i):
                self._mark(a)
        super()._visit_call(e, ctx)

    def _mark(self, e: Expr) -> None:
        """Retain *region* and everything inside it: handing out a nested list
        hands out its rows."""
        depth = 0
        while (inner := self.alias.region_of_expr(e, depth)) is not None:
            self.by_call.add(inner)
            depth += 1
