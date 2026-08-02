"""
Escape summaries: which of a function's list parameters outlive the call.

A caller that hands a list to a callee has to assume the worst — the callee might
keep it — so :mod:`fpy2.analysis.alias` marks any call argument *shared outward*.
That is what makes a compiled-to-compiled boundary keep its handles, and it is
the whole cost of decomposing a program into several functions.

A summary answers the question that assumption stands in for: **after this call
returns, can anything outside still reach the argument's list?**  A caller whose
summary says no can stop marking the argument shared.

Retention is not mutation.  ``zs[0] = 99`` writes the caller's list *during* the
call and keeps nothing afterwards, so it does not retain.  Reading "the callee
touches it" as retention would give up exactly the kernels worth unboxing.

A parameter is retained when its alias region is returned, is reachable from
another parameter (``yss[0] = xs``, so the caller can find it through something
it already holds), or is marked *shared outward* — which covers a retaining
callee, a foreign one, and any expression kind ``alias`` does not model.

Reading routes off regions rather than syntax means aliasing inside the callee
needs no rule of its own: ``ys = xs; return ys`` retains ``xs`` because both
names are one region.

Computed leaves-first, so a callee's summary exists before its callers need it.
No fixpoint is needed and none would help — the parser rejects forward
references, so recursion cannot be written and the order is always a DAG.  A
missing summary reads as *retains everything*.
"""

from dataclasses import dataclass

from ..ast import Argument, FuncDef, NamedId
from .alias import Alias, AliasAnalysis, Region
from .define_use import DefineUse, DefineUseAnalysis


@dataclass(frozen=True)
class EscapeSummary:
    """Which of a function's parameters outlive a call to it, by index."""

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
            summaries: summaries of its callees.  One absent from it is assumed
                to retain everything.
            def_use: reuse an existing def-use analysis.
            alias: reuse an existing alias analysis.  It must have been built
                with the same *summaries*, or every call argument reads as
                shared and the result says nothing.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if alias is None:
            alias = Alias.analyze(
                func, def_use=def_use, summaries=summaries,
            )

        retained: set[int] = set()
        for i, arg in enumerate(func.args):
            if not isinstance(arg.name, NamedId):
                continue
            d = def_use.find_def_from_site(arg.name, arg)
            depth = 0
            while (region := alias.region_of(d, depth)) is not None:
                if _retains(alias, region, arg):
                    retained.add(i)
                    break
                depth += 1
        return EscapeSummary(frozenset(retained))


def _retains(alias: AliasAnalysis, region: Region, arg: Argument) -> bool:
    """Whether *region* outlives a call, given *arg* is the parameter asked
    about."""
    if alias.returned_at(region) or alias.escapes_at(region):
        return True
    # reachable from a *different* parameter: the caller can find it through
    # something it already holds
    return any(
        s.kind == 'param' and s.node is not arg
        for s in alias.sites_at(region)
    )
