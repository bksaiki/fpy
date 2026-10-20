"""
Escape summaries: which of a function's list parameters outlive the call.

A caller that hands a list to a callee must know whether the callee keeps it:
:mod:`fpy2.analysis.alias` marks an argument *shared outward* only where the
summary says it may be retained.

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

``alias`` computes a callee's summary on demand.  The parser rejects forward
references, so the call graph is a DAG and no fixpoint is needed.  A callee that
does not type on its own reads as retaining and writing everything.  A summary
also says which parameters the callee may store into, which is where a list it
keeps may land.
"""

from dataclasses import dataclass

from ..ast import Argument, FuncDef
from .alias import Alias, AliasAnalysis, Region
from .define_use import DefineUse, DefineUseAnalysis


@dataclass(frozen=True)
class EscapeSummary:
    """Which of a function's parameters outlive a call to it, and which it may
    store into, by index."""

    retained: frozenset[int]
    written: frozenset[int]

    def retains(self, index: int) -> bool:
        return index in self.retained

    def writes(self, index: int) -> bool:
        return index in self.written


class Escape:
    """Escape summaries for FPy programs."""

    @staticmethod
    def analyze(
        func: FuncDef,
        summaries: dict[FuncDef, EscapeSummary] | None = None,
        def_use: DefineUseAnalysis | None = None,
        alias: AliasAnalysis | None = None,
    ) -> EscapeSummary:
        """*func*'s :class:`EscapeSummary`.

        Args:
            func: the function to summarize.
            summaries: summaries of its callees, a memo; see
                :meth:`Alias.analyze`.
            def_use: reuse an existing def-use analysis.
            alias: reuse an existing alias analysis.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if alias is None:
            alias = Alias.analyze(
                func, def_use=def_use, summaries=summaries,
            )

        params = alias.param_regions(func, def_use)
        return EscapeSummary(
            frozenset(
                i for i, region in params.items()
                if any(_retains(alias, r, func.args[i]) for r in alias.within(region))
            ),
            frozenset(i for i, region in params.items() if alias.may_write(region)),
        )


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
