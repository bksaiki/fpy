"""
Escape summaries: which of a function's list parameters outlive the call.

A parameter is retained when its alias region is returned, is reachable from
another parameter, or is marked *shared outward*.  Writing a list during the call
does not retain it.
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
