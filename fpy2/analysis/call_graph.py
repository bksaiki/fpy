"""
Call graph analysis.

Builds the call graph of a :class:`~fpy2.ast.FuncDef` and everything it
transitively calls.  The graph is keyed on ``FuncDef`` objects (by
identity); each edge ``caller -> callee`` corresponds to a :class:`Call`
whose ``fn`` resolves to a user-defined :class:`~fpy2.function.Function`.
Calls to primitives, context constructors, or builtins have no FPy body
to recurse into, so they are treated as external leaves and excluded
from the graph.

FPy does not support recursion, so the call graph is a DAG.  This
invariant is *enforced*: a direct (``f -> f``) or mutual
(``a -> b -> a``) cycle raises :class:`CallGraphError`.  Downstream
analyses may therefore assume acyclicity and a valid topological order.
(The parser already rejects forward references, so a cycle cannot arise
through ``@fpy`` decoration; the cycle check is a guard for
programmatically-built ASTs, e.g. FPCore import or AST transforms.)

The analysis exposes a *leaves-first* iteration order (callees before
callers, the root last), which is the natural order for bottom-up
analyses that need a callee's result before processing its callers.
"""

import dataclasses

from ..ast import *
from ..function import Function


class CallGraphError(Exception):
    """Raised when the call graph is not a DAG (FPy forbids recursion)."""
    pass


class _CallCollector(DefaultVisitor):
    """Collects every :class:`Call` in a single function body, in source
    order (outer calls before the calls nested in their arguments).  Does
    not cross into callees — it walks one ``FuncDef`` only."""

    calls: list[Call]

    def __init__(self):
        self.calls = []

    def _visit_call(self, e: Call, ctx: None):
        self.calls.append(e)
        super()._visit_call(e, ctx)

    def collect(self, func: FuncDef) -> list[Call]:
        self.calls = []
        self._visit_function(func, None)
        return self.calls


@dataclasses.dataclass
class CallGraphAnalysis:
    """Result of a :class:`CallGraph` analysis.

    All maps are keyed by ``FuncDef`` identity and have an entry for
    every node in :attr:`nodes` (including the root and leaves)."""

    root: FuncDef
    """the function the graph was built from"""
    nodes: set[FuncDef]
    """every function reachable from the root, including the root"""
    callees: dict[FuncDef, list[FuncDef]]
    """direct callees of each function, deduplicated, in source order"""
    callers: dict[FuncDef, list[FuncDef]]
    """direct callers of each function, deduplicated, in discovery order"""
    call_sites: dict[FuncDef, list[Call]]
    """every ``Call`` to a user-defined function in each body (not deduped)"""
    order: list[FuncDef]
    """leaves-first order: a function appears after all of its callees"""

    def callees_of(self, func: FuncDef) -> list[FuncDef]:
        """Direct callees of *func*."""
        return self.callees[func]

    def callers_of(self, func: FuncDef) -> list[FuncDef]:
        """Direct callers of *func*."""
        return self.callers[func]

    def __iter__(self):
        """Iterate functions leaves-first (callees before callers)."""
        return iter(self.order)

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, func: object) -> bool:
        return func in self.nodes


# DFS colors for cycle detection.
_GRAY = 1   # on the current DFS stack (in progress)
_BLACK = 2  # fully visited


class CallGraph:
    """Call graph analysis.

    Builds the DAG of functions transitively called from a root
    ``FuncDef``.  Raises :class:`CallGraphError` on recursion.
    """

    @staticmethod
    def analyze(func: FuncDef) -> CallGraphAnalysis:
        """Build the call graph rooted at *func*."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')

        nodes: set[FuncDef] = set()
        callees: dict[FuncDef, list[FuncDef]] = {}
        callers: dict[FuncDef, list[FuncDef]] = {}
        call_sites: dict[FuncDef, list[Call]] = {}
        order: list[FuncDef] = []

        color: dict[FuncDef, int] = {}
        stack: list[FuncDef] = []
        # Memoized per-body call collection (a callee reached from
        # multiple sites is only scanned once).
        body_calls: dict[FuncDef, list[Call]] = {}

        def calls_in(fdef: FuncDef) -> list[Call]:
            if fdef not in body_calls:
                body_calls[fdef] = _CallCollector().collect(fdef)
            return body_calls[fdef]

        def visit(fdef: FuncDef):
            color[fdef] = _GRAY
            stack.append(fdef)
            nodes.add(fdef)
            callees.setdefault(fdef, [])
            callers.setdefault(fdef, [])
            call_sites.setdefault(fdef, [])

            seen: set[FuncDef] = set()
            for call in calls_in(fdef):
                fn = call.fn
                if not isinstance(fn, Function):
                    # primitive / builtin / context constructor /
                    # unresolved — an external leaf, not a graph node.
                    continue
                callee = fn.ast
                call_sites[fdef].append(call)
                if callee in seen:
                    continue
                seen.add(callee)
                callees[fdef].append(callee)
                callers.setdefault(callee, []).append(fdef)

                c = color.get(callee)
                if c == _GRAY:
                    cycle = stack[stack.index(callee):] + [callee]
                    path = ' -> '.join(f.name for f in cycle)
                    raise CallGraphError(
                        f'recursion is not supported in FPy, but the call '
                        f'graph contains a cycle: {path}'
                    )
                elif c is None:
                    visit(callee)

            stack.pop()
            color[fdef] = _BLACK
            order.append(fdef)

        visit(func)

        return CallGraphAnalysis(
            root=func,
            nodes=nodes,
            callees=callees,
            callers=callers,
            call_sites=call_sites,
            order=order,
        )
