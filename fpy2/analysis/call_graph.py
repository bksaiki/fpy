"""
Call graph analysis.

Builds the call graph of a :class:`~fpy2.ast.FuncDef` and everything it
transitively calls.  Calls to primitives, context constructors, or builtins are
external leaves, excluded from the graph.  A cycle raises :class:`CallGraphError`.
The analysis exposes a *leaves-first* iteration order (callees before callers).
"""

import dataclasses
from collections.abc import Iterator

from ..ast import *
from ..function import Function


class CallGraphError(Exception):
    """Raised when the call graph is not a DAG (FPy forbids recursion)."""


class _CallCollector(DefaultVisitor):
    """Collects every :class:`Call` in a single function body, in source
    order (outer calls before the calls nested in their arguments).  Does
    not cross into callees — it walks one ``FuncDef`` only."""

    calls: list[Call]

    def __init__(self) -> None:
        self.calls = []

    def _visit_call(self, e: Call, ctx: None) -> None:
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

    def format(self) -> str:
        """An indented-tree rendering rooted at :attr:`root`, callees
        nested under their callers.  A function reached from more than
        one caller is expanded once; later occurrences are marked
        ``(*)`` and not re-expanded (keeping shared subtrees compact —
        and the output finite even if a cycle ever slipped through)."""
        lines: list[str] = []
        self._format_node(self.root, '', False, True, lines, set())
        return '\n'.join(lines)

    def _format_node(
        self,
        func: FuncDef,
        prefix: str,
        is_last: bool,
        is_root: bool,
        lines: list[str],
        seen: set[FuncDef],
    ) -> None:
        """Append the subtree rooted at *func* to *lines*."""
        revisit = func in seen
        marker = ' (*)' if revisit else ''
        connector = '' if is_root else '└─ ' if is_last else '├─ '
        lines.append(f'{prefix}{connector}{func.name}{marker}')
        if revisit:
            return
        seen.add(func)

        # Children continue the vertical line unless this node was the
        # last of its siblings (``└─``), in which case the column is
        # blank below it.  The root's children start at column zero.
        child_prefix = '' if is_root else prefix + (
            '   ' if is_last else '│  '
        )
        callees = self.callees[func]
        for i, callee in enumerate(callees):
            self._format_node(
                callee, child_prefix, i == len(callees) - 1, False, lines, seen,
            )

    def dot(self) -> str:
        """A Graphviz ``digraph`` rendering, pipe-able to ``dot``.

        Nodes are given stable ids (``n0``, ``n1``, … in leaves-first
        order) with the function name as the label, so two distinct
        functions that happen to share a name stay distinct."""
        ids = {func: f'n{i}' for i, func in enumerate(self.order)}
        lines = ['digraph call_graph {']
        for func in self.order:
            lines.append(f'  {ids[func]} [label="{func.name}"];')
        for func in self.order:
            for callee in self.callees[func]:
                lines.append(f'  {ids[func]} -> {ids[callee]};')
        lines.append('}')
        return '\n'.join(lines)

    def __iter__(self) -> Iterator[FuncDef]:
        """Iterate functions leaves-first (callees before callers)."""
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, func: object) -> bool:
        return func in self.nodes


class _CallGraphInstance:
    """Builds the call graph via a DFS, accumulating the edge maps and a
    leaves-first (post-order) traversal order."""

    func: FuncDef
    nodes: set[FuncDef]
    callees: dict[FuncDef, list[FuncDef]]
    callers: dict[FuncDef, list[FuncDef]]
    call_sites: dict[FuncDef, list[Call]]
    order: list[FuncDef]
    _stack: list[FuncDef]
    """functions being visited, outermost first"""

    def __init__(self, func: FuncDef) -> None:
        self.func = func
        self.nodes = set()
        self.callees = {}
        self.callers = {}
        self.call_sites = {}
        self.order = []
        self._stack = []

    def _resolve_callee(self, fdef: FuncDef, call: Call) -> Function | None:
        """The user-defined :class:`Function` a call targets, or ``None`` for
        an external leaf.

        ``call.fn`` is the parse-time resolution and is authoritative when
        set.  When it is ``None`` (a call built without env resolution, e.g.
        FPCore import or an AST transform) fall back to resolving the callee
        *name* against the containing function's environment, so a call to a
        known FPy function still produces an edge (and is subject to the
        acyclicity guard) rather than being dropped as a leaf."""
        if isinstance(call.fn, Function):
            return call.fn
        if call.fn is None and isinstance(call.func, Var):
            name = call.func.name
            if isinstance(name, NamedId):
                val = fdef.env.get(str(name))
                if isinstance(val, Function):
                    return val
        return None

    def _visit(self, fdef: FuncDef) -> None:
        self._stack.append(fdef)
        self.nodes.add(fdef)
        self.callees.setdefault(fdef, [])
        self.callers.setdefault(fdef, [])
        self.call_sites.setdefault(fdef, [])

        seen: set[FuncDef] = set()
        for call in _CallCollector().collect(fdef):
            fn = self._resolve_callee(fdef, call)
            if fn is None:
                # primitive / builtin / context constructor / a name that
                # doesn't resolve to an FPy function — an external leaf, not
                # a graph node.
                continue
            callee = fn.ast
            self.call_sites[fdef].append(call)
            if callee in seen:
                continue
            seen.add(callee)
            self.callees[fdef].append(callee)
            self.callers.setdefault(callee, []).append(fdef)

            if callee in self._stack:
                cycle = self._stack[self._stack.index(callee):] + [callee]
                path = ' -> '.join(f.name for f in cycle)
                raise CallGraphError(
                    f'recursion is not supported in FPy, but the call '
                    f'graph contains a cycle: {path}'
                )
            elif callee not in self.nodes:
                self._visit(callee)

        self._stack.pop()
        self.order.append(fdef)

    def analyze(self) -> CallGraphAnalysis:
        self._visit(self.func)
        return CallGraphAnalysis(
            root=self.func,
            nodes=self.nodes,
            callees=self.callees,
            callers=self.callers,
            call_sites=self.call_sites,
            order=self.order,
        )


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
        return _CallGraphInstance(func).analyze()
