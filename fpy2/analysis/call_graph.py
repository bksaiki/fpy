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
        # ``DefaultVisitor._visit_call`` only walks positional args, so
        # visit keyword-argument values explicitly — a call nested in a
        # kwarg value (and any cycle through it) would otherwise be
        # missed from the graph and the acyclicity guard.
        super()._visit_call(e, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)

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
        revisited = self._format_node(self.root, '', '', True, lines, set())
        if revisited:
            lines.append('')
            lines.append('(*) = callees shown above')
        return '\n'.join(lines)

    def _format_node(
        self,
        func: FuncDef,
        prefix: str,
        connector: str,
        is_root: bool,
        lines: list[str],
        seen: set[FuncDef],
    ) -> bool:
        """Append the subtree rooted at *func* to *lines*.  Returns
        whether any node in the subtree was a revisit, so
        :meth:`format` knows whether to print the ``(*)`` legend."""
        revisit = func in seen
        marker = ' (*)' if revisit else ''
        lines.append(f'{prefix}{connector}{func.name}{marker}')
        if revisit:
            return True
        seen.add(func)

        # Children continue the vertical line unless this node was the
        # last of its siblings (``└─``), in which case the column is
        # blank below it.  The root's children start at column zero.
        child_prefix = '' if is_root else prefix + (
            '   ' if connector == '└─ ' else '│  '
        )
        any_revisit = False
        callees = self.callees[func]
        for i, callee in enumerate(callees):
            last = i == len(callees) - 1
            child_connector = '└─ ' if last else '├─ '
            if self._format_node(
                callee, child_prefix, child_connector, False, lines, seen,
            ):
                any_revisit = True
        return any_revisit

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


class _CallGraphInstance:
    """Builds the call graph via a three-color DFS, accumulating the
    edge maps and a leaves-first (post-order) traversal order."""

    func: FuncDef
    nodes: set[FuncDef]
    callees: dict[FuncDef, list[FuncDef]]
    callers: dict[FuncDef, list[FuncDef]]
    call_sites: dict[FuncDef, list[Call]]
    order: list[FuncDef]

    def __init__(self, func: FuncDef):
        self.func = func
        self.nodes = set()
        self.callees = {}
        self.callers = {}
        self.call_sites = {}
        self.order = []
        # DFS bookkeeping.
        self._color: dict[FuncDef, int] = {}
        self._stack: list[FuncDef] = []
        # Memoized per-body call collection (a callee reached from
        # multiple sites is only scanned once).
        self._body_calls: dict[FuncDef, list[Call]] = {}

    def _calls_in(self, fdef: FuncDef) -> list[Call]:
        if fdef not in self._body_calls:
            self._body_calls[fdef] = _CallCollector().collect(fdef)
        return self._body_calls[fdef]

    def _visit(self, fdef: FuncDef):
        self._color[fdef] = _GRAY
        self._stack.append(fdef)
        self.nodes.add(fdef)
        self.callees.setdefault(fdef, [])
        self.callers.setdefault(fdef, [])
        self.call_sites.setdefault(fdef, [])

        seen: set[FuncDef] = set()
        for call in self._calls_in(fdef):
            fn = call.fn
            if not isinstance(fn, Function):
                # primitive / builtin / context constructor /
                # unresolved — an external leaf, not a graph node.
                continue
            callee = fn.ast
            self.call_sites[fdef].append(call)
            if callee in seen:
                continue
            seen.add(callee)
            self.callees[fdef].append(callee)
            self.callers.setdefault(callee, []).append(fdef)

            c = self._color.get(callee)
            if c == _GRAY:
                cycle = self._stack[self._stack.index(callee):] + [callee]
                path = ' -> '.join(f.name for f in cycle)
                raise CallGraphError(
                    f'recursion is not supported in FPy, but the call '
                    f'graph contains a cycle: {path}'
                )
            elif c is None:
                self._visit(callee)

        self._stack.pop()
        self._color[fdef] = _BLACK
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
