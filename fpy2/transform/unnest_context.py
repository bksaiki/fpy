"""Unnesting context statements."""

from ..analysis import (
    DefAnalysis,
    DefineUse,
    DefineUseAnalysis,
    Purity,
    SyntaxCheck,
)
from ..ast import *


class _Vars(DefaultVisitor):
    """Every `Var` in a subtree."""

    found: list[Var]

    def __init__(self):
        self.found = []

    def _visit_var(self, e: Var, ctx: None):
        self.found.append(e)

    @staticmethod
    def of(node: Expr | Stmt) -> list[Var]:
        inst = _Vars()
        if isinstance(node, Expr):
            inst._visit_expr(node, None)
        else:
            inst._visit_statement(node, None)
        return inst.found


class _Unnester(DefaultTransformVisitor):
    """
    Context statement unnester.
    """

    def_use: DefineUseAnalysis
    changed: bool

    def __init__(self, def_use: DefineUseAnalysis):
        self.def_use = def_use
        self.changed = False

    @staticmethod
    def _leading_run(stmts: list[Stmt]) -> list[Stmt]:
        """The `with`s at the front of *stmts*, never all of them: one
        statement stays behind, so the block cannot empty."""
        n = 0
        while n < len(stmts) - 1 and isinstance(stmts[n], ContextStmt):
            n += 1
        return stmts[:n]

    def _may_lead(self, stmt: ContextStmt) -> bool:
        """Whether the `with`s leading *stmt*'s body may be hoisted out.

        Decided on the *original* body.  `_visit_block` has rebuilt the
        statements by the time this is asked, and the analysis knows nothing
        about the new nodes -- walking them would find no uses and pass every
        guard.  The original leading run covers every statement a rebuilt one
        can come from, so a verdict over it holds for the rewrite.

        The trailing case needs none of this: it moves nothing.
        """
        run = self._leading_run(stmt.body.stmts)
        if not run:
            return False

        # the header is evaluated after the hoisted blocks rather than before
        # them, so neither side of that swap may be observable
        if not Purity.analyze_expr(stmt.ctx, self.def_use):
            return False
        for s in run:
            assert isinstance(s, ContextStmt)
            if not Purity.analyze_expr(s.ctx, self.def_use):
                return False

        # ...and the header must not read a name the hoisted blocks rebind,
        # which would leave it reading the new value instead of the old
        block = StmtBlock(list(run))
        rebound = DefAnalysis.analyze(block)[block]
        if any(v.name in rebound for v in _Vars.of(stmt.ctx)):
            return False

        # the target is bound after the hoisted blocks, so none may read it
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            for s in run:
                if any(self.def_use.use_to_def.get(v) == d for v in _Vars.of(s)):
                    return False

        return True

    def _visit_block(self, block: StmtBlock, ctx: None):
        # a context statement may rewrite to several siblings
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            s, _ = self._visit_statement(stmt, ctx)
            if isinstance(s, StmtBlock):
                stmts.extend(s.stmts)
            else:
                stmts.append(s)
        return StmtBlock(stmts), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        body, _ = self._visit_block(stmt.body, ctx)
        stmts = list(body.stmts)

        # Peel both edges.  Stopping at one statement leaves
        # `with e1: (with e2: B)` alone: `DeadCodeEliminate` drops the outer
        # block outright, which is better than making siblings of the two.
        after: list[Stmt] = []
        while len(stmts) > 1 and isinstance(stmts[-1], ContextStmt):
            after.append(stmts.pop())

        before: list[Stmt] = []
        if self._may_lead(stmt):
            while len(stmts) > 1 and isinstance(stmts[0], ContextStmt):
                before.append(stmts.pop(0))

        outer = ContextStmt(stmt.target, stmt.ctx, StmtBlock(stmts), stmt.loc)
        if not before and not after:
            return outer, ctx

        self.changed = True
        return StmtBlock([*before, outer, *reversed(after)]), ctx

    def apply(self, func: FuncDef) -> tuple[FuncDef, bool]:
        return self._visit_function(func, None), self.changed


class UnnestContext:
    """Unnesting of context statements.

    A `with` block nested directly inside another does not refine it: entering
    one replaces the active context outright rather than merging with the
    context in force, so the parent contributes nothing to the nested block.
    Where that block sits at an *edge* of its parent's body, the two can be
    written as siblings::

        with e1:                    with e1:
            X                           X
            with e2:        ->      with e2:
                B                       B

    which costs no statements and drops a level of nesting.  Nothing else
    moves: every statement stays under the same context, in the same order.

    The two edges are not alike.  Hoisting a *trailing* block changes no
    evaluation order at all and is always allowed.  Hoisting a *leading* one
    moves the enclosing header after it, so that case additionally requires
    the header and the hoisted contexts to be pure, the header to read no name
    the hoisted blocks rebind, and the hoisted blocks not to read the
    enclosing `as` target -- which is bound after them once the rewrite lands.

    A nested block that is its parent's *only* statement is left alone, since
    :class:`DeadCodeEliminate` drops the parent outright instead.

    This pass is for legibility.  Nothing downstream needs the flattened form
    -- `ContextUse` and `FormatInfer` key on scopes rather than on nesting.
    """

    @staticmethod
    def apply(func: FuncDef, *, def_use: DefineUseAnalysis | None = None) -> FuncDef:
        """Unnest the context statements of *func*.  Pass a cached
        ``def_use`` to share it with another pass."""
        func, _ = UnnestContext.apply_with_status(func, def_use=def_use)
        return func

    @staticmethod
    def apply_with_status(
        func: FuncDef, *, def_use: DefineUseAnalysis | None = None
    ) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag
        — ``True`` iff at least one block was unnested."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        func, changed = _Unnester(def_use).apply(func)
        SyntaxCheck.check(func)
        return func, changed
