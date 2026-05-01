"""
Context-use analysis for FPy ASTs.

This analysis links rounding context scopes (introduced by ``with``
statements or function-level context annotations) to the expressions
evaluated under them.  It attempts partial evaluation to resolve
context expressions to concrete :class:`Context` values.  When partial
evaluation does not fully reduce a context expression, or when a
function is entered without an overriding context, a fresh symbolic
context variable (:class:`NamedId`) is generated.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..fpc_context import FPCoreContext
from ..number import Context
from ..types import ContextParam
from ..utils import Gensym, NamedId, default_repr

from .define_use import DefineUse, DefineUseAnalysis
from .partial_eval import PartialEval, PartialEvalInfo

__all__ = [
    'CtxUse',
    'CtxUseAnalysis',
    'ContextScope',
    'CtxScopeSite',
    'CtxUseSite',
]

CtxScopeSite: TypeAlias = FuncDef | ContextStmt
"""AST nodes that introduce a rounding context scope"""

CtxUseSite: TypeAlias = NullaryOp | UnaryOp | BinaryOp | TernaryOp | NaryOp | Call
"""AST nodes that use a rounding context"""


@dataclass(frozen=True)
class ContextScope:
    """A rounding context scope."""

    site: CtxScopeSite
    """the AST node that introduces the scope"""

    ctx: ContextParam
    """the resolved context: a concrete Context or a symbolic NamedId"""


@default_repr
class CtxUseAnalysis:
    """Result of context-use analysis."""

    scopes: list[ContextScope]
    """all context scopes, ordered by introduction"""

    uses: dict[ContextScope, set[CtxUseSite]]
    """mapping from context scope to use sites"""

    use_to_scope: dict[CtxUseSite, ContextScope]
    """mapping from use site to context scope"""

    def __init__(
        self,
        scopes: list[ContextScope],
        uses: dict[ContextScope, set[CtxUseSite]],
    ):
        self.scopes = scopes
        self.uses = uses
        self.use_to_scope = {}
        for s, us in uses.items():
            for u in us:
                self.use_to_scope[u] = s

    def find_scope_from_use(self, site: CtxUseSite) -> ContextScope:
        """Returns the context scope active at a use site."""
        if site in self.use_to_scope:
            return self.use_to_scope[site]
        raise KeyError(f'no context scope found for use site {site}')


class _CtxUseInstance(DefaultVisitor):
    """Per-IR instance of context-use analysis."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym

    scopes: list[ContextScope]
    uses: dict[ContextScope, set[CtxUseSite]]

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym()
        self.scopes = []
        self.uses = {}

    def _fresh_sym_ctx(self) -> NamedId:
        return self.gensym.fresh('ctx')

    def _resolve_ctx_expr(self, expr: Expr) -> ContextParam:
        """
        Attempts to resolve an expression to a concrete Context via partial
        evaluation.  Returns the concrete Context if successful; otherwise
        returns a fresh symbolic NamedId.
        """
        if expr in self.eval_info.by_expr:
            val = self.eval_info.by_expr[expr]
            if isinstance(val, Context):
                return val
        return self._fresh_sym_ctx()

    def _make_scope(self, site: CtxScopeSite, ctx: ContextParam) -> ContextScope:
        s = ContextScope(site, ctx)
        self.scopes.append(s)
        self.uses[s] = set()
        return s

    def _record_use(self, use: CtxUseSite, scope: ContextScope):
        self.uses[scope].add(use)

    # ------------------------------------------------------------------
    # Expression visitors – record context-sensitive operations

    def _visit_nullaryop(self, e: NullaryOp, scope: ContextScope):
        self._record_use(e, scope)

    def _visit_unaryop(self, e: UnaryOp, scope: ContextScope):
        self._visit_expr(e.arg, scope)
        self._record_use(e, scope)

    def _visit_binaryop(self, e: BinaryOp, scope: ContextScope):
        self._visit_expr(e.first, scope)
        self._visit_expr(e.second, scope)
        self._record_use(e, scope)

    def _visit_ternaryop(self, e: TernaryOp, scope: ContextScope):
        self._visit_expr(e.first, scope)
        self._visit_expr(e.second, scope)
        self._visit_expr(e.third, scope)
        self._record_use(e, scope)

    def _visit_naryop(self, e: NaryOp, scope: ContextScope):
        for arg in e.args:
            self._visit_expr(arg, scope)
        self._record_use(e, scope)

    def _visit_call(self, e: Call, scope: ContextScope):
        for arg in e.args:
            self._visit_expr(arg, scope)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, scope)
        self._record_use(e, scope)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_context(self, stmt: ContextStmt, scope: ContextScope):
        # The context expression is evaluated under real arithmetic (not the
        # enclosing floating-point context), so we do not visit it as a use
        # of the enclosing context.  This is consistent with how
        # partial_eval evaluates context expressions under REAL.
        # Resolve the context expression, falling back to a symbolic
        # variable when partial evaluation cannot determine the value.
        ctx = self._resolve_ctx_expr(stmt.ctx)
        # Create a fresh context scope for the body.
        new_scope = self._make_scope(stmt, ctx)
        self._visit_block(stmt.body, new_scope)

    # ------------------------------------------------------------------
    # Function entry point

    def _visit_function(self, func: FuncDef, _scope: None):
        # Determine the body context from the function annotation.
        match func.ctx:
            case None:
                # No overriding context: generate a fresh symbolic variable.
                body_ctx: ContextParam = self._fresh_sym_ctx()
            case FPCoreContext():
                body_ctx = func.ctx.to_context()
            case Context():
                body_ctx = func.ctx
            case _:
                raise RuntimeError(f'unreachable: {func.ctx}')

        func_scope = self._make_scope(func, body_ctx)
        self._visit_block(func.body, func_scope)

    def analyze(self) -> CtxUseAnalysis:
        self._visit_function(self.func, None)
        return CtxUseAnalysis(self.scopes, self.uses)


class CtxUse:
    """
    Context-use analysis.

    This analysis computes, for each rounding context scope in a
    function, the set of context-sensitive expressions evaluated under it.

    Context scopes are introduced by:

    - ``with`` statements (:class:`ContextStmt`), and
    - the function-level context annotation in :class:`FuncDef`.

    The analysis attempts partial evaluation to resolve context expressions
    to concrete :class:`Context` values.  When partial evaluation does not
    fully reduce a context expression, or when a function is entered
    without an overriding context, a fresh symbolic context variable
    (:class:`NamedId`) is generated.
    """

    @staticmethod
    def analyze(func: FuncDef, *, def_use: DefineUseAnalysis | None = None) -> CtxUseAnalysis:
        """
        Runs context-use analysis on a function.

        Parameters
        ----------
        func:
            The function to analyze.
        def_use:
            Optional pre-computed definition-use analysis.  If ``None``,
            it is computed automatically.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        eval_info = PartialEval.apply(func, def_use=def_use)
        return _CtxUseInstance(func, eval_info).analyze()
