"""
Context definition-use analysis for FPy ASTs.

This analysis links rounding context definitions (introduced by ``with``
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
    'CtxDefUse',
    'CtxDefUseAnalysis',
    'ContextDef',
    'CtxDefSite',
    'CtxUseSite',
]

CtxDefSite: TypeAlias = FuncDef | ContextStmt
"""AST nodes that introduce a rounding context"""

CtxUseSite: TypeAlias = NullaryOp | UnaryOp | BinaryOp | TernaryOp | NaryOp | Call
"""AST nodes that use a rounding context"""


@dataclass(frozen=True)
class ContextDef:
    """A rounding context definition."""

    site: CtxDefSite
    """the AST node that introduces the context"""

    ctx: ContextParam
    """the resolved context: a concrete Context or a symbolic NamedId"""


@default_repr
class CtxDefUseAnalysis:
    """Result of context definition-use analysis."""

    defs: list[ContextDef]
    """all context definitions, ordered by introduction"""

    uses: dict[ContextDef, set[CtxUseSite]]
    """mapping from context definition to use sites"""

    use_to_def: dict[CtxUseSite, ContextDef]
    """mapping from use site to context definition"""

    def __init__(
        self,
        defs: list[ContextDef],
        uses: dict[ContextDef, set[CtxUseSite]],
    ):
        self.defs = defs
        self.uses = uses
        self.use_to_def = {}
        for d, us in uses.items():
            for u in us:
                self.use_to_def[u] = d

    def find_def_from_use(self, site: CtxUseSite) -> ContextDef:
        """Returns the context definition active at a use site."""
        if site in self.use_to_def:
            return self.use_to_def[site]
        raise KeyError(f'no context definition found for use site {site}')


class _CtxDefUseInstance(DefaultVisitor):
    """Per-IR instance of context definition-use analysis."""

    func: FuncDef
    eval_info: PartialEvalInfo
    gensym: Gensym

    defs: list[ContextDef]
    uses: dict[ContextDef, set[CtxUseSite]]

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo):
        self.func = func
        self.eval_info = eval_info
        self.gensym = Gensym()
        self.defs = []
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

    def _make_def(self, site: CtxDefSite, ctx: ContextParam) -> ContextDef:
        d = ContextDef(site, ctx)
        self.defs.append(d)
        self.uses[d] = set()
        return d

    def _record_use(self, use: CtxUseSite, ctx_def: ContextDef):
        self.uses[ctx_def].add(use)

    # ------------------------------------------------------------------
    # Expression visitors – record context-sensitive operations

    def _visit_nullaryop(self, e: NullaryOp, ctx_def: ContextDef):
        self._record_use(e, ctx_def)

    def _visit_unaryop(self, e: UnaryOp, ctx_def: ContextDef):
        self._visit_expr(e.arg, ctx_def)
        self._record_use(e, ctx_def)

    def _visit_binaryop(self, e: BinaryOp, ctx_def: ContextDef):
        self._visit_expr(e.first, ctx_def)
        self._visit_expr(e.second, ctx_def)
        self._record_use(e, ctx_def)

    def _visit_ternaryop(self, e: TernaryOp, ctx_def: ContextDef):
        self._visit_expr(e.first, ctx_def)
        self._visit_expr(e.second, ctx_def)
        self._visit_expr(e.third, ctx_def)
        self._record_use(e, ctx_def)

    def _visit_naryop(self, e: NaryOp, ctx_def: ContextDef):
        for arg in e.args:
            self._visit_expr(arg, ctx_def)
        self._record_use(e, ctx_def)

    def _visit_call(self, e: Call, ctx_def: ContextDef):
        for arg in e.args:
            self._visit_expr(arg, ctx_def)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx_def)
        self._record_use(e, ctx_def)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_context(self, stmt: ContextStmt, ctx_def: ContextDef):
        # The context expression is evaluated under real arithmetic (not the
        # enclosing floating-point context), so we do not visit it as a use
        # of the enclosing context.  This is consistent with how
        # partial_eval evaluates context expressions under REAL.
        # Resolve the context expression, falling back to a symbolic
        # variable when partial evaluation cannot determine the value.
        ctx = self._resolve_ctx_expr(stmt.ctx)
        # Create a fresh context definition for the body.
        new_def = self._make_def(stmt, ctx)
        self._visit_block(stmt.body, new_def)

    # ------------------------------------------------------------------
    # Function entry point

    def _visit_function(self, func: FuncDef, _ctx_def: None):
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

        func_def = self._make_def(func, body_ctx)
        self._visit_block(func.body, func_def)

    def analyze(self) -> CtxDefUseAnalysis:
        self._visit_function(self.func, None)
        return CtxDefUseAnalysis(self.defs, self.uses)


class CtxDefUse:
    """
    Context definition-use analysis.

    This analysis computes, for each rounding context definition in a
    function, the set of context-sensitive expressions evaluated under it.

    Context definitions are introduced by:

    - ``with`` statements (:class:`ContextStmt`), and
    - the function-level context annotation in :class:`FuncDef`.

    The analysis attempts partial evaluation to resolve context expressions
    to concrete :class:`Context` values.  When partial evaluation does not
    fully reduce a context expression, or when a function is entered
    without an overriding context, a fresh symbolic context variable
    (:class:`NamedId`) is generated.
    """

    @staticmethod
    def analyze(func: FuncDef, *, def_use: DefineUseAnalysis | None = None) -> CtxDefUseAnalysis:
        """
        Runs context definition-use analysis on a function.

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
        return _CtxDefUseInstance(func, eval_info).analyze()
