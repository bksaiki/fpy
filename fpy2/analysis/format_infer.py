"""
Format analysis for FPy programs.

This analysis determines the number format (bounding the set of representable
values) for each expression and variable definition in an FPy program.

Analysis pipeline
-----------------
1. **Type checking** (:class:`TypeInfer`) — establishes basic types for every
   expression and variable (``bool``, ``real``, ``list``, ``tuple``, …).
2. **Context-use analysis** (:class:`ContextUse`) — maps every context-sensitive
   operation (arithmetic, rounding, etc.) to the rounding context scope that is
   active when it is evaluated.
3. **Format inference** — propagates number format information through the
   program.  For each operation the format is derived from its active rounding
   context; for variables it is propagated from the expression they are bound
   to; at control-flow merge points (``if``/``while``/``for`` phi nodes) formats
   are joined.

Format lattice
--------------
The analysis uses a **flat format lattice**::

    specific_format  <  REAL_FORMAT   (top)

``REAL_FORMAT`` is the top element and represents unrestricted real values
(i.e., the format is unknown or unconstrained).  Any concrete :class:`Format`
object (e.g. ``IEEEFormat(es=8, nbits=32)``) is a lower element representing
values restricted to that format's representable set.

**Join rule** (least upper bound)::

    join(f, f)   = f
    join(f1, f2) = REAL_FORMAT   (when f1 ≠ f2)

For loops, phi nodes are initialised from the pre-loop definition's format,
the loop body is visited, and then the phi format is updated with the join.
Because ``REAL_FORMAT`` is the top element, the fixpoint converges in **at
most two iterations**.

Format inference rules
----------------------
- **Context-sensitive operations** (``NullaryOp``, ``UnaryOp``, ``BinaryOp``,
  ``TernaryOp``, ``NaryOp``): the result is rounded to the active rounding
  context's format, i.e. ``scope.ctx.format()`` when the context is concrete,
  or ``REAL_FORMAT`` when it is a symbolic variable.
- **Function calls** (``Call``): conservatively ``REAL_FORMAT``.
- **Variable references** (``Var``): the format of the variable's definition.
- **Numeric literals** (``Decnum``, ``Integer``, ``Rational``, …):
  ``REAL_FORMAT`` — constants are exact real values and are not rounded.
- **Inline conditionals** (``IfExpr``): ``join(then_fmt, else_fmt)``.
- **Non-real expressions** (``BoolVal``, comparisons, lists, …):
  ``REAL_FORMAT`` (no meaningful number format).
"""

from dataclasses import dataclass

from ..ast.fpyast import *
from ..ast.visitor import Visitor
from ..number import Context
from ..number.context.format import Format
from ..number.context.real import REAL_FORMAT
from ..types import Type

from .context_use import ContextUse, ContextUseAnalysis, ContextScope, ContextUseSite
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import PhiDef, Definition, DefSite
from .type_infer import TypeInfer, TypeAnalysis

__all__ = [
    'FormatInfer',
    'FormatAnalysis',
]


#####################################################################
# Lattice helpers

def _join_formats(f1: Format, f2: Format) -> Format:
    """
    Returns the join (least upper bound) of two formats.

    In the flat format lattice two different concrete formats join to
    ``REAL_FORMAT`` (the top element).
    """
    if f1 == f2:
        return f1
    return REAL_FORMAT


def _format_of_scope(scope: ContextScope) -> Format:
    """
    Returns the number format associated with a context scope.

    If the scope carries a concrete :class:`Context`, the format is
    ``ctx.format()``.  For symbolic (unresolved) context variables the
    analysis falls back to ``REAL_FORMAT``.
    """
    if isinstance(scope.ctx, Context):
        return scope.ctx.format()
    return REAL_FORMAT


#####################################################################
# Analysis result

@dataclass
class FormatAnalysis:
    """
    Result of format analysis for an FPy function.

    Maps each variable definition site and expression node to its inferred
    number format.  ``REAL_FORMAT`` represents the top of the lattice —
    either because the value is genuinely unconstrained (e.g. a function
    argument, a literal constant) or because the analysis is conservative
    (e.g. a function call whose callee is not examined).
    """

    type_info: TypeAnalysis
    """Underlying basic-type analysis (bool, real, list, …)."""

    ctx_use: ContextUseAnalysis
    """Underlying context-use analysis (maps operations to rounding scopes)."""

    by_def: dict[Definition, Format]
    """
    Format inferred for each variable definition site.

    Keys are ``AssignDef`` or ``PhiDef`` objects from the definition-use
    analysis.  For phi nodes the format is the join of the two incoming
    control-flow edge formats.
    """

    by_expr: dict[Expr, Format]
    """
    Format inferred for each expression node.

    For context-sensitive operations the format is the format of the active
    rounding context.  For variable references it is the definition's format.
    For all other expressions it is ``REAL_FORMAT``.
    """


#####################################################################
# Internal analysis visitor

class _FormatInferInstance(Visitor):
    """
    Single-use visitor that performs format inference.

    The visitor walks the entire function body in a single pass (with a
    two-step initialise-then-join for loop phi nodes) and populates
    ``by_def`` and ``by_expr``.
    """

    func: FuncDef
    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis

    by_def: dict[Definition, Format]
    by_expr: dict[Expr, Format]

    def __init__(
        self,
        func: FuncDef,
        type_info: TypeAnalysis,
        ctx_use: ContextUseAnalysis,
    ):
        self.func = func
        self.type_info = type_info
        self.ctx_use = ctx_use
        self.by_def = {}
        self.by_expr = {}

    # ------------------------------------------------------------------
    # Helpers

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    def _set_def_format(self, d: Definition, fmt: Format):
        self.by_def[d] = fmt

    def _format_of_def(self, d: Definition) -> Format:
        return self.by_def.get(d, REAL_FORMAT)

    def _scope_format(self, e: ContextUseSite) -> Format:
        """Returns the format of the rounding context scope for *e*."""
        scope = self.ctx_use.use_to_scope.get(e)
        if scope is None:
            return REAL_FORMAT
        return _format_of_scope(scope)

    def _visit_binding(self, site: DefSite, binding: Id | TupleBinding, fmt: Format):
        """Records *fmt* for every variable introduced by *binding* at *site*."""
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self._set_def_format(d, fmt)
            case UnderscoreId():
                pass
            case TupleBinding():
                # Conservative: each element of a destructured tuple is
                # assigned REAL_FORMAT because we do not track per-element
                # formats for tuples.
                for elt in binding.elts:
                    self._visit_binding(site, elt, REAL_FORMAT)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    # ------------------------------------------------------------------
    # Expression visitors — return the inferred Format for *e*

    def _visit_expr(self, e: Expr, ctx: None) -> Format:  # type: ignore[override]
        """Dispatch, record in ``by_expr``, and return the inferred format."""
        fmt: Format = super()._visit_expr(e, ctx)
        self.by_expr[e] = fmt
        return fmt

    # Numeric literals: exact real values, not rounded
    def _visit_decnum(self, e: Decnum, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_integer(self, e: Integer, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_rational(self, e: Rational, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_digits(self, e: Digits, ctx: None) -> Format:
        return REAL_FORMAT

    # Non-real-valued leaves
    def _visit_bool(self, e: BoolVal, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> Format:
        return REAL_FORMAT

    def _visit_attribute(self, e: Attribute, ctx: None) -> Format:
        self._visit_expr(e.value, ctx)
        return REAL_FORMAT

    # Variable reference: propagate from the definition
    def _visit_var(self, e: Var, ctx: None) -> Format:
        d = self.def_use.find_def_from_use(e)
        return self._format_of_def(d)

    # Context-sensitive operations: format comes from the active scope
    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> Format:
        return self._scope_format(e)

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> Format:
        self._visit_expr(e.arg, ctx)
        return self._scope_format(e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> Format:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return self._scope_format(e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> Format:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return self._scope_format(e)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> Format:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return self._scope_format(e)

    # Function calls: conservatively REAL_FORMAT (callee not analysed)
    def _visit_call(self, e: Call, ctx: None) -> Format:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)
        return REAL_FORMAT

    # Comparison: produces a bool, so no numeric format
    def _visit_compare(self, e: Compare, ctx: None) -> Format:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return REAL_FORMAT

    # Compound expressions
    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> Format:
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        return REAL_FORMAT

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> Format:
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        return REAL_FORMAT

    def _visit_list_comp(self, e: ListComp, ctx: None) -> Format:
        for target, iterable in zip(e.targets, e.iterables):
            self._visit_expr(iterable, ctx)
            # Conservatively assign REAL_FORMAT to loop iteration variables
            self._visit_binding(e, target, REAL_FORMAT)
        self._visit_expr(e.elt, ctx)
        return REAL_FORMAT

    def _visit_list_ref(self, e: ListRef, ctx: None) -> Format:
        self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        # Conservative: REAL_FORMAT for list element dereferences
        return REAL_FORMAT

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> Format:
        self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return REAL_FORMAT

    def _visit_list_set(self, e: ListSet, ctx: None) -> Format:
        self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.expr, ctx)
        return REAL_FORMAT

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> Format:
        self._visit_expr(e.cond, ctx)
        then_fmt = self._visit_expr(e.ift, ctx)
        else_fmt = self._visit_expr(e.iff, ctx)
        return _join_formats(then_fmt, else_fmt)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_assign(self, stmt: Assign, ctx: None):
        fmt = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, fmt)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        # The list variable's definition format is unchanged by element mutation

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs_fmt = self._format_of_def(self.def_use.defs[phi.lhs])
            rhs_fmt = self._format_of_def(self.def_use.defs[phi.rhs])
            self._set_def_format(phi, _join_formats(lhs_fmt, rhs_fmt))

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs_fmt = self._format_of_def(self.def_use.defs[phi.lhs])
            rhs_fmt = self._format_of_def(self.def_use.defs[phi.rhs])
            self._set_def_format(phi, _join_formats(lhs_fmt, rhs_fmt))

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # Initialise phi formats from pre-loop (lhs) definitions so that the
        # loop body can reference them before the back-edge has been processed.
        for phi in self.def_use.phis[stmt]:
            self._set_def_format(phi, self._format_of_def(self.def_use.defs[phi.lhs]))
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        # Recompute phi formats as the join of pre-loop and body values.
        for phi in self.def_use.phis[stmt]:
            lhs_fmt = self._format_of_def(self.def_use.defs[phi.lhs])
            rhs_fmt = self._format_of_def(self.def_use.defs[phi.rhs])
            self._set_def_format(phi, _join_formats(lhs_fmt, rhs_fmt))

    def _visit_for(self, stmt: ForStmt, ctx: None):
        self._visit_expr(stmt.iterable, ctx)
        # Conservatively assign REAL_FORMAT to the iteration variable
        self._visit_binding(stmt, stmt.target, REAL_FORMAT)
        # Initialise phi formats from pre-loop (lhs) definitions
        for phi in self.def_use.phis[stmt]:
            self._set_def_format(phi, self._format_of_def(self.def_use.defs[phi.lhs]))
        self._visit_block(stmt.body, ctx)
        # Recompute phi formats as the join of pre-loop and body values
        for phi in self.def_use.phis[stmt]:
            lhs_fmt = self._format_of_def(self.def_use.defs[phi.lhs])
            rhs_fmt = self._format_of_def(self.def_use.defs[phi.rhs])
            self._set_def_format(phi, _join_formats(lhs_fmt, rhs_fmt))

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # The context expression itself is not a numerical computation.
        # Record the context variable (if named) with REAL_FORMAT.
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self._set_def_format(d, REAL_FORMAT)
        self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        pass

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # Arguments are unrounded inputs — assign REAL_FORMAT
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_def_format(d, REAL_FORMAT)
        # Free variables (captured from outer scope) — conservatively REAL_FORMAT
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            self._set_def_format(d, REAL_FORMAT)
        self._visit_block(func.body, ctx)

    def analyze(self) -> FormatAnalysis:
        self._visit_function(self.func, None)
        return FormatAnalysis(
            type_info=self.type_info,
            ctx_use=self.ctx_use,
            by_def=self.by_def,
            by_expr=self.by_expr,
        )


#####################################################################
# Public API

class FormatInfer:
    """
    Format inference for FPy functions.

    This analysis bounds the number format for each expression and variable
    definition in an FPy program.  It is an **alternative** to
    :class:`ContextInfer`: rather than threading context information through a
    Hindley-Milner style type system, it uses two simpler pre-analyses
    (:class:`TypeInfer` and :class:`ContextUse`) and then performs a single
    forward pass over the AST.

    **Format lattice**::

        specific_format  <  REAL_FORMAT   (top)

    The top element ``REAL_FORMAT`` represents unrestricted real values.
    Any concrete :class:`Format` object (e.g. ``IEEEFormat(es=8, nbits=32)``)
    represents values that are guaranteed to be representable under that format
    after rounding.

    **Join rule**::

        join(f, f)   = f
        join(f1, f2) = REAL_FORMAT   (when f1 ≠ f2)

    This rule is applied at all control-flow merge points (phi nodes), including
    branch merges (``if``/``if1``) and loop back-edges (``while``/``for``).
    Because ``REAL_FORMAT`` is the top element, the fixpoint for loops
    converges in **at most two iterations**.

    **Usage**::

        from fpy2.analysis import FormatInfer

        info = FormatInfer.analyze(func)
        for d, fmt in info.by_def.items():
            print(d.name, '->', fmt)
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
    ) -> FormatAnalysis:
        """
        Performs format analysis on a compiled FPy function.

        All three pre-analyses (``def_use``, ``type_info``, ``ctx_use``) are
        computed automatically when not supplied.  Pre-computing them externally
        is useful when they are also needed for other purposes.

        Args:
            func:      The :class:`FuncDef` AST node to analyse.
            def_use:   Optional pre-computed definition-use analysis.
            type_info: Optional pre-computed type analysis.
            ctx_use:   Optional pre-computed context-use analysis.

        Returns:
            A :class:`FormatAnalysis` result whose ``by_def`` and ``by_expr``
            maps contain the inferred formats for every definition site and
            expression node in *func*.

        Raises:
            TypeError: If *func* is not a :class:`FuncDef`.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {type(func)}")

        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        if ctx_use is None:
            ctx_use = ContextUse.analyze(func, def_use=def_use)

        return _FormatInferInstance(func, type_info, ctx_use).analyze()
