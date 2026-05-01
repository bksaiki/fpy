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

Format shape lattice
--------------------
The analysis tracks a **format shape** that mirrors the basic-type structure::

    FormatShape ::= None                            # non-numeric values
                  | Format                          # scalar real
                  | TupleShape(elts: tuple[Shape])  # heterogeneous tuple
                  | ListShape(elt: Shape)           # homogeneous list

- ``None`` is used for booleans, contexts, foreign values, function values, and
  any other expression for which a number format is not meaningful.
- A scalar :class:`Format` (e.g. ``IEEEFormat(es=8, nbits=32)``) describes a
  real-valued expression.  ``REAL_FORMAT`` is the **scalar top** — unrestricted
  real values (the format is unknown or unconstrained).
- :class:`TupleShape` tracks per-element shapes and is **not** weakened to a
  single scalar.
- :class:`ListShape` tracks a single element shape; the shapes of all elements
  in a list expression are joined to obtain it (lists are homogeneous).

**Join rule** (least upper bound)::

    join(None, None)               = None
    join(f, f)                     = f                 (scalar Format)
    join(f1, f2)                   = REAL_FORMAT       (different scalars)
    join(Tuple(a..), Tuple(b..))   = Tuple(join(ai, bi)..)
    join(List(a), List(b))         = List(join(a, b))

For loops, phi nodes are initialised from the pre-loop definition's shape, the
loop body is visited, and then the phi shape is updated with the join.  Because
the scalar lattice has ``REAL_FORMAT`` as a top element and the structural
lattice is shape-preserving, the fixpoint converges in **at most two
iterations**.

Format inference rules
----------------------
- **Context-sensitive operations** (``NullaryOp``, ``UnaryOp``, ``BinaryOp``,
  ``TernaryOp``, ``NaryOp``): the result is rounded to the active rounding
  context's format, i.e. ``scope.ctx.format()`` when the context is concrete,
  or ``REAL_FORMAT`` when it is a symbolic variable.
- **Function calls** (``Call``): conservatively the top shape of the callee's
  return type.
- **Variable references** (``Var``): the shape of the variable's definition.
- **Numeric literals** (``Decnum``, ``Integer``, ``Rational``, …):
  ``REAL_FORMAT`` — constants are exact real values and are not rounded.
- **Booleans, comparisons, foreign values, attributes**: ``None``.
- **Tuple expressions**: ``TupleShape`` of the per-element shapes.
- **List expressions**: ``ListShape`` of the join of element shapes; the top
  shape of the list's element type when the list is empty.
- **List comprehensions**: ``ListShape`` of the body expression's shape; the
  loop target is bound to the iterable's element shape.
- **Indexing/slicing**: list indexing returns the list's element shape; list
  slicing returns the same ``ListShape`` as the value.
- **Inline conditionals** (``IfExpr``): ``join(then_shape, else_shape)``.
"""

from dataclasses import dataclass
from functools import reduce
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import Visitor
from ..number import Context
from ..number.context.format import Format
from ..number.context.real import REAL_FORMAT
from ..types import (
    Type,
    BoolType,
    RealType,
    ContextType,
    TupleType,
    ListType,
    FunctionType,
    VarType,
)

from .context_use import ContextUse, ContextUseAnalysis, ContextScope, ContextUseSite
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import PhiDef, Definition, DefSite
from .type_infer import TypeInfer, TypeAnalysis

__all__ = [
    'FormatInfer',
    'FormatAnalysis',
    'FormatShape',
    'TupleShape',
    'ListShape',
]


#####################################################################
# Format shape lattice

@dataclass(frozen=True)
class TupleShape:
    """Format shape for a tuple-valued expression."""
    elts: tuple['FormatShape', ...]


@dataclass(frozen=True)
class ListShape:
    """Format shape for a list-valued expression (homogeneous element shape)."""
    elt: 'FormatShape'


FormatShape: TypeAlias = 'None | Format | TupleShape | ListShape'
"""
Inferred format shape for an expression or variable definition.

- ``None`` — no numeric format (booleans, contexts, foreign values, …).
- :class:`Format` — scalar format; ``REAL_FORMAT`` is the top of the scalar
  lattice.
- :class:`TupleShape` — heterogeneous tuple, per-element shapes preserved.
- :class:`ListShape` — homogeneous list, single element shape (the join of
  all element shapes).
"""


def _top_shape(ty: Type) -> FormatShape:
    """
    Returns the top of the format-shape lattice for *ty*.

    For scalar real values this is ``REAL_FORMAT``; for tuples and lists it is
    a structural shape with ``REAL_FORMAT`` (or ``None``) at the leaves.  For
    non-numeric types (bool, context, function, type variable) it is ``None``.
    """
    match ty:
        case RealType():
            return REAL_FORMAT
        case TupleType():
            return TupleShape(tuple(_top_shape(t) for t in ty.elts))
        case ListType():
            return ListShape(_top_shape(ty.elt))
        case BoolType() | ContextType() | FunctionType() | VarType():
            return None
        case _:
            raise RuntimeError(f'unreachable: unknown type {ty!r}')


def _join_shapes(s1: FormatShape, s2: FormatShape) -> FormatShape:
    """
    Returns the join (least upper bound) of two format shapes.

    Joins are structural and only defined for shapes of matching kind — the
    type checker guarantees this for well-typed programs.
    """
    match s1, s2:
        case None, None:
            return None
        case Format(), Format():
            return s1 if s1 == s2 else REAL_FORMAT
        case TupleShape(elts=a), TupleShape(elts=b) if len(a) == len(b):
            return TupleShape(tuple(_join_shapes(x, y) for x, y in zip(a, b)))
        case ListShape(elt=a), ListShape(elt=b):
            return ListShape(_join_shapes(a, b))
        case _:
            raise RuntimeError(
                f'unreachable: cannot join incompatible format shapes {s1!r}, {s2!r}'
            )


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
    format shape.  For real-valued expressions the shape is a :class:`Format`;
    for booleans and other non-numeric expressions it is ``None``; for tuples
    and lists it is a structural :class:`TupleShape` or :class:`ListShape`.
    """

    type_info: TypeAnalysis
    """Underlying basic-type analysis (bool, real, list, …)."""

    ctx_use: ContextUseAnalysis
    """Underlying context-use analysis (maps operations to rounding scopes)."""

    by_def: dict[Definition, FormatShape]
    """
    Format shape inferred for each variable definition site.

    Keys are ``AssignDef`` or ``PhiDef`` objects from the definition-use
    analysis.  For phi nodes the shape is the join of the two incoming
    control-flow edge shapes.
    """

    by_expr: dict[Expr, FormatShape]
    """
    Format shape inferred for each expression node.

    For context-sensitive operations the shape is the format of the active
    rounding context.  For variable references it is the definition's shape.
    For non-numeric expressions it is ``None``.
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

    by_def: dict[Definition, FormatShape]
    by_expr: dict[Expr, FormatShape]

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

    def _set_def_shape(self, d: Definition, shape: FormatShape):
        self.by_def[d] = shape

    def _shape_of_def(self, d: Definition) -> FormatShape:
        if d in self.by_def:
            return self.by_def[d]
        # Unseen definition (e.g. an outer-scope free variable whose binding
        # was not visited): fall back to the top shape of its type.
        ty = self.type_info.by_def.get(d)
        return _top_shape(ty) if ty is not None else None

    def _scope_format(self, e: ContextUseSite) -> Format:
        """Returns the format of the rounding context scope for *e*."""
        scope = self.ctx_use.use_to_scope.get(e)
        if scope is None:
            return REAL_FORMAT
        return _format_of_scope(scope)

    def _op_shape(self, e: Expr) -> FormatShape:
        """
        Shape of a rounded operation's result.

        Real-valued operations are rounded to the active context's format.
        Operations that return any other type (bool, list, context, …) are
        not numerical computations — their shape is derived from the type.
        """
        ret_ty = self.type_info.by_expr.get(e)
        if isinstance(ret_ty, RealType):
            return self._scope_format(e)
        return _top_shape(ret_ty) if ret_ty is not None else REAL_FORMAT

    def _visit_binding(
        self,
        site: DefSite,
        binding: Id | TupleBinding,
        shape: FormatShape,
    ):
        """Records *shape* for every variable introduced by *binding* at *site*."""
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self._set_def_shape(d, shape)
            case UnderscoreId():
                pass
            case TupleBinding():
                if isinstance(shape, TupleShape) and len(shape.elts) == len(binding.elts):
                    for sub_binding, sub_shape in zip(binding.elts, shape.elts):
                        self._visit_binding(site, sub_binding, sub_shape)
                else:
                    # Shape unknown or mismatched: derive per-element shape from
                    # the type of each bound variable instead.
                    for sub_binding in binding.elts:
                        self._visit_binding(site, sub_binding, self._shape_from_binding_type(site, sub_binding))
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _shape_from_binding_type(self, site: DefSite, binding: Id | TupleBinding) -> FormatShape:
        """Top shape derived from the type of variables introduced by *binding*."""
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                ty = self.type_info.by_def.get(d)
                return _top_shape(ty) if ty is not None else None
            case UnderscoreId():
                return None
            case TupleBinding():
                return TupleShape(
                    tuple(
                        self._shape_from_binding_type(site, sub) for sub in binding.elts
                    )
                )
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    # ------------------------------------------------------------------
    # Expression visitors — return the inferred FormatShape for *e*

    def _visit_expr(self, e: Expr, ctx: None) -> FormatShape:  # type: ignore[override]
        """Dispatch, record in ``by_expr``, and return the inferred shape."""
        shape: FormatShape = super()._visit_expr(e, ctx)
        self.by_expr[e] = shape
        return shape

    # Numeric literals: exact real values, not rounded
    def _visit_decnum(self, e: Decnum, ctx: None) -> FormatShape:
        return REAL_FORMAT

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> FormatShape:
        return REAL_FORMAT

    def _visit_integer(self, e: Integer, ctx: None) -> FormatShape:
        return REAL_FORMAT

    def _visit_rational(self, e: Rational, ctx: None) -> FormatShape:
        return REAL_FORMAT

    def _visit_digits(self, e: Digits, ctx: None) -> FormatShape:
        return REAL_FORMAT

    # Non-real-valued leaves
    def _visit_bool(self, e: BoolVal, ctx: None) -> FormatShape:
        return None

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> FormatShape:
        return None

    def _visit_attribute(self, e: Attribute, ctx: None) -> FormatShape:
        self._visit_expr(e.value, ctx)
        return None

    # Variable reference: propagate from the definition
    def _visit_var(self, e: Var, ctx: None) -> FormatShape:
        d = self.def_use.find_def_from_use(e)
        return self._shape_of_def(d)

    # Context-sensitive operations: real-valued results take the active scope's
    # format; non-real results (bool predicates, range, declcontext, …) take
    # the top shape of their inferred type.
    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> FormatShape:
        return self._op_shape(e)

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> FormatShape:
        self._visit_expr(e.arg, ctx)
        return self._op_shape(e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> FormatShape:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return self._op_shape(e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> FormatShape:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return self._op_shape(e)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> FormatShape:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return self._op_shape(e)

    # Function calls: conservatively the top shape of the return type
    def _visit_call(self, e: Call, ctx: None) -> FormatShape:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)
        ret_ty = self.type_info.by_expr.get(e)
        return _top_shape(ret_ty) if ret_ty is not None else None

    # Comparison: produces a bool, so no numeric shape
    def _visit_compare(self, e: Compare, ctx: None) -> FormatShape:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return None

    # Compound expressions
    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> FormatShape:
        return TupleShape(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> FormatShape:
        elt_shapes = [self._visit_expr(elt, ctx) for elt in e.elts]
        if elt_shapes:
            joined = reduce(_join_shapes, elt_shapes)
        else:
            # Empty list: derive the element shape from the inferred list type.
            list_ty = self.type_info.by_expr.get(e)
            if isinstance(list_ty, ListType):
                joined = _top_shape(list_ty.elt)
            else:
                joined = None
        return ListShape(joined)

    def _visit_list_comp(self, e: ListComp, ctx: None) -> FormatShape:
        for target, iterable in zip(e.targets, e.iterables):
            iter_shape = self._visit_expr(iterable, ctx)
            elt_shape = iter_shape.elt if isinstance(iter_shape, ListShape) else None
            self._visit_binding(e, target, elt_shape)
        body_shape = self._visit_expr(e.elt, ctx)
        return ListShape(body_shape)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> FormatShape:
        value_shape = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        if isinstance(value_shape, ListShape):
            return value_shape.elt
        # Unexpected shape (e.g. unknown): fall back to the type's top shape.
        ret_ty = self.type_info.by_expr.get(e)
        return _top_shape(ret_ty) if ret_ty is not None else None

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> FormatShape:
        value_shape = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        # Slice of a list has the same shape as the list itself.
        return value_shape

    def _visit_list_set(self, e: ListSet, ctx: None) -> FormatShape:
        value_shape = self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.expr, ctx)
        return value_shape

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> FormatShape:
        self._visit_expr(e.cond, ctx)
        then_shape = self._visit_expr(e.ift, ctx)
        else_shape = self._visit_expr(e.iff, ctx)
        return _join_shapes(then_shape, else_shape)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_assign(self, stmt: Assign, ctx: None):
        shape = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, shape)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        # The list variable's definition shape is unchanged by element mutation.

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._shape_of_def(self.def_use.defs[phi.lhs])
            rhs = self._shape_of_def(self.def_use.defs[phi.rhs])
            self._set_def_shape(phi, _join_shapes(lhs, rhs))

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._shape_of_def(self.def_use.defs[phi.lhs])
            rhs = self._shape_of_def(self.def_use.defs[phi.rhs])
            self._set_def_shape(phi, _join_shapes(lhs, rhs))

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # Initialise phi shapes from pre-loop (lhs) definitions so that the
        # loop body can reference them before the back-edge has been processed.
        for phi in self.def_use.phis[stmt]:
            self._set_def_shape(phi, self._shape_of_def(self.def_use.defs[phi.lhs]))
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        # Recompute phi shapes as the join of pre-loop and body values.
        for phi in self.def_use.phis[stmt]:
            lhs = self._shape_of_def(self.def_use.defs[phi.lhs])
            rhs = self._shape_of_def(self.def_use.defs[phi.rhs])
            self._set_def_shape(phi, _join_shapes(lhs, rhs))

    def _visit_for(self, stmt: ForStmt, ctx: None):
        iter_shape = self._visit_expr(stmt.iterable, ctx)
        elt_shape = iter_shape.elt if isinstance(iter_shape, ListShape) else None
        self._visit_binding(stmt, stmt.target, elt_shape)
        # Initialise phi shapes from pre-loop (lhs) definitions
        for phi in self.def_use.phis[stmt]:
            self._set_def_shape(phi, self._shape_of_def(self.def_use.defs[phi.lhs]))
        self._visit_block(stmt.body, ctx)
        # Recompute phi shapes as the join of pre-loop and body values
        for phi in self.def_use.phis[stmt]:
            lhs = self._shape_of_def(self.def_use.defs[phi.lhs])
            rhs = self._shape_of_def(self.def_use.defs[phi.rhs])
            self._set_def_shape(phi, _join_shapes(lhs, rhs))

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # The context expression itself is not a numerical computation.
        # Record the context variable (if named) with shape ``None``.
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self._set_def_shape(d, None)
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
        # Arguments: top shape of the declared type (REAL_FORMAT for reals,
        # None for non-numeric, structural for tuples/lists).
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                ty = self.type_info.by_def.get(d)
                self._set_def_shape(d, _top_shape(ty) if ty is not None else None)
        # Free variables (captured from outer scope): top shape of inferred type.
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            ty = self.type_info.by_def.get(d)
            self._set_def_shape(d, _top_shape(ty) if ty is not None else None)
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

    **Format shape lattice**::

        FormatShape ::= None
                      | Format               (REAL_FORMAT is the scalar top)
                      | TupleShape(elts)
                      | ListShape(elt)

    **Join rule**::

        join(None, None)             = None
        join(f, f)                   = f
        join(f1, f2)                 = REAL_FORMAT     (different scalars)
        join(Tuple(a..), Tuple(b..)) = Tuple(join(ai, bi)..)
        join(List(a), List(b))       = List(join(a, b))

    This rule is applied at all control-flow merge points (phi nodes), including
    branch merges (``if``/``if1``) and loop back-edges (``while``/``for``).
    The fixpoint for loops converges in **at most two iterations**.

    **Usage**::

        from fpy2.analysis import FormatInfer

        info = FormatInfer.analyze(func)
        for d, shape in info.by_def.items():
            print(d.name, '->', shape)
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
            maps contain the inferred format shapes for every definition site
            and expression node in *func*.

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
