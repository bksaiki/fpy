"""
Constant folding — thin rewriter over :class:`fpy2.analysis.PartialEval`.

Queries ``partial_eval.by_expr`` at each AST node and substitutes a literal
when a value is available.  No value-tracking dataflow lives here.

The shape queries -- ``len``, ``dim``, ``size`` -- come from
:class:`fpy2.analysis.ArraySizeInfer` instead, since a list can have a known
shape and unknown elements.  They live here rather than in `PartialEval`
because the size analysis consumes it.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeBound,
    ArraySizeInfer,
    DefineUse,
    DefineUseAnalysis,
    ListSize,
    PartialEval,
    PartialEvalInfo,
    Purity,
    TypeAnalysis,
    TypeInfer,
    concrete_size,
)
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..number import Context, Float
from ..types import ListType, Type, VarType


def _rational_literal(val: Fraction, loc):
    """AST literal for an exact rational: an ``Integer`` when integral,
    otherwise a ``rational`` literal."""
    if val.denominator == 1:
        return Integer(int(val), loc)
    return Rational(None, val.numerator, val.denominator, loc)


def value_to_literal(val: object, loc):
    """Convert a value to an AST literal; return ``None`` if ``val`` has no
    FPy literal form (types, functions, modules, containers with
    non-emittable elements)."""
    match val:
        case bool():                       # before int — bool is a subclass
            return BoolVal(val, loc)
        case Float():
            if val.is_nar():
                # Inf and NaN have no rational form, so no literal form either
                return None
            if val.is_zero() and val.s:
                # negative zero has no `Fraction` form; emit a signed literal
                return Decnum('-0.0', loc)
            return _rational_literal(val.as_rational(), loc)
        case float() if not math.isfinite(val):
            return None
        case int() | float():
            return _rational_literal(Fraction(val), loc)
        case Fraction():
            return _rational_literal(val, loc)
        case Context():
            # unlike `free_var_elim.inline_literal`, which keeps a `Context` free
            return ForeignVal(val, loc)
        case tuple() | list():
            elts = [value_to_literal(elt, loc) for elt in val]
            if any(e is None for e in elts):
                return None
            return TupleExpr(elts, loc) if isinstance(val, tuple) else ListExpr(elts, loc)
        case _:
            # includes `Foreign`: opaque values have no literal form
            return None


def _static_dim(ty: Type | None, bound: ArraySizeBound) -> int | None:
    """``dim`` as `ops.dim` computes it, or ``None`` if not provable.

    The nesting is the *type*'s: `ArraySizeBound` spells a scalar element and
    an element it failed to track both ``None``.  The descent through ``x[0]``
    stops at ``x == []``, so an empty level cuts the answer short of that
    nesting -- ``dim([[1.,2.],[3.,4.]][0:0])`` is 1 where the type nests twice
    -- and a level of unknown length might be that empty one.
    """
    depth = 0
    while isinstance(ty, ListType):
        depth += 1
        ty = ty.elt
    if depth == 0 or isinstance(ty, VarType):
        # a non-list, which `dim` rejects, or an element that may yet be one
        return None
    for level in range(depth - 1):
        if not isinstance(bound, ListSize):
            return None
        size = concrete_size(bound.size)
        if size is None:
            return None
        if size == 0:
            return level + 1
        bound = bound.elt
    return depth


class _ShapeScan(DefaultVisitor):
    """Where *func* has a shape query (``len`` / ``size`` / ``dim``): anywhere
    (``found``) and inside an assertion's test (``in_assert``).

    `ArraySizeInfer` runs `TypeInfer` and walks callees, so a function with no
    shape query runs neither reading of it, and one with none in an assert
    runs only the first.
    """

    found: bool
    in_assert: bool
    _depth: int

    def __init__(self):
        self.found = False
        self.in_assert = False
        self._depth = 0

    def _mark(self):
        self.found = True
        if self._depth > 0:
            self.in_assert = True

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if isinstance(e, Len | Dim):
            self._mark()
        super()._visit_unaryop(e, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        if isinstance(e, Size):
            self._mark()
        super()._visit_binaryop(e, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._depth += 1
        try:
            super()._visit_assert(stmt, ctx)
        finally:
            self._depth -= 1

    @staticmethod
    def scan(func: FuncDef) -> '_ShapeScan':
        inst = _ShapeScan()
        inst._visit_function(func, None)
        return inst


@dataclass
class _ShapeFacts:
    """What the shape folds read.  `sizes` is built against `type_info`;
    `sizes_no_assert` is the same analysis told to ignore asserts, built and
    read only where a shape query sits inside one."""

    type_info: TypeAnalysis
    sizes: ArraySizeAnalysis
    sizes_no_assert: ArraySizeAnalysis | None

    def at(self, in_assert: bool) -> ArraySizeAnalysis | None:
        return self.sizes_no_assert if in_assert else self.sizes


class _ConstFoldInstance(DefaultTransformVisitor):
    """ConstFold rewriter — queries ``partial_eval.by_expr`` at each node before
    descent and substitutes a literal on hit.  The ``enable_*`` flags
    are dispatched by the folded value's kind: ``Context`` →
    ``enable_context``, otherwise → ``enable_op``."""

    func: FuncDef
    partial_eval: PartialEvalInfo
    shape: _ShapeFacts | None
    def_use: DefineUseAnalysis
    enable_context: bool
    enable_op: bool
    changed: bool
    _in_assert: bool

    def __init__(
        self,
        func: FuncDef,
        partial_eval: PartialEvalInfo,
        shape: _ShapeFacts | None,
        def_use: DefineUseAnalysis,
        enable_context: bool,
        enable_op: bool,
    ):
        self.func = func
        self.partial_eval = partial_eval
        self.shape = shape
        self.def_use = def_use
        self.enable_context = enable_context
        self.enable_op = enable_op
        self.changed = False
        self._in_assert = False

    def _fold(self, e: Expr) -> Expr | None:
        """Look up ``e`` in ``partial_eval.by_expr`` and convert to a literal,
        gated by ``enable_op`` / ``enable_context``.  Returns ``None``
        when the substitution would be a structural no-op (the AST
        already has the same literal at this position) so ``simplify``
        can detect fixpoint."""
        if e not in self.partial_eval.by_expr:
            return None
        lit = value_to_literal(self.partial_eval.by_expr[e], e.loc)
        if lit is None:
            return None
        # the literal is already at this site
        if type(e) is type(lit) and e.is_equiv(lit):
            return None
        if isinstance(lit, ListExpr | TupleExpr):
            # an aggregate literal allocates: this would put a materialized
            # list where a name or a `range(n)` was, once per use site
            return None
        is_ctx_fold = isinstance(lit, ForeignVal) and isinstance(lit.val, Context)
        if is_ctx_fold:
            if not self.enable_context:
                return None
        elif not self.enable_op:
            return None
        self.changed = True
        return lit

    def _fold_shape(self, e: Expr) -> Expr | None:
        """A shape query -- ``len(xs)``, ``dim(xs)``, ``size(xs, n)`` -- that
        the size analysis answers where `PartialEval`, wanting the whole
        list's value, cannot.

        Inside an assertion's test the sizes come from the reading that did
        not seed itself from asserts.  `ArraySizeInfer` learns from
        ``assert len(ys) == len(xs)``, so folding that test against what it
        learned there would make both sides the same literal, kill the assert
        as trivially true, and take the equality with it.

        The operands must be pure: substituting the literal drops them.
        """
        if not self.enable_op or self.shape is None:
            return None
        sizes = self.shape.at(self._in_assert)
        if sizes is None:
            return None

        operands: tuple[Expr, ...]
        match e:
            case Len():
                operands = (e.arg,)
                bound = sizes.by_expr.get(e.arg)
                val = concrete_size(bound.size) if isinstance(bound, ListSize) else None
            case Dim():
                operands = (e.arg,)
                val = _static_dim(
                    self.shape.type_info.by_expr.get(e.arg), sizes.by_expr.get(e.arg)
                )
            case Size():
                operands = (e.first, e.second)
                val = self._size_of(e.first, e.second, sizes)
            case _:
                return None

        if val is None:
            return None
        if not all(Purity.analyze_expr(arg, self.def_use) for arg in operands):
            return None
        # the literal sits where the query sat, so the same context rounds it
        self.changed = True
        return Integer(val, e.loc)

    def _size_of(self, arg: Expr, index: Expr, sizes: ArraySizeAnalysis) -> int | None:
        """``size(arg, n)`` -- ``len(arg[0]...[0])``, *n* deep -- when *n* is
        known and every level it descends through has a known, non-empty
        length.  ``x[0]`` on an empty level raises, so a size past one is not
        a value to claim.
        """
        if index not in self.partial_eval.by_expr:
            return None
        lit = value_to_literal(self.partial_eval.by_expr[index], None)
        if not isinstance(lit, Integer) or lit.val < 0:
            return None

        bound = sizes.by_expr.get(arg)
        for _ in range(lit.val):
            if not isinstance(bound, ListSize):
                return None
            size = concrete_size(bound.size)
            if size is None or size == 0:
                return None
            bound = bound.elt
        return concrete_size(bound.size) if isinstance(bound, ListSize) else None

    def _visit_expr(self, e: Expr, ctx) -> Expr:
        # the chokepoint: every expression arrives here before the default
        # type-dispatched rewrite
        lit = self._fold(e)
        if lit is not None:
            return lit
        lit = self._fold_shape(e)
        if lit is not None:
            return lit
        return super()._visit_expr(e, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx):
        self._in_assert = True
        try:
            return super()._visit_assert(stmt, ctx)
        finally:
            self._in_assert = False

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """Constant folding and propagation.

    Substitutes any expression whose value is statically known with
    a literal AST node — operators, ``Var`` and ``Attribute`` lookups,
    ``Compare`` chains, context-constructor :class:`Call` s, list
    indexing and slicing.  Static values come from
    :class:`PartialEval`; pass an existing analysis via
    ``partial_eval=`` to avoid re-running it.

    A shape query -- ``len(xs)``, ``dim(xs)``, ``size(xs, n)`` -- also folds
    when :class:`ArraySizeInfer` proves the shape and the operands are pure,
    even where the list's *value* is unknown.  ``dim`` and ``size`` answer
    only where the runtime's descent through ``x[0]`` is known to reach that
    far, since it stops at the first empty level.  An assertion is never
    discharged by a size the analysis learned from that same assertion.  Pass
    ``array_size=`` / ``type_info=`` to reuse the analyses.

    A list or tuple *literal* is never substituted, however statically
    known: it allocates, so putting one at a use site is not the shrink
    a scalar fold is.

    Excluded: values that don't have an FPy literal form (Python
    types, functions, modules, containers whose elements aren't
    literal-emittable) — left as the original AST.
    """

    @staticmethod
    def apply(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        partial_eval: PartialEvalInfo | None = None,
        array_size: ArraySizeAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> FuncDef:
        """Apply constant folding to *func*.  Pass cached ``def_use`` /
        ``partial_eval`` / ``array_size`` / ``type_info`` to share with other
        passes.  Set ``enable_context=False`` to skip ``Context`` folds or
        ``enable_op=False`` to skip everything else."""
        func, _ = ConstFold.apply_with_status(
            func,
            def_use=def_use,
            partial_eval=partial_eval,
            array_size=array_size,
            type_info=type_info,
            enable_context=enable_context,
            enable_op=enable_op,
        )
        return func

    @staticmethod
    def apply_with_status(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        partial_eval: PartialEvalInfo | None = None,
        array_size: ArraySizeAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag
        — ``True`` iff at least one substitution occurred."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if partial_eval is None:
            partial_eval = PartialEval.apply(func, def_use=def_use)
        shape: _ShapeFacts | None = None
        if enable_op:
            scan = _ShapeScan.scan(func)
            if scan.found:
                # shared with `ArraySizeInfer`, which would otherwise run it itself
                if type_info is None:
                    type_info = TypeInfer.check(func, def_use=def_use)
                if array_size is None:
                    array_size = ArraySizeInfer.analyze(
                        func, partial_eval=partial_eval, type_info=type_info
                    )
                no_assert = ArraySizeInfer.analyze(
                    func, partial_eval=partial_eval, type_info=type_info,
                    seed_from_asserts=False,
                ) if scan.in_assert else None
                shape = _ShapeFacts(type_info, array_size, no_assert)

        inst = _ConstFoldInstance(
            func, partial_eval, shape, def_use, enable_context, enable_op
        )
        return inst.apply(), inst.changed
