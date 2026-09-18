"""
Constant folding — thin rewriter over :class:`fpy2.analysis.PartialEval`.

Queries ``partial_eval.by_expr`` at each AST node and substitutes a literal
when a value is available.  No value-tracking dataflow lives here.

``len(xs)`` is the one fold that does not come from a value.  `PartialEval`
reasons only about values, so it folds a `Len` only when it knows the whole
list -- a length it cannot get from a list whose elements are unknown.
:class:`fpy2.analysis.ArraySizeInfer` proves exactly that length, and this is
the layer where both analyses are in scope: the size analysis *consumes*
`PartialEval`, so the query cannot live inside it.
"""

import math
from fractions import Fraction

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    DefineUse,
    DefineUseAnalysis,
    ListSize,
    PartialEval,
    PartialEvalInfo,
    Purity,
    concrete_size,
)
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..number import Context, Float


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
            # A rounding context DOES have a literal form here (folded to a
            # `ForeignVal` for context analysis).  This intentionally differs
            # from `free_var_elim.inline_literal`, which keeps a `Context` free.
            return ForeignVal(val, loc)
        case tuple() | list():
            elts = [value_to_literal(elt, loc) for elt in val]
            if any(e is None for e in elts):
                return None
            return TupleExpr(elts, loc) if isinstance(val, tuple) else ListExpr(elts, loc)
        case _:
            # includes `Foreign`: opaque values have no literal form
            return None


class _LenScan(DefaultVisitor):
    """Where *func* has a ``Len``: anywhere (``found``) and inside an
    assertion's test (``in_assert``).

    `ArraySizeInfer` runs `TypeInfer` and walks callees, so it is far from
    free, and the assert-free reading of it is a second run.  A function with
    no ``len`` pays for neither, one with no ``len`` in an assert pays for the
    first only.
    """

    found: bool
    in_assert: bool
    _depth: int

    def __init__(self):
        self.found = False
        self.in_assert = False
        self._depth = 0

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if isinstance(e, Len):
            self.found = True
            if self._depth > 0:
                self.in_assert = True
        super()._visit_unaryop(e, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._depth += 1
        try:
            super()._visit_assert(stmt, ctx)
        finally:
            self._depth -= 1

    @staticmethod
    def scan(func: FuncDef) -> '_LenScan':
        inst = _LenScan()
        inst._visit_function(func, None)
        return inst


class _ConstFoldInstance(DefaultTransformVisitor):
    """ConstFold rewriter — queries ``partial_eval.by_expr`` at each node before
    descent and substitutes a literal on hit.  The ``enable_*`` flags
    are dispatched by the folded value's kind: ``Context`` →
    ``enable_context``, otherwise → ``enable_op``."""

    func: FuncDef
    partial_eval: PartialEvalInfo
    array_size: ArraySizeAnalysis | None
    array_size_no_assert: ArraySizeAnalysis | None
    def_use: DefineUseAnalysis
    enable_context: bool
    enable_op: bool
    changed: bool
    _in_assert: bool

    def __init__(
        self,
        func: FuncDef,
        partial_eval: PartialEvalInfo,
        array_size: ArraySizeAnalysis | None,
        array_size_no_assert: ArraySizeAnalysis | None,
        def_use: DefineUseAnalysis,
        enable_context: bool,
        enable_op: bool,
    ):
        self.func = func
        self.partial_eval = partial_eval
        self.array_size = array_size
        self.array_size_no_assert = array_size_no_assert
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
        # No-op: the literal we'd substitute is already at this site.
        if type(e) is type(lit) and e.is_equiv(lit):
            return None
        if isinstance(lit, ListExpr | TupleExpr):
            # An aggregate literal is an *allocation*, so substituting one is
            # not the shrink a scalar fold is: it replaces a compact producer
            # (a name, a `range(n)`) with a materialized list, once per use
            # site, and each is a distinct object where the name was one.
            return None
        is_ctx_fold = isinstance(lit, ForeignVal) and isinstance(lit.val, Context)
        if is_ctx_fold:
            if not self.enable_context:
                return None
        elif not self.enable_op:
            return None
        self.changed = True
        return lit

    def _fold_len(self, e: Expr) -> Expr | None:
        """``len(xs)`` where the size analysis proves a concrete length.

        This is the one fold not keyed on a value.  `PartialEval` reaches a
        `Len` only through `_visit_unaryop`, which wants the argument's whole
        value, so a list of known length and unknown elements -- an argument
        with a fixed dimension, the list `UnfoldZip` indexes -- never folds
        there.

        Inside an assertion's test the sizes are read from the analysis that
        did *not* seed itself from asserts.  `ArraySizeInfer` learns from
        ``assert len(ys) == len(xs)``, so folding that test against what it
        learned there is circular: both sides become the same literal, the
        assert dies as trivially true, and the equality it was carrying to
        every later size query dies with it.  Against the assert-free sizes
        it folds only when the lengths were already known some other way, and
        then losing the assert costs nothing.

        The argument must be pure: substituting the literal drops it, and the
        size analysis will happily report the length of a list a call
        returned.  A pure argument is droppable on the same terms
        `DeadCodeEliminate` already drops one.
        """
        sizes = self.array_size_no_assert if self._in_assert else self.array_size
        if sizes is None or not self.enable_op or not isinstance(e, Len):
            return None
        bound = sizes.by_expr.get(e.arg)
        if not isinstance(bound, ListSize):
            return None
        size = concrete_size(bound.size)
        if size is None:
            return None
        if not Purity.analyze_expr(e.arg, self.def_use):
            return None
        # The literal sits where the `len` sat, so it rounds under the same
        # context the `len` would have: no rounding is owed here.
        self.changed = True
        return Integer(size, e.loc)

    def _visit_expr(self, e: Expr, ctx) -> Expr:
        # Single chokepoint: every expression in the tree comes here
        # before the default type-dispatched rewrite.
        lit = self._fold(e)
        if lit is not None:
            return lit
        lit = self._fold_len(e)
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

    ``len(xs)`` also folds when :class:`ArraySizeInfer` proves a concrete
    length and the argument is pure, even though its *value* is unknown --
    the one fold whose source is not :class:`PartialEval`.  Inside an
    assertion's test the sizes come from a second run of the analysis with
    ``seed_from_asserts=False``, so an assertion is never discharged by what
    the analysis learned from it.  Neither run happens unless the function
    has a ``len`` in the corresponding position; pass ``array_size=`` to
    reuse the first.

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
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> FuncDef:
        """Apply constant folding to *func*.  Pass cached ``def_use``
        / ``partial_eval`` / ``array_size`` to share with other passes.
        Set ``enable_context=False`` to skip ``Context`` folds or
        ``enable_op=False`` to skip everything else."""
        func, _ = ConstFold.apply_with_status(
            func,
            def_use=def_use,
            partial_eval=partial_eval,
            array_size=array_size,
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
        array_size_no_assert: ArraySizeAnalysis | None = None
        if enable_op:
            scan = _LenScan.scan(func)
            if array_size is None and scan.found:
                array_size = ArraySizeInfer.analyze(func, partial_eval=partial_eval)
            if scan.in_assert:
                array_size_no_assert = ArraySizeInfer.analyze(
                    func, partial_eval=partial_eval, seed_from_asserts=False
                )
        inst = _ConstFoldInstance(
            func, partial_eval, array_size, array_size_no_assert, def_use,
            enable_context, enable_op,
        )
        return inst.apply(), inst.changed
