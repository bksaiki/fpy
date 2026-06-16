"""
Constant folding — thin rewriter over :class:`fpy2.analysis.PartialEval`.

Queries ``pe.by_expr`` at each AST node and substitutes a literal
when a value is available.  No value-tracking dataflow lives here.
"""

from fractions import Fraction

from ..analysis import DefineUse, DefineUseAnalysis, PartialEval, PartialEvalInfo
from ..analysis.partial_eval import Value
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import Context, Float


class _ConstFoldInstance(DefaultTransformVisitor):
    """ConstFold rewriter — queries ``pe.by_expr`` at each node before
    descent and substitutes a literal on hit.  The ``enable_*`` flags
    are dispatched by the folded value's kind: ``Context`` →
    ``enable_context``, otherwise → ``enable_op``."""

    func: FuncDef
    pe: PartialEvalInfo
    enable_context: bool
    enable_op: bool

    def __init__(
        self,
        func: FuncDef,
        pe: PartialEvalInfo,
        enable_context: bool,
        enable_op: bool,
    ):
        self.func = func
        self.pe = pe
        self.enable_context = enable_context
        self.enable_op = enable_op

    def _value_to_literal(self, val: Value, loc):
        """Convert a :data:`Value` to an AST literal; return ``None`` if
        ``val`` has no FPy literal form (types, functions, modules,
        containers with non-emittable elements)."""
        # bool is checked first — it's a subclass of int.
        if isinstance(val, bool):
            return BoolVal(val, loc)
        if isinstance(val, Float):
            val = val.as_rational()
        elif isinstance(val, (int, float)):
            val = Fraction(val)
        if isinstance(val, Fraction):
            if val.denominator == 1:
                return Integer(int(val), loc)
            func = Attribute(Var(NamedId('fp'), loc), 'rational', loc)
            return Rational(func, val.numerator, val.denominator, loc)
        if isinstance(val, Context):
            return ForeignVal(val, loc)
        if isinstance(val, (tuple, list)):
            elts = [self._value_to_literal(elt, loc) for elt in val]
            if any(e is None for e in elts):
                return None
            return TupleExpr(elts, loc) if isinstance(val, tuple) else ListExpr(elts, loc)
        return None

    def _fold(self, e: Expr) -> Expr | None:
        """Look up ``e`` in ``pe.by_expr`` and convert to a literal,
        gated by ``enable_op`` / ``enable_context``."""
        if e not in self.pe.by_expr:
            return None
        lit = self._value_to_literal(self.pe.by_expr[e], e.loc)
        if lit is None:
            return None
        is_ctx_fold = isinstance(lit, ForeignVal) and isinstance(lit.val, Context)
        if is_ctx_fold:
            return lit if self.enable_context else None
        return lit if self.enable_op else None

    def _visit_expr(self, e: Expr, ctx) -> Expr:
        # Single chokepoint: every expression in the tree comes here
        # before the default type-dispatched rewrite.
        lit = self._fold(e)
        if lit is not None:
            return lit
        return super()._visit_expr(e, ctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """Constant folding and propagation.

    Substitutes any expression whose value is statically known with
    a literal AST node — operators, ``Var`` and ``Attribute`` lookups,
    ``Compare`` chains, context-constructor :class:`Call` s, tuple /
    list literals, list indexing and slicing.  Static values come
    from :class:`PartialEval`; pass an existing analysis via
    ``partial_eval=`` to avoid re-running it.

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
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> FuncDef:
        """Apply constant folding to *func*.  Pass cached ``def_use``
        / ``partial_eval`` to share with other passes.  Set
        ``enable_context=False`` to skip ``Context`` folds or
        ``enable_op=False`` to skip everything else."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if partial_eval is None:
            partial_eval = PartialEval.apply(func, def_use=def_use)
        return _ConstFoldInstance(
            func, partial_eval, enable_context, enable_op,
        ).apply()
