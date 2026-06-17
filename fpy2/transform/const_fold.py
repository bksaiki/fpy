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
    changed: bool

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
        self.changed = False

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
        gated by ``enable_op`` / ``enable_context``.  Returns ``None``
        when the substitution would be a structural no-op (the AST
        already has the same literal at this position) so ``simplify``
        can detect fixpoint."""
        if e not in self.pe.by_expr:
            return None
        lit = self._value_to_literal(self.pe.by_expr[e], e.loc)
        if lit is None:
            return None
        # No-op: the literal we'd substitute is already at this site.
        if type(e) is type(lit) and e.is_equiv(lit):
            return None
        is_ctx_fold = isinstance(lit, ForeignVal) and isinstance(lit.val, Context)
        if is_ctx_fold:
            if not self.enable_context:
                return None
        elif not self.enable_op:
            return None
        self.changed = True
        return lit

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
        func, _ = ConstFold.apply_with_status(
            func,
            def_use=def_use,
            partial_eval=partial_eval,
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
        inst = _ConstFoldInstance(func, partial_eval, enable_context, enable_op)
        return inst.apply(), inst.changed
