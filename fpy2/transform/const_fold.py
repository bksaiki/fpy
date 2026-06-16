"""
Constant folding — thin rewriter over :class:`fpy2.analysis.PartialEval`.

``PartialEval`` is the single source of truth for "what is the static
value, if any, of every expression in the program."  ``ConstFold``
walks the AST and, at each foldable node, queries ``by_expr`` and
substitutes a literal AST when one is available.  No value-tracking
dataflow lives here.
"""

from fractions import Fraction

from ..analysis import DefineUse, DefineUseAnalysis, PartialEval, PartialEvalInfo
from ..analysis.partial_eval import Value
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import Context, Float


class _ConstFoldInstance(DefaultTransformVisitor):
    """ConstFold rewriter.

    For each foldable node kind ``e``, this visitor queries
    ``pe.by_expr[e]`` on the *original* node *before* descending into
    children.  On a hit, it returns a literal AST replacement directly
    (skipping the recursive rebuild for that subtree); on a miss, it
    falls through to the default structural rewrite.

    Substitution policy: every expression whose value is statically
    known is replaceable by a literal — there is no per-node bypass.
    Foldings whose result is a :class:`Context` ride ``enable_context``;
    everything else rides ``enable_op``.  The flag is decided by
    inspecting the *folded* value, not the source AST node — so a
    ``Var`` bound to a ``Context`` (e.g. ``CTX = fp.FP32; with CTX:``)
    is gated as a context fold, and a ``Var`` bound to a numeric is
    gated as an op fold.
    """

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
        """Convert a :data:`Value` from PartialEval back to an AST
        literal.  Returns ``None`` if ``val`` isn't representable as a
        literal in the FPy AST (Python types, functions, modules,
        non-emittable element types in a container, etc.) — the
        caller will leave the original node in place.

        ``bool`` is checked first because ``bool`` is a subclass of
        ``int`` in Python.  Python ``int`` and ``float`` show up when
        a Python-bound free variable is substituted at a ``Var``
        site; they normalize through :class:`Fraction` so they share
        the same AST output shape as folded numeric ops.

        ``tuple`` and ``list`` recurse element-wise: every element
        must itself be literal-emittable, otherwise the container
        falls through to ``None`` (leaving the original AST in
        place).
        """
        if isinstance(val, bool):
            return BoolVal(val, loc)
        if isinstance(val, Float):
            val = val.as_rational()
        elif isinstance(val, (int, float)):
            val = Fraction(val)
        if isinstance(val, Fraction):
            if val.denominator == 1:
                return Integer(int(val), loc)
            # ``fp.rational`` is the surface-level Rational constructor;
            # the parser produces the same AST shape for literal
            # rationals, so downstream consumers don't need to know
            # this node came from a fold.
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
        respecting the ``enable_op`` / ``enable_context`` policy.
        Returns ``None`` when no substitution applies."""
        if e not in self.pe.by_expr:
            return None
        lit = self._value_to_literal(self.pe.by_expr[e], e.loc)
        if lit is None:
            return None
        # Pick the gate by the kind of the folded result.
        is_ctx_fold = isinstance(lit, ForeignVal) and isinstance(lit.val, Context)
        if is_ctx_fold:
            return lit if self.enable_context else None
        return lit if self.enable_op else None

    def _visit_expr(self, e: Expr, ctx) -> Expr:
        """Single chokepoint for substitution: every expression in the
        tree comes through this dispatcher (statements walk their
        expression children via :meth:`_visit_expr`).  We try to fold
        ``e`` here once; on a miss, fall through to the default
        type-dispatched rewrite.

        ``Round`` / ``Cast`` / ``RoundAt`` fold like any other op
        when their result is statically known — the substituted
        literal sits at the target context's format, so the rounding
        intent is preserved by the value rather than by an AST node.
        """
        lit = self._fold(e)
        if lit is not None:
            return lit
        return super()._visit_expr(e, ctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """Constant folding and propagation.

    Substitutes the following with literal AST nodes:

    - **Constant propagation** at ``Var`` and ``Attribute`` sites whose
      value is known statically.
    - **Constant folding** at ``NullaryOp`` / ``UnaryOp`` / ``BinaryOp``
      / ``TernaryOp`` / ``NaryOp`` whose operands are all known.
    - **Context-constructor folding** at ``Call`` sites that build a
      :class:`Context`.

    Static values come from :class:`PartialEval`; pass an existing
    analysis via ``partial_eval=`` to avoid re-running it.

    Excluded:

    - Non-literal-emittable values (types, functions, modules,
      tuples) — left as the original AST.
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
        """Apply constant folding to ``func``.

        Args:
            func: The function to fold.
            def_use: Cached def-use analysis; recomputed if absent.
            partial_eval: Cached partial-eval analysis; recomputed if
                absent.  When recomputed, shares ``def_use`` with this
                call.
            enable_context: When ``False``, suppress folds whose result
                is a :class:`Context`.  Default ``True``.
            enable_op: When ``False``, suppress folds whose result is
                a number or boolean.  Default ``True``.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if partial_eval is None:
            partial_eval = PartialEval.apply(func, def_use=def_use)
        return _ConstFoldInstance(
            func, partial_eval, enable_context, enable_op,
        ).apply()
