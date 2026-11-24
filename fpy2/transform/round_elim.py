"""
Eliminate unnecessary rounding from expressions whose unrounded
format is contained in the active rounding context.

An eliminable arithmetic op is computed under ``with fp.REAL:``; an
eliminable ``Round`` / ``Cast`` collapses to its argument.

.. code-block:: python

   # Before
   with fp.FP64:
       a = g(e_1, ..., e_n)     # g's round is identity

   # After
   with fp.FP64:
       t_1 = e_1                # original-context evaluation
       ...
       t_n = e_n                # of each operand
       with fp.REAL:
           _t = g(t_1, ..., t_n)
       a = _t

The recommended cleanup chain is :class:`fpy2.transform.ConstFold` then
:class:`fpy2.transform.CopyPropagate` then
:class:`fpy2.transform.DeadCodeEliminate`.

Run after :class:`fpy2.transform.Monomorphize` (so format inference
resolves to concrete contexts) and before any pass that depends on
storage selection — those see strictly tighter formats for the
hoisted operations.

**Preserves hoistable form**, which the cpp pipeline relies on: it runs between
:class:`fpy2.transform.Hoistable` and :class:`fpy2.transform.ANF`.
"""

import dataclasses
from typing import Any

from ..analysis import (
    ContextUse,
    ContextUseAnalysis,
    ContextUseSite,
    DefineUse,
    DefineUseAnalysis,
    Hoistability,
    SyntaxCheck,
)
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatAnalysis,
    FormatInfer,
    SetFormat,
    round_is_identity,
    unrounded_format,
)
from ..analysis.hoistability import force_names
from ..ast.fpyast import (
    Abs,
    Add,
    Cast,
    Expr,
    FuncDef,
    Mul,
    NamedId,
    Neg,
    Round,
    Stmt,
    StmtBlock,
    Sub,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import REAL
from ..number.context.context import Context
from ..utils import Gensym
from .utils import lift_op, name_forced


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.  Preamble ``with fp.REAL: _tN = e``
    statements are appended here when the visitor decides to hoist
    a subtree out of an enclosing statement's RHS expression.  The
    enclosing statement (with its hoisted subtree replaced by a
    ``Var`` reference) is then appended after them by
    :meth:`_RoundElimInstance._visit_block`."""
    stmts: list[Stmt]
    force: frozenset[Expr] = frozenset()
    """:func:`~fpy2.analysis.hoistability.force_names` for the statement being
    visited"""

    @staticmethod
    def default() -> '_Ctx':
        return _Ctx(stmts=[])



class _RoundElimInstance(DefaultTransformVisitor):
    """Drives the rewrite.  Single-use — one instance per
    :meth:`RoundElim.apply` call."""

    func: FuncDef
    ctx_use: ContextUseAnalysis
    format_info: FormatAnalysis
    gensym: Gensym
    outer_ctx: Context | None
    strict: set[Expr]

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        ctx_use: ContextUseAnalysis,
        format_info: FormatAnalysis,
    ) -> None:
        super().__init__()
        self.func = func
        self.ctx_use = ctx_use
        self.format_info = format_info
        self.gensym = Gensym(reserved=def_use.names())
        self.strict = Hoistability.analyze(func).strict
        # Outer ctx pinning used to resolve symbolic ``with`` scopes
        # — identical to the resolution :class:`FormatAnalysis` does
        # internally.  Programs analyzed standalone may not have a
        # pin, in which case symbolic scopes are unresolvable here
        # and treated as ineligible for elimination.
        if format_info.fn_fmt is None:
            self.outer_ctx = None
        else:
            self.outer_ctx = format_info.fn_fmt.ctx

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Scope resolution + eliminability decision

    def _resolved_ctx(self, e: ContextUseSite) -> Context | None:
        """Concrete rounding context active at *e*, resolving symbolic scopes
        via :attr:`outer_ctx`; ``None`` if still symbolic."""
        scope = self.ctx_use.find_scope_from_use(e)
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.outer_ctx

    def _is_eliminable(self, e: Expr) -> bool:
        """True when the implicit round on *e* is the identity under its scope.

        The unrounded format must also be *strictly tighter* than the scope's:
        under an unbounded scope (e.g. ``fp.INTEGER``) ``round_is_identity`` is
        vacuously true, but hoisting yields no narrowing and can break storage
        selection downstream."""
        if not isinstance(
            e, (Add, Sub, Mul, Abs, Neg, Round, Cast),
        ):
            return False
        ctx = self._resolved_ctx(e)
        if ctx is None or ctx is REAL:
            # No round to eliminate (REAL is the trivial identity)
            # or unresolvable symbolic scope.  Either way: skip.
            return False
        unrounded = unrounded_format(e, self.format_info.by_expr)
        if not round_is_identity(unrounded, ctx):
            return False
        # Strictly-tighter guard.
        if isinstance(unrounded, SetFormat):
            # SetFormat is always a strict refinement over any Format
            # — a finite value-set vs. a continuous range.  Always
            # worth hoisting.
            return True
        # AbstractFormat case: tighter iff strictly contained in the
        # scope's lifted AF.  Both equal → no benefit, skip.
        scope_fmt = ctx.format()
        if not isinstance(scope_fmt, AbstractableFormat):
            # Scope isn't lifted-comparable; conservative skip.
            return False
        scope_af = AbstractFormat.from_format(scope_fmt)
        # ``a < b`` ≡ ``a <= b and a != b`` for AbstractFormat
        # (uses ``__le__`` for the subset check and ``__eq__`` for
        # parameter-shape equality).
        return unrounded <= scope_af and unrounded != scope_af

    # ------------------------------------------------------------------
    # Block walk — the accumulator pattern of ``iter_elim.Ctx``.

    def _visit_block(
        self, block: StmtBlock, ctx: Any,
    ) -> tuple[StmtBlock, Any]:
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            force = frozenset(force_names(stmt, self._hoisted))
            new_stmt, _ = self._visit_statement(stmt, _Ctx(block_ctx.stmts, force))
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # Expression rewriting

    def _hoisted(self, e: Expr) -> bool:
        return (
            e in self.strict
            and isinstance(e, (Add, Sub, Mul, Abs, Neg))
            and self._is_eliminable(e)
        )

    def _fresh(self) -> NamedId:
        return self.gensym.fresh('_t')

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if isinstance(e, (Round, Cast)) and self._is_eliminable(e):
            rebuilt = self._visit_expr(e.arg, ctx)
        elif self._hoisted(e):
            rebuilt = Var(lift_op(e, REAL, ctx.stmts, lambda x: self._visit_expr(x, ctx), self._fresh), e.loc)
        else:
            rebuilt = super()._visit_expr(e, ctx)
        return name_forced(e, rebuilt, ctx.force, ctx.stmts, self._fresh)


class RoundElim:
    """Rewrite expressions whose implicit rounding to the active
    context is provably an identity.  Arithmetic ops
    (``Add``/``Sub``/``Mul``/``Abs``/``Neg``) hoist into
    ``with fp.REAL:`` preambles; explicit ``Round`` / ``Cast``
    nodes collapse to their argument.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a
        new ``FuncDef``; the input is not mutated.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        # No `use_digit_bounds`: eliminability asks whether the operand's
        # format survives the rounding, which the non-relational pass
        # settles on its own.
        format_info = FormatInfer.analyze(
            func, def_use=def_use, ctx_use=ctx_use,
        )
        out = _RoundElimInstance(
            func, def_use, ctx_use, format_info,
        ).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
