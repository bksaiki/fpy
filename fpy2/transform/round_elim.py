"""
Eliminate unnecessary rounding from expressions whose unrounded
format is contained in the active rounding context.

For a rounded arithmetic op ``f(x_1, ..., x_n)`` (``Add``, ``Sub``,
``Mul``, ``Abs``, ``Neg``) under active context ``C``, the runtime
semantics insert an implicit ``round_C`` around the result.  When
format inference proves ``format(f(...)) ⊆ C.F``, that round is the
identity — the value is unchanged whether or not it fires.  This
transform makes that fact structural by hoisting the operation into
a ``with fp.REAL:`` preamble, binding it to a fresh name, and
threading the name back into the original expression site.

For explicit ``Round`` / ``RoundExact`` / ``Cast`` nodes, the same
notion produces a different rewrite: when the argument's bound
already fits in the target context, the whole node collapses to
its argument (the round was a no-op at runtime).

Rewrite shape:

.. code-block:: python

   # Before
   with fp.FP64:
       a = (1.0 + 2.0) * 3.0    # whole RHS fits FP64 exactly

   # After
   with fp.FP64:
       with fp.REAL:
           _t0 = (1.0 + 2.0) * 3.0
       a = _t0

Greedy at the *outermost* fitting subtree: if a parent op is
eliminable, the whole subtree hoists as one unit (children are not
separately considered).  Only when the parent isn't eliminable do
we recurse and consider its children individually.

Soundness pitfall the implementation guards against:

If the hoisted subtree contains an explicit ``Round`` / ``RoundExact``
/ ``Cast`` node whose round is *not* itself the identity, moving the
subtree under ``with fp.REAL:`` silently turns that round into a
no-op — changing the value of the program.  The transform therefore
hoists only when every explicit round node inside the subtree is
also eliminable.  In practice, eliminable round nodes are collapsed
to their argument *at their own site* by the same pass, so a hoisted
subtree never contains an explicit round node — the check is the
belt-and-suspenders that documents the invariant.

Run after :class:`fpy2.transform.Monomorphize` (so format inference
resolves to concrete contexts) and before any pass that depends on
storage selection — those see strictly tighter formats for the
hoisted operations.
"""

import dataclasses
import operator

from typing import Any, Callable

from ..analysis import (
    ContextUse, ContextUseAnalysis, ContextUseSite, DefineUse,
    DefineUseAnalysis, SyntaxCheck,
)
from ..analysis.format_infer import (
    FormatAnalysis, FormatInfer, exact_binop, exact_unop, round_is_identity,
)
from ..ast.fpyast import (
    Abs, Add, Assign, Cast, ContextStmt, Expr, ForeignVal, FuncDef, IfExpr,
    ListComp, Mul, Neg, Round, RoundExact, Stmt, StmtBlock, Sub, UnaryOp,
    UnderscoreId, Var,
)
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..number import REAL
from ..number.context.context import Context
from ..utils import Gensym


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.  Preamble ``with fp.REAL: _tN = e``
    statements are appended here when the visitor decides to hoist
    a subtree out of an enclosing statement's RHS expression.  The
    enclosing statement (with its hoisted subtree replaced by a
    ``Var`` reference) is then appended after them by
    :meth:`_RoundElimInstance._visit_block`."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> '_Ctx':
        return _Ctx(stmts=[])


# Operator dispatch for the arithmetic ops we handle.  The unrounded
# value-set of an op is computed by `exact_binop` / `exact_unop` on
# the children's stored (post-round) bounds.
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    Add: operator.add,
    Sub: operator.sub,
    Mul: operator.mul,
}

_UNOPS: dict[type, Callable[[Any], Any]] = {
    Abs: abs,
    Neg: operator.neg,
}


class _RoundElimInstance(DefaultTransformVisitor):
    """Drives the rewrite.  Single-use — one instance per
    :meth:`RoundElim.apply` call."""

    func: FuncDef
    ctx_use: ContextUseAnalysis
    format_info: FormatAnalysis
    gensym: Gensym
    outer_ctx: Context | None

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        ctx_use: ContextUseAnalysis,
        format_info: FormatAnalysis,
    ):
        super().__init__()
        self.func = func
        self.ctx_use = ctx_use
        self.format_info = format_info
        self.gensym = Gensym(reserved=def_use.names())
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
        """Concrete rounding context active at *e*, with symbolic
        scopes substituted via :attr:`outer_ctx` when the caller
        pinned one.  Returns ``None`` for scopes that remain
        symbolic — the eliminability decision treats those as
        ineligible (we can't claim the round is identity without
        a concrete target).

        *e* must be a :data:`ContextUseSite` (op-typed expression);
        only op nodes have context scopes registered."""
        try:
            scope = self.ctx_use.find_scope_from_use(e)
        except KeyError:
            return None
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.outer_ctx

    def _unrounded_format(self, e: Expr):
        """Return the unrounded value-set ``F`` for a rounded op, or
        ``None`` for ops we don't handle.  Pulls the children's
        stored (post-round) bounds from :attr:`format_info.by_expr`
        and applies the corresponding ``exact_binop`` / ``exact_unop``
        primitive.  For ``Round`` / ``RoundExact`` / ``Cast``, the
        unrounded value *is* the argument."""
        match e:
            case Add() | Sub() | Mul():
                binop = _BINOPS[type(e)]
                lhs = self.format_info.by_expr.get(e.first)
                rhs = self.format_info.by_expr.get(e.second)
                return exact_binop(lhs, rhs, binop)
            case Abs() | Neg():
                unop = _UNOPS[type(e)]
                arg = self.format_info.by_expr.get(e.arg)
                return exact_unop(arg, unop)
            case Round() | RoundExact() | Cast():
                return self.format_info.by_expr.get(e.arg)
            case _:
                return None

    def _is_eliminable(self, e: Expr) -> bool:
        """True when the implicit round on *e* is the identity under
        its active scope — meaning the round produces no value
        change and can be skipped.

        Only meaningful for rounded ops (arithmetic + explicit
        ``Round`` / ``RoundExact`` / ``Cast``); other expressions
        don't carry a context-driven round, so the question is
        ill-posed and the answer is ``False``."""
        if not isinstance(
            e, (Add, Sub, Mul, Abs, Neg, Round, RoundExact, Cast),
        ):
            return False
        ctx = self._resolved_ctx(e)
        if ctx is None or ctx is REAL:
            # No round to eliminate (REAL is the trivial identity)
            # or unresolvable symbolic scope.  Either way: skip.
            return False
        return round_is_identity(self._unrounded_format(e), ctx)

    def _safe_to_hoist(self, e: Expr) -> bool:
        """Whether *e*'s subtree can be moved under ``with fp.REAL:``
        without changing semantics.  All explicit
        ``Round`` / ``RoundExact`` / ``Cast`` nodes in the subtree
        must already be eliminable at their original scope — moving
        them under REAL would silently drop a real rounding step
        otherwise.  In practice this is a redundant check because
        eliminable round nodes are collapsed *at their own site* by
        the same pass before any enclosing arithmetic op decides to
        hoist; non-eliminable ones remain and trip the check.
        """
        finder = _NonEliminableRoundFinder(self._is_eliminable)
        finder.scan(e)
        return not finder.found

    # ------------------------------------------------------------------
    # Block walk — the ``_Ctx``-accumulator pattern from ``ZipElim``.

    def _visit_block(
        self, block: StmtBlock, ctx: Any,
    ) -> tuple[StmtBlock, Any]:
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # Expression rewriting
    #
    # The decision logic lives in ``_visit_expr``.  At each node:
    #
    #  - Eliminable ``Round`` / ``RoundExact`` / ``Cast`` → replace
    #    with the (recursively-rewritten) argument.  No statement-
    #    level hoist needed.
    #  - Eliminable arithmetic op AND every inner round-node is also
    #    eliminable AND we're at a statement-level expression
    #    position (``ctx`` is a ``_Ctx``) → hoist the whole subtree
    #    into a ``with fp.REAL: _tN = e`` preamble, return ``Var(_tN)``.
    #  - Anything else → recurse via the base visitor (children may
    #    still be individually eliminable).

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        # Round / RoundExact / Cast collapse: works regardless of
        # ctx since it's a pure node-level rewrite (no preamble).
        if (
            isinstance(e, (Round, RoundExact, Cast))
            and self._is_eliminable(e)
        ):
            return self._visit_expr(e.arg, ctx)
        # Arithmetic-op hoist: only at statement-level positions
        # where ``ctx`` carries a preamble buffer.  Comprehension
        # element expressions and conditional branches pass a
        # different sentinel (``None``), suppressing hoists where
        # statement-level preambles wouldn't be sound.
        if (
            isinstance(ctx, _Ctx)
            and isinstance(e, (Add, Sub, Mul, Abs, Neg))
            and self._is_eliminable(e)
            and self._safe_to_hoist(e)
        ):
            return self._hoist(e, ctx)
        return super()._visit_expr(e, ctx)

    def _hoist(self, e: Expr, ctx: _Ctx) -> Expr:
        """Mint ``_tN``, emit ``with fp.REAL: _tN = e`` into *ctx*'s
        preamble buffer, return ``Var(_tN)`` for the original
        expression site.

        The hoisted subtree *e* is not separately rewritten — it
        now lives under REAL, where no round-elim applies (REAL is
        the trivial identity, so :meth:`_is_eliminable` returns
        False for every op inside).  Idempotence falls out of
        that: a re-run of the transform sees only ``Var`` references
        at the previously-hoisted sites and nothing to hoist inside
        the REAL blocks.
        """
        name = self.gensym.fresh('_t')
        assign = Assign(name, None, e, None)
        block = StmtBlock([assign])
        wrapped = ContextStmt(
            UnderscoreId(), ForeignVal(REAL, None), block, None,
        )
        ctx.stmts.append(wrapped)
        return Var(name, None)

    # ------------------------------------------------------------------
    # Sentinel-ctx propagation for nested expression positions where
    # statement-level hoisting would be unsound.

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> ListComp:
        # ``[elt for t in iter]``: the elt sees the loop targets,
        # and successive iterables in multi-stage comps reference
        # earlier targets.  Neither can be hoisted to the enclosing
        # block — pass ``None`` to disable hoisting inside the comp.
        # Round-node collapse still works (it doesn't need a
        # statement-level position).
        targets = [self._visit_binding(t, ctx) for t in e.targets]
        iterables = [self._visit_expr(i, None) for i in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # ``cond ? ift : iff``: the cond is evaluated unconditionally
        # so it can carry through ``ctx``.  The branches are
        # conditional; hoisting either of them unconditionally would
        # change semantics under contexts that can raise.  Disable
        # branch hoisting via ``None``.
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)


class _NonEliminableRoundFinder(DefaultVisitor):
    """Scan a subtree for an explicit ``Round`` / ``RoundExact`` /
    ``Cast`` node whose round is *not* the identity at its original
    scope.  Early-exits via the :attr:`found` flag — the scan stops
    descending once any such node is hit."""

    found: bool

    def __init__(self, is_eliminable: Callable[[Expr], bool]):
        super().__init__()
        self.is_eliminable = is_eliminable
        self.found = False

    def scan(self, e: Expr) -> None:
        self._visit_expr(e, None)

    def _visit_expr(self, e: Expr, ctx: Any) -> None:
        if self.found:
            return
        super()._visit_expr(e, ctx)

    def _visit_round(self, e: Round | RoundExact, ctx: Any) -> None:
        if not self.is_eliminable(e):
            self.found = True
            return
        super()._visit_round(e, ctx)

    def _visit_unaryop(self, e: UnaryOp, ctx: Any) -> None:
        if isinstance(e, Cast) and not self.is_eliminable(e):
            self.found = True
            return
        super()._visit_unaryop(e, ctx)


class RoundElim:
    """Rewrite expressions whose implicit rounding to the active
    context is provably an identity.  Arithmetic ops
    (``Add``/``Sub``/``Mul``/``Abs``/``Neg``) hoist into
    ``with fp.REAL:`` preambles; explicit ``Round`` / ``RoundExact``
    / ``Cast`` nodes collapse to their argument.

    See the module docstring for the rewrite shape, the
    greedy-outermost policy, and the soundness invariants.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a
        new ``FuncDef``; the input is not mutated.

        Runs :class:`DefineUse`, :class:`ContextUse`, and
        :class:`FormatInfer` internally — these are pre-analyses the
        rewrite needs to decide eliminability per expression.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")
        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        format_info = FormatInfer.analyze(
            func, def_use=def_use, ctx_use=ctx_use,
        )
        out = _RoundElimInstance(
            func, def_use, ctx_use, format_info,
        ).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
