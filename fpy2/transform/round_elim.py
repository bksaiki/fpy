"""
Eliminate unnecessary rounding from expressions whose unrounded
format is contained in the active rounding context.

For a rounded arithmetic op ``g(e_1, ..., e_n)`` (``Add``, ``Sub``,
``Mul``, ``Abs``, ``Neg``) under active context ``C``, the runtime
semantics insert an implicit ``round_C`` around the result.  When
format inference proves ``format(g(...)) ⊆ C.F``, that round is the
identity — the value is unchanged whether or not it fires.  This
transform makes that fact structural by computing ``g`` under
``with fp.REAL:`` and binding the result to a fresh name that
replaces the original expression site.

For explicit ``Round`` / ``RoundExact`` / ``Cast`` nodes, the same
notion produces a different rewrite: when the argument's bound
already fits in the target context, the whole node collapses to
its argument (the round was a no-op at runtime).

Rewrite shape — per-op hoisting with unconditional operand binding:

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

Each operand is bound to a temp under the *current* context
before being passed into the REAL block.  The bind preserves
every round inside the operand at its original scope — explicit
``Round`` / ``Cast`` nodes, implicit rounds on nested arithmetic
ops, eliminable or non-eliminable, all fire as originally
written.  The REAL block then consumes only ``Var`` references,
so it can never accidentally re-evaluate a subexpression at the
wrong context.

Operands that are already ``Var`` references are passed through
without binding — a copy ``_t = v`` has no rounds to preserve
and just adds a redundant alias.  This is the only case-split;
every other operand kind is bound unconditionally for safety.

Trade-off: the unconditional bind produces redundant copies for
literals and already-clean subtrees.  ``CopyPropagate`` and
``DeadCodeEliminate`` (both available in :mod:`fpy2.transform`)
clean these up downstream — the local rewrite stays simple at
the cost of slack in the intermediate AST.

Suppressed positions: hoisting needs a statement-level preamble
slot, so it is disabled inside ``ListComp`` element / iterable
expressions and inside ``IfExpr`` branches (the latter would
evaluate operands unconditionally, changing exception semantics
under contexts that can raise).  ``Round`` / ``Cast`` collapse
still applies inside these positions — it's a pure node-level
rewrite that needs no preamble slot.

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
    AbstractFormat, AbstractableFormat, FormatAnalysis, FormatInfer, SetFormat,
    exact_binop, exact_unop, round_is_identity,
)
from ..ast.fpyast import (
    Abs, Add, Assign, Cast, ContextStmt, Expr, ForeignVal, FuncDef, IfExpr,
    ListComp, Mul, Neg, Round, RoundExact, Stmt, StmtBlock, Sub,
    UnderscoreId, Var,
)
from ..ast.visitor import DefaultTransformVisitor
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
                # The unrounded "value" of an explicit round node is
                # the argument itself.  Stored bounds may be a
                # ``Format`` (e.g., ``IEEEFormat`` when the arg's
                # post-round bound widened to the scope's format);
                # lift to ``AbstractFormat`` so the result fits
                # ``round_is_identity``'s expected input shape.
                arg_fmt = self.format_info.by_expr.get(e.arg)
                if isinstance(arg_fmt, SetFormat):
                    return arg_fmt
                if isinstance(arg_fmt, AbstractableFormat):
                    return AbstractFormat.from_format(arg_fmt)
                return None
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
    #    with the (recursively-rewritten) argument.  Pure node-level
    #    rewrite, no preamble required, works at any expression
    #    position (including inside ListComp / IfExpr branches).
    #  - Eliminable arithmetic op AND we're at a statement-level
    #    expression position (``ctx`` is a ``_Ctx``) → per-op hoist
    #    (see :meth:`_hoist`).
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
        ):
            return self._hoist(e, ctx)
        return super()._visit_expr(e, ctx)

    def _operands(self, e: Expr) -> list[Expr]:
        """Return *e*'s direct operands (left-to-right) for the
        rounded ops the hoist handles.  Used to enumerate the
        positions :meth:`_hoist` may need to bind to temps."""
        match e:
            case Add() | Sub() | Mul():
                return [e.first, e.second]
            case Abs() | Neg():
                return [e.arg]
            case _:
                raise RuntimeError(
                    f'_operands called on non-rounded-arithmetic op: {e!r}'
                )

    def _rebuild(self, e: Expr, operands: list[Expr]) -> Expr:
        """Reconstruct an arithmetic op *e* with new *operands*.
        Mirrors the inverse of :meth:`_operands`."""
        match e:
            case Add() | Sub() | Mul():
                return type(e)(operands[0], operands[1], e.loc)
            case Abs() | Neg():
                return type(e)(operands[0], e.loc)
            case _:
                raise RuntimeError(
                    f'_rebuild called on non-rounded-arithmetic op: {e!r}'
                )

    def _hoist(self, e: Expr, ctx: _Ctx) -> Expr:
        """Per-op hoist: compute *e* under ``with fp.REAL:`` and
        return ``Var(_tN)`` for the original expression site.

        Each operand is visited with the same *ctx* so nested
        eliminable ops hoist at their own level — the result is a
        ``Var`` that we can plug directly into our reconstructed
        op.  Operands that survive the visit as non-``Var``
        expressions (literals, non-eliminable ops, ``Round`` nodes
        that weren't collapsed) are bound to a temp under the
        *current* context before being passed into the REAL block.
        The bind preserves every round inside the operand at its
        original scope: explicit ``Round`` / ``Cast`` nodes,
        implicit rounds on non-eliminable arithmetic, all fire as
        originally written.

        The REAL block consumes only ``Var`` references — it can
        never accidentally re-evaluate a subexpression at the wrong
        context.  Idempotence falls out: a second pass sees only
        ``Var``-argumented ops under REAL (and bare assigns under
        the original context), none of which trigger another hoist.

        Trade-off: the unconditional bind produces redundant
        ``_t = literal`` lines and per-op REAL blocks for nested
        eliminations.  :class:`fpy2.transform.CopyPropagate` and
        :class:`fpy2.transform.DeadCodeEliminate` (both in
        :mod:`fpy2.transform`) clean these up downstream — the
        local rewrite stays simple at the cost of slack in the
        intermediate AST.
        """
        new_operands: list[Expr] = []
        for operand in self._operands(e):
            new_operand = self._visit_expr(operand, ctx)
            if isinstance(new_operand, Var):
                # Bind would be a pure copy — no rounds to preserve
                # inside a name lookup.  Use the ``Var`` directly.
                new_operands.append(new_operand)
                continue
            # Bind under the current context.  Anything not a
            # ``Var`` is conservatively assumed to contain a round
            # (literal, non-eliminable op, ``Round`` node, …);
            # binding fires those rounds at their original scope
            # before the REAL block sees the value.
            t_op = self.gensym.fresh('_t')
            ctx.stmts.append(Assign(t_op, None, new_operand, None))
            new_operands.append(Var(t_op, None))

        rebuilt = self._rebuild(e, new_operands)
        result_name = self.gensym.fresh('_t')
        block = StmtBlock([Assign(result_name, None, rebuilt, None)])
        wrapped = ContextStmt(
            UnderscoreId(), ForeignVal(REAL, None), block, None,
        )
        ctx.stmts.append(wrapped)
        return Var(result_name, None)

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
