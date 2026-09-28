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

For explicit ``Round`` / ``Cast`` nodes, the same
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
literals and already-clean subtrees.  The recommended cleanup
chain is :class:`fpy2.transform.ConstFold` then
:class:`fpy2.transform.CopyPropagate` then
:class:`fpy2.transform.DeadCodeEliminate` — the first inlines
literal-bound temps (``_t = 1`` → uses of ``_t`` become ``1``)
via the value flow from :class:`fpy2.analysis.PartialEval`, the
second collapses any remaining Var→Var copies, and the third
removes the now-unused assigns and any block left with no
rounding to install.  The local rewrite stays simple at the cost
of slack in the intermediate AST.

An operation is hoisted only where it is strict (see
:class:`~fpy2.analysis.Hoistability`), and the operands its statement evaluates
before it are named first.  ``Round`` / ``Cast`` collapse applies anywhere --
it's a pure node-level rewrite that needs no preamble slot.

Run after :class:`fpy2.transform.Monomorphize` (so format inference
resolves to concrete contexts) and before any pass that depends on
storage selection — those see strictly tighter formats for the
hoisted operations.

**Preserves hoistable form**, which the cpp pipeline relies on: it runs between
:class:`fpy2.transform.Hoistable` and :class:`fpy2.transform.ANF`.  Nothing here
can break it — every rewrite
either collapses a node in place or appends statements to a preamble slot, and
neither introduces a ternary, an ``and``/``or`` or a ``while`` condition.  Given
the form, this pass also *gains* by it: a ternary arm is an ``IfStmt`` branch by
the time it arrives, so an operation in it is strict.
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
    Assign,
    Cast,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    Mul,
    Neg,
    Round,
    Stmt,
    StmtBlock,
    Sub,
    UnderscoreId,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import REAL
from ..number.context.context import Context
from ..utils import Gensym
from .utils import name_forced, operands, rebuild


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
    ):
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
        """Concrete rounding context active at *e*, with symbolic
        scopes substituted via :attr:`outer_ctx` when the caller
        pinned one.  Returns ``None`` for scopes that remain
        symbolic — the eliminability decision treats those as
        ineligible (we can't claim the round is identity without
        a concrete target).

        *e* must be a :data:`ContextUseSite` (op-typed expression)
        that's been seen by :class:`ContextUseAnalysis`.  We don't
        catch ``KeyError`` here: a node that should have a scope
        but doesn't indicates a bug elsewhere (the node was
        constructed without going through scope analysis), and
        failing loudly is more useful than silently treating it
        as ineligible."""
        scope = self.ctx_use.find_scope_from_use(e)
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.outer_ctx

    def _is_eliminable(self, e: Expr) -> bool:
        """True when the implicit round on *e* is the identity under
        its active scope AND the unrounded format is *strictly
        tighter* than the scope's format — meaning the round
        produces no value change *and* moving the op to REAL gives
        downstream consumers a more precise bound than the scope
        provides.

        Only meaningful for rounded ops (arithmetic + explicit
        ``Round`` / ``Cast``); other expressions
        don't carry a context-driven round, so the question is
        ill-posed and the answer is ``False``.

        The "strictly tighter" guard is the subtle one: under an
        unbounded scope (e.g. ``fp.INTEGER`` / ``MPFixedContext``),
        ``round_is_identity`` is vacuously true for any value, but
        the unrounded format is *also* unbounded — hoisting yields
        no narrowing and can break downstream consumers whose
        storage selection can't pick a finite type for a saturated
        format (e.g. the cpp emitter's lossless-widening dispatch).
        Skipping these saves both pointless rewrites and the
        cascading storage failures."""
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
    # Block walk — the ``_Ctx``-accumulator pattern from ``ZipElim``.

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
    #
    # The decision logic lives in ``_visit_expr``.  At each node:
    #
    #  - Eliminable ``Round`` / ``Cast`` → replace
    #    with the (recursively-rewritten) argument.  Pure node-level
    #    rewrite, no preamble required, works at any expression
    #    position (including inside ListComp / IfExpr branches).
    #  - Eliminable arithmetic op that is strict → per-op hoist (see
    #    :meth:`_hoist`).
    #  - Anything else → recurse via the base visitor (children may
    #    still be individually eliminable).
    #
    # Whatever it becomes, an operand the prefix rule forces is then named.

    def _hoisted(self, e: Expr) -> bool:
        return (
            e in self.strict
            and isinstance(e, (Add, Sub, Mul, Abs, Neg))
            and self._is_eliminable(e)
        )

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if isinstance(e, (Round, Cast)) and self._is_eliminable(e):
            rebuilt = self._visit_expr(e.arg, ctx)
        elif self._hoisted(e):
            rebuilt = self._hoist(e, ctx)
        else:
            rebuilt = super()._visit_expr(e, ctx)
        fresh = lambda: self.gensym.fresh('_t')
        return name_forced(e, rebuilt, ctx.force, ctx.stmts, fresh)

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
        eliminations.  The recommended cleanup chain —
        :class:`fpy2.transform.ConstFold` then
        :class:`fpy2.transform.CopyPropagate` then
        :class:`fpy2.transform.DeadCodeEliminate` — collapses both,
        the last dropping a block once nothing under it reads the
        context; the local rewrite stays simple at the cost of slack
        in the intermediate AST.
        """
        # Inherit *e*'s location for hoist-introduced nodes so
        # downstream errors that reference the temps point back at
        # the original source position.
        loc = e.loc
        new_operands: list[Expr] = []
        for operand in operands(e):
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
            ctx.stmts.append(Assign(t_op, None, new_operand, loc))
            new_operands.append(Var(t_op, loc))

        rebuilt = rebuild(e, new_operands)
        result_name = self.gensym.fresh('_t')
        block = StmtBlock([Assign(result_name, None, rebuilt, loc)])
        wrapped = ContextStmt(
            UnderscoreId(), ForeignVal(REAL, loc), block, loc,
        )
        ctx.stmts.append(wrapped)
        return Var(result_name, loc)


class RoundElim:
    """Rewrite expressions whose implicit rounding to the active
    context is provably an identity.  Arithmetic ops
    (``Add``/``Sub``/``Mul``/``Abs``/``Neg``) hoist into
    ``with fp.REAL:`` preambles; explicit ``Round`` / ``Cast``
    nodes collapse to their argument.

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
