"""
Machinery shared by the *iterable-elimination* transforms —
:class:`fpy2.transform.ZipElim` and :class:`fpy2.transform.EnumerateElim`.
Both rewrite iteration over a *derived* sequence (``zip(...)``,
``enumerate(...)``) into an indexed traversal of the underlying source vectors,
differing only in which pattern they match and which sources they index.

Each has two paths, with different capabilities:

*Statement (for-loop) path.*  A statement slot exists before the loop, so each
source binds to a fresh ``_srcK`` temp — evaluated once, and immune to the body
rebinding the name it came from.  A binding slot of any arity lowers to a
destructuring assign the backends already handle.  :class:`Ctx` is what lets a
statement visitor emit that preamble.

*Expression (comprehension) path.*  No statement slot, so sources are *inlined*
into the element expression and re-evaluated per iteration.  Hence two
restrictions: a source must be an :func:`is_access_path`, and a destructuring
slot is reached by a pair-only ``fst``/``snd`` chain — see
:func:`comp_binding_is_pairs`.  :func:`destructure_subst` builds the
substitution, :class:`SubstNames` applies it scope-aware.

Anything failing a guard is left alone for the backend to materialize.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

from ..analysis import DefineUseAnalysis
from ..ast.fpyast import (
    Assign,
    Expr,
    ForStmt,
    Fst,
    FuncDef,
    Integer,
    Len,
    ListComp,
    ListRef,
    NamedId,
    Range1,
    Snd,
    Stmt,
    StmtBlock,
    TupleBinding,
    TupleExpr,
    UnderscoreId,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym, Id
from .utils import clone


@dataclasses.dataclass
class Ctx:
    """Block-walk accumulator: a rewriting statement visitor appends its
    ``_srcK = ...`` preamble here and returns the rewritten statement, which
    the block visitor then appends — one-to-many statement expansion without a
    custom statement visitor."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> 'Ctx':
        return Ctx(stmts=[])


# A binding slot: a name, a discard, or a nested destructure.
Slot = Id | TupleBinding


@dataclasses.dataclass
class Plan:
    """Which sources to index, and which bindings read them.

    ``args[0]`` also supplies the loop bound.  With ``tupled``, one slot reads
    the whole element — rebuilt as ``(args[0][i], ..., args[n][i])``; without
    it, each slot reads its own ``args[k][i]``.
    """
    args: list[Expr]
    slots: list[Slot]
    tupled: bool

    def __post_init__(self) -> None:
        assert self.args, 'need a source for the bound'
        assert len(self.slots) == (1 if self.tupled else len(self.args))


def plan_for_zip(args: list[Expr], slot: Slot) -> Plan | None:
    """How *slot* reads one element of ``zip(*args)`` while indexing *args*
    directly, or ``None`` if it cannot.

    A slot destructuring the tuple positionally reads one source each; a slot
    bound to the whole tuple has it rebuilt per iteration.  ``None`` means the
    slot's arity disagrees with the ``zip``'s — an already ill-typed program,
    where leaving the ``zip`` in place keeps its diagnostic rather than trading
    it for an arity-mismatched destructure.
    """
    if not args:
        # ``zip()`` has no source to take the bound from.
        return None
    if (
        isinstance(slot, TupleBinding)
        and len(slot.elts) == len(args)
        and all(
            isinstance(e, (NamedId, UnderscoreId, TupleBinding))
            for e in slot.elts
        )
    ):
        return Plan(list(args), list(slot.elts), tupled=False)
    if isinstance(slot, (NamedId, UnderscoreId)):
        return Plan(list(args), [slot], tupled=True)
    return None


def is_access_path(e: Expr) -> bool:
    """Whether *e* is safe to inline into a comprehension body, which
    re-evaluates it once per iteration.

    True for a ``Var`` or a pure, O(1) projection/index chain rooted at one: no
    side effects, same value each time.  Excludes allocating arguments (slices,
    calls, a nested ``zip``) whose re-evaluation would turn O(n) into O(n^2);
    those are left for the backend to materialize.
    """
    match e:
        case Var():
            return True
        case Fst() | Snd():
            return is_access_path(e.arg)
        case ListRef():
            return is_access_path(e.value) and is_access_path(e.index)
        case Integer():                       # a constant index in ``arg[i]``
            return True
        case _:
            return False


def index_access(arg: Expr, idx: NamedId) -> Callable[[], Expr]:
    """A thunk building ``arg[idx]`` with a fresh copy of ``arg`` each call.

    *arg* must be an :func:`is_access_path` (safe to re-evaluate per call)."""
    return lambda: ListRef(clone(arg), Var(idx, None), None)


def comp_binding_is_pairs(binding: Id | TupleBinding) -> bool:
    """Whether the comprehension path can lower the binding slot *binding*.

    A slot's own value is reached by direct indexing (``srcK[_i]``), but a slot
    that *destructures* has its leaves reached by a pair-only ``fst``/``snd``
    chain (see :func:`destructure_subst`), so it — and everything nested within
    it — must have arity 2.  A name or underscore needs no accessor at all.

    The enclosing target is exempt: its elements *are* the slots, each reached
    by its own ``srcK[_i]``.
    """
    match binding:
        case TupleBinding():
            return (
                len(binding.elts) == 2
                and all(comp_binding_is_pairs(e) for e in binding.elts)
            )
        case _:
            return True


def destructure_subst(
    binding: Id | TupleBinding,
    make_access: Callable[[], Expr],
    subst: dict[NamedId, Expr],
) -> None:
    """Map every :class:`NamedId` in *binding* to an accessor expression.

    *make_access* builds a *fresh* expression for the value bound to
    *binding* (the per-iteration source element).  It is invoked once per
    leaf, so no AST node is shared between substitutions.
    """
    match binding:
        case NamedId():
            subst[binding] = make_access()
        case UnderscoreId():
            pass
        case TupleBinding():
            # Guaranteed a pair by `comp_binding_is_pairs`; `fst`/`snd` are its
            # two projections.
            assert len(binding.elts) == 2, 'comp-path nested slot must be a pair'
            head, tail = binding.elts
            destructure_subst(head, lambda: Fst(None, make_access(), None), subst)
            destructure_subst(tail, lambda: Snd(None, make_access(), None), subst)
        case _:
            raise RuntimeError(f'unexpected binding element: {binding!r}')


class SubstNames(DefaultTransformVisitor):
    """Replace every :class:`Var` reference to a name in *subst* with the
    corresponding expression.  Scope-aware: a comprehension target that shadows
    a substituted name disables the substitution inside that comprehension, so
    the inner uses bind to the shadowing iteration variable.
    """

    _subst: dict[NamedId, Expr]

    def __init__(self, subst: dict[NamedId, Expr]) -> None:
        super().__init__()
        self._subst = dict(subst)

    def _visit_var(self, e: Var, ctx: Any) -> Expr:
        replacement = self._subst.get(e.name)
        if replacement is not None:
            return replacement
        return super()._visit_var(e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> Expr:
        # Disable any substitution this comp's targets shadow, then restore.
        bound = TupleBinding(e.targets, None).names()
        shadowed = {n: self._subst.pop(n) for n in bound if n in self._subst}
        try:
            return super()._visit_list_comp(e, ctx)
        finally:
            self._subst.update(shadowed)


class IterElimInstance(DefaultTransformVisitor):
    """Single-use visitor driving an iterable elimination; a subclass says
    which pattern it rewrites (`_match`)."""

    func: FuncDef
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis) -> None:
        super().__init__()
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _match(self, target: Slot, iterable: Expr) -> tuple[Id, Plan] | None:
        """The index slot and plan for iterating *iterable* into *target*, or
        ``None`` when this transform does not apply."""
        raise NotImplementedError

    def _index_name(self, idx: Id) -> NamedId:
        """The counter standing in for the index slot: a named slot *is* the
        counter, a discarded one gets a fresh name nothing reads."""
        if isinstance(idx, NamedId):
            return idx
        return self.gensym.fresh('_i')

    # ------------------------------------------------------------------
    # Block walk with stmt → stmts expansion

    def _visit_block(self, block: StmtBlock, ctx: Any) -> tuple[StmtBlock, Any]:
        # A local Ctx per block: preambles belong to the block introducing them.
        block_ctx = Ctx.default()
        for stmt in block.stmts:
            new_stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(new_stmt)
        return StmtBlock(block_ctx.stmts), ctx

    # ------------------------------------------------------------------
    # For loops

    def _visit_for(self, stmt: ForStmt, ctx: Ctx) -> tuple[Stmt, Ctx]:
        matched = self._match(stmt.target, stmt.iterable)
        if matched is None:
            return super()._visit_for(stmt, ctx)
        # Recursively rewrite the body first, in case it contains nested
        # patterns.
        body, _ = self._visit_block(stmt.body, ctx)
        return self._rewrite_for(stmt, *matched, body, ctx), ctx

    def _rewrite_for(
        self, stmt: ForStmt, idx_slot: Id, plan: Plan, new_body: StmtBlock, ctx: Ctx,
    ) -> ForStmt:
        # Bind each source to a fresh `_srcK` before the loop — a discarded
        # slot's source too, so any side-effect in it still fires once.
        src_names: list[NamedId] = []
        for arg in plan.args:
            src = self.gensym.fresh('_src')
            ctx.stmts.append(Assign(src, None, arg, None))
            src_names.append(src)

        idx = self._index_name(idx_slot)
        # A tupled plan feeds every source into one slot; otherwise each slot
        # reads its own.  `plan.tupled` is the discriminator, not the source
        # count: `zip(xs)` is tupled over one source and still yields 1-tuples.
        groups = [src_names] if plan.tupled else [[s] for s in src_names]
        per_iter: list[Stmt] = []
        for slot, srcs in zip(plan.slots, groups):
            if isinstance(slot, UnderscoreId):
                continue
            refs = [ListRef(Var(s, None), Var(idx, None), None) for s in srcs]
            # A nested slot becomes `(a, b) = src[i]` — the backends already
            # destructure, so a statement context needs no `fst`/`snd`.
            value = TupleExpr(refs, None) if plan.tupled else refs[0]
            per_iter.append(Assign(slot, None, value, None))

        return ForStmt(
            idx,
            Range1(None, Len(None, Var(src_names[0], None), None), None),
            StmtBlock(per_iter + list(new_body.stmts)),
            stmt.loc,
        )

    # ------------------------------------------------------------------
    # List comprehensions

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> ListComp:
        new_targets: list[Id | TupleBinding] = []
        new_iterables: list[Expr] = []
        subst: dict[NamedId, Expr] = {}

        for target, iterable in zip(e.targets, e.iterables):
            new_iter = self._visit_expr(iterable, ctx)
            # A later stage's iterable may reference an earlier stage's target
            # (`[... for a, b in zip(xs, ys) for c in a]`), whose name no
            # longer exists once that stage is rewritten.
            if subst:
                new_iter = SubstNames(subst)._visit_expr(new_iter, ctx)
            rewritten = self._rewrite_comp_stage(target, new_iter, subst)
            if rewritten is None:
                new_targets.append(self._visit_binding(target, ctx))
                new_iterables.append(new_iter)
            else:
                new_target, new_iterable = rewritten
                new_targets.append(new_target)
                new_iterables.append(new_iterable)

        # Walk `elt` normally first, so nested patterns inside it are
        # rewritten too, then substitute.
        new_elt = self._visit_expr(e.elt, ctx)
        if subst:
            new_elt = SubstNames(subst)._visit_expr(new_elt, ctx)
        return ListComp(new_targets, new_iterables, new_elt, e.loc)

    def _rewrite_comp_stage(
        self,
        target: Id | TupleBinding,
        iterable: Expr,
        subst: dict[NamedId, Expr],
    ) -> tuple[Id, Expr] | None:
        """The rewritten ``(target, iterable)`` for one comprehension stage,
        extending *subst* with the accessors its slots need; ``None`` to leave
        the stage as it is."""
        matched = self._match(target, iterable)
        if matched is None:
            return None
        idx_slot, plan = matched
        # Sources are inlined and re-evaluated per iteration (a comp has no
        # preamble to bind them once), so each must be pure and O(1).
        if not all(is_access_path(a) for a in plan.args):
            return None

        idx = self._index_name(idx_slot)
        if plan.tupled:
            # A whole-element slot is a name or a discard, so no `fst`/`snd`
            # chain is involved and `comp_binding_is_pairs` has nothing to say.
            slot = plan.slots[0]
            if isinstance(slot, NamedId):
                subst[slot] = TupleExpr(
                    [index_access(arg, idx)() for arg in plan.args], None,
                )
        else:
            # `fst`/`snd` are pair-only, so a destructuring slot of arity != 2
            # can't be reached; leave the stage rather than emit an ill-typed
            # `snd`-of-non-pair.
            if not all(comp_binding_is_pairs(slot) for slot in plan.slots):
                return None
            for slot, arg in zip(plan.slots, plan.args):
                destructure_subst(slot, index_access(arg, idx), subst)
        return idx, Range1(None, Len(None, clone(plan.args[0]), None), None)
