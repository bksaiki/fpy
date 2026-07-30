"""
Machinery shared by the *iterable-elimination* transforms —
:class:`fpy2.transform.ZipElim` and :class:`fpy2.transform.EnumerateElim`.

Both rewrite iteration over a *derived* sequence (``zip(...)``,
``enumerate(...)``) into an indexed traversal of the underlying source
vectors, so no backend has to materialize the intermediate list of tuples.
They differ only in which pattern they match and which sources they index;
everything below is common to both.

Two shapes of rewrite show up in each transform, and they have different
capabilities:

*Statement (for-loop) path.*  There is a statement slot before the loop, so
each source can be bound to a fresh ``_srcK`` temporary — evaluated exactly
once, and immune to the loop body rebinding the name it came from.  A binding
slot of any arity lowers to a destructuring assignment ``(a, b) = _srcK[_i]``,
which the backends already handle.  :class:`Ctx` is the accumulator that lets
a statement visitor emit that preamble.

*Expression (comprehension) path.*  A comprehension has no statement slot, so
every source is instead *inlined* into the element expression and re-evaluated
once per iteration.  That forces two restrictions: a source must be an
:func:`is_access_path` (pure and O(1), so re-evaluation is neither observable
nor asymptotically costly), and a binding slot that destructures is reached by
an ``fst``/``snd`` chain — pair-only accessors, hence
:func:`comp_binding_is_pairs`.  :func:`destructure_subst` builds the
substitution and :class:`SubstNames` applies it scope-aware.

Anything failing a guard is left alone for the backend to materialize.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

from ..ast.fpyast import (
    Expr,
    Fst,
    Integer,
    ListComp,
    ListRef,
    NamedId,
    Snd,
    Stmt,
    TupleBinding,
    UnderscoreId,
    Var,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Id


@dataclasses.dataclass
class Ctx:
    """Block-walk accumulator.  When a statement visitor decides to rewrite a
    statement, it appends the ``_srcK = ...`` preamble assignments to ``stmts``
    and returns the rewritten statement; the block visitor then appends that,
    producing one-to-many statement expansion without a custom statement
    visitor."""
    stmts: list[Stmt]

    @staticmethod
    def default() -> 'Ctx':
        return Ctx(stmts=[])


def is_access_path(e: Expr) -> bool:
    """Whether *e* is safe to inline into a comprehension body (re-evaluated
    once per iteration, since a comp has no preamble to bind it once).

    True for a ``Var`` or a pure, O(1) projection/index chain rooted at one
    (``fst``/``snd``/``arg[i]``): no side effects, same value each time.
    Excludes allocating/expensive args (slices, calls, a nested ``zip``) whose
    re-evaluation would turn O(n) into O(n^2); those are left for the backend
    to materialize.
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


def clone(e: Expr) -> Expr:
    """A structurally fresh copy of *e* (no AST shared between substituted
    occurrences); ``DefaultTransformVisitor`` rebuilds every node."""
    return DefaultTransformVisitor()._visit_expr(e, None)


def index_access(arg: Expr, idx: NamedId) -> Callable[[], Expr]:
    """A thunk building ``arg[idx]`` with a fresh copy of ``arg`` each call.

    *arg* must be an :func:`is_access_path` (safe to re-evaluate per call)."""
    return lambda: ListRef(clone(arg), Var(idx, None), None)


def fst(arg: Expr) -> Expr:
    """Build ``fst(arg)``."""
    return Fst(None, arg, None)


def snd(arg: Expr) -> Expr:
    """Build ``snd(arg)``."""
    return Snd(None, arg, None)


def comp_binding_is_pairs(binding: Id | TupleBinding) -> bool:
    """Whether the comprehension path can lower the binding *slot* *binding*.

    A slot's own value is reached by direct indexing (``srcK[_i]``), but a slot
    that *destructures* has its leaves reached by an ``fst``/``snd`` chain (see
    :func:`destructure_subst`), and those accessors are pair-only.  So a
    ``TupleBinding`` slot — and every binding nested within it — must have
    arity 2.  A plain name or an underscore needs no accessor at all.

    The enclosing target is not a slot and is exempt: its elements *are* the
    slots, each reached by its own ``srcK[_i]``.
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
            # The comp path only reaches a nested slot that is a pair
            # (guaranteed by `comp_binding_is_pairs`); `fst`/`snd` are its two
            # projections.
            assert len(binding.elts) == 2, 'comp-path nested slot must be a pair'
            head, tail = binding.elts
            destructure_subst(head, lambda: fst(make_access()), subst)
            destructure_subst(tail, lambda: snd(make_access()), subst)
        case _:
            raise RuntimeError(f'unexpected binding element: {binding!r}')


class SubstNames(DefaultTransformVisitor):
    """Replace every :class:`Var` reference to a name in *subst*
    with the corresponding expression.  Scope-aware: comprehension
    targets that shadow a substituted name disable the substitution
    inside that comprehension's ``elt`` (the inner uses bind to the
    shadowing iteration variable, not to the outer one).
    """

    def __init__(self, subst: dict[NamedId, Expr]):
        super().__init__()
        # Active substitutions; `_visit_list_comp` shadows/restores entries
        # around a nested comp that rebinds a substituted name.
        self._subst = dict(subst)

    def _visit_var(self, e: Var, ctx: Any):
        # Substitution targets are ``NamedId``s, keyed by structural equality.
        replacement = self._subst.get(e.name)
        if replacement is not None:
            return replacement
        return super()._visit_var(e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        # A target NamedId inside this comp shadows any outer
        # substitution for the same name.  Save the shadowed entries,
        # disable them, recurse, then restore.
        shadowed: dict[NamedId, Expr] = {}
        for target in e.targets:
            for name in binding_names(target):
                if name in self._subst:
                    shadowed[name] = self._subst.pop(name)
        try:
            return super()._visit_list_comp(e, ctx)
        finally:
            self._subst.update(shadowed)


def binding_names(target: Id | TupleBinding) -> list[NamedId]:
    """Flatten a binding into the named identifiers it binds.  Underscore
    slots and nested bindings contribute zero or more names."""
    match target:
        case NamedId():
            return [target]
        case UnderscoreId():
            return []
        case TupleBinding():
            out: list[NamedId] = []
            for elt in target.elts:
                out.extend(binding_names(elt))
            return out
        case _:
            return []
