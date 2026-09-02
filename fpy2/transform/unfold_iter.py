"""
Unfolding ``zip`` and ``enumerate`` into comprehensions.

Both are derived forms, and `derived-semantics.rst` gives each as a
comprehension over ``range(len(...))``.  Stating that trades a node a consumer
must special-case for one `CompToLoop` already lowers totally.

Both rewrites read their argument twice, in ``xs[i]`` and in ``len(xs)``, so an
argument that is not already a name is bound above first: rebuilding it would
make two lists where the program had one.  That needs a statement slot, which
hoistable form guarantees and a comprehension element or a ``while`` condition
does not have.
"""

from ..analysis import DefineUse
from ..ast.fpyast import (
    AssertStmt,
    Assign,
    Compare,
    CompareOp,
    Enumerate,
    Expr,
    FuncDef,
    Len,
    ListComp,
    ListExpr,
    ListRef,
    NamedId,
    Range1,
    Stmt,
    TupleExpr,
    Var,
    Zip,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym
from .cursor import Cursor, EditLog
from .utils import PreambleScoped, check_where

_NO_SLOT = (
    'the iterable has no statement-level position for the bindings the '
    'rewrite emits'
)


class _UnfoldIterInstance(PreambleScoped):
    """Rewrites one derived iterable into the comprehension it stands for.

    A subclass says which node it takes (`_node`) and what the comprehension
    element is (`_elt`); everything else -- binding a non-name argument,
    minting the index, refusing where no statement reaches -- is shared.
    """

    _expr_sited = True

    _node: type
    """the surface node this pass unfolds"""

    temp_id: NamedId
    """the name every temporary this pass mints is a refresh of"""

    def __init__(
        self, func: FuncDef, temp_id: NamedId | None = None, where=None,
    ):
        self.func = func
        self.temp_id = NamedId('t') if temp_id is None else temp_id
        self.gensym = Gensym(reserved=DefineUse.analyze(func).names())
        self.where = where

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # What a subclass supplies

    def _elt(self, args: list[Expr], i: NamedId, loc) -> Expr:
        """The comprehension's element, over the bound arguments and index."""
        raise NotImplementedError

    def _pre(self, args: list[Expr], loc) -> list[Stmt]:
        """Statements the unfolding claims before the comprehension."""
        return []

    # ------------------------------------------------------------------

    def _emit(self, e: Expr, out: list[Stmt]) -> Expr:
        """The comprehension *e* stands for, with its arguments named."""
        loc = e.loc
        if not e.args:   # type: ignore[attr-defined]
            # `zip()` is the empty list.  The backend refuses one for having no
            # element type, which is a diagnostic where reading `args[0]` would
            # be a crash.
            return ListExpr([], loc)
        args: list[Expr] = []
        for arg in e.args:   # type: ignore[attr-defined]
            new = self._visit_expr(arg, out)
            if isinstance(new, Var):
                # a name is read twice for free; anything else would be built
                # twice, and the two lists would be distinct objects
                args.append(new)
                continue
            t = self.gensym.refresh(self.temp_id)
            out.append(Assign(t, None, new, loc))
            args.append(Var(t, loc))

        out.extend(self._pre(args, loc))
        i = self.gensym.refresh(self.temp_id)
        return ListComp(
            [i], [Range1(None, Len(None, args[0], loc), loc)],
            self._elt(args, i, loc), loc,
        )

    # `PreambleScoped` seals every compound statement's sub-expression, which a
    # `while` condition needs and these three do not: each is evaluated exactly
    # once, where the preamble runs, and the `for` iterable is where a derived
    # iterable appears -- sealing it would refuse the site that matters.  A
    # nested block builds its own preamble in `_visit_block` either way, so
    # un-sealing is the base implementation back.  A comprehension's own
    # positions stay sealed.
    _visit_for = DefaultTransformVisitor._visit_for
    _visit_if1 = DefaultTransformVisitor._visit_if1
    _visit_if = DefaultTransformVisitor._visit_if
    _visit_context = DefaultTransformVisitor._visit_context

    def _visit_expr(self, e: Expr, ctx):
        if not isinstance(e, self._node):
            return super()._visit_expr(e, ctx)

        # a refusal is not a site, so it is decided before an index is spent
        if ctx is None:
            self.refused.append((e, _NO_SLOT))
            if self._target_expr is e:
                self.declined.append(_NO_SLOT)
            return super()._visit_expr(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(e, idx):
            return super()._visit_expr(e, ctx)

        self._matched += 1
        if self.listing:
            self.found_exprs.append(e)
            return super()._visit_expr(e, ctx)

        emitted = self._emit(e, ctx)
        self._replaced = True
        return emitted


class _UnfoldZipInstance(_UnfoldIterInstance):
    _node = Zip

    def _pre(self, args: list[Expr], loc) -> list[Stmt]:
        """``len(xsk) == len(xs1)`` for each argument past the first.

        The length is the first iterable's and unequal lengths are undefined,
        so this is a claim about a well-defined program rather than a check
        with a defined failure.  It is also what carries the node's own
        strictness: `ArraySizeInfer` reads an assert as an equality, and
        without it a proven length on one iterable stops reaching the others.
        """
        head = Len(None, args[0], loc)
        return [
            AssertStmt(
                Compare([CompareOp.EQ], [Len(None, a, loc), head], loc),
                None, loc,
            )
            for a in args[1:]
        ]

    def _elt(self, args: list[Expr], i: NamedId, loc) -> Expr:
        return TupleExpr([ListRef(a, Var(i, loc), loc) for a in args], loc)


class _UnfoldEnumerateInstance(_UnfoldIterInstance):
    _node = Enumerate

    def _elt(self, args: list[Expr], i: NamedId, loc) -> Expr:
        return TupleExpr([Var(i, loc), ListRef(args[0], Var(i, loc), loc)], loc)


class _Unfold:
    """The `sites` / `refusals` / `apply` surface both unfolds share."""

    _instance: type[_UnfoldIterInstance]
    _what: str

    @classmethod
    def sites(cls, func: FuncDef, within: Cursor | None = None) -> list[Cursor]:
        """The iterables of `func` this pass would unfold, in visit order --
        what a `where` index counts, and what `within` narrows."""
        return cls._instance(func).list_sites(within)

    @classmethod
    def refusals(
        cls, func: FuncDef, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each one that is not a site was left alone."""
        return cls._instance(func).list_refusals(within)

    @classmethod
    def apply(
        cls, func: FuncDef, *,
        where: int | Cursor | None = None,
        temp_id: NamedId | None = None,
    ) -> FuncDef:
        """`func` with the selected iterables stated as comprehensions.

        `where` selects one by index in visit order, or by cursor; `None` takes
        every one that has a statement slot.
        """
        return cls.apply_with_edits(func, where=where, temp_id=temp_id).result

    @classmethod
    def apply_with_edits(
        cls, func: FuncDef, *,
        where: int | Cursor | None = None,
        temp_id: NamedId | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)
        vtor = cls._instance(func, temp_id, where)
        out = vtor.apply()
        vtor.check_site(cls._what)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=False)


class UnfoldZip(_Unfold):
    """Transformation pass to state `zip` as a comprehension."""

    _instance = _UnfoldZipInstance
    _what = 'a `zip`'


class UnfoldEnumerate(_Unfold):
    """Transformation pass to state `enumerate` as a comprehension."""

    _instance = _UnfoldEnumerateInstance
    _what = 'an `enumerate`'
