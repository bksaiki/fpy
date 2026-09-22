"""Transformation pass to rewrite if statements to if expressions."""

from ..analysis import ContextUse, DefineUse, DefineUseAnalysis, SyntaxCheck
from ..analysis.context_use import ContextUseAnalysis
from ..ast import *
from ..function import Function
from ..number import (
    Context,
    EFloatContext,
    ExpContext,
    FixedContext,
    IEEEContext,
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    MPFloatContext,
    MPSFloatContext,
    OverflowMode,
    SMFixedContext,
)
from ..primitive import Primitive
from ..utils import Gensym
from .copy_propagate import CopyPropagate
from .cursor import Cursor, EditLog, StmtPath
from .error import TransformDeclined
from .rename_target import RenameTarget
from .utils import SiteRewriter, check_where

# Scoped statements (`with`) are absent on purpose: the scan descends into
# them, since their contents become unconditional too.
_POLE_OPS: frozenset[type[Expr]] = frozenset({
    Acos, Acosh, Asin, Atanh, Div, Fmod, Lgamma, Log, Log10, Log1p, Log2,
    Logb, Mod, Pow, Remainder, Sqrt, Tgamma,
})
"""Operations with a pole or branch cut at a finite operand.

IEEE 754 §7.2 (invalid) and §7.3 (divideByZero) give these a case where a
finite input yields an infinity or NaN whatever the format's range: `logb(0)`,
`sqrt(-1)`, `acos(2)`.  An operand can never *already* be a special under a
context that cannot hold one, since producing it would have raised earlier.
Overflow is the other route and is handled separately -- it turns on the
context, not the operation.
"""


_MAX_INLINE_GROWTH = 4
"""How far substituting an arm's temps may grow it before it is hoisted instead.

A name bound in an arm and read twice is duplicated into both readers; this
bounds the blowup when arms nest.
"""


class _Clone(DefaultTransformVisitor):
    """Deep copy.  A substituted expression cannot be shared between two
    occurrences: the analyses key on node identity."""

    def apply(self, e: Expr) -> Expr:
        return self._visit_expr(e, None)


class _Subst(DefaultTransformVisitor):
    """Replaces names by expressions, one fresh copy per occurrence."""

    def __init__(self, env: dict[NamedId, Expr]):
        super().__init__()
        self.env = env

    def _visit_var(self, e: Var, ctx: None):
        sub = self.env.get(e.name)
        return Var(e.name, e.loc) if sub is None else _Clone().apply(sub)

    def apply(self, e: Expr) -> Expr:
        return self._visit_expr(e, None)


class _Names(DefaultVisitor):
    """Names an expression reads."""

    def __init__(self):
        super().__init__()
        self.names: set[NamedId] = set()

    def _visit_var(self, e: Var, ctx):
        self.names.add(e.name)


def _reads(e: Expr) -> set[NamedId]:
    v = _Names()
    v._visit_expr(e, None)
    return v.names


class _Size(DefaultVisitor):
    """Expression nodes beneath a node, to bound substitution growth."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def _visit_expr(self, e: Expr, ctx):
        self.n += 1
        return super()._visit_expr(e, ctx)


def _size(nodes) -> int:
    v = _Size()
    for n in nodes:
        v._visit_expr(n, None) if isinstance(n, Expr) else v._visit_statement(n, None)
    return v.n


def _inline_arm(body: StmtBlock) -> dict[NamedId, Expr] | None:
    """Each name the arm assigns, as an expression over the pre-`if` names --
    or `None` if the arm cannot be reduced to one.

    An arm expressed this way goes *inside* the `IfExpr`, which is lazy, so it
    runs only when the guard held, exactly as the `if` statement did.  Hoisting
    it instead would make it unconditional, which is what the refusals in
    :class:`_Unhoistable` are about; an arm that inlines needs none of them.
    """
    env: dict[NamedId, Expr] = {}
    for stmt in body.stmts:
        # anything but a plain assignment -- a loop, a list write, an `if`
        # left unrewritten -- has no expression form
        if not isinstance(stmt, Assign) or not isinstance(stmt.target, NamedId):
            return None
        env[stmt.target] = _Subst(env).apply(stmt.expr)
    if _size(env.values()) > _MAX_INLINE_GROWTH * _size(body.stmts):
        return None
    return env


_UNHOISTABLE: dict[type[Stmt], str] = {
    ReturnStmt: 'a `return` escapes the branch and has no expression form',
    AssertStmt: 'an `assert` would run unconditionally and can abort',
    EffectStmt: 'an effect would run unconditionally',
    IndexedAssign: 'a list write would run unconditionally',
    WhileStmt: 'a `while` would run unconditionally and may not terminate',
    ForStmt: 'a `for` would run unconditionally',
}


class _Unhoistable(DefaultVisitor):
    """Why a branch body cannot be hoisted, in two categories.

    ``aborts`` can change which value the function produces, or aborts where
    the program asked to -- an `assert`, an `fp.cast`, an `ASSERT` overflow.
    No mode admits it.

    ``unproven`` is what ``strict`` governs: an observable effect that may
    differ, and a trap the program did not ask for -- a context with nowhere
    to put an infinity raises, but that is the format's limit, not a
    requested abort.  The default admits these; :func:`_inline_arm` is what
    keeps most of them from arising, since an arm it reduces to expressions
    keeps its guard and is never hoisted.
    """

    def __init__(self, ctx_use: ContextUseAnalysis):
        super().__init__()
        self.ctx_use = ctx_use
        self.aborts: str | None = None
        self.unproven: str | None = None

    def _abort(self, why: str) -> None:
        if self.aborts is None:
            self.aborts = why

    def _cannot_prove(self, why: str) -> None:
        if self.unproven is None:
            self.unproven = why

    def _visit_statement(self, stmt: Stmt, ctx):
        why = _UNHOISTABLE.get(type(stmt))
        if why is not None:
            self._abort(why)
        return super()._visit_statement(stmt, ctx)

    def _visit_list_ref(self, e: ListRef, ctx):
        self._cannot_prove('a subscript may be out of range outside its guard')
        super()._visit_list_ref(e, ctx)

    def _visit_list_slice(self, e: ListSlice, ctx):
        self._cannot_prove('a slice may be out of range outside its guard')
        super()._visit_list_slice(e, ctx)

    def _visit_call(self, e: Call, ctx):
        """A callee's body is not scanned, so an abort inside it -- an
        `assert`, a rounding that overflows -- would reach the hoist unseen.
        Refused rather than analyzed interprocedurally.

        A context constructor is exempt: it builds a value and the `with` it
        heads is where any rounding happens.  Mirrors the callee taxonomy of
        :class:`fpy2.analysis.Purity`.
        """
        match e.fn:
            case type() if issubclass(e.fn, Context):
                pass
            case Function():
                self._abort(
                    f'`{e.fn.name}` may abort, and a callee is not scanned'
                )
            case Primitive() if e.fn.pure:
                pass
            case _:
                self._abort('a call to a foreign function may abort')
        super()._visit_call(e, ctx)

    def _visit_expr(self, e: Expr, ctx):
        if isinstance(e, Cast):
            self._abort(
                '`fp.cast` asserts its result is exact, so hoisting it can abort'
            )
        self._check_context(e)
        return super()._visit_expr(e, ctx)

    def _check_context(self, e: Expr) -> None:
        """Whether *e* can abort through the context it rounds under.

        Keyed on whether `ContextUse` records *e* as a use site, not on its node
        class: every rounded operation consults the ambient context, so `x * x`
        overflows under an `ASSERT` context exactly as `fp.round(x)` does.  An
        earlier version asked `isinstance(e, Round | Cast)` and let arithmetic,
        and `fp.round_at`, through.

        A context that does not resolve to a concrete one -- a function with no
        ``ctx=`` inherits its caller's -- cannot be shown either way, so it is
        ``unproven``.  That makes ``strict`` conservative in an unannotated
        function, which is the honest reading: the equivalence it promises is
        exactly what an unknown context denies.
        """
        try:
            resolved = self.ctx_use.find_scope_from_use(e).ctx
        except KeyError:
            return
        if isinstance(resolved, Context):
            # The tuple is literal rather than a named constant: mypy
            # narrows on a literal and so type-checks `.overflow`, which the
            # base `Context` does not declare.  `getattr` would answer `None`
            # on a rename and silently stop refusing.
            if isinstance(resolved, (
                EFloatContext, ExpContext, FixedContext, IEEEContext,
                MPBFixedContext, MPBFloatContext, SMFixedContext,
            )) and resolved.overflow is OverflowMode.ASSERT:
                self._abort(
                    'an operation under an `ASSERT` overflow context can abort'
                )
            # Such a context raises rather than yield the special.  Nobody
            # asked it to -- unlike `assert`, `fp.cast` or `ASSERT`, the trap
            # is the context having nowhere to put the result -- so this is
            # `strict`'s to decline, not an abort.
            if isinstance(resolved, (
                MPBFixedContext, MPBFloatContext, MPFixedContext,
                MPFloatContext, MPSFloatContext,
            )):
                if type(e) in _POLE_OPS and not (
                    resolved.enable_inf and resolved.enable_nan
                ):
                    self._cannot_prove(
                        f'`{type(e).__name__.lower()}` can produce an infinity '
                        'or NaN, which this context cannot hold'
                    )
                # IEEE 754 §7.4: overflow rounds to an infinity, and any
                # rounded operation can overflow a bounded format
                if isinstance(resolved, (
                    MPBFixedContext, MPBFloatContext,
                )) and not resolved.enable_inf \
                        and resolved.overflow is OverflowMode.OVERFLOW:
                    self._cannot_prove(
                        'an operation can overflow to an infinity, which this '
                        'context cannot hold'
                    )
        else:
            self._cannot_prove(
                'an operation under an unresolved context may abort on overflow'
            )


def _why_unhoistable(
    block: StmtBlock, ctx_use: ContextUseAnalysis, strict: bool,
) -> str | None:
    v = _Unhoistable(ctx_use)
    v._visit_block(block, None)
    return v.aborts or (v.unproven if strict else None)


class _SimplifyIfInstance(SiteRewriter):
    """Single-use instance of the SimplifyIf pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        ctx_use: ContextUseAnalysis,
        strict: bool,
        where: int | Cursor | None = None,
    ):
        super().__init__()
        self.func = func
        self.def_use = def_use
        self.ctx_use = ctx_use
        self.strict = strict
        self.where = where
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self):
        func = self._visit_function(self.func, None)
        self.check_site('an `if` statement')
        return func, self.gensym.generated

    def _claims(self, stmt: Stmt, *bodies: StmtBlock) -> bool:
        """Whether to rewrite here.  A refusal that was aimed at is raised
        rather than skipped: a named site this pass cannot take is an error."""
        block, pos = self._site
        why: str | None = None
        for body in bodies:
            why = _why_unhoistable(body, self.ctx_use, self.strict)
            if why is not None:
                break

        if why is not None:
            self.refused.append((stmt, why))
            if self._selects(block, pos, -1):
                self.declined.append(why)
                if not self.listing:
                    raise TransformDeclined(
                        f'cannot rewrite `if` to `if` expression: {why}'
                    )
            return False

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects(block, pos, idx):
            return False
        self._matched += 1
        if self.listing:
            self.found.append(StmtPath(self._paths[id(block)], pos))
            return False
        return True

    def _emit(self, ctx: list[Stmt], stmts: list[Stmt]):
        """Give the enclosing block *stmts* in place of the `if`, returning the
        last as the replacement per :meth:`SiteRewriter._visit_block`."""
        self._replaced = True
        ctx.extend(stmts[:-1])
        return stmts[-1], None

    def _rewrite(
        self, cond_e: Expr, ift: StmtBlock, iff: StmtBlock | None, ctx,
    ) -> list[Stmt]:
        """The statements replacing an `if`, with *iff* `None` for a one-armed
        one -- which is this rewrite with an empty else throughout: no names
        merge from that side, so each takes its pre-`if` value there.

        `None` rather than an empty block because `mutated_in` / `introed_in`
        index by block identity, and a synthesized one is in neither.
        """
        stmts: list[Stmt] = []
        cond = self._visit_expr(cond_e, ctx)
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            stmts.append(Assign(t, BoolTypeAnn(None), cond, None))
            cond = Var(t, None)

        bodies = [
            None if b is None else self._visit_block(b, ctx)[0]
            for b in (ift, iff)
        ]

        # FPy semantics: a name is introduced in both arms or in neither
        intros = sorted(
            self.def_use.introed_in(ift) & self.def_use.introed_in(iff)
        ) if iff is not None else []

        renames: list[dict[NamedId, NamedId]] = []
        inlined: list[dict[NamedId, Expr]] = []
        merged: set[NamedId] = set()
        for src, body in zip((ift, iff), bodies):
            if src is None or body is None:
                renames.append({})
                inlined.append({})
                continue
            mutated = sorted(self.def_use.mutated_in(src))
            merged |= set(mutated) | set(intros)
            # an arm that reduces to expressions goes inside the `IfExpr`,
            # which is lazy, so it keeps its guard
            env = _inline_arm(body)
            if env is not None:
                renames.append({})
                inlined.append(env)
                continue
            rename = {v: self.gensym.refresh(v) for v in mutated + intros}
            renames.append(rename)
            inlined.append({})
            # a mutated name carries its pre-`if` value in; an introduced one
            # has none to carry
            for v in mutated:
                stmts.append(Assign(rename[v], None, Var(v, None), None))
            stmts.extend(RenameTarget.apply_block(body, rename).stmts)

        # Over the union: a name mutated in one arm only still needs a merge,
        # and takes its pre-`if` value on the other side.
        def side(i: int, var: NamedId) -> Expr:
            """*var* as arm *i* leaves it: the expression an inlined arm
            reduced it to, else the name the hoisted arm assigned."""
            e = inlined[i].get(var)
            if e is None:
                return Var(renames[i].get(var, var), None)
            return e

        exprs: dict[NamedId, Expr] = {
            var: IfExpr(cond, side(0, var), side(1, var), None)
            for var in sorted(merged)
        }

        # An inlined arm leaves pre-`if` names in the merges, so a merge can
        # read one an earlier merge has already overwritten.  Those go through
        # a temporary: the merges happen at once.
        reads = {var: _reads(e) for var, e in exprs.items()}
        shared = {
            var for var in exprs
            if any(var in reads[o] for o in exprs if o != var)
        }
        tmp = {var: self.gensym.refresh(var) for var in sorted(shared)}
        for var, e in exprs.items():
            stmts.append(Assign(tmp.get(var, var), None, e, None))
        for var in sorted(shared):
            stmts.append(Assign(var, None, Var(tmp[var], None), None))
        return stmts

    def _visit_if1(self, stmt: If1Stmt, ctx: list[Stmt]):
        if not self._claims(stmt, stmt.body):
            return super()._visit_if1(stmt, ctx)
        return self._emit(ctx, self._rewrite(stmt.cond, stmt.body, None, ctx))

    def _visit_if(self, stmt: IfStmt, ctx: list[Stmt]):
        if not self._claims(stmt, stmt.ift, stmt.iff):
            return super()._visit_if(stmt, ctx)
        return self._emit(
            ctx, self._rewrite(stmt.cond, stmt.ift, stmt.iff, ctx)
        )


class SimplifyIf:
    """Rewrites `if` statements into `if` expressions::

        if <cond>:          t = <cond>
            S1 ...    ⇝     S1 ...
        else:               S2 ...
            S2 ...          x = x_S1 if t else x_S2

    Both bodies are hoisted into the enclosing block and each merged name is
    made explicit with an `IfExpr`.

    Hoisting makes a branch body unconditional, so any construct that could
    change whether -- or which -- value the function produces is declined.
    ``strict`` additionally declines what is preserved in value but not
    provably in observable effect.  :func:`fpy2.strategies.simplify_if`
    documents both.
    """

    @staticmethod
    def _instance(
        func: FuncDef, strict: bool, where: 'int | Cursor | None',
    ) -> _SimplifyIfInstance:
        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        return _SimplifyIfInstance(func, def_use, ctx_use, strict, where)

    @staticmethod
    def sites(
        func: FuncDef, within: 'Cursor | None' = None, *, strict: bool = False,
    ) -> list[Cursor]:
        """The `if` statements this pass would rewrite, in visit order --
        what a `where` index counts, and what `within` narrows.

        `strict` is taken because it decides what is a site: a branch this pass
        would decline is not one.
        """
        return SimplifyIf._instance(func, strict, None).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: 'Cursor | None' = None, *, strict: bool = False,
    ) -> list[tuple[Cursor, str]]:
        """Why each `if` this pass could have rewritten is not a site."""
        return SimplifyIf._instance(func, strict, None).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef,
        where: 'int | Cursor | None' = None,
        *,
        strict: bool = False,
    ) -> FuncDef:
        """Rewrite `if` statements into `if` expressions.

        `where` names one site: an index counting the `if` statements this
        rewrite acts on, in visit order, or a cursor or region, which takes the
        sites at or beneath it.  `None` rewrites every one.
        :func:`fpy2.strategies.simplify_if` documents the rest.
        """
        return SimplifyIf.apply_with_edits(func, where, strict=strict).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        where: 'int | Cursor | None' = None,
        *,
        strict: bool = False,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced.

        A cursor naming a statement *inside* a rewritten branch does not
        forward: that subtree was rebuilt and renamed.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        check_where(where)
        inst = SimplifyIf._instance(func, strict, where)
        ast, new_ids = inst.apply()
        ast = CopyPropagate.apply(ast, names=new_ids)
        SyntaxCheck.check(ast, ignore_unknown=True)
        return EditLog(func, ast, tuple(inst.edits), exprs_preserved=True)
