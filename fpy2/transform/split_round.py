"""
Rounding splitting: one correctly-rounded operation into two roundings.

A correctly-rounded operation is a real computation followed by a rounding,
``f_F1,rm1 = rnd_F1,rm1 . f_R``.  Where the pair of formats and modes satisfies
Figure 8 of *When Double Rounding is Correct*, that single rounding equals a
rounding to an intermediate followed by a rounding to the target, so either may
replace the other.  This pass reads that left to right, computing under the
intermediate and re-rounding to the target.

The candidates are the operations that *already* have a format, which is the
complement of :class:`.RoundInsert`'s.  An assignment does not round in FPy, so
the second rounding is emitted as an explicit ``round`` in the enclosing block,
where it picks up the target's own mode.

.. code-block:: python

    # before                       # after, split through an RTO intermediate
    with fp.FP32:                  with fp.FP32:
        t = x * y                      with <intermediate>:
                                           _t = x * y
                                       t = round(_t)

Which pairs are admissible is decided by
:func:`fpy2.analysis.format_infer.double_round_ok`; the intermediate is the
caller's, and :func:`fpy2.analysis.format_infer.derive_intermediate` computes a
suitable one.  An intermediate that represents the operation's *exact* result is
admitted whatever the modes, since rounding to it is then the identity -- which is
how a nearest-to-nearest split becomes possible at all.  Explicit ``Round`` / ``Cast`` nodes are deliberately not
candidates: splitting a rounding is the inverse of merging two, and admitting
them makes a second application grow the tree twice as fast.
"""

import operator
from collections.abc import Callable
from typing import Any

from ..analysis import SyntaxCheck
from ..analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    SetFormat,
    double_round_ok,
    exact_binop,
    exact_unop,
)
from ..ast.fpyast import (
    Abs,
    Add,
    BinaryOp,
    Cast,
    ConstInf,
    ConstNan,
    Dim,
    Expr,
    Fst,
    FuncDef,
    Len,
    Max,
    Min,
    Mul,
    NamedId,
    NaryOp,
    Neg,
    NullaryOp,
    Round,
    RoundAt,
    RoundInt,
    Size,
    Snd,
    Sub,
    TernaryOp,
    UnaryOp,
    Var,
)
from ..number import REAL, Context, Float, RealFloat
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    RoundingRewriter,
    RoundingScopes,
    check_where,
    operands,
)

_OPS = (NullaryOp, UnaryOp, BinaryOp, TernaryOp, NaryOp)
"""The nodes that round under the active context.  A ``Call`` is not one: it
inherits the context, but its callee's operations each round separately."""

_NOT_SPLITTABLE = (
    # the inverse rewrite's territory, not this one's
    Round, Cast, RoundAt, RoundInt,
    # exact queries and projections: nothing is rounded
    Len, Size, Dim, Fst, Snd,
    # `min`/`max` *select* an argument and hand it back with its own format
    # rather than rounding to the active context, so there is no rounding here
    # to split -- measured: every other real-valued operation does round
    Min, Max,
    # not finite reals, which is what the theorems quantify over
    ConstInf, ConstNan,
)
"""Operations that are not a real computation followed by a rounding."""

_EXACT_BINOP: dict[type, Callable[[Any, Any], Any]] = {
    Add: operator.add, Sub: operator.sub, Mul: operator.mul,
}
_EXACT_UNOP: dict[type, Callable[[Any], Any]] = {
    Abs: operator.abs, Neg: operator.neg,
}
"""The operators `FormatInfer` computes each unrounded result with."""


class _SplitRoundInstance(RoundingRewriter):
    """Splits selected rounded operations through an intermediate."""

    def _candidate(self, e: Expr) -> bool:
        """Any operation that is a real computation followed by a rounding.

        The rules quantify over an arbitrary real, so what produced it does not
        matter -- ``sqrt``, ``fma``, a transcendental and ``pi`` are all as
        splittable as a multiply.  A scalar format bound is what says the
        operation is real-valued: a boolean has none and a list or tuple has a
        bound of another kind.
        """
        if not isinstance(e, _OPS) or isinstance(e, _NOT_SPLITTABLE):
            return False
        bound = self.scopes.format_info.by_expr.get(e)
        return isinstance(bound, (AbstractableFormat, AbstractFormat, SetFormat))

    def _verify(self, e: Expr) -> None | Declined:
        """`None` where *e*'s rounding may be split, else why not."""
        target = self.scopes.scope_ctx(e)
        if target is None:
            return Declined(
                'the operation\'s scope is symbolic, so the rounding it '
                'performs is unknown'
            )
        if target is REAL:
            return Declined(
                'the operation rounds exactly, so there is no rounding to '
                'split; `insert_round` gives it one first'
            )
        if target.is_stochastic() or self.ctx.is_stochastic():
            return Declined(
                'a stochastic rounding is not a function of its input, so the '
                'composition cannot be checked'
            )

        f1, f2 = target.format(), self.ctx.format()
        if not isinstance(f1, AbstractableFormat) or not isinstance(f2, AbstractableFormat):
            return Declined(
                'one of the formats has no abstract form, so the premise '
                'cannot be checked'
            )
        f2a = AbstractFormat.from_format(f2)

        # `rndExact`: where the intermediate represents this operation's exact
        # result, rounding to it is the identity, so the composition *is* the
        # original computation -- under any pair of modes, overflow included.
        exact = self._exact_result(e)
        if exact is not None and exact.contained_in(f2a):
            return None

        rm1, rm2 = target.rounding_mode(), self.ctx.rounding_mode()
        if rm1 is None or rm2 is None:
            return Declined('a context without a rounding mode has no rule')
        if not double_round_ok(AbstractFormat.from_format(f1), rm1, f2a, rm2):
            return Declined(
                f'rounding to {rm2.name} and then {rm1.name} is not the same '
                f'as rounding to {rm1.name} for these formats'
            )

        # Figure 8 covers finite values inside the intermediate's range.  A
        # bounded intermediate can also be handed a special or a value past its
        # range, and the rule stays valid exactly where the composition agrees
        # there too -- which is a finite check.
        if isinstance(f2a.bound, RealFloat) and not (
            self._composes_special(target, f2a) or self._within(exact, f2a)
        ):
            return Declined(
                'the composition disagrees with the single rounding on a '
                'special or a value past the intermediate\'s range'
            )
        return None

    def _composes_special(self, target: Context, f2: AbstractFormat) -> bool:
        """Whether the composition agrees with the single rounding on
        ``{NaN, +Inf, -Inf, +Huge, -Huge}``.

        Those are the inputs the premise says nothing about: it quantifies over
        finite values, and over values the intermediate can represent.  *Huge* is
        one step past the intermediate's bound -- the smallest magnitude it
        cannot hold, and so where the two paths first diverge, since above that
        the intermediate's answer is constant while the target's is not.
        """
        probes: list[Float | RealFloat] = [
            Float(isnan=True), Float(isinf=True), Float(isinf=True, s=True),
        ]
        for bound in (f2.pos_bound, f2.neg_bound):
            if isinstance(bound, RealFloat) and not bound.is_zero():
                probes.append(bound.next_away_zero())

        try:
            for v in probes:
                once = target.round(v)
                twice = target.round(self.ctx.round(v))
                # `str` rather than `==`: it separates the zeros and is total on
                # NaN, which compares equal to nothing
                if str(once) != str(twice):
                    return False
        except (ValueError, OverflowError):
            # a context that cannot represent a probe -- `enable_nan=False`, or
            # `OverflowMode.ASSERT`
            return False
        return True

    def _exact_result(self, e: Expr) -> AbstractFormat | None:
        """*e*'s result before the target rounds it, as a format.

        The *unrounded* one: `by_expr` holds the result after the rounding, which
        is inside the target's format by construction and would prove nothing.
        `None` where it has no abstract form -- an operation with no exact rule,
        a constant-folded set, or an operand whose format is unknown.
        """
        args = [self.scopes.format_info.by_expr.get(a) for a in operands(e)]
        # the arity is implied by the table: unpacking asserts it
        if type(e) in _EXACT_BINOP:
            lhs, rhs = args
            out = exact_binop(lhs, rhs, _EXACT_BINOP[type(e)])
        elif type(e) in _EXACT_UNOP:
            arg, = args
            out = exact_unop(arg, _EXACT_UNOP[type(e)])
        else:
            return None

        if isinstance(out, AbstractableFormat):
            out = AbstractFormat.from_format(out)
        return out if isinstance(out, AbstractFormat) else None

    @staticmethod
    def _within(exact: AbstractFormat | None, f2: AbstractFormat) -> bool:
        """Whether *exact* provably stays inside *f2*'s finite range, so that
        rounding to *f2* cannot overflow."""
        if exact is None:
            return False
        pos, neg = exact.pos_bound, exact.neg_bound
        if isinstance(pos, float) or isinstance(neg, float):
            return False        # unbounded result: nothing to prove
        return pos <= f2.pos_bound and neg >= f2.neg_bound

    def _wrap(self, t: NamedId, loc) -> Expr:
        # an assignment rounds nothing in FPy, so the second rounding is
        # explicit -- and sits in the enclosing block, which applies `rm1`
        return Round(None, Var(t, loc), loc)


class SplitRound:
    """
    Transformation pass to split one rounding into two that compose to the
    same answer.
    """

    @staticmethod
    def sites(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[Cursor]:
        """The operations of `func` whose rounding would be split through
        `ctx`, in visit order -- what a `where` index counts, and what `within`
        narrows.

        `ctx` is required because it decides the answer: whether an operation is
        a site is whether *its* rounding composes with a rounding to *this*
        intermediate.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        scopes = RoundingScopes(func)
        return _SplitRoundInstance(func, ctx, scopes).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None, *, ctx: Context
    ) -> list[tuple[Cursor, str]]:
        """Why each operation of `func` that is not a site was left alone."""
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        scopes = RoundingScopes(func)
        return _SplitRoundInstance(func, ctx, scopes).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, ctx: Context, *, where: int | Cursor | None = None
    ) -> FuncDef:
        """
        Splits every rounding of `func` that composes with a rounding to `ctx`,
        and leaves the rest alone.

        `where` selects one operation by index in visit order; `None` takes
        every one it can split.
        """
        return SplitRound.apply_with_edits(func, ctx, where=where).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, ctx: Context, *, where: int | Cursor | None = None
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected a \'Context\', got {ctx}')
        check_where(where)

        scopes = RoundingScopes(func)
        vtor = _SplitRoundInstance(func, ctx, scopes, where)
        out = vtor.apply()
        vtor.check_site('a rounded operation')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
