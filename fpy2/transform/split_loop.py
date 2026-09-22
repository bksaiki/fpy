"""
Loop splitting transformation.

This transformation is inspired by the `split()` procedure
from Halide (https://halide-lang.org/).

Rounding-context safety
-----------------------
Splitting introduces loop-control and index arithmetic (``len(t)``, the
remainder ``fmod(n, f)``, the chunk bound ``i + f``).  These are
*integer* computations, but FPy rounds every arithmetic operation under
the active context (**E-Add**): under a low-precision float context a
bound could round to the wrong value and read out of bounds.  Every such
inserted computation is therefore wrapped in ``with fp.INTEGER:``.  The
element reads then index with the resulting exact values, and each read
is *adjacent to its body* — the loop target is reassigned per element,
so a body that mutates the iterated list in place observes its own
writes exactly as the original loop does.  The iterable and the loop
body run under the *ambient* context so their rounding is unchanged.
"""

import enum

from ..analysis import (
    ArraySizeAnalysis,
    ReachingDefs,
    ReachingDefsAnalysis,
    SyntaxCheck,
)
from ..ast.fpyast import *
from ..utils import Gensym
from .cursor import Cursor, EditLog, StmtCursor
from .path import StmtPath
from .utils import (
    SiteRewriter,
    check_where,
    clone_block,
    copy_target,
    infer_array_size,
    integer_ctx,
    static_size,
)


class SplitLoopStrategy(enum.Enum):
    """Strategy for dealing with the loop remainder."""

    STRICT = 0
    """Require the length to be an exact multiple of the factor:
    verified at compile time when the length and factor are statically
    known (a provably-indivisible length raises), else guarded by a
    runtime ``assert fmod(len(t), f) == 0``."""

    PEEL = 1
    """Split the largest multiple-of-factor prefix into chunks, then
    run the remaining ``len % f`` iterations in a residual loop.
    Correct for any length."""

    MASK = 2
    """Chunk a length rounded *up* to a multiple of the factor, so every
    chunk is a full ``f`` wide, and guard the body with ``j < n`` so the
    over-run iterations do nothing.  Correct for any length, and unlike
    ``PEEL`` it emits the body once -- the tail is a predicate rather than a
    second loop, which is the shape a SIMD target masks directly."""


class _SplitLoop(SiteRewriter):
    """
    Split loop visitor.
    """

    func: FuncDef
    factor: Expr
    where: int | Cursor | None
    strategy: SplitLoopStrategy
    temp_id: NamedId
    outer_id: NamedId
    inner_id: NamedId

    gensym: Gensym
    # Static list sizes of iterables; with a literal factor, enables
    # discharging the remainder handling at compile time.
    array_size: ArraySizeAnalysis | None
    use_fmod: bool

    def __init__(
        self,
        func: FuncDef,
        factor: Expr,
        where: int | Cursor | None,
        strategy: SplitLoopStrategy,
        reaching_defs: ReachingDefsAnalysis,
        temp_id: NamedId,
        outer_id: NamedId,
        inner_id: NamedId,
        array_size: ArraySizeAnalysis | None,
        use_fmod: bool = True
    ):
        super().__init__()
        self.func = func
        self.factor = factor
        self.where = where
        self.strategy = strategy
        self.temp_id = temp_id
        self.outer_id = outer_id
        self.inner_id = inner_id
        self.array_size = array_size
        self.use_fmod = use_fmod

        self.gensym = Gensym(reaching_defs.names())

    def _static_factor(self) -> int | None:
        """The factor as a positive compile-time constant, or ``None``."""
        if isinstance(self.factor, Integer) and self.factor.val >= 1:
            return self.factor.val
        return None

    def _rem(self, a: Expr, b: Expr) -> Expr:
        """``a`` remainder ``b``, spelled as the caller asked.

        The two differ on a negative dividend (`-5 fmod 4 = -1`, `-5 % 4 = 3`)
        and every dividend here is a length or a value in ``[0, f)``, so they
        agree; which one a backend can emit is what differs, and that is the
        caller's to know."""
        if self.use_fmod:
            return Fmod(None, a, b, None)
        return Mod(a, b, None)

    @staticmethod
    def _ref(x: NamedId | int) -> Expr:
        """A fresh reference node per use site — a ``Var`` for a bound
        name, an ``Integer`` for a compile-time constant — so no node is
        shared between positions (analyses key results by node identity)."""
        if isinstance(x, NamedId):
            return Var(x, None)
        return Integer(x, None)

    def _dynamic_prelude(
        self,
        t: NamedId,
        f: NamedId,
        n: NamedId,
        factor: Expr,
        extra: list[Stmt],
        loc: Location | None
    ) -> ContextStmt:
        """The loop-control bindings shared by both dynamic paths.
        A non-positive runtime factor would silently skip iterations
        (``range(0, n, f)`` is empty), so it is rejected loudly."""
        return integer_ctx([
            # `factor` is an arbitrary caller Expr feeding `range`/`fmod`,
            # so it is evaluated exactly (unlike the iterable, kept ambient)
            Assign(f, None, factor, None),
            AssertStmt(
                Compare([CompareOp.GE], [Var(f, None), Integer(1, None)], None),
                None,
                None
            ),
            Assign(n, None, Len(None, Var(t, None), None), None),
            *extra,
        ], loc)

    def _chunk_loop(
        self,
        t: NamedId,
        f: NamedId | int,
        bound: NamedId | int,
        target: Id | TupleBinding,
        body: StmtBlock,
        loc: Location | None,
        mask: NamedId | int | None = None
    ) -> ForStmt:
        """The chunked loop over ``range(0, bound, f)``; each chunk runs
        an inner loop whose target is reassigned per element, so every
        read stays adjacent to its body (see the module docstring).

        With *mask*, the inner body is guarded by ``j < mask``.  The guard
        wraps the element read as well as the body: reading ``t[j]`` past the
        end is the access it exists to prevent."""
        outer = self.gensym.refresh(self.outer_id)
        inner = self.gensym.refresh(self.inner_id)
        hi = self.gensym.refresh(self.temp_id)

        # the chunk bound `i + f` must be exact, so it is precomputed
        # under `INTEGER` rather than inlined into the `range`
        hi_bind = integer_ctx([
            Assign(hi, None, Add(Var(outer, None), self._ref(f), None), None)
        ], None)

        guarded = StmtBlock([
            Assign(copy_target(target), None, ListRef(Var(t, None), Var(inner, None), None), None),
            *body.stmts
        ])
        if mask is None:
            inner_body = guarded
        else:
            inner_body = StmtBlock([If1Stmt(
                Compare([CompareOp.LT], [Var(inner, None), self._ref(mask)], None),
                guarded,
                None
            )])
        inner_loop = ForStmt(
            inner,
            Range3(None, Var(outer, None), Var(hi, None), Integer(1, None), None),
            inner_body,
            loc
        )

        return ForStmt(
            outer,
            Range3(None, Integer(0, None), self._ref(bound), self._ref(f), None),
            StmtBlock([hi_bind, inner_loop]),
            loc
        )

    def _residual_loop(
        self,
        t: NamedId,
        lo: NamedId | int,
        hi: NamedId | int,
        target: Id | TupleBinding,
        body: StmtBlock,
        loc: Location | None
    ) -> ForStmt:
        """The ``[lo, hi)`` remainder loop, over a fresh copy of the
        target and body (they also appear in the chunked loop)."""
        rem = self.gensym.refresh(self.inner_id)
        rem_body = StmtBlock([
            Assign(
                copy_target(target), None,
                ListRef(Var(t, None), Var(rem, None), None), None
            ),
            *clone_block(body).stmts
        ])
        return ForStmt(
            rem,
            Range3(None, self._ref(lo), self._ref(hi), Integer(1, None), None),
            rem_body,
            loc
        )

    def _refuses(self, stmt: ForStmt) -> str | None:
        """Why this loop cannot be split, or `None` where it can.

        Only ``STRICT`` refuses, and only where both the length and the factor
        are statically known and indivisible -- decided before an index is
        spent, so a listing and the rewrite agree.
        """
        if self.strategy is not SplitLoopStrategy.STRICT:
            return None
        size = static_size(self.array_size, stmt.iterable)
        fval = self._static_factor()
        if size is None or fval is None or size % fval == 0:
            return None
        return (
            f'a STRICT split by {fval} needs the iterable length to be a '
            f'multiple of {fval}, and its statically-known length is {size}'
        )

    def _build_strict(self, stmt: ForStmt, iterable: Expr, factor: Expr, body: StmtBlock) -> list[Stmt]:
        # STRICT: the length must be an exact multiple of `f`, so the
        # chunked loop covers the whole (asserted-divisible) length.
        t = self.gensym.refresh(self.temp_id)
        emitted: list[Stmt] = [Assign(t, None, iterable, None)]   # ambient materialize

        size = static_size(self.array_size, stmt.iterable)
        fval = self._static_factor()
        if size is not None and fval is not None:
            # Statically-known length and factor: verify divisibility at
            # compile time and drop the runtime `len`/`assert` entirely.
            # `_refuses` established divisibility before this site was counted
            assert size % fval == 0
            if size > 0:
                emitted.append(self._chunk_loop(t, fval, size, stmt.target, body, stmt.loc))
        else:
            f = self.gensym.refresh(self.temp_id)
            n = self.gensym.refresh(self.temp_id)
            emitted.append(self._dynamic_prelude(t, f, n, factor, [
                AssertStmt(
                    Compare(
                        [CompareOp.EQ],
                        [
                            self._rem(Var(n, None), Var(f, None)),
                            Integer(0, None)
                        ],
                        None
                    ),
                    None,
                    None
                ),
            ], stmt.loc))
            emitted.append(self._chunk_loop(t, f, n, stmt.target, body, stmt.loc))

        return emitted

    def _build_peel(self, stmt: ForStmt, iterable: Expr, factor: Expr, body: StmtBlock) -> list[Stmt]:
        # PEEL: chunk the `[0, m)` prefix (largest multiple of `f`) and
        # run the `[m, n)` remainder in a residual loop.  Correct for
        # any length.
        t = self.gensym.refresh(self.temp_id)
        emitted: list[Stmt] = [Assign(t, None, iterable, None)]   # ambient materialize

        size = static_size(self.array_size, stmt.iterable)
        fval = self._static_factor()
        if size is not None and fval is not None:
            # Statically-known length and factor: the bounds are
            # compile-time constants, so no `len`, no `fmod`, and empty
            # regions are dropped entirely.
            m = (size // fval) * fval
            if m > 0:
                emitted.append(self._chunk_loop(t, fval, m, stmt.target, body, stmt.loc))
            if m < size:
                emitted.append(self._residual_loop(t, m, size, stmt.target, body, stmt.loc))
        else:
            f = self.gensym.refresh(self.temp_id)
            n = self.gensym.refresh(self.temp_id)
            m_id = self.gensym.refresh(self.temp_id)
            emitted.append(self._dynamic_prelude(t, f, n, factor, [
                Assign(m_id, None, Sub(
                    Var(n, None),
                    self._rem(Var(n, None), Var(f, None)),
                    None
                ), None),
            ], stmt.loc))
            emitted.append(self._chunk_loop(t, f, m_id, stmt.target, body, stmt.loc))
            emitted.append(self._residual_loop(t, m_id, n, stmt.target, body, stmt.loc))

        return emitted

    def _build_mask(self, stmt: ForStmt, iterable: Expr, factor: Expr, body: StmtBlock) -> list[Stmt]:
        # MASK: chunk a length rounded up to a multiple of `f`, so every
        # chunk is full width, and guard the body so the over-run does
        # nothing.  One copy of the body, and the tail is a predicate.
        t = self.gensym.refresh(self.temp_id)
        emitted: list[Stmt] = [Assign(t, None, iterable, None)]   # ambient materialize

        size = static_size(self.array_size, stmt.iterable)
        fval = self._static_factor()
        if size is not None and fval is not None:
            # Statically-known length and factor: the padded bound is a
            # compile-time constant, and where `f` divides the length the
            # guard can never fail, so it is not emitted at all.
            padded = -(-size // fval) * fval
            if padded > 0:
                emitted.append(self._chunk_loop(
                    t, fval, padded, stmt.target, body, stmt.loc,
                    mask=None if size % fval == 0 else size,
                ))
        else:
            f = self.gensym.refresh(self.temp_id)
            n = self.gensym.refresh(self.temp_id)
            p_id = self.gensym.refresh(self.temp_id)
            # padded = n + rem(f - rem(n, f), f); every intermediate is an
            # exact integer, so nothing here depends on the rounding mode
            emitted.append(self._dynamic_prelude(t, f, n, factor, [
                Assign(p_id, None, Add(
                    Var(n, None),
                    self._rem(
                        Sub(Var(f, None), self._rem(Var(n, None), Var(f, None)), None),
                        Var(f, None),
                    ),
                    None
                ), None),
            ], stmt.loc))
            emitted.append(self._chunk_loop(
                t, f, p_id, stmt.target, body, stmt.loc, mask=n,
            ))

        return emitted

    def _visit_for(self, stmt: ForStmt, ctx: list[Stmt]) -> tuple[Stmt, None]:
        block, pos = self._site
        reason = self._refuses(stmt)
        if reason is not None:
            # a refusal is not a site, so it takes no index
            self.refused.append((stmt, reason))
            if self._target is not None and self._selects(block, pos, -1):
                self.declined.append(reason)
            return super()._visit_for(stmt, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects(block, pos, idx):
            return super()._visit_for(stmt, ctx)
        self._matched += 1
        if self.listing:
            self.found.append(StmtPath(self._paths[id(block)], pos))
            return super()._visit_for(stmt, ctx)

        iterable = self._visit_expr(stmt.iterable, ctx)
        factor = self._visit_expr(self.factor, ctx)
        body, _ = self._visit_block(stmt.body, ctx)

        match self.strategy:
            case SplitLoopStrategy.STRICT:
                emitted = self._build_strict(stmt, iterable, factor, body)
            case SplitLoopStrategy.PEEL:
                emitted = self._build_peel(stmt, iterable, factor, body)
            case SplitLoopStrategy.MASK:
                emitted = self._build_mask(stmt, iterable, factor, body)
            case _:
                raise RuntimeError(f'unknown strategy `{self.strategy}`')

        # The loop expands to several statements: emit all but the last
        # into the enclosing block and return the last as the replacement.
        self._replaced = True
        ctx.extend(emitted[:-1])
        return emitted[-1], None

    def apply(self):
        return self._visit_function(self.func, None)


def _lister(
    func: FuncDef, factor: 'Expr | None', strategy: SplitLoopStrategy
) -> '_SplitLoop':
    """The pass instance a listing walks `func` with.

    `use_fmod` is absent on purpose: it changes which node a remainder is
    spelled with, never whether a loop is a site."""
    return _SplitLoop(
        func,
        Integer(1, None) if factor is None else factor,
        None,
        strategy,
        ReachingDefs.analyze(func),
        NamedId('t'), NamedId('i'), NamedId('j'),
        infer_array_size(func),
    )


class SplitLoop:
    """
    Split loop transformation.

    This transformation rewrites a single ``for`` loop::

        for x1, ..., xk in xs:
            BODY[x1, ..., xk]

    into a nested loop (``PEEL``, the default)::

        t = xs
        with INTEGER:
            f = factor
            assert f >= 1
            n = len(t)
            m = n - fmod(n, f)
        for i in range(0, m, f):
            with INTEGER:
                hi = i + f
            for j in range(i, hi, 1):
                x1, ..., xk = t[j]
                BODY[x1, ..., xk]
        for j2 in range(m, n, 1):
            x1, ..., xk = t[j2]
            BODY[x1, ..., xk]

    ``STRICT`` instead chunks the whole length, guarded by a runtime
    ``assert fmod(n, f) == 0``, and emits no residual loop.  ``MASK`` chunks a
    length rounded *up* to a multiple of the factor and guards the body with
    ``j < n``, so every chunk is full width and the body is emitted once.

    When the array-size analysis proves the iterable's length and the factor is
    a literal, the remainder handling is resolved at compile time: no
    ``len``/``fmod``, empty regions are dropped, and ``MASK`` emits no guard
    where the factor divides the length.
    """

    @staticmethod
    def sites(
        func: FuncDef,
        within: Cursor | None = None,
        *,
        factor: Expr | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
    ) -> list[Cursor]:
        """The `for` loops of `func` this rewrite would split, in visit order --
        what a `where` index counts.

        Pass the `factor` and `strategy` the rewrite will get: only ``STRICT``
        refuses a loop, and only where the length and factor are both known and
        indivisible.  Under the default ``PEEL`` nothing is refused, so neither
        argument changes the answer.
        """
        return _lister(func, factor, strategy).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef,
        within: Cursor | None = None,
        *,
        factor: Expr | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
    ) -> list[tuple[Cursor, str]]:
        """Why each `for` loop of `func` that is not a site was refused."""
        return _lister(func, factor, strategy).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef,
        factor: Expr,
        where: int | Cursor | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
        reaching_defs: ReachingDefsAnalysis | None = None,
        array_size: ArraySizeAnalysis | None = None,
        temp_id: NamedId | None = None,
        outer_id: NamedId | None = None,
        inner_id: NamedId | None = None,
        use_fmod: bool = True
    ) -> FuncDef:
        """
        Apply the transformation.

        Parameters
        ----------
        factor : Expr
            The chunk size; a literal ``Integer`` must be positive, and
            any other expression is guarded by a runtime ``assert``.
        where : int | Cursor | None
            Which `for` loop to split: an index counting loops in visit
            order, or a cursor or region naming a program point, which
            takes every loop at or beneath it. If `None`, split every
            `for` loop.
        strategy : SplitLoopStrategy
            How to handle a length that is not a multiple of the factor
            (see :class:`SplitLoopStrategy`). Defaults to ``PEEL``,
            which is correct for any length; ``STRICT`` instead
            requires divisibility.
        reaching_defs : ReachingDefsAnalysis | None
            Pre-computed reaching-definitions analysis (for fresh names).
        array_size : ArraySizeAnalysis | None
            Pre-computed array-size analysis, used to discharge the
            remainder handling when an iterable's length is statically
            known.
        use_fmod : bool
            Spell the remainder with ``fp.fmod`` (the default) rather than
            ``%``.  The two agree on every value this emits, so the choice is
            which one the consuming backend can lower.
        """
        return SplitLoop.apply_with_edits(
            func,
            factor=factor,
            where=where,
            strategy=strategy,
            reaching_defs=reaching_defs,
            array_size=array_size,
            temp_id=temp_id,
            outer_id=outer_id,
            inner_id=inner_id,
            use_fmod=use_fmod,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        factor: Expr,
        where: int | Cursor | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
        reaching_defs: ReachingDefsAnalysis | None = None,
        array_size: ArraySizeAnalysis | None = None,
        temp_id: NamedId | None = None,
        outer_id: NamedId | None = None,
        inner_id: NamedId | None = None,
        use_fmod: bool = True
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        if not isinstance(factor, Expr):
            raise TypeError(f"Expected an \'Expr\' for factor, got {factor}")
        if isinstance(factor, Integer) and factor.val < 1:
            raise ValueError(f"Expected a positive factor, got {factor.val}")
        check_where(where)

        if reaching_defs is None:
            reaching_defs = ReachingDefs.analyze(func)
        if array_size is None:
            array_size = infer_array_size(func)
        if temp_id is None:
            temp_id = NamedId('t')
        if outer_id is None:
            outer_id = NamedId('i')
        if inner_id is None:
            inner_id = NamedId('j')

        vtor = _SplitLoop(
            func, factor, where, strategy, reaching_defs,
            temp_id, outer_id, inner_id, array_size, use_fmod
        )
        out = vtor.apply()
        # `site_idx` is the true loop count: generated loops are never re-visited
        vtor.check_site('a `for` loop')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)
