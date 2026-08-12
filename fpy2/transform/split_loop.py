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

import dataclasses
import enum

from ..analysis import ReachingDefs, ReachingDefsAnalysis, SyntaxCheck
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import INTEGER
from ..utils import Gensym


class SplitLoopStrategy(enum.Enum):
    """Strategy for dealing with the loop remainder."""

    STRICT = 0
    """Require the length to be an exact multiple of the factor:
    guarded by a runtime ``assert fmod(len(t), f) == 0``."""

    PEEL = 1
    """Split the largest multiple-of-factor prefix into chunks, then
    run the remaining ``len % f`` iterations in a residual loop.
    Correct for any length."""


@dataclasses.dataclass
class _Ctx:
    stmts: list[Stmt]

    @staticmethod
    def default():
        return _Ctx(stmts=[])


def _copy_target(target: Id | TupleBinding) -> Id | TupleBinding:
    """A fresh copy of a loop target with the *same* names.  ``Id``s are
    value-like and shared verbatim; a ``TupleBinding`` is rebuilt so no
    node is shared between the copies it appears in."""
    match target:
        case Id():
            return target
        case TupleBinding():
            return TupleBinding([_copy_target(e) for e in target.elts], target.loc)
        case _:
            raise RuntimeError(f'Unexpected target {target}')


def _clone_block(block: StmtBlock) -> StmtBlock:
    """A structurally-fresh copy of *block*, so the main and residual
    loops occupy distinct AST nodes (a plain transform visit rebuilds
    every node)."""
    block, _ = DefaultTransformVisitor()._visit_block(block, None)
    return block


class _SplitLoop(DefaultTransformVisitor):
    """
    Split loop visitor.
    """

    func: FuncDef
    factor: Expr
    where: int | None
    strategy: SplitLoopStrategy
    tmp_id: NamedId
    outer_id: NamedId
    inner_id: NamedId

    gensym: Gensym
    index: int

    def __init__(
        self,
        func: FuncDef,
        factor: Expr,
        where: int | None,
        strategy: SplitLoopStrategy,
        reaching_defs: ReachingDefsAnalysis,
        tmp_id: NamedId,
        outer_id: NamedId,
        inner_id: NamedId
    ):
        super().__init__()
        self.func = func
        self.factor = factor
        self.where = where
        self.strategy = strategy
        self.tmp_id = tmp_id
        self.outer_id = outer_id
        self.inner_id = inner_id

        self.gensym = Gensym(reaching_defs.names())
        self.index = 0

    def _integer_ctx(self, stmts: list[Stmt], loc: Location | None) -> ContextStmt:
        return ContextStmt(UnderscoreId(), ForeignVal(INTEGER, None), StmtBlock(stmts), loc)

    def _chunk_loop(
        self,
        t: NamedId,
        f: NamedId,
        bound: NamedId,
        target: Id | TupleBinding,
        body: StmtBlock,
        loc: Location | None
    ) -> ForStmt:
        """The chunked loop over ``range(0, bound, f)``; each chunk runs
        an inner loop whose target is reassigned per element, so every
        read stays adjacent to its body (see the module docstring)."""
        outer = self.gensym.refresh(self.outer_id)
        inner = self.gensym.refresh(self.inner_id)
        hi = self.gensym.refresh(self.tmp_id)

        # the chunk bound `i + f` must be exact, so it is precomputed
        # under `INTEGER` rather than inlined into the `range`
        hi_bind = self._integer_ctx([
            Assign(hi, None, Add(Var(outer, None), Var(f, None), None), None)
        ], None)

        inner_body = StmtBlock([
            Assign(target, None, ListRef(Var(t, None), Var(inner, None), None), None),
            *body.stmts
        ])
        inner_loop = ForStmt(
            inner,
            Range3(None, Var(outer, None), Var(hi, None), Integer(1, None), None),
            inner_body,
            loc
        )

        return ForStmt(
            outer,
            Range3(None, Integer(0, None), Var(bound, None), Var(f, None), None),
            StmtBlock([hi_bind, inner_loop]),
            loc
        )

    def _build_strict(self, stmt: ForStmt, iterable: Expr, factor: Expr, body: StmtBlock) -> list[Stmt]:
        # STRICT: the length must be an exact multiple of `f`, so the
        # chunked loop covers the whole (asserted-divisible) length.
        t = self.gensym.refresh(self.tmp_id)
        f = self.gensym.refresh(self.tmp_id)
        n = self.gensym.refresh(self.tmp_id)

        return [
            Assign(t, None, iterable, None),   # ambient materialize
            self._integer_ctx([
                Assign(f, None, factor, None),
                Assign(n, None, Len(None, Var(t, None), None), None),
                AssertStmt(
                    Compare(
                        [CompareOp.EQ],
                        [
                            Fmod(None, Var(n, None), Var(f, None), None),
                            Integer(0, None)
                        ],
                        None
                    ),
                    None,
                    None
                ),
            ], stmt.loc),
            self._chunk_loop(t, f, n, stmt.target, body, stmt.loc),
        ]

    def _build_peel(self, stmt: ForStmt, iterable: Expr, factor: Expr, body: StmtBlock) -> list[Stmt]:
        # PEEL: chunk the `[0, m)` prefix (largest multiple of `f`) and
        # run the `[m, n)` remainder in a residual loop.  Correct for
        # any length.
        t = self.gensym.refresh(self.tmp_id)
        f = self.gensym.refresh(self.tmp_id)
        n = self.gensym.refresh(self.tmp_id)
        m = self.gensym.refresh(self.tmp_id)

        rem = self.gensym.refresh(self.inner_id)
        rem_body = StmtBlock([
            Assign(
                _copy_target(stmt.target), None,
                ListRef(Var(t, None), Var(rem, None), None), None
            ),
            *_clone_block(body).stmts
        ])

        return [
            Assign(t, None, iterable, None),   # ambient materialize
            self._integer_ctx([
                Assign(f, None, factor, None),
                Assign(n, None, Len(None, Var(t, None), None), None),
                Assign(m, None, Sub(
                    Var(n, None),
                    Fmod(None, Var(n, None), Var(f, None), None),
                    None
                ), None),
            ], stmt.loc),
            self._chunk_loop(t, f, m, stmt.target, body, stmt.loc),
            ForStmt(
                rem,
                Range3(None, Var(m, None), Var(n, None), Integer(1, None), None),
                rem_body,
                stmt.loc
            ),
        ]

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx) -> tuple[Stmt, None]:
        selected = self.where is None or self.index == self.where
        self.index += 1
        if not selected:
            return super()._visit_for(stmt, ctx)

        iterable = self._visit_expr(stmt.iterable, ctx)
        factor = self._visit_expr(self.factor, ctx)
        body, _ = self._visit_block(stmt.body, ctx)

        match self.strategy:
            case SplitLoopStrategy.STRICT:
                emitted = self._build_strict(stmt, iterable, factor, body)
            case SplitLoopStrategy.PEEL:
                emitted = self._build_peel(stmt, iterable, factor, body)
            case _:
                raise RuntimeError(f'unknown strategy `{self.strategy}`')

        # The loop expands to several statements: emit all but the last
        # into the enclosing block and return the last as the replacement.
        ctx.stmts.extend(emitted[:-1])
        return emitted[-1], None

    def _visit_block(self, block: StmtBlock, ctx: _Ctx | None):
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(stmt)
        b = StmtBlock(block_ctx.stmts)
        return b, None

    def apply(self):
        return self._visit_function(self.func, None)


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
    ``assert fmod(n, f) == 0``, and emits no residual loop.
    """

    @staticmethod
    def apply(
        func: FuncDef,
        factor: Expr,
        where: int | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.PEEL,
        reaching_defs: ReachingDefsAnalysis | None = None,
        tmp_id: NamedId | None = None,
        outer_id: NamedId | None = None,
        inner_id: NamedId | None = None
    ) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        if not isinstance(factor, Expr):
            raise TypeError(f"Expected an \'Expr\' for factor, got {factor}")
        if where is not None and not isinstance(where, int):
            raise TypeError(f"Expected an \'int\' or None for where, got {where}")

        if reaching_defs is None:
            reaching_defs = ReachingDefs.analyze(func)
        if tmp_id is None:
            tmp_id = NamedId('t')
        if outer_id is None:
            outer_id = NamedId('i')
        if inner_id is None:
            inner_id = NamedId('j')

        vtor = _SplitLoop(func, factor, where, strategy, reaching_defs, tmp_id, outer_id, inner_id)
        func = vtor.apply()
        # A `where` that named no loop leaves the function unchanged;
        # fail rather than silently no-op.  `index` is the true loop
        # count: generated loops are never re-visited.
        if where is not None and not (0 <= where < vtor.index):
            raise ValueError(
                f'where={where} does not correspond to a `for` loop; '
                f'the function has {vtor.index} `for` loop(s)'
            )
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
