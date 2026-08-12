"""
Loop splitting transformation.

This transformation is inspired by the `split()` procedure
from Halide (https://halide-lang.org/).

Rounding-context safety
-----------------------
Splitting introduces loop-control and index arithmetic (``len(t)``, the
divisibility check ``fmod(n, f)``, the chunk bound ``i + f``).  These are
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
    """Asserts at runtime that the loop splits without remainder
    (``fmod(len(t), factor) == 0``)."""


@dataclasses.dataclass
class _Ctx:
    stmts: list[Stmt]

    @staticmethod
    def default():
        return _Ctx(stmts=[])


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

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx) -> tuple[Stmt, None]:
        selected = self.where is None or self.index == self.where
        self.index += 1
        if not selected:
            return super()._visit_for(stmt, ctx)

        iterable = self._visit_expr(stmt.iterable, ctx)
        factor = self._visit_expr(self.factor, ctx)
        body, _ = self._visit_block(stmt.body, ctx)

        t = self.gensym.refresh(self.tmp_id)
        f = self.gensym.refresh(self.tmp_id)
        n = self.gensym.refresh(self.tmp_id)
        outer = self.gensym.refresh(self.outer_id)
        inner = self.gensym.refresh(self.inner_id)
        hi = self.gensym.refresh(self.tmp_id)

        # materialize the iterable under the ambient context
        ctx.stmts.append(Assign(t, None, iterable, None))

        # loop-control values: the factor binding sits under `INTEGER`
        # so an arithmetic factor expression is never rounded by the
        # ambient context (a `Var`/`Integer` factor is unaffected)
        prelude: list[Stmt] = [
            Assign(f, None, factor, None),
            Assign(n, None, Len(None, Var(t, None), None), None),
        ]
        match self.strategy:
            case SplitLoopStrategy.STRICT:
                prelude.append(AssertStmt(
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
                ))
            case _:
                raise RuntimeError(f'unknown strategy `{self.strategy}`')
        ctx.stmts.append(ContextStmt(
            UnderscoreId(),
            ForeignVal(INTEGER, None),
            StmtBlock(prelude),
            stmt.loc
        ))

        # inner loop: the target is reassigned per element so each read
        # stays adjacent to its body (see the module docstring)
        inner_body = StmtBlock([
            Assign(stmt.target, None, ListRef(Var(t, None), Var(inner, None), None), None),
            *body.stmts
        ])
        inner_loop = ForStmt(
            inner,
            Range3(None, Var(outer, None), Var(hi, None), Integer(1, None), None),
            inner_body,
            stmt.loc
        )

        # the chunk bound `i + f` must be exact, so it is precomputed
        # under `INTEGER` rather than inlined into the `range`
        hi_bind = ContextStmt(
            UnderscoreId(),
            ForeignVal(INTEGER, None),
            StmtBlock([
                Assign(hi, None, Add(Var(outer, None), Var(f, None), None), None)
            ]),
            None
        )

        outer_loop = ForStmt(
            outer,
            Range3(None, Integer(0, None), Var(n, None), Var(f, None), None),
            StmtBlock([hi_bind, inner_loop]),
            stmt.loc
        )
        return outer_loop, None

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

    into a nested loop::

        t = xs
        with INTEGER:
            f = factor
            n = len(t)
            assert fmod(n, f) == 0
        for i in range(0, n, f):
            with INTEGER:
                hi = i + f
            for j in range(i, hi, 1):
                x1, ..., xk = t[j]
                BODY[x1, ..., xk]
    """

    @staticmethod
    def apply(
        func: FuncDef,
        factor: Expr,
        where: int | None = None,
        strategy: SplitLoopStrategy = SplitLoopStrategy.STRICT,
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
