"""
Loop splitting transformation.

This transformation is inspired by the `split()` procedure
from Halide (https://halide-lang.org/).
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
    """Asserts that the loop can be split without remainder."""



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
    len_id: NamedId
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

        self.gensym = Gensym()
        self.index = 0

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx) -> tuple[Stmt, None]:
        if self.where is None or self.index == self.where:
            self.index += 1

            t1 = self.gensym.refresh(self.tmp_id)
            t2 = self.gensym.refresh(self.tmp_id)
            n = self.gensym.refresh(self.tmp_id)
            outer = self.gensym.refresh(self.outer_id)
            inner = self.gensym.refresh(self.inner_id)

            iterable = self._visit_expr(stmt.iterable, ctx)
            factor = self._visit_expr(self.factor, ctx)
            body, _ = self._visit_block(stmt.body, ctx)

            # bind the iterable and `factor`
            ctx.stmts.append(Assign(t1, None, iterable, None))
            ctx.stmts.append(Assign(t2, None, factor, None))

            # check the length divides evenly
            match self.strategy:
                case SplitLoopStrategy.STRICT:
                    # need to check that len(t) % factor == 0
                    ctx.stmts.append(ContextStmt(
                        UnderscoreId(),
                        ForeignVal(INTEGER, None),
                        StmtBlock([
                            Assign(n, None, Len(Var(NamedId('len'), None), Var(t1, None), None), None),
                            AssertStmt(
                                Compare(
                                    [CompareOp.EQ], 
                                    [
                                        Fmod(Var(NamedId('fmod'), None), Var(n, None), Var(t2, None), None),
                                        Integer(0, None)
                                    ],
                                    None
                                ),
                                None,
                                None
                            )
                        ]),
                        stmt.loc
                    ))
                case _:
                    raise RuntimeError(f'unknown strategy `{self.strategy}`')

            # inner loop
            inner_loop = ForStmt(stmt.target, Var(inner, None), body, stmt.loc)

            # slice context
            slice_ctx = ContextStmt(
                UnderscoreId(),
                ForeignVal(INTEGER, None),
                StmtBlock([
                    Assign(
                        inner, None,
                        ListSlice(
                            Var(t1, None),
                            Var(outer, None),
                            Add(Var(outer, None), Var(t2, None), None),
                            None
                        ),
                        None
                    )
                ]),
                None
            )

            # outer loop
            stmt = ForStmt(
                outer,
                Range3(Var(NamedId('range'), None), Integer(0, None), Var(n, None), Var(t2, None), None),
                StmtBlock([slice_ctx, inner_loop]),
                stmt.loc
            )

            return stmt, None
        else:
            self.index += 1
            return super()._visit_for(stmt, ctx)

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

    This transformation rewrites a single `for` loop:
    ```
    for x1, ..., xk in xs:
        BODY[x1, ..., xk ]
    ```
    into a nested loop:
    ```
    t1 = xs
    t2 = factor
    with Z:
        n = len(t1)
        assert n % t2 == 0
    for i in range(0, n, t2):
        with Z:
            slice = t1[i:i+t2]
        for x1, ..., xk in slice:
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

        if reaching_defs is None:
            reaching_defs = ReachingDefs.analyze(func)
        if tmp_id is None:
            tmp_id = NamedId('t')
        if outer_id is None:
            outer_id = NamedId('t', None)
        if inner_id is None:
            inner_id = NamedId('t', None)

        vtor = _SplitLoop(func, factor, where, strategy, tmp_id, outer_id, inner_id)
        func = vtor.apply()

        SyntaxCheck.check(func, ignore_unknown=True)
        return func
