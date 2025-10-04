"""
Unroller for `for` loops.
"""

import dataclasses
import enum

from ..analysis import ReachingDefs, ReachingDefsAnalysis, SyntaxCheck
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import INTEGER
from ..utils import Gensym

from .rename_target import RenameTarget

class ForUnrollStrategy(enum.Enum):
    """Strategy for dealing with the loop remainder."""

    STRICT = 0
    """Asserts that the loop can be unrolled without remainder."""

@dataclasses.dataclass
class _Ctx:
    stmts: list[Stmt]

    @staticmethod
    def default():
        return _Ctx(stmts=[])


class _ForUnroll(DefaultTransformVisitor):
    """
    Unroll visitor.

    ```
    for x in <iterable>:
        BODY[x]
    ```
    into
    ```
    with Z:
        t = <iterable>
        n = len(t)
    for i in range(0, n, k):
        with Z:
            x_1 = t[i]
            ...
            x_k = t[i + k - 1]
        BODY[x_1 -> x]
        ...
        BODY[x_k -> x]
    ```
    """

    func: FuncDef
    where: int | None
    times: int
    index: int
    strategy: ForUnrollStrategy
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        where: int | None,
        times: int,
        strategy: ForUnrollStrategy,
        reaching_defs: ReachingDefsAnalysis
    ):
        super().__init__()
        self.func = func
        self.where = where
        self.times = times
        self.index = 0
        self.strategy = strategy
        self.gensym = Gensym(reaching_defs.names())

    def _refresh(self, target: Id | TupleBinding) -> tuple[Id | TupleBinding, dict[NamedId, NamedId]]:
        match target:
            case UnderscoreId():
                return target, {}
            case NamedId():
                new_target = self.gensym.refresh(target)
                return new_target, {target: new_target}
            case TupleBinding():
                subst: dict[NamedId, NamedId] = {}
                new_elts: list[Id | TupleBinding] = []
                for elt in target.elts:
                    new_elt, elt_subst = self._refresh(elt)
                    new_elts.append(new_elt)
                    subst |= elt_subst
                return TupleBinding(new_elts, target.loc), subst
            case _:
                raise RuntimeError(f'Unexpected target {target}')

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx) -> tuple[Stmt, None]:
        if self.where is None or self.index == self.where:
            self.index += 1
            iterable = self._visit_expr(stmt.iterable, ctx)
            body, _ = self._visit_block(stmt.body, ctx)

            t = self.gensym.fresh('t')
            n = self.gensym.fresh('n')
            idx = self.gensym.fresh('i')
            match self.strategy:
                case ForUnrollStrategy.STRICT:
                    # need to check that len(t) % times == 0
                    ctx.stmts.append(ContextStmt(
                        UnderscoreId(),
                        ForeignVal(INTEGER, None),
                        StmtBlock([
                            Assign(t, None, iterable, None),
                            Assign(n, None, Len(Var(NamedId('len'), None), Var(t, None), None), None),
                            AssertStmt(
                                Compare(
                                    [CompareOp.EQ], 
                                    [
                                        Fmod(Var(NamedId('fmod'), None), Var(n, None), Integer(self.times, None), None),
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

            assign_stmts: list[Stmt] = []
            body_stmts: list[Stmt] = []
            for i in range(self.times):
                target, subst = self._refresh(stmt.target)
                renamed_body = RenameTarget.apply_block(body, subst)
                body_stmts.extend(renamed_body.stmts)
                assign_stmts.append(Assign(
                    target,
                    None,
                    ListRef(
                        Var(t, None),
                        Var(idx, None) if i == 0 else Add(Var(idx, None), Integer(i, None), None),
                        None
                    ),
                    None
                ))

            unpack_stmt = ContextStmt(
                UnderscoreId(),
                ForeignVal(INTEGER, None),
                StmtBlock(assign_stmts),
                stmt.loc
            )

            stmt = ForStmt(
                idx,
                Range3(Var(NamedId('range'), None), Integer(0, None), Var(n, None), Integer(self.times, None), None),
                StmtBlock([unpack_stmt] + body_stmts),
                stmt.loc
            )

            return stmt, None
        else:
            self.index += 1
            return super()._visit_for(stmt, ctx)

    def _visit_block(self, block: StmtBlock, ctx: _Ctx | None):
        block_ctx = _Ctx.default()
        for stmt in block.stmts:
            if isinstance(stmt, ForStmt):
                stmt, _ = self._visit_for(stmt, block_ctx)
                if isinstance(stmt, PassStmt):
                    continue
                block_ctx.stmts.append(stmt)
            else:
                stmt, _ = self._visit_statement(stmt, block_ctx)
                block_ctx.stmts.append(stmt)

        b = StmtBlock(block_ctx.stmts)
        return b, None

    def apply(self):
        return self._visit_function(self.func, None)


class ForUnroll:
    """
    Unrolling for `for` loops.
    """

    @staticmethod
    def apply(
        func: FuncDef,
        where: int | None = None,
        times: int = 1,
        strategy: ForUnrollStrategy = ForUnrollStrategy.STRICT,
        reaching_defs: ReachingDefsAnalysis | None = None
    ):
        """
        Apply the transformation.

        Parameters
        ----------
        where : int | None
            The index of the `while` loop to unroll. If `None`, unroll all
            `while` loops.
        times : int
            The number of times to unroll the loop.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        if not isinstance(times, int):
            raise TypeError(f"Expected an \'int\' for times, got {times}")
        if times < 1:
            raise ValueError(f"Expected a positive integer for times, got {times}")

        if reaching_defs is None:
            reaching_defs = ReachingDefs.analyze(func)

        unroller = _ForUnroll(func, where, times, strategy, reaching_defs)
        func = unroller.apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
