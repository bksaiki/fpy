"""
Unroller for `for` loops.

Rounding-context safety
-----------------------
Unrolling introduces loop-control and index arithmetic (``len(t)``, the
remainder ``n % k``, index offsets ``t[i + j]``).  These are *integer*
computations, but FPy rounds every arithmetic operation under the active
context (**E-Add**): under a low-precision float context an offset or bound
could round to the wrong value and read out of bounds.  Every such inserted
computation is therefore wrapped in ``with fp.INTEGER:`` (see
:func:`_integer_ctx`).  The iterable and the loop body, by contrast, run
under the *ambient* context so their rounding is unchanged.
"""

import dataclasses
import enum

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ReachingDefs,
    ReachingDefsAnalysis,
    SyntaxCheck,
)
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import INTEGER
from ..utils import Gensym

from .rename_target import RenameTarget

class ForUnrollStrategy(enum.Enum):
    """Strategy for dealing with the loop remainder."""

    STRICT = 0
    """Asserts that the loop can be unrolled without remainder."""

    PEEL = 1
    """Unroll the largest multiple-of-``k`` prefix, then handle the
    remaining ``len % k`` iterations separately (a residual loop, or —
    when the length is statically known — straight-line peeled copies).
    Correct for any length."""


@dataclasses.dataclass
class _Ctx:
    stmts: list[Stmt]

    @staticmethod
    def default():
        return _Ctx(stmts=[])


#####################################################################
# AST construction helpers
#
# Small builders for the nodes the unroller emits, so the transformation
# reads close to the FPy surface syntax it produces.

def _var(name: NamedId) -> Var:
    return Var(name, None)

def _int(value: int) -> Integer:
    return Integer(value, None)

def _add(a: Expr, b: Expr) -> Add:
    return Add(a, b, None)

def _len(xs: Expr) -> Len:
    return Len(_var(NamedId('len')), xs, None)

def _fmod(a: Expr, b: Expr) -> Fmod:
    return Fmod(_var(NamedId('fmod')), a, b, None)

def _range(start: Expr, stop: Expr, step: Expr) -> Range3:
    return Range3(_var(NamedId('range')), start, stop, step, None)

def _eq(a: Expr, b: Expr) -> Compare:
    return Compare([CompareOp.EQ], [a, b], None)

def _assign(target: Id | TupleBinding, expr: Expr) -> Assign:
    return Assign(target, None, expr, None)

def _integer_ctx(stmts: list[Stmt], loc: Location | None) -> ContextStmt:
    """A ``with fp.INTEGER:`` block: the exact integer context under which
    loop-control and index arithmetic must be evaluated (see the module
    docstring on rounding-context safety)."""
    return ContextStmt(UnderscoreId(), ForeignVal(INTEGER, None), StmtBlock(stmts), loc)


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
    temp_id: NamedId
    len_id: NamedId
    idx_id: NamedId
    # Static list sizes of iterables; enables discharging the remainder
    # check at compile time (used by the static-size specialization).
    array_size: ArraySizeAnalysis | None

    def __init__(
        self,
        func: FuncDef,
        where: int | None,
        times: int,
        strategy: ForUnrollStrategy,
        reaching_defs: ReachingDefsAnalysis,
        temp_id: NamedId,
        len_id: NamedId,
        idx_id: NamedId,
        array_size: ArraySizeAnalysis | None
    ):
        super().__init__()
        self.func = func
        self.where = where
        self.times = times
        self.index = 0
        self.strategy = strategy
        self.gensym = Gensym(reaching_defs.names())
        self.temp_id = temp_id
        self.len_id = len_id
        self.idx_id = idx_id
        self.array_size = array_size

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
        # ``k`` consecutive elements are consumed per iteration of the
        # rewritten loop (``times`` extra copies on top of the original).
        k = self.times + 1
        if (self.where is None or self.index == self.where) and self.times > 0:
            self.index += 1
            iterable = self._visit_expr(stmt.iterable, ctx)
            body, _ = self._visit_block(stmt.body, ctx)

            t = self.gensym.refresh(self.temp_id)
            n = self.gensym.refresh(self.len_id)
            idx = self.gensym.refresh(self.idx_id)
            match self.strategy:
                case ForUnrollStrategy.STRICT:
                    # need to check that len(t) % k == 0
                    ctx.stmts.append(_integer_ctx([
                        _assign(t, iterable),
                        _assign(n, _len(_var(t))),
                        AssertStmt(_eq(_fmod(_var(n), _int(k)), _int(0)), None, None),
                    ], stmt.loc))
                case ForUnrollStrategy.PEEL:
                    raise NotImplementedError('PEEL strategy is not yet implemented')
                case _:
                    raise RuntimeError(f'unknown strategy `{self.strategy}`')

            # original iteration uses (target, body) as is
            body_stmts: list[Stmt] = list(body.stmts)
            assign_stmts: list[Stmt] = [
                _assign(stmt.target, ListRef(_var(t), _var(idx), None))
            ]

            for i in range(self.times):
                # unrolled iteration uses (target, body) with target renamed
                target, subst = self._refresh(stmt.target)
                renamed_body = RenameTarget.apply_block(body, subst)
                body_stmts.extend(renamed_body.stmts)
                assign_stmts.append(_assign(
                    target,
                    ListRef(_var(t), _add(_var(idx), _int(i + 1)), None)
                ))

            unpack_stmt = _integer_ctx(assign_stmts, stmt.loc)

            stmt = ForStmt(
                idx,
                _range(_int(0), _var(n), _int(k)),
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
        reaching_defs: ReachingDefsAnalysis | None = None,
        array_size: ArraySizeAnalysis | None = None,
        temp_id: NamedId | None = None,
        len_id: NamedId | None = None,
        idx_id: NamedId | None = None
    ):
        """
        Apply the transformation.

        Parameters
        ----------
        where : int | None
            The index of the `for` loop to unroll. If `None`, unroll all
            `for` loops.
        times : int
            The number of times to unroll the loop.
        strategy : ForUnrollStrategy
            How to handle a length that is not a multiple of the unroll
            factor (see :class:`ForUnrollStrategy`).
        reaching_defs : ReachingDefsAnalysis | None
            Pre-computed reaching-definitions analysis (for fresh names).
        array_size : ArraySizeAnalysis | None
            Pre-computed array-size analysis, used to discharge the
            remainder check when an iterable's length is statically known.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a \'FuncDef\', got {func}")
        if not isinstance(times, int):
            raise TypeError(f"Expected an \'int\' for times, got {times}")
        if times < 0:
            raise ValueError(f"Expected a non-negative integer for times, got {times}")

        if reaching_defs is None:
            reaching_defs = ReachingDefs.analyze(func)
        if array_size is None:
            # Auxiliary: a failed size analysis only disables the static
            # optimization, so never let it break the transformation.
            try:
                array_size = ArraySizeInfer.analyze(func)
            except Exception:
                array_size = None
        if temp_id is None:
            temp_id = NamedId('t')
        if len_id is None:
            len_id = NamedId('n')
        if idx_id is None:
            idx_id = NamedId('i')

        unroller = _ForUnroll(
            func, where, times, strategy, reaching_defs,
            temp_id, len_id, idx_id, array_size
        )
        func = unroller.apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
