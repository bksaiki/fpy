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
    ListSize,
    ReachingDefs,
    ReachingDefsAnalysis,
    SyntaxCheck,
    concrete_size,
)
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..number import INTEGER
from ..utils import Gensym

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

def _sub(a: Expr, b: Expr) -> Sub:
    return Sub(a, b, None)

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

def _clone_block(block: StmtBlock) -> StmtBlock:
    """A structurally-fresh copy of *block*, so each unrolled body occupies
    distinct AST nodes (a plain transform visit rebuilds every node)."""
    block, _ = DefaultTransformVisitor()._visit_block(block, None)
    return block


class _ForUnroll(DefaultTransformVisitor):
    """
    Unroll visitor.

    ```
    for x in <iterable>:
        BODY[x]
    ```
    into (PEEL, unknown length; `k = times + 1`)
    ```
    t = <iterable>
    with INTEGER:
        n = len(t)
        m = n - fmod(n, k)
    for i in range(0, m, k):
        with INTEGER:            # offset indices grouped here
            i_1 = i + 1
            ...
            i_{k-1} = i + (k - 1)
        x = t[i];     BODY       # each read stays adjacent to its body
        x = t[i_1];   BODY
        ...
    for i in range(m, n):        # remainder (peeled straight-line if static)
        x = t[i]; BODY
    ```

    The loop target is *reassigned* per copy (not renamed) and each read sits
    immediately before its body: reordering reads ahead of bodies would be
    unsound if a body mutates the iterable in place.  STRICT is the same main
    loop over the whole (asserted-divisible) length, with no remainder.
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

    def _copy_target(self, target: Id | TupleBinding) -> Id | TupleBinding:
        """A fresh copy of a loop target with the *same* names.  ``Id``s are
        value-like and shared verbatim (as the base transform visitor does);
        a ``TupleBinding`` is rebuilt so no node is shared between the copies
        it appears in."""
        match target:
            case Id():
                return target
            case TupleBinding():
                return TupleBinding([self._copy_target(e) for e in target.elts], target.loc)
            case _:
                raise RuntimeError(f'Unexpected target {target}')

    def _static_size(self, iterable: Expr) -> int | None:
        """The statically-known length of *iterable* (the original AST node),
        or ``None`` if the array-size analysis could not pin it down."""
        if self.array_size is None:
            return None
        bound = self.array_size.by_expr.get(iterable)
        if isinstance(bound, ListSize):
            return concrete_size(bound.size)
        return None

    def _body_copy(
        self,
        target: Id | TupleBinding,
        t: NamedId,
        index: Expr,
        *,
        body: StmtBlock
    ) -> list[Stmt]:
        """One unrolled iteration: bind *target* to ``t[index]`` then run a
        fresh copy of *body*.  *index* must already be exact (a bare variable
        or a literal); any arithmetic offset is precomputed under the integer
        context by the caller, so no rounding occurs in the read itself.  The
        read must stay adjacent to its body — reordering reads ahead of bodies
        is unsound when a body mutates the iterable in place."""
        stmts: list[Stmt] = [_assign(self._copy_target(target), ListRef(_var(t), index, None))]
        stmts.extend(_clone_block(body).stmts)
        return stmts

    def _main_loop(
        self,
        t: NamedId,
        bound: Expr,
        k: int,
        target: Id | TupleBinding,
        body: StmtBlock,
        loc: Location | None
    ) -> ForStmt:
        """The unrolled loop over ``range(0, bound, k)`` consuming ``k``
        consecutive elements (``t[i]``, ``t[i+1]``, …) per iteration.

        The ``k - 1`` offset indices are computed together in a single
        exact-integer block, so the element reads use plain-variable indices
        (no per-read rounding context).  The reads stay interleaved with their
        bodies — see :meth:`_body_copy`."""
        idx = self.gensym.refresh(self.idx_id)

        # group the index arithmetic: off_j = idx + j, all under `with INTEGER`
        offsets = [self.gensym.refresh(self.idx_id) for _ in range(1, k)]
        offset_defs = [_assign(off, _add(_var(idx), _int(j + 1))) for j, off in enumerate(offsets)]

        main_body: list[Stmt] = []
        if offset_defs:
            main_body.append(_integer_ctx(offset_defs, loc))
        # `idx` (straight from `range`) and each `off_j` are exact integers
        for index in [_var(idx)] + [_var(off) for off in offsets]:
            main_body.extend(self._body_copy(target, t, index, body=body))

        return ForStmt(idx, _range(_int(0), bound, _int(k)), StmtBlock(main_body), loc)

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx) -> tuple[Stmt, None]:
        if (self.where is None or self.index == self.where) and self.times > 0:
            self.index += 1
            # ``k`` consecutive elements are consumed per iteration of the
            # rewritten loop (``times`` extra copies on top of the original).
            k = self.times + 1
            iterable = self._visit_expr(stmt.iterable, ctx)
            body, _ = self._visit_block(stmt.body, ctx)

            match self.strategy:
                case ForUnrollStrategy.STRICT:
                    emitted = self._build_strict(stmt, iterable, body, k)
                case ForUnrollStrategy.PEEL:
                    emitted = self._build_peel(stmt, iterable, body, k)
                case _:
                    raise RuntimeError(f'unknown strategy `{self.strategy}`')

            # The loop expands to several statements: emit all but the last
            # into the enclosing block and return the last as the replacement.
            for s in emitted[:-1]:
                ctx.stmts.append(s)
            return emitted[-1], None
        else:
            self.index += 1
            return super()._visit_for(stmt, ctx)

    def _build_strict(self, stmt: ForStmt, iterable: Expr, body: StmtBlock, k: int) -> list[Stmt]:
        # STRICT: the length must be an exact multiple of `k`, so the whole
        # loop is the main region with no remainder.  Reuses the shared,
        # interleaved `_main_loop` (reads stay adjacent to their bodies).
        t = self.gensym.refresh(self.temp_id)
        emitted: list[Stmt] = [_assign(t, iterable)]   # ambient materialize

        size = self._static_size(stmt.iterable)
        if size is not None:
            # Statically known length: verify divisibility at compile time and
            # drop the runtime `len`/`assert` entirely.
            if size % k != 0:
                raise ValueError(
                    f'STRICT unroll by {k} requires the iterable length to be '
                    f'a multiple of {k}, but its statically-known length is {size}'
                )
            if size > 0:
                emitted.append(self._main_loop(t, _int(size), k, stmt.target, body, stmt.loc))
        else:
            # Unknown length: assert `len(t) % k == 0` at runtime.
            n = self.gensym.refresh(self.len_id)
            emitted.append(_integer_ctx([
                _assign(n, _len(_var(t))),
                AssertStmt(_eq(_fmod(_var(n), _int(k)), _int(0)), None, None),
            ], stmt.loc))
            emitted.append(self._main_loop(t, _var(n), k, stmt.target, body, stmt.loc))

        return emitted

    def _build_peel(self, stmt: ForStmt, iterable: Expr, body: StmtBlock, k: int) -> list[Stmt]:
        # Unroll the largest multiple-of-``k`` prefix ``[0, m)`` and run the
        # remaining ``[m, n)`` elements separately.  Correct for any length.
        #
        # Materialize the iterable under the *ambient* context (its elements
        # must round as they did in the original loop); everything downstream
        # indexes the temporary ``t``.
        t = self.gensym.refresh(self.temp_id)
        emitted: list[Stmt] = [_assign(t, iterable)]

        size = self._static_size(stmt.iterable)
        if size is not None:
            # Statically-known length: the bound and remainder indices are
            # compile-time constants, so no `len`, no `fmod`, and the leftover
            # is peeled straight-line (no residual loop).
            m = (size // k) * k
            if m > 0:
                emitted.append(self._main_loop(t, _int(m), k, stmt.target, body, stmt.loc))
            for p in range(m, size):
                # literal index: exact by construction, no integer context
                emitted.extend(
                    self._body_copy(stmt.target, t, _int(p), body=body)
                )
        else:
            # Unknown length: compute the main-region bound `m = n - n % k`
            # under the exact integer context and run a residual loop.
            n = self.gensym.refresh(self.len_id)
            m = self.gensym.fresh('m')
            emitted.append(_integer_ctx([
                _assign(n, _len(_var(t))),
                _assign(m, _sub(_var(n), _fmod(_var(n), _int(k)))),
            ], stmt.loc))
            emitted.append(self._main_loop(t, _var(m), k, stmt.target, body, stmt.loc))

            rem_idx = self.gensym.refresh(self.idx_id)
            rem_body = self._body_copy(
                stmt.target, t, _var(rem_idx), body=body
            )
            emitted.append(ForStmt(
                rem_idx, _range(_var(m), _var(n), _int(1)), StmtBlock(rem_body), stmt.loc
            ))

        return emitted

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
        strategy: ForUnrollStrategy = ForUnrollStrategy.PEEL,
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
