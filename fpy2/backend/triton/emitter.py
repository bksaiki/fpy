"""
Triton backend: emitting kernel source.

This module holds the scalar half -- expressions, and the straight-line
statements that bind them.  Loops, memory and the kernel wrapper come later.

**The cast discipline is the point.**  Triton types `fp16 op fp16` as fp16, so
spelling a product of two fp16 operands as `x * y` computes it *in fp16* and
rounds it.  Where the program says the product is exact, that is a different
operation, and it was measured differing on 2000 of 2000 sampled inputs.  The
casts do not come from Triton, which asks for nothing; they come from the
storage the pipeline chose.  So an operand is cast *into* the storage its
signature wants before the operation, never after it.

A cast that would lose information is refused rather than emitted.  The op
table is the other half of that: every omission in it is a refusal, so an
operation with no signature under the active context is an error, not a
fallback.
"""

from collections.abc import Sequence

from ...analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    FormatAnalysis,
    FormatInfer,
)
from ...analysis.array_size import trip_count
from ...analysis.format_infer import rounds_exactly
from ...ast import (
    Assign,
    BinaryOp,
    BoolVal,
    Cast,
    ContextStmt,
    Decnum,
    Expr,
    ForStmt,
    FuncDef,
    Hexnum,
    Id,
    IfExpr,
    Integer,
    NamedId,
    Rational,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    TernaryOp,
    UnaryOp,
    Var,
)
from ...number import REAL, Context
from ..backend import CompileError
from .storage import choose_storage_scalar, scalar_fits_in
from .target import ScalarOpTable, TritonOp, is_native_ctx, make_op_table
from .types import TritonScalar

__all__ = ['TritonEmitError', 'emit_block', 'emit_expr']


class TritonEmitError(CompileError):
    """A program this backend declines to emit."""


class _IndentedWriter:
    """Line-oriented Triton source builder."""

    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0

    def add_line(self, line: str = ''):
        self._lines.append('    ' * self._depth + line if line else '')

    def indent(self):
        self._depth += 1

    def dedent(self):
        self._depth -= 1

    def render(self) -> str:
        return '\n'.join(self._lines)


class _ExprEmitter:
    """Emits one scalar expression."""

    def __init__(
        self,
        func: FuncDef,
        format_info: FormatAnalysis,
        ctx_use: ContextUseAnalysis,
        op_table: ScalarOpTable,
    ):
        self.func = func
        self.format_info = format_info
        self.ctx_use = ctx_use
        self.op_table = op_table

    # -- storage and context -------------------------------------------

    def _storage(self, e: Expr) -> TritonScalar:
        """The scalar storage the pipeline chose for *e*."""
        bound = self.format_info.by_expr.get(e)
        if bound is None:
            raise TritonEmitError(
                f'no inferred format for `{type(e).__name__}`, so its storage '
                'is unknown'
            )
        return choose_storage_scalar(bound)

    def _active_ctx(self, e: Expr) -> Context:
        """The rounding context *e* is evaluated under."""
        try:
            scope = self.ctx_use.find_scope_from_use(e)
        except KeyError:
            raise TritonEmitError(
                f'the context of `{type(e).__name__}` does not resolve'
            ) from None
        if not isinstance(scope.ctx, Context):
            raise TritonEmitError(
                f'the context of `{type(e).__name__}` is not concrete'
            )
        return scope.ctx

    # -- casts ---------------------------------------------------------

    def _maybe_cast(
        self, code: str, have: TritonScalar, want: TritonScalar,
    ) -> str:
        """*code*, in *want*'s storage, or a refusal.

        An implicit narrowing is the fp16 trap, so it is never emitted; a
        program that needs one has to say so with `fp.cast`.
        """
        if have == want:
            return code
        if not scalar_fits_in(have, want):
            raise TritonEmitError(
                f'emitting this would narrow {have.format()} to '
                f'{want.format()} implicitly, which rounds'
            )
        return self._explicit_cast(code, want)

    def _explicit_cast(self, code: str, want: TritonScalar) -> str:
        return f'{code}.to({want.format()})'

    # -- dispatch ------------------------------------------------------

    def _dispatch(
        self,
        e: UnaryOp | BinaryOp | TernaryOp,
        table: dict,
        operands: Sequence[tuple[str, Expr]],
    ) -> str:
        """Emit *e* through the op table.

        A **direct match** needs no conversion.  Failing that,
        **cast-to-active** takes a signature all of whose slots are the active
        context's storage, and widens each operand into it -- which the cast
        discipline permits only where nothing is lost.  Failing that,
        **widening**, sound only under ``REAL``; see :meth:`_try_widen`.
        """
        sigs = table.get(type(e))
        if not sigs:
            raise TritonEmitError(
                f'no signatures for op: {type(e).__name__}'
            )
        codes = [code for code, _ in operands]
        storages = tuple(self._storage(src) for _, src in operands)
        active = self._active_ctx(e)

        for sig in sigs:
            if sig.matches(storages, active):
                return sig.format(*codes)

        target = self._active_storage(sigs, active, len(operands))
        if target is not None:
            want = (target,) * len(operands)
            for sig in sigs:
                if sig.in_tys == want and sig.out_ctx == active:
                    return sig.format(*[
                        self._maybe_cast(code, have, target)
                        for code, have in zip(codes, storages)
                    ])

        if active is REAL:
            widened = self._try_widen(e, sigs, codes, storages)
            if widened is not None:
                return widened

        raise TritonEmitError(
            f'no matching signature for {type(e).__name__} under context '
            f'`{active}`: {[s.format() for s in storages]}'
        )

    def _try_widen(
        self,
        e: Expr,
        sigs: list[TritonOp],
        codes: list[str],
        storages: tuple[TritonScalar, ...],
    ) -> str | None:
        """Under ``REAL``, compute at the width that holds the exact result.

        ``REAL`` has no storage of its own, so no signature names it.  But the
        storage the pipeline chose for *e* holds the operation's *unrounded*
        result -- that is what choosing it means -- so the operation at that
        width rounds to itself, and is therefore the ``REAL`` operation.

        This is what the running example turns on: an fp16 product needs 22
        bits and fp32 carries 24, so `x.to(tl.float32) * y.to(tl.float32)` is
        exact where `x * y` on fp16 operands is not.
        """
        target = self._storage(e)
        want = (target,) * len(codes)
        for sig in sigs:
            if sig.in_tys == want:
                return sig.format(*[
                    self._maybe_cast(code, have, target)
                    for code, have in zip(codes, storages)
                ])
        return None

    @staticmethod
    def _active_storage(
        sigs: list[TritonOp], active: Context, arity: int,
    ) -> TritonScalar | None:
        """The storage a same-type signature of *active* uses, if there is
        one."""
        for sig in sigs:
            if (sig.out_ctx == active and len(sig.in_tys) == arity
                    and len(set(sig.in_tys)) == 1):
                return sig.in_tys[0]
        return None

    def _emit_round(self, e: Round | Cast) -> str:
        """An explicit `fp.round` / `fp.cast`, which is a *cast*, not an
        operation the table dispatches.

        Where the round changes nothing it emits nothing -- a literal already
        representable in the target needs no `.to`, and Triton has no spelling
        for casting one.  Otherwise it is a cast, which is sound only under a
        context whose `round` *is* the hardware conversion; `is_native_ctx`
        answers exactly that question.
        """
        arg = self.emit(e.arg)
        ctx = self._active_ctx(e)
        if rounds_exactly(e, self.format_info.by_expr, ctx):
            return arg
        if not is_native_ctx(ctx):
            raise TritonEmitError(
                f'`{type(e).__name__.lower()}` to `{ctx}` is not a hardware '
                'conversion, so it has no cast spelling'
            )
        return self._explicit_cast(arg, self._storage(e))

    def _emit_where(self, e: IfExpr) -> str:
        """``tl.where``, with both arms in the result's storage.

        **`tl.where` evaluates both arms where the interpreter evaluates one.**
        That does not cost the contract: the arms are effect-free by
        `SimplifyIf`'s own refusals, and the GPU does not trap where the
        interpreter would -- `logb(0)` is `-inf` in hardware and `tl.where`
        discards it -- so the two agree on the value, which is what the
        contract asks.
        """
        want = self._storage(e)
        arms = [
            self._maybe_cast(self.emit(a), self._storage(a), want)
            for a in (e.ift, e.iff)
        ]
        return f'tl.where({self.emit(e.cond)}, {arms[0]}, {arms[1]})'

    # -- expressions ---------------------------------------------------

    def emit(self, e: Expr) -> str:
        match e:
            case Var():
                return str(e.name)
            case BoolVal():
                return 'True' if e.val else 'False'
            case Integer():
                return str(e.val)
            case Decnum() | Hexnum():
                return str(e.val)
            case Rational():
                raise TritonEmitError(
                    'a rational literal has no Triton spelling; round it first'
                )
            case Round() | Cast():
                return self._emit_round(e)
            case IfExpr():
                return self._emit_where(e)
            case UnaryOp():
                return self._dispatch(
                    e, self.op_table.unary, [(self.emit(e.arg), e.arg)],
                )
            case BinaryOp():
                return self._dispatch(e, self.op_table.binary, [
                    (self.emit(e.first), e.first),
                    (self.emit(e.second), e.second),
                ])
            case TernaryOp():
                return self._dispatch(e, self.op_table.ternary, [
                    (self.emit(e.first), e.first),
                    (self.emit(e.second), e.second),
                    (self.emit(e.third), e.third),
                ])
            case _:
                raise TritonEmitError(
                    f'no Triton spelling for `{type(e).__name__}`'
                )


def emit_expr(e: Expr, func: FuncDef) -> str:
    """*e*, as Triton source.

    *func* is the function it belongs to; its analyses decide the storage each
    operand is held in and the context the operation rounds under.
    """
    if not isinstance(e, Expr):
        raise TypeError(f"Expected an 'Expr', got {e}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    def_use = DefineUse.analyze(func)
    return _ExprEmitter(
        func,
        FormatInfer.analyze(func),
        ContextUse.analyze(func, def_use=def_use),
        make_op_table(),
    ).emit(e)


def _static_count(stmt: ForStmt, sizes: ArraySizeAnalysis) -> int:
    """How many times *stmt* runs, as a compile-time constant.

    ``tl.static_range`` needs the count as a ``constexpr``, so an unproven one
    is refused.  The limit is `trip_count`'s modeling rather than the
    program's: it answers only for a `range`, so a `zip` or a bare list is
    declined even where `ArraySizeInfer` knows the length.
    """
    n = trip_count(stmt.iterable, sizes)
    if not isinstance(n, int):
        raise TritonEmitError(
            f'`tl.static_range` needs a compile-time trip count, and this '
            f'`{type(stmt.iterable).__name__}` has none that `trip_count` '
            'models'
        )
    return n


def _emit_stmts(
    block: StmtBlock,
    emitter: _ExprEmitter,
    sizes: ArraySizeAnalysis,
    out: _IndentedWriter,
) -> None:
    """Statements, less the ones that touch memory or the launch grid.

    A `with` emits nothing of its own: a context change is a change of
    *storage*, which the dispatch already reads per expression.
    """
    for stmt in block.stmts:
        match stmt:
            case Assign():
                if not isinstance(stmt.target, NamedId):
                    raise TritonEmitError(
                        'a destructuring assignment has no Triton spelling'
                    )
                out.add_line(f'{stmt.target} = {emitter.emit(stmt.expr)}')
            case ReturnStmt():
                out.add_line(f'return {emitter.emit(stmt.expr)}')
            case ContextStmt():
                _emit_stmts(stmt.body, emitter, sizes, out)
            case ForStmt():
                if not isinstance(stmt.target, Id):
                    raise TritonEmitError(
                        'a destructuring loop target has no Triton spelling'
                    )
                n = _static_count(stmt, sizes)
                out.add_line(f'for {stmt.target} in tl.static_range({n}):')
                out.indent()
                _emit_stmts(stmt.body, emitter, sizes, out)
                out.dedent()
            case _:
                raise TritonEmitError(
                    f'no Triton spelling for `{type(stmt).__name__}`'
                )


def emit_block(block: StmtBlock, func: FuncDef) -> str:
    """*block*, as Triton source, for a block of straight-line statements."""
    if not isinstance(block, StmtBlock):
        raise TypeError(f"Expected a 'StmtBlock', got {block}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    def_use = DefineUse.analyze(func)
    emitter = _ExprEmitter(
        func,
        FormatInfer.analyze(func),
        ContextUse.analyze(func, def_use=def_use),
        make_op_table(),
    )
    out = _IndentedWriter()
    _emit_stmts(block, emitter, ArraySizeInfer.analyze(func), out)
    return out.render()
