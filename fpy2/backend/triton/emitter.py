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
from ...analysis.array_size import ListSize, trip_count
from ...analysis.format_infer import rounds_exactly
from ...ast import (
    And,
    Assign,
    BinaryOp,
    BoolVal,
    Cast,
    Compare,
    CompareOp,
    ContextStmt,
    Decnum,
    Expr,
    ForStmt,
    FuncDef,
    Hexnum,
    Id,
    If1Stmt,
    IfExpr,
    IndexedAssign,
    Integer,
    ListRef,
    NamedId,
    NaryOp,
    Not,
    Or,
    Rational,
    ReturnStmt,
    Round,
    Stmt,
    StmtBlock,
    TernaryOp,
    UnaryOp,
    Var,
)
from ...ast.visitor import Visitor
from ...number import REAL, Context
from ..backend import CompileError
from .storage import choose_storage_scalar, scalar_fits_in
from .target import ScalarOpTable, TritonOp, is_native_ctx, make_op_table
from .types import TritonScalar

__all__ = ['TritonEmitError', 'emit_block', 'emit_expr']


class TritonEmitError(CompileError):
    """A program this backend declines to emit."""


_COMPARE: dict[CompareOp, str] = {
    CompareOp.LT: '<',
    CompareOp.LE: '<=',
    CompareOp.GT: '>',
    CompareOp.GE: '>=',
    CompareOp.EQ: '==',
    CompareOp.NE: '!=',
}


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


class _Emitter(Visitor):
    """Produces Triton source.

    Dispatch is the framework's, not a `match` of its own: every node this
    backend cannot spell then has a *named* refusal, and a new AST node is a
    build error rather than something a catch-all absorbs.

    Dispatch walks the MRO, so `Round` -- a `NamedUnaryOp` -- arrives at
    :meth:`_visit_unaryop` rather than at an entry of its own; separating a
    cast from a table operation is an explicit test there.
    """

    def __init__(
        self,
        func: FuncDef,
        format_info: FormatAnalysis,
        ctx_use: ContextUseAnalysis,
        sizes: ArraySizeAnalysis,
        op_table: ScalarOpTable,
    ):
        self.func = func
        self.format_info = format_info
        self.ctx_use = ctx_use
        self.sizes = sizes
        self.op_table = op_table
        self.mask: str | None = None
        """The guard in force, as a Triton predicate.

        A `tile_loops` mask is not a branch: its body runs for every lane and
        the guard becomes the `mask=` of each load and store.  `None` where
        nothing encloses the access.
        """

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
        """*code* cast to *want*.

        A numeric literal is parenthesized first: `2.to(...)` lexes as `2.`
        followed by `to`, which is a different program and usually a syntax
        error.  A name or a call needs no parentheses, and `dot_exact` spells
        it bare.
        """
        if not (code.isidentifier() or code.endswith(')')):
            code = f'({code})'
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

    # -- memory --------------------------------------------------------

    def _flatten(self, e: ListRef) -> tuple[NamedId, list[Expr]]:
        """A subscript chain as its base and indices, outermost first."""
        indices: list[Expr] = []
        cur: Expr = e
        while isinstance(cur, ListRef):
            indices.append(cur.index)
            cur = cur.value
        if not isinstance(cur, Var):
            raise TritonEmitError(
                'a subscript of something other than a name has no Triton '
                'spelling'
            )
        indices.reverse()
        return cur.name, indices

    def _strides(self, base: NamedId, rank: int) -> list[int]:
        """Row-major strides for *base*, from its proven shape.

        A kernel argument is a flat pointer, so an unproven length has no
        offset arithmetic to emit -- which is why this backend stores no list
        whose length it cannot prove.
        """
        bound = next(
            (b for defn, b in self.sizes.by_def.items() if defn.name == base),
            None,
        )
        dims: list[int] = []
        while isinstance(bound, ListSize):
            if not isinstance(bound.size, int):
                raise TritonEmitError(
                    f'`{base}` has no proven length, so its offsets cannot be '
                    'computed'
                )
            dims.append(bound.size)
            bound = bound.elt
        if len(dims) < rank:
            raise TritonEmitError(
                f'`{base}` is subscripted {rank} deep but only {len(dims)} '
                'dimensions are proven'
            )
        strides = [1] * rank
        for i in range(rank - 2, -1, -1):
            strides[i] = strides[i + 1] * dims[i + 1]
        return strides

    def _offset(self, base: NamedId, indices: list[Expr]) -> str:
        """The flat element offset, row-major."""
        strides = self._strides(base, len(indices))
        terms = [
            self.emit(idx) if st == 1 else f'{self.emit(idx)} * {st}'
            for idx, st in zip(indices, strides)
        ]
        return ' + '.join(terms)

    def _emit_load(self, e: ListRef) -> str:
        """`tl.load`, masked by whatever guard is in force.

        ``other=0.0`` is safe rather than meaningful: a masked-off lane's
        value feeds only that lane's arithmetic, and the store that would
        commit it carries the same mask, so it is discarded.
        """
        base, indices = self._flatten(e)
        addr = f'{base}_ptr + {self._offset(base, indices)}'
        if self.mask is None:
            return f'tl.load({addr})'
        return f'tl.load({addr}, mask={self.mask}, other=0.0)'

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

    def _emit_compare(self, e: Compare) -> str:
        """A comparison, which rounds nothing and so is not in the op table.

        FPy chains like Python does, and a chain is the conjunction of its
        links -- but `and` short-circuits and has no elementwise meaning on a
        tile, so the links join with `&`.  Each operand is emitted once per
        link it appears in; they are names by the time the normal form is
        done, so nothing is recomputed.
        """
        codes = [self.emit(a) for a in e.args]
        links = [
            f'({codes[i]} {_COMPARE[op]} {codes[i + 1]})'
            for i, op in enumerate(e.ops)
        ]
        return links[0] if len(links) == 1 else '(' + ' & '.join(links) + ')'

    def _emit_connective(self, e: And | Or) -> str:
        """`and` / `or`, spelled elementwise.

        Python's keywords short-circuit and return an operand rather than a
        tile, so a lane-wise connective has to be `&` / `|`.  Both arms are
        evaluated, which is sound for the same reason `tl.where` is.
        """
        op = '&' if isinstance(e, And) else '|'
        return '(' + f' {op} '.join(self.emit(a) for a in e.args) + ')'

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
        """*e* as Triton source."""
        return self._visit_expr(e, None)

    # -- expressions ---------------------------------------------------

    def _visit_var(self, e: Var, ctx) -> str:
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx) -> str:
        return 'True' if e.val else 'False'

    def _visit_integer(self, e: Integer, ctx) -> str:
        return str(e.val)

    def _visit_decnum(self, e: Decnum, ctx) -> str:
        return str(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return str(e.val)

    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        # `Round` and `Cast` are casts, not table operations, and they arrive
        # here because dispatch walks the MRO
        if isinstance(e, (Round, Cast)):
            return self._emit_round(e)
        if isinstance(e, Not):
            return f'(~{self.emit(e.arg)})'
        return self._dispatch(
            e, self.op_table.unary, [(self.emit(e.arg), e.arg)],
        )

    def _visit_binaryop(self, e: BinaryOp, ctx) -> str:
        return self._dispatch(e, self.op_table.binary, [
            (self.emit(e.first), e.first),
            (self.emit(e.second), e.second),
        ])

    def _visit_ternaryop(self, e: TernaryOp, ctx) -> str:
        return self._dispatch(e, self.op_table.ternary, [
            (self.emit(e.first), e.first),
            (self.emit(e.second), e.second),
            (self.emit(e.third), e.third),
        ])

    def _visit_naryop(self, e: NaryOp, ctx) -> str:
        if isinstance(e, (And, Or)):
            return self._emit_connective(e)
        raise TritonEmitError(f'no Triton spelling for `{type(e).__name__}`')

    def _visit_compare(self, e: Compare, ctx) -> str:
        return self._emit_compare(e)

    def _visit_list_ref(self, e: ListRef, ctx) -> str:
        return self._emit_load(e)

    def _visit_if_expr(self, e: IfExpr, ctx) -> str:
        return self._emit_where(e)

    # -- expressions with no Triton spelling ---------------------------

    def _visit_rational(self, e: Rational, ctx):
        raise TritonEmitError(
            'a rational literal has no Triton spelling; round it first'
        )

    def _visit_digits(self, e, ctx):
        raise TritonEmitError(
            'a `digits` literal has no Triton spelling; round it first'
        )

    def _visit_foreign(self, e, ctx):
        raise TritonEmitError(
            'a foreign value is not a Triton value; fold it first'
        )

    def _visit_nullaryop(self, e, ctx):
        raise TritonEmitError(
            f'`{type(e).__name__.lower()}` is not in the op table'
        )

    def _visit_call(self, e, ctx):
        raise TritonEmitError(
            'a call survives only if inlining failed; this backend inlines '
            'everything'
        )

    def _visit_tuple_expr(self, e, ctx):
        raise TritonEmitError('a tuple has no Triton storage')

    def _visit_list_expr(self, e, ctx):
        raise TritonEmitError(
            'a list literal has no Triton storage; a list lives in memory a '
            'kernel argument points at'
        )

    def _visit_list_comp(self, e, ctx):
        raise TritonEmitError(
            'a comprehension has no Triton spelling; lower it to a loop first'
        )

    def _visit_list_slice(self, e, ctx):
        raise TritonEmitError('a slice has no Triton spelling')

    def _visit_attribute(self, e, ctx):
        raise TritonEmitError('an attribute has no Triton spelling')

    # -- statements ----------------------------------------------------

    def _visit_assign(self, stmt: Assign, ctx: _IndentedWriter):
        if not isinstance(stmt.target, NamedId):
            raise TritonEmitError(
                'a destructuring assignment has no Triton spelling'
            )
        ctx.add_line(f'{stmt.target} = {self.emit(stmt.expr)}')

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _IndentedWriter):
        addr = f'{stmt.var}_ptr + {self._offset(stmt.var, list(stmt.indices))}'
        val = self.emit(stmt.expr)
        mask = '' if self.mask is None else f', mask={self.mask}'
        ctx.add_line(f'tl.store({addr}, {val}{mask})')

    def _visit_return(self, stmt: ReturnStmt, ctx: _IndentedWriter):
        ctx.add_line(f'return {self.emit(stmt.expr)}')

    def _visit_context(self, stmt: ContextStmt, ctx: _IndentedWriter):
        # a context change is a change of *storage*, which the dispatch reads
        # per expression; it has no statement of its own
        self._visit_block(stmt.body, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: _IndentedWriter):
        # a `tile_loops` guard is a mask, not a branch: the body runs for
        # every lane and the predicate rides on each access
        prev = self.mask
        self.mask = self.emit(stmt.cond)
        self._visit_block(stmt.body, ctx)
        self.mask = prev

    def _visit_for(self, stmt: ForStmt, ctx: _IndentedWriter):
        if not isinstance(stmt.target, Id):
            raise TritonEmitError(
                'a destructuring loop target has no Triton spelling'
            )
        n = _static_count(stmt, self.sizes)
        ctx.add_line(f'for {stmt.target} in tl.static_range({n}):')
        ctx.indent()
        self._visit_block(stmt.body, ctx)
        ctx.dedent()

    def _visit_block(self, block: StmtBlock, ctx: _IndentedWriter):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    # -- statements with no Triton spelling ----------------------------

    def _visit_if(self, stmt, ctx):
        raise TritonEmitError(
            'an `if` statement remains; the normal form makes them `if` '
            'expressions'
        )

    def _visit_while(self, stmt, ctx):
        raise TritonEmitError('a `while` has no Triton spelling')

    def _visit_assert(self, stmt, ctx):
        raise TritonEmitError(
            'an `assert` has no Triton spelling; a kernel cannot raise'
        )

    def _visit_effect(self, stmt, ctx):
        raise TritonEmitError('an effect has no Triton spelling')

    def _visit_pass(self, stmt, ctx):
        pass

    def _visit_function(self, func: FuncDef, ctx):
        raise TritonEmitError('emitting a whole kernel is not implemented yet')


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
    return _Emitter(
        func,
        FormatInfer.analyze(func),
        ContextUse.analyze(func, def_use=def_use),
        ArraySizeInfer.analyze(func),
        make_op_table(),
    ).emit(e)


def emit_block(block: StmtBlock, func: FuncDef) -> str:
    """*block*, as Triton source, for a block of straight-line statements."""
    if not isinstance(block, StmtBlock):
        raise TypeError(f"Expected a 'StmtBlock', got {block}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    def_use = DefineUse.analyze(func)
    sizes = ArraySizeInfer.analyze(func)
    emitter = _Emitter(
        func,
        FormatInfer.analyze(func),
        ContextUse.analyze(func, def_use=def_use),
        sizes,
        make_op_table(),
    )
    out = _IndentedWriter()
    emitter._visit_block(block, out)
    return out.render()
