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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from ...analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    FormatAnalysis,
    FormatInfer,
)
from ...analysis.array_size import ListSize, static_trip_count
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
    Digits,
    Expr,
    ForStmt,
    FuncDef,
    Hexnum,
    Id,
    If1Stmt,
    IfExpr,
    IndexedAssign,
    Integer,
    IsFinite,
    IsInf,
    IsNan,
    Len,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    ListTypeAnn,
    Max,
    Min,
    Mul,
    NamedId,
    NaryOp,
    Not,
    Or,
    Range1,
    Range3,
    Rational,
    ReturnStmt,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    TernaryOp,
    TupleBinding,
    TupleExpr,
    UnaryOp,
    UnderscoreId,
    Var,
)
from ...ast.visitor import DefaultVisitor, Visitor
from ...number import REAL, Context
from ..backend import CompileError
from .storage import choose_storage_scalar, scalar_fits_in
from .target import ScalarOpTable, TritonOp, is_native_ctx, make_op_table
from .types import TritonScalar

__all__ = ['KernelSource', 'TritonEmitError', 'emit_block', 'emit_expr', 'emit_kernel']


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


def _reads_name(code: str, names: set[str]) -> bool:
    """Whether *code* mentions any of *names* as an identifier."""
    return any(re.search(rf'\b{re.escape(n)}\b', code) for n in names)


def _as_literal(code: str) -> float | None:
    """*code* as a number, if that is all it is."""
    try:
        return float(code)
    except ValueError:
        return None


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
        tiled: Sequence[ForStmt] = (),
        drop_asserts: bool = False,
    ):
        self.func = func
        self.format_info = format_info
        self.ctx_use = ctx_use
        self.sizes = sizes
        self.op_table = op_table
        self.tiled = tiled
        """The loops carrying a tile, as `tile_loops` reported them.

        Identity, not shape: recognizing a tiled loop by what it looks like
        would be pattern-matching another pass's output.
        """
        self.ranges: dict[NamedId, tuple[str, str]] = {}
        """Names bound to a `range`, as (start, step).

        `SplitLoop` materializes the iterable it splits -- `t = range(n)` --
        and then indexes it.  A range holds no memory: `t[j]` is arithmetic,
        `start + j * step`, so neither the binding nor the subscript is an
        access.
        """
        self.drop_asserts = drop_asserts
        """Whether an `assert` is dropped rather than refused.

        A kernel cannot raise, so an assert has no spelling either way; the
        flag is which of the two answers the caller wants.  Dropping one is a
        *semantic* change -- the program said to abort and the kernel will
        not -- so it is opt-in, and a launcher that wants the check runs it
        host-side.
        """
        self.consts: dict[str, int] = {}
        """Names this kernel binds to an integer constant.

        The tiled loop's bound is a proven length, but it reaches the source
        through a temporary -- `t10 = 4` -- so the launcher's grid extent has
        to be read back from the binding rather than from the expression.
        """
        self.copies: dict[str, str] = {}
        """Names bound to another name.

        `SplitLoop` binds the factor to a temporary, and a `tl.constexpr`
        does not survive the copy -- `tl.arange`'s arguments must be
        `constexpr`, and a plain variable holding one is not.  So a tile's
        width is resolved back to the name it came from.
        """
        self.seqs: dict[NamedId, list[str]] = {}
        """Names holding a *scalarized* sequence, as one code per element.

        Triton has no list value: a tile is not scalar-indexable and a Python
        list is compile-time metaprogramming.  A sequence of proven length
        therefore stops existing -- it becomes that many ordinary values, and
        every access is resolved here rather than emitted.
        """
        self.subst: dict[NamedId, str] = {}
        """Names bound to a code while a comprehension is unrolled."""
        self._next_tmp = 0
        self.grid_extent: int | None = None
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

        **A literal is retyped, not cast.**  `2.to(...)` lexes as `2.` then
        `to`, and parenthesizing it to `(2).to(...)` only moves the problem:
        it is valid Python, and Triton rejects it with *"'int' object has no
        attribute 'to'"* because a Python scalar is a `constexpr`, not a
        tile.  Writing the literal in the target's own spelling avoids the
        conversion entirely.
        """
        literal = _as_literal(code)
        if literal is not None:
            return f'{float(literal)}' if want.is_float() else f'{int(literal)}'
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

    def _range_index(self, e: ListRef) -> str | None:
        """*e* as index arithmetic, if it subscripts a `range`."""
        if not isinstance(e.value, Var):
            return None
        bounds = self.ranges.get(e.value.name)
        if bounds is None:
            return None
        start, step = bounds
        idx = self.emit(e.index)
        term = idx if step == '1' else f'{idx} * {step}'
        return term if start == '0' else f'({start} + {term})'

    def _elements(self, e: Expr) -> list[str] | None:
        """*e* as one code per element, or `None` if it is not a sequence this
        backend can take apart."""
        match e:
            case Var() if e.name in self.seqs:
                return list(self.seqs[e.name])

            case ListExpr():
                return [self.emit(elt) for elt in e.elts]
            case ListSlice():
                return self._slice_elements(e)
            case ListComp():
                return self._comp_elements(e)
        return None

    def _range_elements(self, e: Expr) -> list[str] | None:
        """A `range`, as its index values.

        Unrolling a comprehension means substituting each index in turn, so
        the range has to come apart the same way a list does -- its elements
        are the indices themselves.
        """
        lo: int | None
        hi: int | None
        step: int | None
        if isinstance(e, Range1):
            lo, hi, step = 0, self._const_index(e.arg), 1
        elif isinstance(e, Range3):
            lo = self._const_index(e.first)
            hi = self._const_index(e.second)
            step = self._const_index(e.third)
        else:
            return None
        if lo is None or hi is None or step is None or step == 0:
            return None
        return [str(i) for i in range(lo, hi, step)]

    def _source_elements(self, e: Expr) -> list[str] | None:
        """*e* as elements, where it is being *consumed* as a sequence.

        Wider than :meth:`_elements` by one case: a pointer-backed list, whose
        elements are that many loads.  Kept separate because expanding one is
        only wanted where a sequence is taken apart -- iterated or sliced --
        and never for `t = xs`, which is a copy of a pointer rather than a
        request for its contents.
        """
        elems = self._elements(e)
        if elems is not None:
            return elems
        # a `range` expands only where a comprehension consumes it: bound to
        # a name it stays a range, which is index arithmetic rather than
        # values
        if isinstance(e, (Range1, Range3)):
            return self._range_elements(e)
        if not isinstance(e, Var):
            return None
        bound = self.sizes.by_expr.get(e)
        if isinstance(bound, ListSize) and isinstance(bound.size, int):
            return [self._load_at(e.name, str(i)) for i in range(bound.size)]
        return None

    def _slice_elements(self, e: ListSlice) -> list[str] | None:
        """A slice, as its elements.

        The length comes from the size analysis, which cancels a common base:
        `xs[k:k + L]` is `L` whatever `k` is, provided the index arithmetic is
        exact.  The *offsets* are then `start + 0 .. start + n - 1`.
        """
        bound = self.sizes.by_expr.get(e)
        if not isinstance(bound, ListSize) or not isinstance(bound.size, int):
            return None
        inner = self._elements(e.value)
        start = 0 if e.start is None else self._const_index(e.start)
        if inner is not None:
            if start is None:
                return None
            return inner[start:start + bound.size]
        # a pointer-backed list: the slice is that many loads
        if not isinstance(e.value, Var):
            return None
        base = self.emit(e.start) if e.start is not None else '0'
        return [
            self._load_at(e.value.name, base if i == 0 else f'{base} + {i}')
            for i in range(bound.size)
        ]

    def _comp_elements(self, e: ListComp) -> list[str] | None:
        """A comprehension, unrolled.

        Each source is taken apart first, then the element expression is
        emitted once per index with the targets bound to that index's codes.
        Binding rather than assigning keeps the unrolled copies from needing
        names of their own.
        """
        if len(e.iterables) != len(e.targets):
            return None
        sources = [self._source_elements(it) for it in e.iterables]
        if any(src is None for src in sources):
            return None
        n = min(len(src) for src in sources if src is not None)
        out: list[str] = []
        saved = dict(self.subst)
        try:
            for i in range(n):
                for target, src in zip(e.targets, sources):
                    assert src is not None
                    if not isinstance(target, NamedId):
                        return None
                    self.subst[target] = src[i]
                out.append(self.emit(e.elt))
        finally:
            self.subst = saved
        return out

    def _const_index(self, e: Expr) -> int | None:
        """*e* as a compile-time index, or `None`."""
        code = self.emit(e)
        try:
            return int(code)
        except ValueError:
            return None

    def _load_at(self, base: NamedId, offset: str) -> str:
        addr = f'{base}_ptr + {offset}'
        if self.mask is None:
            return f'tl.load({addr})'
        return f'tl.load({addr}, mask={self.mask}, other=0.0)'

    def _emit_load(self, e: ListRef) -> str:
        """`tl.load`, masked by whatever guard is in force.

        ``other=0.0`` is safe rather than meaningful: a masked-off lane's
        value feeds only that lane's arithmetic, and the store that would
        commit it carries the same mask, so it is discarded.
        """
        direct = self._range_index(e)
        if direct is not None:
            return direct
        elems = self._elements(e.value)
        if elems is not None:
            i = self._const_index(e.index)
            if i is None:
                raise TritonEmitError(
                    'a scalarized sequence can only be indexed by a '
                    'compile-time constant; Triton has no addressable local '
                    'array'
                )
            if not 0 <= i < len(elems):
                raise TritonEmitError(
                    f'index {i} is outside a sequence of {len(elems)}'
                )
            return elems[i]
        base, indices = self._flatten(e)
        addr = f'{base}_ptr + {self._offset(base, indices)}'
        if self.mask is None:
            return f'tl.load({addr})'
        return f'tl.load({addr}, mask={self.mask}, other=0.0)'

    def _emit_len(self, e: Len) -> str:
        """A length, as the constant the pipeline proved it to be.

        A kernel argument is a bare pointer and carries no length, so the only
        length available is the proven one.  That makes the kernel specific to
        the shape it was compiled for -- which `Specialize` has already made
        it, since a proven length is what lets any of this be emitted at all.
        """
        if isinstance(e.arg, Var) and e.arg.name in self.seqs:
            return str(len(self.seqs[e.arg.name]))
        bound = self.sizes.by_expr.get(e.arg)
        if not isinstance(bound, ListSize) or not isinstance(bound.size, int):
            raise TritonEmitError(
                'a length this backend cannot prove has no Triton spelling; '
                'a kernel argument is a pointer and carries no length'
            )
        return str(bound.size)

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

    def _emit_predicate(self, e: UnaryOp) -> str:
        """A classification predicate, spelled from comparisons.

        Triton has no `isnan`, `isinf` or `isfinite`, so each is written out.
        All three are exact -- they read the value rather than computing one
        -- and each relies on a NaN comparing unequal to everything, which is
        what makes the spellings fall out:

        - `isnan(x)`    is `x != x`
        - `isinf(x)`    is `|x| == inf`, false for a NaN because the
          comparison is
        - `isfinite(x)` is `|x| < inf`, false for a NaN for the same reason

        `signbit` is the exception: it must distinguish `-0.0` from `0.0`,
        which no comparison does, so it is refused rather than approximated.
        """
        arg = self.emit(e.arg)
        if isinstance(e, IsNan):
            return f'({arg} != {arg})'
        if isinstance(e, IsInf):
            return f"(tl.abs({arg}) == float('inf'))"
        if isinstance(e, IsFinite):
            return f"(tl.abs({arg}) < float('inf'))"
        raise TritonEmitError(
            '`signbit` cannot be spelled from comparisons: it has to separate '
            '`-0.0` from `0.0`, which no comparison does'
        )

    def _emit_select_op(self, e: Max | Min) -> str:
        """`max` / `min`, folded pairwise and made NaN-propagating.

        **`tl.maximum` is the wrong operation on its own.**  FPy follows IEEE
        754-2019 `maximum`, where a NaN operand propagates; Triton's
        `tl.maximum` follows `maximumNumber`, which returns the *other*
        operand.  Checked on hardware: FPy gives `nan` for `max(nan, 1.0)`
        and `tl.maximum` gives `1.0`.  Emitting it bare would be a miscompile
        on any input containing a NaN.

        So the fold is guarded *once*, over every operand: if any is a NaN the
        result is a NaN, else the chain of hardware selects.  `x != x` is the
        NaN test.  Guarding each step instead would put the accumulator inside
        its own test and grow the expression exponentially in the number of
        operands.

        **The guard repeats its operands**, which is free for a name and two
        extra loads for a subscript.  They are pure and identically masked, so
        it is correct either way and a CSE pass should collapse them -- but
        that is unverified here.  The principled fix is not a cleverer
        expansion: it is to drop the guard where a NaN cannot arise, which a
        context with `enable_nan=False` already states and `ValueClassInfer`
        could prove more widely.

        Folding pairwise is sound because both are associative *and* exact:
        they return an operand rather than computing one, so no grouping
        rounds differently.
        """
        name = 'tl.maximum' if isinstance(e, Max) else 'tl.minimum'
        want = self._storage(e)
        args = [
            self._maybe_cast(self.emit(a), self._storage(a), want)
            for a in e.args
        ]
        if not args:
            raise TritonEmitError(f'`{name}` needs at least one operand')
        acc = args[0]
        for rhs in args[1:]:
            acc = f'{name}({acc}, {rhs})'
        if len(args) == 1:
            return acc
        # one NaN test over all the operands, not one per fold step: guarding
        # each step would put the accumulator inside its own test and grow
        # the expression exponentially in the number of operands
        nan = ' | '.join(f'({a} != {a})' for a in args)
        return f"tl.where({nan}, float('nan'), {acc})"

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
        bound = self.subst.get(e.name)
        if bound is not None:
            return bound
        if e.name in self.seqs:
            # unreachable from well-typed source -- `TypeInfer` rejects a
            # sequence in a scalar position first -- so this guards against an
            # internal slip rather than a user program
            raise TritonEmitError(
                f'`{e.name}` is a sequence, which has no Triton value; it can '
                'only be indexed, sliced, measured or copied'
            )
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx) -> str:
        return 'True' if e.val else 'False'

    def _visit_integer(self, e: Integer, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_decnum(self, e: Decnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _emit_numeric_literal(self, v: Fraction) -> str:
        """A literal, as Triton source.

        An FPy literal is an exact rational rounded where it is *used*, which
        Triton has no spelling for -- so a value the target holds exactly
        prints as itself, and one it cannot is refused rather than emitted as
        `num / denom`, which would be an *operation* where FPy has a constant.

        Ported from the cpp emitter's `_emit_numeric_literal`, which had
        already answered this: the question is whether the target holds the
        value, not what context surrounds it, so no scope lookup is involved.
        """
        if v.denominator == 1:
            return str(v.numerator)
        exact = float(v)
        if Fraction(exact) == v:
            return repr(exact)
        raise TritonEmitError(
            f'`{v.numerator}/{v.denominator}` is not representable here; '
            'wrap it in `fp.round(...)` to round it to a format that is'
        )

    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        # `Round` and `Cast` are casts, not table operations, and they arrive
        # here because dispatch walks the MRO
        if isinstance(e, (Round, Cast)):
            return self._emit_round(e)
        if isinstance(e, Not):
            return f'(~{self.emit(e.arg)})'
        if isinstance(e, Len):
            return self._emit_len(e)
        if isinstance(e, (IsNan, IsInf, IsFinite, Signbit)):
            return self._emit_predicate(e)
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
        if isinstance(e, (Max, Min)):
            return self._emit_select_op(e)
        raise TritonEmitError(f'no Triton spelling for `{type(e).__name__}`')

    def _visit_compare(self, e: Compare, ctx) -> str:
        return self._emit_compare(e)

    def _visit_list_ref(self, e: ListRef, ctx) -> str:
        return self._emit_load(e)

    def _visit_if_expr(self, e: IfExpr, ctx) -> str:
        return self._emit_where(e)

    # -- expressions with no Triton spelling ---------------------------

    def _visit_rational(self, e: Rational, ctx) -> str:
        """`FreeVarElim` materializes a captured `2.5` as `fp.rational(5, 2)`,
        so refusing every rational would refuse every captured non-integer."""
        return self._emit_numeric_literal(e.as_rational())

    def _visit_digits(self, e: Digits, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

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
        if isinstance(stmt.target, TupleBinding):
            return self._emit_destructure(stmt, ctx)
        if not isinstance(stmt.target, NamedId):
            raise TritonEmitError(
                f'a `{type(stmt.target).__name__}` assignment target has no '
                'Triton spelling'
            )
        elems = self._elements(stmt.expr)
        if elems is not None:
            # the sequence stops existing: each element becomes a value of
            # its own, and every later access resolves against them
            names = []
            for i, code in enumerate(elems):
                name = f'{stmt.target}_{i}'
                ctx.add_line(f'{name} = {code}')
                names.append(name)
            self.seqs[stmt.target] = names
            return
        match stmt.expr:
            case Range1():
                self.ranges[stmt.target] = ('0', '1')
                return
            case Range3():
                self.ranges[stmt.target] = (
                    self.emit(stmt.expr.first), self.emit(stmt.expr.third),
                )
                return
        code = self.emit(stmt.expr)
        if code.isdigit():
            self.consts[str(stmt.target)] = int(code)
        elif isinstance(stmt.expr, Var):
            self.copies[str(stmt.target)] = self._root(code)
        ctx.add_line(f'{stmt.target} = {code}')

    def _emit_destructure(self, stmt: Assign, ctx: _IndentedWriter):
        """`a, b = (x, y)`, as one assignment per element.

        Triton has no tuple *value* to bind, so the binding has to come apart.
        Only a literal tuple does: a name holding one would need this emitter
        to track components it never built, and a primitive returning one
        needs that primitive in the op table first.

        **The elements are simultaneous, and sequential assignment is not.**
        `a, b = (b, a)` is a swap, which `a = b; b = a` turns into a copy.
        Where a target is read by the right-hand side, the values go through
        temporaries first.
        """
        target = stmt.target
        assert isinstance(target, TupleBinding)
        names = [elt for elt in target.elts]
        if not all(isinstance(n, (NamedId, UnderscoreId)) for n in names):
            raise TritonEmitError(
                'a nested destructuring target has no Triton spelling'
            )
        if not isinstance(stmt.expr, TupleExpr):
            raise TritonEmitError(
                f'destructuring a `{type(stmt.expr).__name__}` has no Triton '
                'spelling; only a literal tuple comes apart here'
            )
        if len(names) != len(stmt.expr.elts):
            raise TritonEmitError(
                f'destructuring {len(names)} names from '
                f'{len(stmt.expr.elts)} elements'
            )

        codes = [self.emit(e) for e in stmt.expr.elts]
        bound = {str(n) for n in names if isinstance(n, NamedId)}
        if any(_reads_name(code, bound) for code in codes):
            tmps = []
            for code in codes:
                tmp = f'_t{self._next_tmp}'
                self._next_tmp += 1
                ctx.add_line(f'{tmp} = {code}')
                tmps.append(tmp)
            codes = tmps
        for name, code in zip(names, codes):
            if isinstance(name, UnderscoreId):
                continue
            ctx.add_line(f'{name} = {code}')

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
        if any(stmt is t for t in self.tiled):
            return self._emit_tile(stmt, ctx)
        if not isinstance(stmt.target, Id):
            raise TritonEmitError(
                'a destructuring loop target has no Triton spelling'
            )
        # `tl.static_range` yields *indices*.  That is what the loop
        # variable means only when the iterable is a `range`; over a list it
        # binds an element, and emitting the index in its place is a silent
        # miscompile -- `for xi in xs` would max against 0, 1, 2 rather than
        # against the values.  Loading the element instead needs the
        # iterable's base and stride, which a slice or a `zip` does not
        # supply, so this refuses rather than guesses.
        if not isinstance(stmt.iterable, (Range1, Range3)):
            raise TritonEmitError(
                f'a `for` over a `{type(stmt.iterable).__name__}` binds an '
                'element, and `tl.static_range` yields an index; iterate a '
                '`range` and subscript instead'
            )
        n = _static_count(stmt, self.sizes)
        ctx.add_line(f'for {stmt.target} in tl.static_range({n}):')
        ctx.indent()
        self._visit_block(stmt.body, ctx)
        ctx.dedent()

    def _root(self, name: str) -> str:
        """*name* followed through the copies that bound it."""
        seen: set[str] = set()
        while name in self.copies and name not in seen:
            seen.add(name)
            name = self.copies[name]
        return name

    def _emit_tile(self, stmt: ForStmt, ctx: _IndentedWriter):
        """A tiled loop is not a loop: it is the launch grid plus a tile.

        `tile_loops` leaves `for i in range(0, n, B)` around
        `for j in range(i, i + B)`, and the chunk index becomes the program
        instance while the inner index becomes the lane vector::

            i = tl.program_id(0) * B
            j = i + tl.arange(0, B)

        The guard inside is already a mask, so nothing here emits a branch.
        The shape is destructured rather than assumed: a mismatch is a
        refusal, since the only thing that produces it is `tile_loops`.
        """
        outer = stmt.target
        it = stmt.iterable
        if not isinstance(it, Range3) or not isinstance(outer, NamedId):
            raise TritonEmitError(
                'a tiled loop should iterate a three-argument `range`'
            )
        width = self._root(self.emit(it.third))
        inner = next(
            (s for s in stmt.body.stmts if isinstance(s, ForStmt)), None,
        )
        if inner is None or not isinstance(inner.target, NamedId):
            raise TritonEmitError(
                'a tiled loop should hold the tile loop it was split into'
            )
        bound = self.emit(it.second)
        self.grid_extent = (
            int(bound) if bound.isdigit() else self.consts.get(bound)
        )
        ctx.add_line(f'{outer} = tl.program_id(0) * {width}')
        ctx.add_line(f'{inner.target} = {outer} + tl.arange(0, {width})')
        self._visit_block(inner.body, ctx)

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
        if self.drop_asserts:
            return
        raise TritonEmitError(
            'an `assert` has no Triton spelling; a kernel cannot raise. '
            'Pass `drop_asserts` to skip it instead'
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
    is refused.  `static_trip_count` falls back to the iterable's inferred
    length, so a `zip` or a bare list answers where a `range` would --
    provided the size analysis proved it, which specialization is what makes
    true.
    """
    n = static_trip_count(stmt.iterable, sizes)
    if not isinstance(n, int):
        raise TritonEmitError(
            f'`tl.static_range` needs a compile-time trip count, and this '
            f'`{type(stmt.iterable).__name__}` has no proven length'
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


def emit_block(
    block: StmtBlock,
    func: FuncDef,
    tiled: Sequence[ForStmt] = (),
    *,
    drop_asserts: bool = False,
) -> str:
    """*block*, as Triton source.

    *tiled* names the loops carrying a tile, as `tile_loops` reported them.
    *drop_asserts* skips an `assert` rather than refusing it.
    """
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
        tiled,
        drop_asserts,
    )
    out = _IndentedWriter()
    emitter._visit_block(block, out)
    return out.render()


@dataclass
class KernelSource:
    """An emitted kernel, and what the launcher has to know to run it."""

    name: str
    source: str
    """The `@triton.jit` function, as text."""

    params: tuple[str, ...]
    """Its parameters in order, `_ptr`-suffixed where the argument is a list."""

    grid_extent: int | None
    """How many elements the tiled dimension covers, if one was tiled.

    The launcher needs it to size the grid -- ``cdiv(extent, BLOCK)`` program
    instances -- and it is the proven length the kernel was compiled for, not
    a runtime value.  ``None`` where nothing was tiled.
    """

    enable_fp_fusion: bool
    """Whether contracting a multiply-add is unobservable here.

    Not part of the source: Triton takes it at the *launch*, so it is derived
    and handed over rather than emitted.  Contracting `acc + x * y` rounds
    once over an exact product where the unfused form rounds twice, so the two
    agree exactly where every product is already exact -- and differ where one
    is not.  Measured: an FP16-in kernel is unchanged by fusion, an all-FP32
    one differs on 590 of 2000 inputs.
    """


def _products_are_exact(func: FuncDef, emitter: _Emitter) -> bool:
    """Whether every product in *func* is exact, so fusion cannot be seen."""
    found: list[bool] = []

    class _V(DefaultVisitor):
        def _visit_binaryop(self, e, ctx):
            if isinstance(e, Mul):
                try:
                    ctx_at = emitter._active_ctx(e)
                except TritonEmitError:
                    found.append(False)
                else:
                    found.append(rounds_exactly(
                        e, emitter.format_info.by_expr, ctx_at,
                    ))
            return super()._visit_binaryop(e, ctx)

    _V()._visit_function(func, None)
    return all(found)


def emit_kernel(
    func: FuncDef,
    tiled: Sequence[ForStmt] = (),
    *,
    block: str | None = None,
    drop_asserts: bool = False,
) -> KernelSource:
    """*func* as a ``@triton.jit`` kernel.

    A list argument becomes a pointer, *block* becomes a ``tl.constexpr``, and
    a trailing `return` is dropped -- a kernel writes through its pointers and
    returns nothing, which is why the program it is emitted from takes its
    output as an argument.
    """
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
        tiled,
        drop_asserts,
    )

    params: list[str] = []
    for arg in func.args:
        name = str(arg.name)
        if name == block:
            params.append(f'{name}: tl.constexpr')
        elif isinstance(arg.type, ListTypeAnn):
            params.append(f'{name}_ptr')
        else:
            params.append(name)

    body = StmtBlock([
        stmt for stmt in func.body.stmts
        if not isinstance(stmt, ReturnStmt)
    ])
    out = _IndentedWriter()
    out.add_line('@triton.jit')
    out.add_line(f'def {func.name}({", ".join(params)}):')
    out.indent()
    emitter._visit_block(body, out)
    out.dedent()

    return KernelSource(
        name=func.name,
        source=out.render(),
        params=tuple(params),
        grid_extent=emitter.grid_extent,
        enable_fp_fusion=_products_are_exact(func, emitter),
    )
