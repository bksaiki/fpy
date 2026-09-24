"""
Triton backend: emitting kernel source.

**This module transliterates; it does not rewrite.**  Control flow is
`normalize`'s job, so an `if` statement arriving here is a bug in the normal
form, not a case to handle -- predicating one here would duplicate a decision
`SimplifyIf` and `_emit_where` already make between them.

What it decides is what has no expression in the FPy AST: storage and casts,
the `mask=` on an access, how a literal is spelled, which op-table signature
applies, and how a *sequence* is spelled at all -- Triton has no list value,
so one is expanded into its elements here whatever `Scalarize` left behind.  What it cannot spell it refuses -- never a fallback, and never
source that fails at `triton.jit`, since a refusal at least names its cause.

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
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from ...analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    AssignDef,
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    DefineUseAnalysis,
    Definition,
    FormatAnalysis,
    FormatInfer,
    PhiDef,
    TypeAnalysis,
    TypeInfer,
)
from ...analysis.array_size import ListSize, static_trip_count
from ...analysis.format_infer import (
    FormatBound,
    ListFormat,
    is_bottom,
    rounds_exactly,
    to_abstract,
)
from ...analysis.reaching_defs import same_object_defs
from ...analysis.storage_infer import StorageSelectionError, join, of_bound
from ...ast import (
    AllOf,
    AMax,
    AMin,
    And,
    AnyOf,
    Assign,
    BinaryOp,
    BoolVal,
    Cast,
    Compare,
    CompareOp,
    ConstInf,
    ConstNan,
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
    IfStmt,
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
    Logb,
    Max,
    Min,
    Mul,
    NamedId,
    NaryOp,
    Not,
    Or,
    Pow,
    Range1,
    Range3,
    Rational,
    RationalVal,
    ReturnStmt,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    Sum,
    TernaryOp,
    TupleBinding,
    TupleExpr,
    UnaryOp,
    UnderscoreId,
    Var,
)
from ...ast.visitor import DefaultVisitor, Visitor
from ...number import REAL, Context, Float, RealFloat, RoundingMode
from ...number.context.mp_fixed import MPFixedContext
from ...types import BoolType, ListType, RealType
from ...utils import Unionfind
from ..backend import CompileError
from .storage import (
    TritonStorageDomain,
    bound_fits_in_scalar,
    choose_storage_scalar,
    scalar_fits_in,
    to_triton,
)
from .target import (
    ScalarOpTable,
    TritonOp,
    downcast_rounding,
    is_native_ctx,
    make_op_table,
)
from .types import TritonScalar
from .vectorize import carried_scalars

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


_SPECIALS = frozenset({"float('nan')", "float('inf')", "(-float('inf'))"})
"""How `fp.nan()` and `fp.inf()` are spelled: Python floats, so weakly typed."""


def _as_literal(code: str) -> float | None:
    """*code* as a number, if that is all it is."""
    try:
        return float(code)
    except ValueError:
        return None


_ASSIGN = re.compile(r'(\w+) = (.*)')
_IDENT = re.compile(r'\b\w+\b')


class _IndentedWriter:
    """Line-oriented Triton source builder."""

    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0
        self.lanes: set[str] = set()
        """The names holding a tile rather than a scalar.

        Read off the emitted lines, since every runtime value is a name one
        of them binds: a name is a tile where its line reads `tl.arange` or
        another tile.  Taken to be a scalar in error, an address is only
        broadcast where it did not need to be."""

    def add_line(self, line: str = ''):
        self._lines.append('    ' * self._depth + line if line else '')
        if (m := _ASSIGN.fullmatch(line)) is not None:
            name, code = m.groups()
            if 'tl.arange(' in code or self.lanes & set(_IDENT.findall(code)):
                self.lanes.add(name)
            else:
                self.lanes.discard(name)

    def indent(self):
        self._depth += 1

    def dedent(self):
        self._depth -= 1

    def render(self) -> str:
        return '\n'.join(self._lines)


_SIGN_BITS: dict[TritonScalar, str] = {
    TritonScalar.F16: 'tl.int16',
    TritonScalar.F32: 'tl.int32',
    TritonScalar.F64: 'tl.int64',
}
"""The integer a float is bitcast to so its sign bit can be read."""

_INT_ROUND: dict[RoundingMode, str] = {
    RoundingMode.RTZ: 'libdevice.trunc',
    RoundingMode.RTN: 'libdevice.floor',
    RoundingMode.RTP: 'libdevice.ceil',
    RoundingMode.RNE: 'libdevice.nearbyint',
    RoundingMode.RNA: 'libdevice.round',
}
"""Rounding to the integers, by mode.

`nearbyint` is ties-to-even and C's `round` is ties-away, which is what
separates ``RNE`` from ``RNA``.  The modes with no C function -- ``RAZ``,
``RTO``, ``RTE`` -- are absent, so they refuse rather than round differently.
"""


def _integral_round(ctx: Context) -> str | None:
    """How to round to *ctx*, where it is the integers, or `None`.

    A bounded fixed-point context is deliberately not matched: it can
    overflow, and the check C++ emits for that is an `assert`, which a kernel
    cannot raise.
    """
    if not isinstance(ctx, MPFixedContext) or ctx.nmin != -1:
        return None
    return _INT_ROUND.get(ctx.rm)


_LOGB: dict[TritonScalar, tuple[int, int, int, float, int, str]] = {
    TritonScalar.F16: (15, 10, 0x1F, 2.0 ** -14, 11, 'tl.int16'),
    TritonScalar.F32: (127, 23, 0xFF, 2.0 ** -126, 24, 'tl.int32'),
    TritonScalar.F64: (1023, 52, 0x7FF, 2.0 ** -1022, 53, 'tl.int64'),
}
"""Per format: (bias, mantissa bits, exponent mask, smallest normal, the
power of two a subnormal is scaled by, the integer to bitcast through).

The scale is chosen so the *smallest* subnormal becomes normal: fp32's is
`2**-149`, and `2**-149 * 2**24` is `2**-125`.
"""


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
        types: TypeAnalysis,
        ctx_use: ContextUseAnalysis,
        sizes: ArraySizeAnalysis,
        op_table: ScalarOpTable,
        tiled: Sequence[ForStmt] = (),
        drop_asserts: bool = False,
        guards: Sequence[If1Stmt] = (),
        def_use: DefineUseAnalysis | None = None,
    ):
        self.func = func
        self.format_info = format_info
        self.types = types
        self.ctx_use = ctx_use
        self.sizes = sizes
        self.op_table = op_table
        self.tiled = tiled
        """The loops carrying a tile, as `tile_loops` reported them.

        Identity, not shape: recognizing a tiled loop by what it looks like
        would be pattern-matching another pass's output.
        """
        self.guards = guards
        """The tiles' `j < n` guards, as `tile_loops` reported them: a mask on
        the tile, where any other `if` is a branch."""
        self.def_use = def_use or DefineUse.analyze(func)
        # the classes `StorageInfer` coalesces: the definitions a phi joins
        defs = self.def_use.defs
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            for i in same_object_defs(d):
                uf.union(d, defs[i])
        self._class_of = {d: uf.find(d) for d in defs}
        self._members: dict[Definition, list[Definition]] = defaultdict(list)
        for d, c in self._class_of.items():
            self._members[c].append(d)
        self._class_ty: dict[Definition, TritonScalar | None] = {}
        """One storage per class, as the cpp backend declares a name: every
        read of a scalar name is in it and every assignment is cast into it,
        so a merge or a loop needs no cast.  Chosen per class on demand, since
        a class no read needs may have none."""
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
        self.rows: dict[NamedId, tuple[NamedId, list[Expr]]] = {}
        """Names bound to a *row* of a pointer-backed list, as (base, prefix).

        `A = Ass[r]` names a sub-list, which is neither a value Triton has nor
        a sequence that can scalarize -- the row lives behind a pointer.  So
        the binding is not emitted at all: it records that `A` is `Ass` at a
        prefix, and `A[i]` flattens back to one load off `Ass_ptr`.  That
        keeps the loop rolled, where scalarizing the row would unroll every
        load in it.
        """
        self.slices: dict[NamedId, tuple[NamedId, list[Expr], str]] = {}
        """Names bound to a *slice* of a pointer-backed list, as
        (base, prefix, start).

        A slice of something in memory is still in memory -- the same list at
        an offset -- so like a row it is an address rather than a value.
        Scalarizing it into that many loads throws the fact away, and then a
        subscript by anything but a constant has nothing to resolve against.
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
        self.grid_extent: int | str | None = None
        self.mask: str | None = None
        """The guard in force, as a Triton predicate.

        Code under a guard runs on every lane, and the guard becomes the
        `mask=` of each load and store.  So **an inactive lane may compute
        garbage**, provided it is masked off before it is observed: by the
        mask on an access, or by the `tl.where` that merges a branch.  A format
        bounds what an *active* lane computes.  Nothing emitted under a mask
        may trap or be undefined on garbage -- float overflow is `inf`, and
        integers wrap.  `None` where nothing encloses the access.
        """
        self._branches = 0
        """How many flattened branches enclose the statement being emitted."""
        self._lanes: set[str] = set()
        """The writer's `lanes`, bound when emission starts."""
        self._tile: str | None = None
        """The tile's index, while its body is emitted."""
        self._guard_mask: str | None = None
        self.size_params: dict[NamedId, str] = {}
        """Each unproven length a list argument has, by its size variable, to
        the kernel parameter the launcher fills it from."""
        """The mask of the tile's own guard while inside it: lanes past the
        end, and no branch."""

    # -- storage and context -------------------------------------------

    def _storage(self, e: Expr) -> TritonScalar:
        """The scalar storage the pipeline chose for *e*.

        Two analyses, each asked what it is for: the **type** says whether
        this is a boolean or a real, and only for a real does the **format**
        say which width.  Format inference is defined over real-valued
        expressions and structures of them, so reading "no format" as "must
        be a boolean" would be inferring a type from the absence of one --
        and wrong for the other things with no format, a rounding context or
        any other foreign value.
        """
        ty = self.types.by_expr.get(e)
        if isinstance(ty, BoolType):
            return TritonScalar.BOOL
        if not isinstance(ty, RealType):
            raise TritonEmitError(
                f'a `{type(ty).__name__ if ty else "?"}` has no Triton '
                f'storage, so `{type(e).__name__}` cannot be held'
            )
        if isinstance(e, Var) and (d := self.def_use.use_to_def.get(e)):
            # what the name holds, not what it can be here -- a read under a
            # branch is refined, where the storage is the class's
            held = self._class_storage(d)
            if held is not None:
                return held
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
        bound: FormatBound = None,
    ) -> str:
        """*code*, in *want*'s storage, or a refusal.

        An implicit narrowing is the fp16 trap, so it is never emitted; a
        program that needs one has to say so with `fp.cast`.

        *bound* is what format inference proved about the value, and it is
        the question soundness actually turns on -- the cpp backend's
        `_value_fits` says it best: `scalar_fits_in` asks whether the two
        *types* nest, where a conversion only needs the *values* to.  They
        come apart wherever storage is wider than the bound it was chosen to
        hold: an `e_zero = -132` reports `int16`, which no `float16` holds in
        general, and which this one holds exactly.
        """
        if have == want:
            return code
        if not self._value_fits(bound, have, want):
            raise TritonEmitError(
                f'emitting this would narrow {have.format()} to '
                f'{want.format()} implicitly, which rounds'
            )
        return self._explicit_cast(code, want)

    def _value_fits(
        self, bound: FormatBound, have: TritonScalar, want: TritonScalar,
    ) -> bool:
        """Can a value bounded by *bound*, held as *have*, live in *want*?"""
        return scalar_fits_in(have, want) or bound_fits_in_scalar(bound, want)

    def _explicit_cast(self, code: str, want: TritonScalar) -> str:
        """*code* cast to *want*.

        **A literal is retyped, not cast.**  `2.to(...)` lexes as `2.` then
        `to`, and parenthesizing it to `(2).to(...)` only moves the problem:
        it is valid Python, and Triton rejects it with *"'int' object has no
        attribute 'to'"* because a Python scalar is a `constexpr`, not a
        tile.  Writing the literal in the target's own spelling avoids the
        conversion entirely.
        """
        if code in _SPECIALS:
            return code     # a Python float, typed where it is used
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
        bounds = [self.format_info.by_expr.get(src) for _, src in operands]
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
                        self._maybe_cast(code, have, target, bound)
                        for code, have, bound in zip(codes, storages, bounds)
                    ])

        if active is REAL:
            widened = self._try_widen(e, sigs, codes, storages, bounds)
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
        bounds: list[FormatBound],
    ) -> str | None:
        """Under ``REAL``, compute at the width that holds the exact result.

        ``REAL`` has no storage of its own, so no signature names it.  But the
        storage the pipeline chose for *e* holds the operation's *unrounded*
        result -- that is what choosing it means -- so the operation at that
        width rounds to itself, and is therefore the ``REAL`` operation.

        This is what the running example turns on: an fp16 product needs 22
        bits and fp32 carries 24, so `x.to(tl.float32) * y.to(tl.float32)` is
        exact where `x * y` on fp16 operands is not.

        The result's width need not hold an operand -- `z * 0` is ±0, `2^139 *
        t` can fit `f32` -- so any width holding the operands and the result
        will do, the narrowest first, and the exact result is then cast down,
        as the cpp backend's `_try_widen` does.
        """
        target = self._storage(e)
        result = self.format_info.by_expr.get(e)

        def fits(have: TritonScalar, bound: FormatBound, slot: TritonScalar):
            return self._value_fits(bound, have, slot)

        slots = sorted(
            {sig.in_tys[0] for sig in sigs
             if len(sig.in_tys) == len(codes) and len(set(sig.in_tys)) == 1},
            # the result's own first: it needs no cast back
            key=lambda t: (t != target, t.float_bits() or t.int_bits() or 0),
        )
        for slot in slots:
            if slot != target and not fits(target, result, slot):
                continue
            if not all(fits(h, b, slot) for h, b in zip(storages, bounds)):
                continue
            sig = next(g for g in sigs if g.in_tys == (slot,) * len(codes))
            out = sig.format(*[
                self._maybe_cast(code, have, slot, bound)
                for code, have, bound in zip(codes, storages, bounds)
            ])
            return out if slot == target else self._explicit_cast(out, target)
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

    def _is_list(self, e: Expr) -> bool:
        """Whether *e* is a sequence rather than a scalar."""
        return isinstance(self.sizes.by_expr.get(e), ListSize)

    def _flatten(self, e: ListRef) -> tuple[NamedId, list[Expr], str | None]:
        """A subscript chain as its base and indices, outermost first.

        A base bound to a row resolves through to the pointer it came from,
        so `A = Ass[r]; A[i]` gives `Ass` at `[r, i]` -- one load, not a load
        of a load.
        """
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
        return self._resolve(cur.name, indices)

    def _resolve(
        self, name: NamedId, indices: list[Expr],
    ) -> tuple[NamedId, list[Expr], str | None]:
        """*name* followed through the rows and slices that bound it.

        Returns the base, its indices, and any constant offset a slice
        contributed -- kept as *code* rather than an expression, because it is
        address arithmetic rather than a program operation and has no business
        going through the op table.

        Both a load and a store go through here, so naming a sub-sequence
        works the same either side of the assignment.
        """
        seen: set[NamedId] = set()
        extra: list[str] = []
        while name not in seen:
            seen.add(name)
            if name in self.rows:
                name, prefix = self.rows[name]
                indices = prefix + indices
            elif name in self.slices:
                name, prefix, start = self.slices[name]
                indices = prefix + indices
                extra.append(start)
            else:
                break
        root = self._root(str(name))
        # every other list has stopped existing by here -- scalarized, or
        # resolved as a row -- so anything else would emit a `_ptr` that is
        # not a parameter
        if not any(str(a.name) == root and isinstance(a.type, ListTypeAnn)
                   for a in self.func.args):
            raise TritonEmitError(
                f'`{name}` is subscripted but is not a kernel argument, so '
                'there is no pointer to load from'
            )
        return name, indices, ' + '.join(extra) if extra else None

    def _slice_base(
        self, e: ListSlice,
    ) -> tuple[NamedId, list[Expr], str] | None:
        """*e* as (base, prefix, start) where it slices something in memory.

        `None` where it does not -- a slice of a local sequence of computed
        values has no address, and scalarizing it is right.  What provenance
        decides is which of the two a list is; it is not a choice.
        """
        if not isinstance(e.value, (Var, ListRef)):
            return None
        try:
            if isinstance(e.value, Var):
                base, prefix, extra = self._resolve(e.value.name, [])
            else:
                base, prefix, extra = self._flatten(e.value)
        except TritonEmitError:
            return None
        if extra is not None:
            # a slice of a slice: one offset is all this carries for now
            return None
        start = '0' if e.start is None else self.emit(e.start)
        return base, prefix, start

    def _size_code(self, size: object) -> str | None:
        """A length as code: a constant, or the kernel parameter holding it."""
        if isinstance(size, int):
            return str(size)
        if isinstance(size, NamedId):
            return self.size_params.get(size)
        return None

    def _strides(self, base: NamedId, rank: int) -> list[str]:
        """Row-major strides for *base*, as code: from its proven shape, or a
        size the kernel takes as a parameter.

        A kernel argument is a flat pointer, so a length neither proven nor a
        parameter has no offset arithmetic to emit.
        """
        bound = next(
            (b for defn, b in self.sizes.by_def.items() if defn.name == base),
            None,
        )
        dims: list[str | None] = []
        while isinstance(bound, ListSize):
            dims.append(self._size_code(bound.size))
            bound = bound.elt
        if len(dims) < rank:
            raise TritonEmitError(
                f'`{base}` is subscripted {rank} deep but only {len(dims)} '
                'dimensions are proven'
            )
        strides = ['1'] * rank
        for i in range(rank - 2, -1, -1):
            dim = dims[i + 1]
            if dim is None:
                raise TritonEmitError(
                    f'`{base}` has no proven length at depth {i + 1}, so its '
                    'offsets cannot be computed'
                )
            strides[i] = _times(strides[i + 1], dim)
        return strides

    def _offset(
        self, base: NamedId, indices: list[Expr], extra: str | None = None,
    ) -> str:
        """The flat element offset, row-major, plus any slice start."""
        strides = self._strides(base, len(indices))
        terms = [
            self.emit(idx) if st == '1' else f'{self.emit(idx)} * {st}'
            for idx, st in zip(indices, strides)
        ]
        if extra is not None:
            terms.append(extra)
        # a zero term is address arithmetic noise, not a value
        terms = [t for t in terms if t != '0']
        return ' + '.join(terms) if terms else '0'

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
        return self._masked_load(f'{base}_ptr + {offset}')

    def _masked_load(self, addr: str) -> str:
        """`tl.load` at *addr*, under whatever guard is in force.

        Triton rejects a tile of a mask on a scalar address, so a scalar one
        under a branch is broadcast to the tile rather than unmasked: a branch
        can guard an address on every lane at once.  Under the tile's own
        guard alone it is a scalar load.
        """
        if self.mask is None:
            return f'tl.load({addr})'
        if self._tile is not None and not (
            self._lanes & set(_IDENT.findall(addr))
        ):
            if self.mask == self._guard_mask:
                # every launched program has a live lane, which reads it:
                # one scalar load, where a vector one costs Triton's
                # coalescing superlinearly
                return f'tl.load({addr})'
            addr = f'{addr} + tl.zeros_like({self._tile})'
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
        base, indices, extra = self._flatten(e)
        return self._masked_load(
            f'{base}_ptr + {self._offset(base, indices, extra)}')

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
        code = self._size_code(bound.size) if isinstance(bound, ListSize) else None
        if code is None:
            raise TritonEmitError(
                'a length neither proven nor an argument\'s has no Triton '
                'spelling; a kernel argument is a pointer and carries no length'
            )
        return code

    def _emit_round(self, e: Round | Cast) -> str:
        """An explicit `fp.round` / `fp.cast`, which is a *cast*, not an
        operation the table dispatches.

        Where the round changes nothing it emits nothing -- a literal already
        representable in the target needs no `.to`, and Triton has no spelling
        for casting one.  Otherwise it is a cast, which is sound only under a
        context whose `round` *is* the hardware conversion; `is_native_ctx`
        answers exactly that question.

        A fixed-point context sitting at position zero is the exception: its
        values are the integers, so the round is one of the C integral
        roundings rather than a conversion.  That is the form `RescaleFixed`
        leaves behind, and it is how a fixed-point grid whose position is
        computed at *runtime* is spelled at all -- the scale moves into
        `ldexp` either side, and what is left rounds to a concrete context.
        """
        arg = self.emit(e.arg)
        ctx = self._active_ctx(e)
        if rounds_exactly(e, self.format_info.by_expr, ctx):
            return arg
        if isinstance(e, Cast) and not self.drop_asserts:
            # the cpp backend checks the claim at runtime; a kernel cannot
            raise TritonEmitError(
                '`fp.cast` asserts its result is exact, which is not proven '
                'here and a kernel cannot check.  Pass `drop_asserts` to round '
                'without the check'
            )
        integral = _integral_round(ctx)
        if integral is not None:
            return f'{integral}({arg})'
        if not is_native_ctx(ctx):
            return self._emit_downcast(e, arg, ctx)
        return self._explicit_cast(arg, self._storage(e))

    def _emit_downcast(self, e: Round | Cast, arg: str, ctx: Context) -> str:
        """A `round` Triton spells as a narrowing cast under a rounding mode.

        `tl.cast` takes an `fp_downcast_rounding` that reaches one context
        more than `is_native_ctx`: `x.to(tl.float16,
        fp_downcast_rounding="rtz")` **is** FP16's round-toward-zero.  Which
        conversions it covers is `downcast_rounding`'s to say.  A literal is
        excluded for the reason `_explicit_cast` gives -- it would be retyped
        rather than cast, and retyping rounds to nearest whatever mode is
        asked for.
        """
        want = self._storage(e)
        rm = downcast_rounding(ctx, self._storage(e.arg))
        if rm is None or _as_literal(arg) is not None:
            raise TritonEmitError(
                f'`{type(e).__name__.lower()}` to `{ctx}` is not a hardware '
                'conversion, so it has no cast spelling'
            )
        return f'{arg}.to({want.format()}, fp_downcast_rounding="{rm}")'

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

        `signbit` is not one of them: it has to separate `-0.0` from `0.0`,
        which no *float* comparison does.  It reads the sign bit instead --
        a bitcast to the same-width integer, tested for negative -- checked
        on the card against the interpreter over both zeros, both infinities
        and finite values of each sign.  For an integer there is no `-0`, so
        the test is direct.

        **A NaN's sign is not among them.**  FPy does not distinguish one:
        rounding normalizes it away, so there is no signed NaN for this to
        agree or disagree with.  What the hardware reports for one is
        therefore unspecified rather than wrong, and nothing here should rely
        on it.
        """
        arg = self.emit(e.arg)
        if isinstance(e, IsNan):
            return f'({arg} != {arg})'
        if isinstance(e, IsInf):
            return f"(tl.abs({arg}) == float('inf'))"
        if isinstance(e, IsFinite):
            return f"(tl.abs({arg}) < float('inf'))"
        if isinstance(e, Signbit):
            have = self._storage(e.arg)
            if have.is_integer():
                return f'({arg} < 0)'
            bits = _SIGN_BITS.get(have)
            if bits is None:
                raise TritonEmitError(
                    f'`signbit` needs an integer the width of {have.format()} '
                    'to read the sign bit from, and there is none'
                )
            return f'({arg}.to({bits}, bitcast=True) < 0)'
        raise TritonEmitError(
            f'no Triton spelling for `{type(e).__name__}`'
        )

    def _emit_ldexp(self, e: Mul) -> str | None:
        """``2 ** n * v`` as ``libdevice.ldexp(v, n)``, or `None`.

        `ldexp` is IEEE 754's `scaleB`: multiplication by an integral power of
        two, exact but for overflow and underflow.  The product it replaces
        rounds twice and rests on `exp2` returning `2 ** n` exactly, which
        IEEE only *recommends* -- so a library within one ulp would give a
        scale that is not the power.  This is the shape `RescaleFixed` emits
        for every rounding it moves, so it is how a fixed-point context with a
        *runtime* scale reaches this backend at all: the data dependence moves
        out of the context, which becomes concrete, and into this arithmetic.

        Because `ldexp` computes the *exact* product, it stands in for the
        multiply only where the context would not round it -- otherwise it
        would skip a rounding the program asked for.  Taken from the cpp
        backend's `_ldexp`, which draws the same line.
        """
        for scale, value in ((e.first, e.second), (e.second, e.first)):
            if not isinstance(scale, Pow) or len(scale.args) != 2:
                continue
            base, exp = scale.args
            if not (isinstance(base, RationalVal)
                    and base.as_rational() == 2):
                continue
            # only where the context does not round: `ldexp` is exact, and
            # standing in for a rounded product would drop the rounding
            active = self._active_ctx(e)
            if active is not REAL and not rounds_exactly(
                e, self.format_info.by_expr, active,
            ):
                return None
            n = self.emit(exp)
            held = self._storage(exp)
            if held.is_integer() and held != TritonScalar.S32:
                n = self._maybe_cast(
                    n, held, TritonScalar.S32,
                    self.format_info.by_expr.get(exp))
            elif held.is_float():
                # `ldexp` takes an `int32`: exact where every finite value
                # is an integer it holds; a special one is on a dead lane
                af = to_abstract(self.format_info.by_expr.get(exp))
                if af is None or af.exp < 0 or not all(
                    isinstance(b, RealFloat) and abs(b) < 2 ** 31
                    for b in (af.pos_bound, af.neg_bound)
                ):
                    return None
                n = f'{n}.to(tl.int32)'
            # in the product's storage, which may be wider than the operand's:
            # `ldexp` computes in its argument's type.  An unbounded product
            # has none, and stays in the operand's.
            v = self.emit(value)
            have = self._storage(value)
            try:
                want = self._storage(e)
            except StorageSelectionError:
                want = have
            v = self._maybe_cast(
                v, have, want, self.format_info.by_expr.get(value))
            return f'libdevice.ldexp({v}, {n})'
        return None

    def _emit_logb(self, e: Logb) -> str:
        """IEEE 754 `logB`: the exponent of *x*, read from its bits.

        There is no correctly-rounded primitive to call -- `tl.log2` is a
        transcendental, which the op table excludes by design -- so the
        exponent field is taken directly.  Exact, because it reads the value
        rather than computing one.

        **A subnormal is scaled into range rather than counted.**  Its
        exponent field is zero and its true exponent depends on where the
        leading one sits, which would want a count-leading-zeros; multiplying
        by `2**k` makes it normal and the `k` comes back off afterwards.  The
        multiply is exact -- it only moves the exponent -- and `k` is chosen
        so the smallest subnormal lands on a normal.

        The three specials are `logB`'s own: `+/-0` is `-inf`, `+/-inf` is
        `+inf`, a NaN is a NaN.  Checked against the interpreter on the card
        over every fp32 exponent boundary of both signs and 3000 random bit
        patterns.

        The operand is repeated, which is free for a name and a load for a
        subscript.  It is pure and identically masked, so a CSE pass should
        collapse the copies -- unverified here, as for the `max` fold.
        """
        have = self._storage(e.arg)
        spec = _LOGB.get(have)
        if spec is None:
            raise TritonEmitError(
                f'`logb` reads an exponent field, and {have.format()} has '
                'none to read'
            )
        want = self._storage(e)
        if want.is_integer():
            # `logb(0)` is `-inf`, which an integer cannot hold: the program
            # aborts there and a kernel cannot
            raise TritonEmitError(
                f'`logb` gives `-inf` at zero, which {want.format()} cannot '
                'hold'
            )
        bias, mant, mask, min_normal, scale, ity = spec
        x = self.emit(e.arg)
        fty = want.format()
        sub = f'(tl.abs({x}) < {min_normal!r})'
        scaled = f'tl.where({sub}, {x} * {float(2 ** scale)!r}, {x})'
        bits = f'({scaled}).to({ity}, bitcast=True)'
        exp = f'((({bits} >> {mant}) & {mask}) - {bias})'
        adj = f'tl.where({sub}, {exp} - {scale}, {exp}).to({fty})'
        at_zero = f"tl.where(tl.abs({x}) == 0.0, float('-inf'), {adj})"
        at_inf = (f"tl.where(tl.abs({x}) == float('inf'), float('inf'), "
                  f'{at_zero})')
        return f"tl.where({x} != {x}, float('nan'), {at_inf})"

    def _emit_reduction(self, e: UnaryOp) -> str:
        """`max`, `min` or `sum` over a sequence, folded over its elements.

        The sequence has already stopped existing, so a reduction is a fold
        over values -- which is also what keeps `sum` exact: FPy's is a *left*
        fold seeded with the first element unrounded, and folding the
        scalarized elements left to right is that, not an approximation of it.

        A tile reduction would be the alternative and is not available: it
        reassociates, which `sum` does not permit and which the roadmap
        retired tile reductions over.
        """
        elems = self._source_elements(e.arg)
        if elems is None:
            raise TritonEmitError(
                f'`{type(e).__name__.lower()}` needs a sequence of proven '
                'length to fold over'
            )
        if isinstance(e, (AnyOf, AllOf)):
            # `&` / `|` rather than Python's keywords, which short-circuit and
            # return an operand rather than a tile -- the same reason
            # `_emit_connective` spells `and` / `or` that way.  The empty fold
            # is each one's identity, which is what FPy gives.
            if not elems:
                return 'False' if isinstance(e, AnyOf) else 'True'
            op = '|' if isinstance(e, AnyOf) else '&'
            return '(' + f' {op} '.join(elems) + ')'
        name = 'tl.maximum' if isinstance(e, AMax) else 'tl.minimum'
        if isinstance(e, Sum):
            if not elems:
                # FPy's empty sum is an exact `+0`; as a literal it retypes
                # where it is used, like any other, and broadcasts over a tile
                return self._emit_numeric_literal(Fraction(0))
            if self._active_ctx(e) is REAL:
                # every partial sum lies in the sum's own format, so its
                # storage holds each exactly -- where every element fits it.
                # A sum with none is refused where it is assigned.
                try:
                    want = self._storage(e)
                except StorageSelectionError:
                    want = None
                seq = self.format_info.by_expr.get(e.arg)
                if want is not None and isinstance(seq, ListFormat) and (
                    bound_fits_in_scalar(seq.elt, want)
                ):
                    elems = [self._explicit_cast(c, want) for c in elems]
            acc = elems[0]
            for rhs in elems[1:]:
                acc = f'({acc} + {rhs})'
            return acc
        if not elems:
            # FPy raises `ValueError` here, so there is nothing to emit
            raise TritonEmitError(
                f'`{name}` of an empty sequence has no value'
            )
        return self._fold_select(name, elems)

    def _fold_select(self, name: str, args: list[str]) -> str:
        """A `max`/`min` fold.

        Pairwise is sound because both are associative *and* exact: they
        return an operand rather than computing one, so no grouping rounds
        differently.
        """
        acc = args[0]
        for rhs in args[1:]:
            acc = f'{name}({acc}, {rhs}, propagate_nan=tl.PropagateNan.ALL)'
        return acc

    def _emit_select_op(self, e: Max | Min) -> str:
        """`max` / `min`, folded pairwise.

        `propagate_nan` is not optional.  FPy follows IEEE 754-2019
        `maximum`, where a NaN operand propagates; Triton's default is
        `PropagateNan.NONE`, which is `maximumNumber` and returns the *other*
        operand.  Checked on hardware: FPy gives `nan` for `max(nan, 1.0)`
        and a bare `tl.maximum` gives `1.0`.
        """
        name = 'tl.maximum' if isinstance(e, Max) else 'tl.minimum'
        want = self._storage(e)
        args = [
            self._maybe_cast(self.emit(a), self._storage(a), want,
                             self.format_info.by_expr.get(a))
            for a in e.args
        ]
        if not args:
            raise TritonEmitError(f'`{name}` needs at least one operand')
        return self._fold_select(name, args)

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
            self._maybe_cast(self.emit(a), self._storage(a), want,
                              self.format_info.by_expr.get(a))
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
        return self._emit_real_literal(e)

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return self._emit_real_literal(e)

    def _emit_real_literal(self, e: Decnum | Hexnum) -> str:
        """A negative zero, which no `Fraction` holds, is a `Float`.  Triton
        folds a `-0.0` constant to `+0.0` -- even through `tl.full` -- so it is
        a negated zero tile, which is not folded."""
        r = e.as_real()
        if isinstance(r, Float):
            return f'(-tl.zeros((), {self._storage(e).format()}))'
        return self._emit_numeric_literal(r)

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
        # Triton refuses an integer literal no `int64` holds
        if v.denominator == 1 and -2 ** 63 <= v.numerator < 2 ** 63:
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
        if isinstance(e, Logb):
            return self._emit_logb(e)
        if isinstance(e, (AMax, AMin, Sum, AnyOf, AllOf)):
            return self._emit_reduction(e)
        return self._dispatch(
            e, self.op_table.unary, [(self.emit(e.arg), e.arg)],
        )

    def _visit_binaryop(self, e: BinaryOp, ctx) -> str:
        if isinstance(e, Mul):
            scaled = self._emit_ldexp(e)
            if scaled is not None:
                return scaled
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

    def _visit_nullaryop(self, e, ctx) -> str:
        """`nan` and `inf`, which are *values* rather than operations.

        So they are spelled like any other literal and retyped where used,
        not cast -- `float('nan')` is a Python float and Triton types it in
        context.

        **`fp.nan()` means `C.round(nan)`**, so a context that cannot hold one
        makes it an abort rather than a value: `fp.INTEGER.round(nan)` raises
        *"Cannot round NaN under this context"*, as `1 / 0` there does.  A
        kernel cannot raise, so this is refused.

        The rest of the table is transcendental (`pi`, `e`, `log2(e)`, ...),
        which this backend excludes by design: there is no correctly-rounded
        constant to emit, and a rounded one is a different program.
        """
        if isinstance(e, (ConstNan, ConstInf)):
            what = 'NaN' if isinstance(e, ConstNan) else 'infinity'
            ctx_ = self._active_ctx(e)
            ok = getattr(
                ctx_, 'enable_nan' if isinstance(e, ConstNan) else 'enable_inf',
                True,
            )
            if not ok:
                raise TritonEmitError(
                    f'`{type(e).__name__.lower()}` is `round({what.lower()})` '
                    f'under this context, which cannot hold one -- the '
                    f'program aborts there and a kernel cannot'
                )
            return "float('nan')" if isinstance(e, ConstNan) else "float('inf')"
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
        # ahead of scalarizing: a slice of something in memory is an
        # address, and taking it apart into loads loses that
        if isinstance(stmt.expr, ListSlice):
            bound = self._slice_base(stmt.expr)
            if bound is not None:
                self.slices[stmt.target] = bound
                return
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
        if isinstance(stmt.expr, ListRef) and self._is_list(stmt.expr):
            base, indices, extra = self._flatten(stmt.expr)
            if extra is not None:
                return  # a row of a slice: no place to keep the offset yet
            self.rows[stmt.target] = (base, indices)
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
        code = self._into_class(stmt, self.emit(stmt.expr))
        if code.isdigit():
            self.consts[str(stmt.target)] = int(code)
        elif isinstance(stmt.expr, Var):
            self.copies[str(stmt.target)] = self._root(code)
        ctx.add_line(f'{stmt.target} = {code}')

    def _into_class(self, stmt: Assign, code: str) -> str:
        """*code*, the value *stmt* assigns, in its target's class storage."""
        assert isinstance(stmt.target, NamedId)
        d = self.def_use.find_def_from_site(stmt.target, stmt)
        if not isinstance(self.types.by_def.get(d), RealType):
            return code
        want = self._class_storage(d)
        if want is None:
            # a name is held in one storage, as a declaration has one type
            raise TritonEmitError(
                f'no storage holds every value `{stmt.target}` is assigned'
            )
        return self._maybe_cast(
            code, self._storage(stmt.expr), want,
            self.format_info.by_expr.get(stmt.expr),
        )

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
        base, indices, extra = self._resolve(stmt.var, list(stmt.indices))
        addr = f'{base}_ptr + {self._offset(base, indices, extra)}'
        val = self.emit(stmt.expr)
        mask = '' if self.mask is None else f', mask={self.mask}'
        ctx.add_line(f'tl.store({addr}, {val}{mask})')

    def _visit_return(self, stmt: ReturnStmt, ctx: _IndentedWriter):
        if self._branches:
            raise TritonEmitError(
                'a `return` in a branch has no Triton spelling; a flattened '
                'branch runs on every lane'
            )
        ctx.add_line(f'return {self.emit(stmt.expr)}')

    def _visit_context(self, stmt: ContextStmt, ctx: _IndentedWriter):
        # a context change is a change of *storage*, which the dispatch reads
        # per expression; it has no statement of its own
        self._visit_block(stmt.body, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: _IndentedWriter):
        if not any(stmt is g for g in self.guards):
            return self._emit_branch(stmt, stmt.body, None, ctx)
        # a tile's guard only drops the over-run: nothing merges out of it
        prev, prev_guard = self.mask, self._guard_mask
        self.mask = self._guard_mask = self.emit(stmt.cond)
        self._visit_block(stmt.body, ctx)
        self.mask, self._guard_mask = prev, prev_guard

    def _visit_if(self, stmt: IfStmt, ctx: _IndentedWriter):
        self._emit_branch(stmt, stmt.ift, stmt.iff, ctx)

    def _bind(self, code: str, ctx: _IndentedWriter) -> str:
        """*code*, evaluated once into a temporary."""
        name = f'__t{self._next_tmp}'
        self._next_tmp += 1
        ctx.add_line(f'{name} = {code}')
        return name

    def _class_storage(self, d: Definition) -> TritonScalar | None:
        """The storage of *d*'s class, or `None` where it holds no scalar or
        no storage holds it."""
        if isinstance(self.types.by_def.get(d), BoolType):
            return TritonScalar.BOOL
        c = self._class_of[d]
        if c not in self._class_ty:
            # `StorageInfer`'s join: an empty list's bottom constrains nothing
            bounds = [self.format_info.by_def.get(m) for m in self._members[c]]
            kept = [b for b in bounds if not is_bottom(b)] or bounds
            dom = TritonStorageDomain()
            try:
                ty = to_triton(join(dom, [of_bound(dom, b) for b in kept]))
            except StorageSelectionError:
                ty = None
            self._class_ty[c] = ty if isinstance(ty, TritonScalar) else None
        return self._class_ty[c]

    def _emit_branch(
        self, stmt: IfStmt | If1Stmt, ift: StmtBlock, iff: StmtBlock | None,
        ctx: _IndentedWriter,
    ):
        """An `if`, flattened.

        Each arm runs on every lane, under the enclosing mask and its guard;
        each name the `if` merges is then chosen by `tl.where`.  The arms are
        emitted with their own names -- the analyses key on nodes, so the AST
        is not renamed -- and a name the first arm overwrites is saved before
        it and restored after, for the second arm and the merge to read.
        """
        # a list behind a pointer merges by its masked stores alone
        phis = []
        for p in self.def_use.phis[stmt]:
            if not isinstance(self.types.by_def.get(p), ListType):
                phis.append(p)
            elif p.name in self.seqs:
                raise TritonEmitError(
                    f'`{p.name}` is a list chosen by a branch, which has no '
                    'Triton value'
                )
        merged = {p.name for p in phis}
        named: tuple[dict, ...] = (self.consts, self.copies, self.ranges,
                                   self.rows, self.slices, self.seqs)
        saved_named = [dict(m) for m in named]

        def restore(drop: set[NamedId]):
            # what an arm bound is gone, and a merged name is no constant
            for m, prev in zip(named, saved_named):
                m.clear()
                m.update(prev)
                for v in drop:
                    m.pop(v, None)
                    m.pop(str(v), None)

        outer = self.mask

        cond = self._bind(self.emit(stmt.cond), ctx)
        saved = {
            v: self._bind(str(v), ctx)
            for v in sorted(self.def_use.mutated_in(ift)) if v in merged
        }
        self._branches += 1
        self.mask = cond if outer is None else f'({outer} & {cond})'
        self._visit_block(ift, ctx)
        taken = {p.name: self._bind(str(p.name), ctx) for p in phis}
        for v, code in saved.items():
            ctx.add_line(f'{v} = {code}')
        restore(set())
        if iff is not None:
            self.mask = f'(~{cond})' if outer is None else f'({outer} & ~{cond})'
            self._visit_block(iff, ctx)
        self._branches -= 1
        self.mask = outer

        # a phi's sides share its class, so they are in one storage already
        for p in phis:
            ctx.add_line(f'{p.name} = tl.where({cond}, {taken[p.name]}, {p.name})')
        restore(merged)

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
        n = static_trip_count(stmt.iterable, self.sizes)
        if not isinstance(n, int):
            # a length the kernel takes as a parameter: a loop at runtime,
            # which Triton needs to carry each value at one type -- where
            # `tl.static_range`, unrolled while tracing, did not
            if carried := carried_scalars(stmt, self.def_use):
                raise TritonEmitError(
                    f'a loop with a runtime count carries `{min(carried)}`, '
                    'which Triton needs at one type on every iteration; this '
                    'backend does not yet arrange that'
                )
            it = stmt.iterable
            args = ([self.emit(it.arg)] if isinstance(it, Range1)
                    else [self.emit(a) for a in (it.first, it.second, it.third)])
            ctx.add_line(f'for {stmt.target} in range({", ".join(args)}):')
            ctx.indent()
            self._visit_block(stmt.body, ctx)
            ctx.dedent()
            return
        if isinstance(stmt.iterable, Range1):
            ctx.add_line(f'for {stmt.target} in tl.static_range({n}):')
            ctx.indent()
        else:
            # `static_range` counts from zero: the target is where the
            # count lands in `range(start, stop, step)`
            k = f'__t{self._next_tmp}'
            self._next_tmp += 1
            start = self.emit(stmt.iterable.first)
            step = self.emit(stmt.iterable.third)
            ctx.add_line(f'for {k} in tl.static_range({n}):')
            ctx.indent()
            ctx.add_line(f'{stmt.target} = {start} + {k} * {step}')
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
        if carried := carried_scalars(stmt, self.def_use):
            raise TritonEmitError(
                f'a tiled loop carrying `{min(carried)}` needs a reduction '
                "across the tile's lanes, which this backend does not lower"
            )
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
        stop = it.second
        if isinstance(stop, Var):
            # the split bound the length to a name; the grid wants the length
            d = self.def_use.find_def_from_use(stop)
            if (isinstance(d, AssignDef) and isinstance(d.site, Assign)
                    and isinstance(d.site.expr, Len)):
                stop = d.site.expr
        bound = self._root(self.emit(stop))
        self.grid_extent = (
            int(bound) if bound.isdigit()
            else bound if bound in self.size_params.values()
            else self.consts.get(bound)
        )
        ctx.add_line(f'{outer} = tl.program_id(0) * {width}')
        ctx.add_line(f'{inner.target} = {outer} + tl.arange(0, {width})')
        prev, self._tile = self._tile, str(inner.target)
        self._visit_block(inner.body, ctx)
        self._tile = prev

    def _visit_block(self, block: StmtBlock, ctx: _IndentedWriter):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    # -- statements with no Triton spelling ----------------------------

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
        FormatInfer.analyze(func, use_digit_bounds=True),
        TypeInfer.check(func),
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
    guards: Sequence[If1Stmt] = (),
) -> str:
    """*block*, as Triton source.

    *tiled* and *guards* name the loops carrying a tile and the guards on
    them, as `tile_loops` reported them.  *drop_asserts* skips an `assert`
    rather than refusing it.
    """
    if not isinstance(block, StmtBlock):
        raise TypeError(f"Expected a 'StmtBlock', got {block}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    def_use = DefineUse.analyze(func)
    sizes = ArraySizeInfer.analyze(func)
    emitter = _Emitter(
        func,
        FormatInfer.analyze(func, use_digit_bounds=True),
        TypeInfer.check(func),
        ContextUse.analyze(func, def_use=def_use),
        sizes,
        make_op_table(),
        tiled,
        drop_asserts,
        guards,
        def_use,
    )
    out = _IndentedWriter()
    emitter._lanes = out.lanes
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

    grid_extent: int | str | None
    """How many elements the tiled dimension covers, if one was tiled.

    The launcher needs it to size the grid -- ``cdiv(extent, BLOCK)`` program
    instances: a proven length, or the name of the size parameter holding it.
    ``None`` where nothing was tiled.
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

    sizes: tuple[tuple[str, int, int], ...] = ()
    """Each size parameter, after the arguments: its name, and the argument
    position and dimension whose length it is."""


def _times(a: str, b: str) -> str:
    """``a * b`` as code, folded where both are constants."""
    if a.isdigit() and b.isdigit():
        return str(int(a) * int(b))
    return b if a == '1' else a if b == '1' else f'{a} * {b}'


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
    guards: Sequence[If1Stmt] = (),
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
        FormatInfer.analyze(func, use_digit_bounds=True),
        TypeInfer.check(func),
        ContextUse.analyze(func, def_use=def_use),
        sizes,
        make_op_table(),
        tiled,
        drop_asserts,
        guards,
        def_use,
    )

    params: list[str] = []
    size_params: list[tuple[str, int, int]] = []
    for pos, arg in enumerate(func.args):
        name = str(arg.name)
        if name == block:
            params.append(f'{name}: tl.constexpr')
        elif isinstance(arg.type, ListTypeAnn):
            params.append(f'{name}_ptr')
            # an unproven length is the kernel's to be told, as Triton's
            # own kernels take theirs
            bound = sizes.by_def.get(def_use.find_def_from_site(arg.name, arg))
            depth = 0
            while isinstance(bound, ListSize):
                if isinstance(bound.size, NamedId) and bound.size not in emitter.size_params:
                    emitter.size_params[bound.size] = f'{name}_n{depth}'
                    size_params.append((f'{name}_n{depth}', pos, depth))
                bound, depth = bound.elt, depth + 1
        else:
            params.append(name)
    params.extend(name for name, _, _ in size_params)

    body = StmtBlock([
        stmt for stmt in func.body.stmts
        if not isinstance(stmt, ReturnStmt)
    ])
    out = _IndentedWriter()
    emitter._lanes = out.lanes
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
        sizes=tuple(size_params),
    )
