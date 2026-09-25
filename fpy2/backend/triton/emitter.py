"""
Triton backend: emitting kernel source.

Transliterates the normal form into a `@triton.jit` kernel, deciding only
what the FPy AST leaves open: each value's storage and the casts between
them, the `mask=` on each access, how a literal is spelled, which op-table
signature applies, and how a list is held -- behind a kernel argument's
pointer, as a register tile, or as one value per element.  Operands are cast
into a signature's storage before the operation, since Triton computes
`fp16 * fp16` in fp16.  A lossy cast or a missing signature is refused, never
a fallback.
"""

import math
import re
import struct
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
from ...analysis.format_infer.format import AbstractFormat
from ...analysis.reaching_defs import same_object_defs
from ...analysis.storage_infer import StorageSelectionError, join, of_bound
from ...ast import (
    AllOf,
    AMax,
    AMin,
    And,
    AnyOf,
    AssertStmt,
    Assign,
    Attribute,
    BinaryOp,
    BoolVal,
    Call,
    Cast,
    Compare,
    CompareOp,
    ConstInf,
    ConstNan,
    ContextStmt,
    Decnum,
    Digits,
    EffectStmt,
    Empty,
    Expr,
    ForeignVal,
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
    NullaryOp,
    Or,
    PassStmt,
    Pow,
    Range1,
    Range3,
    Rational,
    RationalVal,
    ReturnStmt,
    Round,
    Signbit,
    StmtBlock,
    Sum,
    TernaryOp,
    TupleBinding,
    TupleExpr,
    UnaryOp,
    UnderscoreId,
    Var,
    WhileStmt,
)
from ...ast.visitor import Visitor
from ...number import INTEGER, REAL, Context, Float, RealFloat, RoundingMode
from ...number.context.mp_fixed import MPFixedContext
from ...transform.path import walk_exprs, walk_stmts
from ...transform.simplify_if import _reads
from ...types import BoolType, ListType, RealType
from ...utils import Unionfind
from ..backend import CompileError
from .storage import (
    TritonStorageDomain,
    bound_fits_in_scalar,
    choose_storage_scalar,
    scalar_fits_in,
    scalar_sup,
    to_triton,
)
from .target import (
    ScalarOpTable,
    TritonOp,
    _int_ctxs,
    directed_cast,
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


_SPECIALS: dict[str, float] = {
    "float('nan')": math.nan,
    "float('inf')": math.inf,
    "(-float('inf'))": -math.inf,
}
"""How `fp.nan()` and `fp.inf()` are spelled: Python floats, so weakly typed."""

_NUMBER = re.compile(r'-?\d[\w.+-]*')


def _as_literal(code: str) -> float | None:
    """*code* as a number, if that is all it is."""
    if not _NUMBER.fullmatch(code):
        return None
    try:
        return float(code)
    except ValueError:
        return None


def _as_number(code: str) -> int | float | bool | None:
    """*code* as the Python number Triton sees, if that is all it is."""
    if code in _SPECIALS:
        return _SPECIALS[code]
    if code in ('True', 'False'):
        return code == 'True'
    n = _literal_int(code)
    return n if n is not None else _as_literal(code)


def _fp32_exact(v: float) -> bool:
    """Whether Triton types the Python float *v* as an `fp32` holding it: it
    takes `fp32` for a zero, a special or an `fp32` normal, else `fp64`."""
    if math.isnan(v) or v == 0 or math.isinf(v):
        return True
    return (2.0 ** -126 <= abs(v) <= 3.4028234663852886e38
            and struct.unpack('f', struct.pack('f', v))[0] == v)


def _literal_int(code: str) -> int | None:
    """*code* as an integer, if that is all it is."""
    try:
        return int(code)
    except ValueError:
        return None


_ASSIGN = re.compile(r'(\w+) = (.*)')
_IDENT = re.compile(r'\b\w+\b')


class _IndentedWriter:
    """Line-oriented Triton source builder, tracking the shape each line
    binds."""

    _lines: list[str]
    _depth: int
    rows: set[str]
    """The names holding a row vector: a line reading `tl.arange` or another
    one.  An address wrongly taken for a scalar is only broadcast needlessly."""
    wide: set[str]
    """The names holding a `[rows, lanes]` tile: a line reading one, or
    broadcasting across the lanes."""

    def __init__(self) -> None:
        self._lines = []
        self._depth = 0
        self.rows = set()
        self.wide = set()

    def add_line(self, line: str = '', shape: str | None = None) -> None:
        """*line*; *shape* ('wide', 'row' or 'scalar') says what it binds
        where reading it off the line would not."""
        self._lines.append('    ' * self._depth + line if line else '')
        if (m := _ASSIGN.fullmatch(line)) is not None:
            name, code = m.groups()
            idents = set(_IDENT.findall(code))
            if shape is None:
                if '[None, :]' in code or '[:, None]' in code or self.wide & idents:
                    shape = 'wide'
                elif 'tl.arange(' in code or self.rows & idents:
                    shape = 'row'
            self.wide.discard(name)
            self.rows.discard(name)
            if shape == 'wide':
                self.wide.add(name)
            elif shape == 'row':
                self.rows.add(name)

    def indent(self) -> None:
        self._depth += 1

    def dedent(self) -> None:
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

    A bounded fixed-point context is not matched: its overflow aborts, which
    a kernel cannot.
    """
    if not isinstance(ctx, MPFixedContext) or ctx.nmin != -1:
        return None
    return _INT_ROUND.get(ctx.rm)


_TO_ZERO_OR_NEAREST = frozenset({RoundingMode.RTZ, RoundingMode.RNE, RoundingMode.RNA})
"""Integral rounds that send anything below one half to zero."""


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


def _scaled(e: Mul) -> tuple[Expr, Expr] | None:
    """*e* as ``(n, x)`` where it is ``2 ** n * x``, else `None`."""
    for scale, value in ((e.first, e.second), (e.second, e.first)):
        if (isinstance(scale, Pow) and len(scale.args) == 2
                and isinstance(scale.args[0], RationalVal)
                and scale.args[0].as_rational() == 2):
            return scale.args[1], value
    return None


def _magnitude(af: AbstractFormat | None) -> float:
    """The largest finite magnitude *af* admits; infinite where unknown."""
    if af is None or not all(isinstance(b, RealFloat) for b in (af.pos_bound, af.neg_bound)):
        return math.inf
    return float(max(abs(af.pos_bound), abs(af.neg_bound)))


def _joined(bounds: list[FormatBound]) -> TritonScalar | None:
    """`StorageInfer`'s join of *bounds*, or `None` where no storage holds
    it: an empty list's bottom constrains nothing."""
    kept = [b for b in bounds if not is_bottom(b)] or bounds
    dom = TritonStorageDomain()
    try:
        return to_triton(join(dom, [of_bound(dom, b) for b in kept]))
    except StorageSelectionError:
        return None


class _Emitter(Visitor):
    """Produces Triton source.

    Dispatch is the framework's, so every node this backend cannot spell has
    a named refusal.  It walks the MRO: `Round` and `Cast` arrive at
    :meth:`_visit_unaryop`.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    format_info: FormatAnalysis
    types: TypeAnalysis
    ctx_use: ContextUseAnalysis
    sizes: ArraySizeAnalysis
    op_table: ScalarOpTable
    drop_asserts: bool
    """Whether an `assert` is dropped rather than refused."""
    tiled: Sequence[ForStmt]
    """The loops carrying a tile, as `tile_loops` reported them; matched by
    identity, not shape."""
    guards: Sequence[If1Stmt]
    """The tiles' `j < n` guards: a mask, where any other `if` is a branch."""
    lane_loops: Sequence[ForStmt]
    """The loops across a tile's lanes."""
    grid_loops: Sequence[ForStmt]
    """The loop the grid's second axis takes."""
    tiles: dict[NamedId, tuple[int, int, TritonScalar]]
    """Local lists of static length, as (width, tail length, element storage).

    A list of `N` is a `[rows, P]` tile, `P` the largest power of two in `N`,
    and a tail of the `N - P` elements past it, each a row value named
    `{list}_t{k}`.  A write rebinds the tile to a select into a new one.
    """
    ranges: dict[NamedId, tuple[str, str]]
    """Names bound to a `range`, as (start, step): `t[j]` is
    `start + j * step`, not an access."""
    consts: dict[str, int]
    """Names bound to an integer constant, for reading a grid extent back."""
    copies: dict[str, str]
    """Names bound to another name, so a tile's width resolves back to the
    `tl.constexpr` it came from."""
    rows: dict[NamedId, tuple[NamedId, list[Expr | str]]]
    """Names bound to a row of a pointer-backed list, as (base, prefix).  The
    binding is not emitted: `A = Ass[r]; A[i]` is one load off `Ass_ptr`."""
    slices: dict[NamedId, tuple[NamedId, list[Expr | str], str]]
    """Names bound to a slice of a pointer-backed list, as
    (base, prefix, start): an address, like a row."""
    seqs: dict[NamedId, list[str]]
    """Names bound to a list literal, as one code per element: every access
    resolves here."""
    subst: dict[NamedId, str]
    """Names bound to a code: a lane loop's target, on a tail element."""
    grid_extent: int | str | None
    """How far the grid's first axis runs, as :class:`KernelSource` has it."""
    grid_outer: int | str | None
    """How far the grid's second axis runs, likewise."""
    written: set[str]
    """The pointer parameters a store goes through."""
    mask: str | None
    """The guard in force, as a Triton predicate; `None` where nothing
    encloses the access.

    Code under a guard runs everywhere, and the guard becomes the `mask=` of
    each load and store.  So **a masked-off element may compute garbage**,
    provided it is masked off before it is observed: by the mask on an
    access, or by the `tl.where` that merges a branch.  Nothing emitted under
    a mask may trap or be undefined on garbage -- float overflow is `inf`,
    and integers wrap.
    """
    size_params: dict[NamedId, str]
    """Each unproven length a list argument has, by its size variable, to
    the kernel parameter the launcher fills it from."""
    _out: _IndentedWriter
    _lane: tuple[NamedId, int] | None
    """The lane loop's target and width, while its body is emitted."""
    _tile: str | None
    """The tiled loop's index, while its body is emitted."""
    _row_width: str
    """How many rows a tile holds: the tiled loop's width inside one."""
    _guard_mask: str | None
    """The tile's own guard while inside it: the rows past the end, and no
    branch."""
    _carried: dict[NamedId, TritonScalar]
    """The names a runtime loop in a tile carries as rows, by storage."""
    _class_of: dict[Definition, Definition]
    """Each definition's class: the definitions a phi joins, as
    `StorageInfer` coalesces them."""
    _members: dict[Definition, list[Definition]]
    _class_ty: dict[Definition, TritonScalar | None]
    """Each class's storage, chosen on demand: every read of a scalar name is
    in it and every assignment is cast into it, so a merge or a loop needs no
    cast."""
    _once: set[str]
    """The names assigned exactly once."""
    _next_tmp: int
    _branches: int
    """How many flattened branches enclose the statement being emitted."""
    _merging: set[NamedId]
    """The lists those branches merge."""
    _fused: dict[int, tuple[Expr, Expr, TritonScalar]]
    """Rounds of a scale-in `2 ** n * x`, by `id`, lowered as one operation in
    `x`'s storage: `(n, x, storage)`; see :meth:`_fuse_rounds`."""
    _unmaterialized: set[Definition]
    """The scale-ins those rounds compute themselves."""

    def __init__(
        self,
        func: FuncDef,
        tiled: Sequence[ForStmt] = (),
        drop_asserts: bool = False,
        guards: Sequence[If1Stmt] = (),
        lanes: Sequence[ForStmt] = (),
        grid: Sequence[ForStmt] = (),
    ) -> None:
        self.func = func
        self.def_use = DefineUse.analyze(func)
        self.format_info = FormatInfer.analyze(func, use_digit_bounds=True)
        self.types = TypeInfer.check(func)
        self.ctx_use = ContextUse.analyze(func, def_use=self.def_use)
        self.sizes = ArraySizeInfer.analyze(func)
        self.op_table = make_op_table()
        self.drop_asserts = drop_asserts
        self.tiled = tiled
        self.guards = guards
        self.lane_loops = lanes
        self.grid_loops = grid
        self.tiles = {}
        self.ranges = {}
        self.consts = {}
        self.copies = {}
        self.rows = {}
        self.slices = {}
        self.seqs = {}
        self.subst = {}
        self.grid_extent = None
        self.grid_outer = None
        self.written = set()
        self.mask = None
        self.size_params = {}
        self._out = _IndentedWriter()
        self._lane = None
        self._tile = None
        self._row_width = '1'
        self._guard_mask = None
        self._carried = {}
        defs = self.def_use.defs
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            for i in same_object_defs(d):
                uf.union(d, defs[i])
        self._class_of = {d: uf.find(d) for d in defs}
        self._members = defaultdict(list)
        for d, c in self._class_of.items():
            self._members[c].append(d)
        self._class_ty = {}
        counts: dict[str, int] = defaultdict(int)
        for n, ds in self.def_use.name_to_defs.items():
            counts[str(n)] += len(ds)
        self._once = {n for n, c in counts.items() if c == 1}
        self._next_tmp = 0
        self._branches = 0
        self._merging = set()
        self._fused, self._unmaterialized = {}, set()
        self._fuse_rounds()

    # -- storage and context -------------------------------------------

    def _storage(self, e: Expr) -> TritonScalar:
        """The scalar storage the pipeline chose for *e*: the type says
        whether it is a boolean or a real, and for a real the format says
        which width."""
        ty = self.types.by_expr.get(e)
        if isinstance(ty, BoolType):
            return TritonScalar.BOOL
        if not isinstance(ty, RealType):
            raise TritonEmitError(
                f'a `{type(ty).__name__ if ty else "?"}` has no Triton '
                f'storage, so `{type(e).__name__}` cannot be held'
            )
        if isinstance(e, ListRef) and (tile := self._tile_of(e.value)):
            return self.tiles[tile][2]
        if isinstance(e, Var) and (d := self.def_use.use_to_def.get(e)):
            # the class's storage, not the read's refined format
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
        """*code*, in *want*'s storage, or a refusal where the cast may lose
        information.  *bound* is the value's proven format, which may fit
        *want* where *have* does not (see :meth:`_value_fits`)."""
        if have == want:
            return code
        if not self._value_fits(bound, have, want):
            raise TritonEmitError(
                f'emitting this would narrow {have.format()} to '
                f'{want.format()} implicitly, which rounds'
            )
        return self._explicit_cast(code, want)

    def _emit_as(self, e: Expr, want: TritonScalar) -> str:
        """*e*, emitted into *want*'s storage."""
        return self._maybe_cast(self.emit(e), self._storage(e), want,
                                self.format_info.by_expr.get(e))

    def _value_fits(
        self, bound: FormatBound, have: TritonScalar, want: TritonScalar,
    ) -> bool:
        """Whether a value bounded by *bound*, held as *have*, fits *want*:
        the storages nest, or the value does."""
        return scalar_fits_in(have, want) or bound_fits_in_scalar(bound, want)

    def _explicit_cast(self, code: str, want: TritonScalar) -> str:
        """*code* cast to *want*.  A literal is respelled instead: a Python
        number is a `constexpr`, which has no `.to`."""
        if code in _SPECIALS:
            return code     # a Python float, typed where it is used
        literal = _as_literal(code)
        if literal is not None:
            return f'{float(literal)}' if want.is_float() else f'{int(literal)}'
        return f'{code}.to({want.format()})'

    def _typed(self, code: str, want: TritonScalar, force: bool = False) -> str:
        """*code* as a *want* tensor where it is a Python number no tensor
        operand types: Triton makes a float `fp32`, rounding it.  An integer
        is left to Triton, which holds it; *force* types any number."""
        v = _as_number(code)
        if v is None or not (force or want.is_float()):
            return code
        if (not force and want is TritonScalar.F32 and isinstance(v, float)
                and _fp32_exact(v)):
            return code
        return f'tl.full((), {code}, dtype={want.format()})'

    def _weak(self, codes: list[str], wants: Sequence[TritonScalar]) -> list[str]:
        """*codes*, typed where every one is a Python number, so no tensor
        among them types the rest.  Integers stay Python's, whose arithmetic
        is exact."""
        if all(_as_number(c) is not None for c in codes):
            return [self._typed(c, w, w.is_float()) for c, w in zip(codes, wants)]
        return codes

    # -- dispatch ------------------------------------------------------

    def _dispatch(
        self,
        e: UnaryOp | BinaryOp | TernaryOp,
        table: dict,
        operands: Sequence[tuple[str, Expr]],
    ) -> str:
        """Emit *e* through the op table: a signature matching the operands'
        storage; else one whose slots are all the active context's, each
        operand cast into it losslessly; else, under ``REAL``,
        :meth:`_try_widen`."""
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
                return sig.format(*self._weak(codes, storages))

        for sig in sigs:
            if (sig.out_ctx == active and len(sig.in_tys) == len(codes)
                    and len(set(sig.in_tys)) == 1):
                return sig.format(*self._weak([
                    self._maybe_cast(code, have, sig.in_tys[0], bound)
                    for code, have, bound in zip(codes, storages, bounds)
                ], sig.in_tys))

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
        """Under ``REAL``, compute at a width that holds the exact result.

        The storage chosen for *e* holds its unrounded result, so the
        operation at that width is exact: `x.to(tl.float32) *
        y.to(tl.float32)` for an fp16 product.  Any width holding the operands
        and the result will do, the result's own first, then the narrowest;
        the exact result is then cast into *e*'s storage.
        """
        target = self._storage(e)
        result = self.format_info.by_expr.get(e)

        slots = sorted(
            {sig.in_tys[0] for sig in sigs
             if len(sig.in_tys) == len(codes) and len(set(sig.in_tys)) == 1},
            # the result's own first: it needs no cast back
            key=lambda t: (t != target, t.float_bits() or t.int_bits() or 0),
        )
        for slot in slots:
            if slot != target and not self._value_fits(result, target, slot):
                continue
            if not all(self._value_fits(b, h, slot) for h, b in zip(storages, bounds)):
                continue
            sig = next(g for g in sigs if g.in_tys == (slot,) * len(codes))
            out = sig.format(*self._weak([
                self._maybe_cast(code, have, slot, bound)
                for code, have, bound in zip(codes, storages, bounds)
            ], sig.in_tys))
            return out if slot == target else self._explicit_cast(out, target)
        return None

    # -- memory --------------------------------------------------------

    def _is_list(self, e: Expr) -> bool:
        """Whether *e* is a sequence rather than a scalar."""
        return isinstance(self.sizes.by_expr.get(e), ListSize)

    def _flatten(self, e: ListRef) -> tuple[NamedId, list[Expr | str], str | None]:
        """A subscript chain as its base, indices (outermost first), and any
        slice offset, resolved as :meth:`_resolve` does."""
        indices: list[Expr | str] = []
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
        self, name: NamedId, indices: list[Expr | str],
    ) -> tuple[NamedId, list[Expr | str], str | None]:
        """*name* followed through the rows and slices that bound it: the
        base, its indices, and the slices' offset as code (address arithmetic,
        not an op-table operation)."""
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
        # any other list is a tile or a literal's elements by now
        if not any(str(a.name) == root and isinstance(a.type, ListTypeAnn)
                   for a in self.func.args):
            raise TritonEmitError(
                f'`{name}` is subscripted but is not a kernel argument, so '
                'there is no pointer to load from'
            )
        return name, indices, ' + '.join(extra) if extra else None

    def _slice_base(
        self, e: ListSlice,
    ) -> tuple[NamedId, list[Expr | str], str] | None:
        """*e* as (base, prefix, start) where it slices a pointer-backed list,
        else `None`."""
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
            # a slice of a slice: only one offset is carried
            return None
        start = '0' if e.start is None else self._pin(self.emit(e.start))
        return base, self._pinned(prefix), start

    def _pin(self, code: str) -> str:
        """*code*, bound to a temporary unless it is a number or a name
        assigned once, since a recorded binding is re-emitted at each use.  In
        a tile the temporary is a row, as its source need not be one on every
        iteration of a `static_range`."""
        if _as_number(code) is not None or code in self._once:
            return code
        if self._tile is not None and not self._out.wide & set(_IDENT.findall(code)):
            return self._bind(f'tl.broadcast_to({code}, ({self._row_width},))', 'row')
        return self._bind(code)

    def _pinned(self, indices: list[Expr | str]) -> list[Expr | str]:
        """*indices*, each emitted and pinned."""
        return [i if isinstance(i, str) else self._pin(self.emit(i)) for i in indices]

    def _size_code(self, size: object) -> str | None:
        """A length as code: a constant, or the kernel parameter holding it."""
        if isinstance(size, int):
            return str(size)
        if isinstance(size, NamedId):
            return self.size_params.get(size)
        return None

    def _strides(self, base: NamedId, rank: int) -> list[str]:
        """Row-major strides for *base*, as code: from its proven shape, or a
        size the kernel takes as a parameter."""
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
        self, base: NamedId, indices: list[Expr | str], extra: str | None = None,
    ) -> str:
        """The flat element offset, row-major, plus any slice start."""
        strides = self._strides(base, len(indices))
        # a pinned row index, as a column against a lane loop's tile
        codes = [
            self.emit(i) if not isinstance(i, str)
            else f'{i}[:, None]' if self._lane is not None and i in self._out.rows
            else i
            for i in indices
        ]
        terms = [
            code if st == '1' else f'{code} * {st}'
            for code, st in zip(codes, strides)
        ]
        if extra is not None:
            terms.append(extra)
        terms = [t for t in terms if t != '0']
        return ' + '.join(terms) if terms else '0'

    # -- tiles ---------------------------------------------------------

    def _tile_of(self, e: Expr) -> NamedId | None:
        """The local list *e* names, or `None`."""
        return e.name if isinstance(e, Var) and e.name in self.tiles else None

    def _as_col(self, code: str) -> str:
        """*code* broadcastable against a tile: a row value as a column."""
        if not _IDENT.fullmatch(code):
            code = self._bind(code)
        return f'{code}[:, None]' if code in self._out.rows else code

    @staticmethod
    def _lane_vec(width: int) -> str:
        """The lane indices of a tile *width* wide, as a `[1, width]` row."""
        return f'tl.arange(0, {width})[None, :]'

    def _at_lane(self, tile: NamedId, idx: Expr) -> bool:
        """Whether *idx* is the lane loop's own index, into a tile its
        width."""
        if self._lane is None or not (
            isinstance(idx, Var) and idx.name == self._lane[0]
        ):
            return False
        if self.tiles[tile][0] != self._lane[1]:
            raise TritonEmitError(
                f'`{tile}` is {self.tiles[tile][0]} lanes wide and its loop '
                f'{self._lane[1]}; tiles of different widths do not combine'
            )
        return True

    def _extract(self, tile: NamedId, idx: str) -> str:
        """Element *idx* of *tile*, a row value: one lane, selected by a
        reduction whose other lanes are the identity.  For a float that is
        `-0.0`, spelled as a negated zero tile since Triton folds a `-0.0`
        literal to `+0.0`."""
        width, tail, elt = self.tiles[tile]
        k = _literal_int(idx)
        if k is not None and k >= width:
            return f'{tile}_t{k - width}'
        at = f'({self._lane_vec(width)} == {self._as_col(idx)})'
        if elt is TritonScalar.BOOL:
            code = (f'(tl.max(tl.where({at}, {tile}, False).to(tl.int32), '
                    'axis=1) != 0)')
        else:
            zero = f'(-tl.zeros_like({tile}))' if elt.is_float() else '0'
            code = (f'tl.sum(tl.where({at}, {tile}, {zero}), axis=1)'
                    f'.to({elt.format()})')
        code = self._bind(code, 'row')
        if k is None:
            for j in reversed(range(tail)):
                code = f'tl.where({idx} == {width + j}, {tile}_t{j}, {code})'
        return code

    def _insert(self, tile: NamedId, idx: str, v: str) -> None:
        """*tile* with element *idx* set to *v*, under any branch."""
        width, tail, _ = self.tiles[tile]
        mask = self.mask if self._branches else None
        k = _literal_int(idx)
        if k is None or k < width:
            at = f'({self._lane_vec(width)} == {self._as_col(idx)})'
            if mask is not None:
                at = f'({at} & {self._as_col(mask)})'
            self._out.add_line(
                f'{tile} = tl.where({at}, {self._as_col(v)}, {tile})', 'wide')
        for j in range(tail):
            if k is not None and k != width + j:
                continue
            name = f'{tile}_t{j}'
            cond = None if k is not None else f'({idx} == {width + j})'
            if mask is not None:
                cond = mask if cond is None else f'({cond} & {mask})'
            self._out.add_line(
                f'{name} = {v}' if cond is None
                else f'{name} = tl.where({cond}, {v}, {name})')

    def _elt_storage(self, stmt: Assign) -> TritonScalar:
        """The storage every element of the list *stmt* allocates is held
        in: its class's elements, joined."""
        assert isinstance(stmt.target, NamedId)
        d = self.def_use.find_def_from_site(stmt.target, stmt)
        ty = self.types.by_def.get(d)
        if isinstance(ty, ListType) and isinstance(ty.elt, BoolType):
            return TritonScalar.BOOL
        held = _joined([
            b.elt for m in self._members[self._class_of[d]]
            if isinstance(b := self.format_info.by_def.get(m), ListFormat)
        ])
        if held is None:
            raise TritonEmitError(
                f'no storage holds every element `{stmt.target}` is assigned'
            )
        return held

    def _allocate(self, stmt: Assign, ctx: _IndentedWriter) -> None:
        """`fp.empty(n)` of a static length, as a tile and its tail."""
        assert isinstance(stmt.target, NamedId)
        self._no_branch(stmt.target)
        bound = self.sizes.by_expr.get(stmt.expr)
        if not (isinstance(bound, ListSize) and isinstance(bound.size, int)
                and bound.size > 0 and not isinstance(bound.elt, ListSize)):
            raise TritonEmitError(
                'a local list needs one dimension of static, nonzero length '
                'to be held in registers'
            )
        n = bound.size
        width = 1 << (n.bit_length() - 1)
        elt = self._elt_storage(stmt)
        self.tiles[stmt.target] = (width, n - width, elt)
        ctx.add_line(
            f'{stmt.target} = tl.zeros(({self._row_width}, {width}), '
            f'dtype={elt.format()})', 'wide')
        zero = 'False' if elt is TritonScalar.BOOL else '0.0' if elt.is_float() else '0'
        for j in range(n - width):
            ctx.add_line(f'{stmt.target}_t{j} = {self._typed(zero, elt)}')

    def _no_branch(self, name: NamedId) -> None:
        """Refuse rebinding a list a branch merges: a tile merges by its
        masked writes alone."""
        if name in self._merging:
            raise TritonEmitError(
                f'`{name}` is a list held in registers, rebound under a branch '
                'it is merged out of'
            )

    def _aligned_block(self, start: Expr | None, width: int) -> str | None:
        """*start* as `k` where it is `k * width`, else `None`."""
        if start is None:
            return '0'
        if isinstance(start, Integer):
            return str(start.val // width) if start.val % width == 0 else None
        if isinstance(start, Mul):
            for c, k in ((start.first, start.second), (start.second, start.first)):
                if isinstance(c, Integer) and c.val % width == 0:
                    scale = c.val // width
                    code = self.emit(k)
                    return code if scale == 1 else f'({code} * {scale})'
        return None

    def _within(self, e: Expr | None, hi: int) -> bool:
        """Whether *e* is proven in `[0, hi]`; `None` is `0`."""
        if e is None:
            return True
        af = to_abstract(self.format_info.by_expr.get(e))
        return (af is not None
                and isinstance(af.neg_bound, RealFloat) and af.neg_bound >= 0
                and isinstance(af.pos_bound, RealFloat) and af.pos_bound <= hi)

    def _slice_tile(self, stmt: Assign, tile: NamedId, ctx: _IndentedWriter) -> None:
        """`xs[k * W:(k + 1) * W]` of a tile, as a tile of its own: block `k`
        of its lanes, selected like an extract by a sum whose other blocks are
        `-0.0`."""
        assert isinstance(stmt.target, NamedId) and isinstance(stmt.expr, ListSlice)
        self._no_branch(stmt.target)
        width, _, elt = self.tiles[tile]
        bound = self.sizes.by_expr.get(stmt.expr)
        w = bound.size if isinstance(bound, ListSize) else None
        k = None
        if isinstance(w, int) and w > 0 and width % w == 0:
            k = self._aligned_block(stmt.expr.start, w)
        if k is None or not isinstance(w, int):
            raise TritonEmitError(
                f'a slice of `{tile}` is a tile only where it is a power of two '
                'wide and starts at a multiple of that'
            )
        if not self._within(stmt.expr.start, width - w):
            raise TritonEmitError(
                f'a slice of `{tile}` is a tile only where it is proven to lie '
                f'in its first {width} elements'
            )
        self.tiles[stmt.target] = (w, 0, elt)
        if w == width:
            ctx.add_line(f'{stmt.target} = {tile}', 'wide')
            return
        nblocks = width // w
        shape = f'({self._row_width}, {nblocks}, {w})'
        blocks = f'tl.reshape({tile}, {shape})'
        at = f'(tl.arange(0, {nblocks})[None, :, None] == {k})'
        if elt is TritonScalar.BOOL:
            code = f'(tl.max(tl.where({at}, {blocks}, False).to(tl.int32), axis=1) != 0)'
        else:
            zero = f'(-tl.zeros({shape}, dtype={elt.format()}))' if elt.is_float() else '0'
            code = f'tl.sum(tl.where({at}, {blocks}, {zero}), axis=1).to({elt.format()})'
        ctx.add_line(f'{stmt.target} = {code}', 'wide')

    def _emit_lanes(self, stmt: ForStmt, ctx: _IndentedWriter) -> None:
        """A lane loop: its body once across the lanes of a tile, then once
        per element of the tail, each at its own index."""
        n = static_trip_count(stmt.iterable, self.sizes)
        target = stmt.target
        if not (isinstance(stmt.iterable, Range1) and isinstance(n, int)
                and n > 0 and isinstance(target, NamedId)):
            raise TritonEmitError(
                'a lane loop should count a static, nonzero `range`'
            )
        width = 1 << (n.bit_length() - 1)
        prev = (self.mask, self._guard_mask, self._lane)
        if self.mask is not None:
            col = self._as_col(self.mask)
            if self._guard_mask == self.mask:
                self._guard_mask = col
            self.mask = col
        self._lane = (target, width)
        ctx.add_line(f'{target} = {self._lane_vec(width)}', 'wide')
        self._visit_block(stmt.body, ctx)
        self.mask, self._guard_mask, self._lane = prev
        for k in range(width, n):
            self.subst[target] = str(k)
            self._visit_block(stmt.body, ctx)
            del self.subst[target]

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
        """*e* as one code per element, where it is a list literal or a name
        bound to one; else `None`."""
        match e:
            case Var() if e.name in self.seqs:
                return list(self.seqs[e.name])
            case ListExpr():
                return [self.emit(elt) for elt in e.elts]
        return None

    def _masked_load(self, addr: str) -> str:
        """`tl.load` at *addr*, under whatever guard is in force.

        Triton rejects a tensor mask on a scalar address, so under a branch a
        scalar address is broadcast to the rows.  Under the tile's own guard
        alone it is a scalar load.
        """
        if self.mask is None:
            return f'tl.load({addr})'
        if self._tile is not None and not (
            (self._out.rows | self._out.wide) & set(_IDENT.findall(addr))
        ):
            if self.mask == self._guard_mask:
                # every launched program has a live row, so the load is safe
                return f'tl.load({addr})'
            addr = f'{addr} + tl.zeros_like({self._tile})'
        return f'tl.load({addr}, mask={self.mask}, other=0.0)'

    def _visit_list_ref(self, e: ListRef, ctx: None) -> str:
        """A subscript: range arithmetic, an element of a tile or a literal,
        or a masked `tl.load`, whose ``other=0.0`` reaches only masked-off
        rows."""

        direct = self._range_index(e)
        if direct is not None:
            return direct
        if (tile := self._tile_of(e.value)) is not None:
            if self._at_lane(tile, e.index):
                return str(tile)
            idx = self.emit(e.index)
            if self._out.wide & set(_IDENT.findall(idx)):
                raise TritonEmitError(
                    f'`{tile}` is read at an index that varies across its '
                    'lanes, which is a gather'
                )
            code = self._extract(tile, idx)
            return code if self._lane is None else self._as_col(code)
        elems = self._elements(e.value)
        if elems is not None:
            idx = self.emit(e.index)
            i = _literal_int(idx)
            if i is None:
                # an index the trace fixes, as a `static_range` target is:
                # the element it selects
                code = elems[-1]
                for j in reversed(range(len(elems) - 1)):
                    code = f'tl.where({idx} == {j}, {elems[j]}, {code})'
                return code
            if not 0 <= i < len(elems):
                raise TritonEmitError(
                    f'index {i} is outside a sequence of {len(elems)}'
                )
            return elems[i]
        base, indices, extra = self._flatten(e)
        return self._masked_load(
            f'{base}_ptr + {self._offset(base, indices, extra)}')

    def _emit_len(self, e: Len) -> str:
        """A length: a proven constant, or the size parameter holding it."""
        if isinstance(e.arg, Var) and e.arg.name in self.seqs:
            return str(len(self.seqs[e.arg.name]))
        bound = self.sizes.by_expr.get(e.arg)
        code = self._size_code(bound.size) if isinstance(bound, ListSize) else None
        if code is None:
            raise TritonEmitError(
                'a length neither proven nor a size parameter has no Triton '
                'spelling'
            )
        return code

    def _emit_round(self, e: Round | Cast) -> str:
        """An explicit `fp.round` / `fp.cast`: a cast, not an op-table
        operation.

        An exact round emits nothing, and a literal is rounded here.  Rounding
        to the integers (`RescaleFixed`'s output) is a C integral rounding;
        otherwise the context must be a hardware conversion, native or
        directed.
        """
        if isinstance(e, Round) and (fused := self._fused.get(id(e))) is not None:
            return self._emit_fused_round(e, *fused)
        arg = self.emit(e.arg)
        ctx = self._active_ctx(e)
        if rounds_exactly(e, self.format_info.by_expr, ctx):
            return arg
        if isinstance(e, Cast) and not self.drop_asserts:
            raise TritonEmitError(
                '`fp.cast` asserts its result is exact, which is not proven '
                'here and a kernel cannot check.  Pass `drop_asserts` to round '
                'without the check'
            )
        if (v := _as_number(arg)) is not None and not isinstance(v, bool):
            return self._emit_rounded(e, v, ctx)
        integral = _integral_round(ctx)
        if integral is not None:
            want = self._storage(e)
            code = f'{integral}({arg})'
            if want.is_float() and not getattr(ctx, 'enable_neg_zero', True):
                code = f'({code} + 0.0)'    # `-0` is not an integer
            return self._maybe_cast(code, self._storage(e.arg), want,
                                    self.format_info.by_expr.get(e))
        if not is_native_ctx(ctx):
            return self._emit_downcast(e, arg, ctx)
        if ctx is not INTEGER and ctx in _int_ctxs() and self._storage(e.arg).is_float():
            raise TritonEmitError(
                f'a cast from a float into `{ctx}` saturates where the context wraps'
            )
        return self._explicit_cast(arg, self._storage(e))

    def _fuse_rounds(self) -> None:
        """Find the integral rounds of a scale-in `RescaleFixed` split off, to
        lower each ``round(2 ** n * x)`` as one operation in `x`'s float
        storage `S`.

        A round under a mode that sends anything below one half to zero reads
        its argument only where it is one half or more, and there
        `2 ** n * x` is `x` shifted, exact in `S` while it cannot overflow.  So
        every round of the scale-in must be under such a mode and bounded
        below `2 ** bias_S`, and `|n| <= 2 * (bias_S - 1)`, which keeps both
        factors of :meth:`_scale_by_halves` normal.  The scale-in is not
        materialized: what it reads must reach each round unchanged."""
        stmt_of = {id(s.expr): s for _, s in walk_stmts(self.func) if isinstance(s, Assign)}
        round_of = {
            id(e.arg): e for _, e in walk_exprs(self.func)
            if isinstance(e, Round) and isinstance(e.arg, Var) and id(e) in stmt_of
        }
        for d in self.def_use.defs:
            site = d.site
            if not isinstance(site, Assign) or not isinstance(site.expr, Mul):
                continue
            parts = _scaled(site.expr)
            uses = self.def_use.uses.get(d, set())
            if parts is None or not uses or any(id(u) not in round_of for u in uses):
                continue
            n, x = parts
            rounds = [round_of[id(u)] for u in uses]
            try:
                held = self._storage(x)
                exact = self._active_ctx(site.expr) is REAL
                ctxs = [self._active_ctx(r) for r in rounds]
            except (TritonEmitError, StorageSelectionError):
                continue
            if not exact or held not in _LOGB:
                continue
            bias = _LOGB[held][0]
            af = to_abstract(self.format_info.by_expr.get(n))
            if af is None or af.exp < 0 or _magnitude(af) > 2 * (bias - 1):
                continue
            here = self.def_use.reach[site]
            read = _reads(site.expr)
            if all(
                isinstance(c, MPFixedContext) and _integral_round(c) is not None
                and c.rm in _TO_ZERO_OR_NEAREST
                and _magnitude(to_abstract(self.format_info.by_expr.get(r))) < 2 ** bias
                and all(self.def_use.reach[stmt_of[id(r)]].get(v) is here.get(v) for v in read)
                for r, c in zip(rounds, ctxs)
            ):
                self._unmaterialized.add(d)
                for r in rounds:
                    self._fused[id(r)] = (n, x, held)

    def _emit_fused_round(self, e: Round, n: Expr, x: Expr, held: TritonScalar) -> str:
        """``round(2 ** n * x)`` in `x`'s storage; :meth:`_fuse_rounds` says
        when."""
        ctx = self._active_ctx(e)
        integral = _integral_round(ctx)
        assert integral is not None
        code = f'{integral}({self._scale_by_halves(self._emit_as(x, held), self.emit(n), held)})'
        want = self._storage(e)
        if want.is_float() and not getattr(ctx, 'enable_neg_zero', True):
            code = f'({code} + 0.0)'    # `-0` is not an integer
        return self._maybe_cast(code, held, want, self.format_info.by_expr.get(e))

    def _scale_by_halves(self, x: str, n: str, held: TritonScalar) -> str:
        """``x * 2 ** n`` as two multiplies by powers of two built from their
        bits, where one power of two would not be normal.  Both scale the same
        way, so the product between them lies between `x` and the result:
        it is exact wherever the result is."""
        bias, mbits, _, _, _, ity = _LOGB[held]
        t = self._bind(f'{n}.to({ity})')
        h = self._bind(f'({t} >> 1)')

        def pow2(k: str) -> str:
            return f'(({bias} + {k}) << {mbits}).to({held.format()}, bitcast=True)'
        return f'({x} * {pow2(h)} * {pow2(f"({t} - {h})")})'

    def _emit_rounded(self, e: Round | Cast, v: float, ctx: Context) -> str:
        """A literal rounded, folded here: Triton would retype it rather than
        round it."""
        try:
            r = ctx.round(v)
        except ValueError as exc:
            raise TritonEmitError(f'rounding `{v}` aborts: {exc}') from None
        if r.isnan:
            return "float('nan')"
        if r.isinf:
            return "(-float('inf'))" if r.s else "float('inf')"
        if r.is_zero() and r.s:
            return f'(-tl.zeros((), {self._storage(e).format()}))'
        return self._emit_numeric_literal(r.as_rational())

    def _emit_downcast(self, e: Round | Cast, arg: str, ctx: Context) -> str:
        """A `round` Triton spells as a conversion under a directed rounding;
        :func:`directed_cast` says which."""
        cast = directed_cast(ctx, self._storage(e.arg))
        if cast is None:
            raise TritonEmitError(
                f'`{type(e).__name__.lower()}` to `{ctx}` is not a hardware '
                'conversion, so it has no cast spelling'
            )
        spell, clear = cast
        code = spell.format(arg)
        if not clear:
            return code
        t = self._bind(code)
        bits = f'({t}.to(tl.int32, bitcast=True) & {-(1 << clear)}).to(tl.float32, bitcast=True)'
        return f'tl.where({t} != {t}, {t}, {bits})'    # a NaN keeps its payload

    def _visit_compare(self, e: Compare, ctx: None) -> str:
        """A comparison, which rounds nothing and so is not in the op table.

        A chain's links join with `&`, which is elementwise where `and` is
        not.  Floats against floats, or integers against integers, Triton
        promotes exactly, and an integer literal every side holds takes the
        others' type; a mix is cast into one storage holding each, since
        Triton would compare an `int32` against a float in `fp32`.
        """
        codes = [self.emit(a) for a in e.args]
        try:
            haves = [self._storage(a) for a in e.args]
        except StorageSelectionError as exc:
            raise TritonEmitError(f'a compared value has no storage: {exc}') from None
        bounds = [self.format_info.by_expr.get(a) for a in e.args]
        kinds = {
            h.is_float() for c, h, b in zip(codes, haves, bounds)
            if _literal_int(c) is None
            or not all(self._value_fits(b, h, w) for w in haves)
        }
        if len(kinds) <= 1:
            return self._chain(e, self._weak(codes, haves))
        want = next((w for w in haves if all(
            self._value_fits(b, h, w) for h, b in zip(haves, bounds))), None)
        if want is None:
            try:
                want = scalar_sup(haves)
            except StorageSelectionError:
                raise TritonEmitError(
                    'no storage holds both sides of a comparison'
                ) from None
        return self._chain(e, self._weak([
            self._maybe_cast(c, h, want, b)
            for c, h, b in zip(codes, haves, bounds)
        ], [want] * len(haves)))

    @staticmethod
    def _chain(e: Compare, codes: list[str]) -> str:
        """The links of *e* over *codes*, joined with `&`."""
        links = [
            f'({codes[i]} {_COMPARE[op]} {codes[i + 1]})'
            for i, op in enumerate(e.ops)
        ]
        return links[0] if len(links) == 1 else '(' + ' & '.join(links) + ')'

    def _emit_connective(self, e: And | Or) -> str:
        """`and` / `or` as `&` / `|`, which are elementwise where Python's
        keywords are not.  Every operand is evaluated, as for `tl.where`."""
        op = '&' if isinstance(e, And) else '|'
        return '(' + f' {op} '.join(self.emit(a) for a in e.args) + ')'

    def _emit_predicate(self, e: UnaryOp) -> str:
        """A classification predicate, spelled from comparisons, each false
        for a NaN where it should be: `isnan(x)` is `x != x`, `isinf(x)` is
        `|x| == inf`, `isfinite(x)` is `|x| < inf`.

        `signbit` must separate `-0.0` from `0.0`, which no float comparison
        does, so it bitcasts to the same-width integer and tests for negative.
        A NaN's sign is unspecified: FPy does not distinguish one.
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

        `ldexp` scales exactly, where the product rests on `exp2` returning
        `2 ** n` exactly, which IEEE 754 only recommends.  This is the shape
        `RescaleFixed` emits for a runtime fixed-point scale.  Since `ldexp`
        is exact, it replaces the product only where the context does not
        round it.
        """
        if (parts := _scaled(e)) is not None:
            exp, value = parts
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
                # is an integer it holds; a special one is on a masked-off row
                af = to_abstract(self.format_info.by_expr.get(exp))
                if af is None or af.exp < 0 or not all(
                    isinstance(b, RealFloat) and abs(b) < 2 ** 31
                    for b in (af.pos_bound, af.neg_bound)
                ):
                    return None
                n = f'{n}.to(tl.int32)'
            # `ldexp` computes in its argument's type: the product's storage,
            # else (unbounded) the operand's
            try:
                want = self._storage(e)
            except StorageSelectionError:
                want = self._storage(value)
            return f'libdevice.ldexp({self._emit_as(value, want)}, {n})'
        return None

    def _emit_logb(self, e: Logb) -> str:
        """IEEE 754 `logB`: the exponent of *x*, read from its bits.

        A subnormal is first scaled exactly by `2**k` into the normals, and
        `k` taken back off.  The specials are `logB`'s own: `+/-0` is `-inf`,
        `+/-inf` is `+inf`, a NaN is a NaN.
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
        """`max`, `min`, `any`, `all` or `sum` over a sequence.

        Over a tile, a reduction across its lanes and then its tail, where
        the grouping cannot be seen (:meth:`_reduce_tile`).  Otherwise a left
        fold over the elements, as FPy's `sum` is.
        """
        elems: list[str] | None
        if (tile := self._tile_of(e.arg)) is not None:
            reduced = self._reduce_tile(e, tile)
            if reduced is not None:
                return reduced
            width, tail, _ = self.tiles[tile]
            elems = [self._extract(tile, str(k)) for k in range(width + tail)]
        else:
            elems = self._elements(e.arg)
        if elems is None:
            raise TritonEmitError(
                f'`{type(e).__name__.lower()}` folds over a list held in '
                'registers or a literal one, and this is neither'
            )
        if isinstance(e, (AnyOf, AllOf)):
            # elementwise, as in `_emit_connective`; empty is the identity
            if not elems:
                return 'False' if isinstance(e, AnyOf) else 'True'
            op = '|' if isinstance(e, AnyOf) else '&'
            return '(' + f' {op} '.join(elems) + ')'
        name = 'tl.maximum' if isinstance(e, AMax) else 'tl.minimum'
        if isinstance(e, Sum):
            if not elems:
                # FPy's empty sum is an exact `+0`
                return self._emit_numeric_literal(Fraction(0))
            ctx = self._active_ctx(e)
            try:
                want: TritonScalar | None = self._storage(e)
            except StorageSelectionError:
                want = None
            seq = self.format_info.by_expr.get(e.arg)
            fits = want is not None and isinstance(seq, ListFormat) and (
                bound_fits_in_scalar(seq.elt, want))
            # under `REAL`, every partial sum lies in the sum's own format,
            # so its storage holds each exactly; under a native context, an
            # add in its storage is its rounding
            if ctx is not REAL and not (fits and is_native_ctx(ctx)):
                raise TritonEmitError(
                    f'`sum` under `{ctx}` rounds each partial sum, which needs '
                    "every element held exactly in that context's storage"
                )
            if want is None or not fits:
                raise TritonEmitError(
                    '`sum` is exact, and no storage holds its elements and '
                    'partial sums'
                )
            elems = self._weak([self._explicit_cast(c, want) for c in elems],
                               [want] * len(elems))
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

    def _reduce_tile(self, e: UnaryOp, tile: NamedId) -> str | None:
        """*e* across *tile*'s lanes, then its tail in order; `None` for a
        `sum` whose grouping could be seen.

        `tl.max` and `tl.min` order `-0.0` below `+0.0` in any grouping, as
        the pairwise fold does, and differ from it only in dropping a NaN,
        which is tested for apart.  A `sum` regroups only under `REAL`, in a
        storage holding its elements and its result, whose format covers every
        partial sum, and so every sub-sum.
        """
        _, tail, elt = self.tiles[tile]
        tails = [f'{tile}_t{j}' for j in range(tail)]
        if isinstance(e, (AnyOf, AllOf)):
            red, op = ('tl.max', '|') if isinstance(e, AnyOf) else ('tl.min', '&')
            code = self._bind(
                f'({red}({tile}.to(tl.int32), axis=1) != 0)', 'row')
            return code if not tails else '(' + f' {op} '.join([code, *tails]) + ')'
        if isinstance(e, (AMax, AMin)):
            red, name = (('tl.max', 'tl.maximum') if isinstance(e, AMax)
                         else ('tl.min', 'tl.minimum'))
            code = f'{red}({tile}, axis=1)'
            if elt.is_float():
                nan = f'(tl.max(({tile} != {tile}).to(tl.int32), axis=1) != 0)'
                code = f"tl.where({nan}, float('nan'), {code})"
            return self._fold_select(name, [self._bind(code, 'row'), *tails])
        if not isinstance(e, Sum):
            return None
        try:
            want = self._storage(e)
        except StorageSelectionError:
            return None
        seq = self.format_info.by_expr.get(e.arg)
        if not (self._active_ctx(e) is REAL
                and isinstance(seq, ListFormat)
                and bound_fits_in_scalar(seq.elt, want)
                and bound_fits_in_scalar(self.format_info.by_expr.get(e), want)):
            return None
        def cast(code: str) -> str:
            return self._maybe_cast(code, elt, want, seq.elt)

        acc = self._bind(f'tl.sum({cast(str(tile))}, axis=1)', 'row')
        for t in tails:
            acc = f'({acc} + {cast(t)})'
        return acc

    def _fold_select(self, name: str, args: list[str]) -> str:
        """A `max`/`min` fold, NaN-propagating; sound pairwise, since both
        are associative and exact."""
        acc = args[0]
        for rhs in args[1:]:
            acc = f'{name}({acc}, {rhs}, propagate_nan=tl.PropagateNan.ALL)'
        return acc

    def _emit_select_op(self, e: Max | Min) -> str:
        """`max` / `min`, folded pairwise.  FPy's propagates a NaN (IEEE
        754-2019 `maximum`), where Triton's default returns the other
        operand."""
        name = 'tl.maximum' if isinstance(e, Max) else 'tl.minimum'
        want = self._storage(e)
        args = self._weak([self._emit_as(a, want) for a in e.args],
                          [want] * len(e.args))
        if not args:
            raise TritonEmitError(f'`{name}` needs at least one operand')
        return self._fold_select(name, args)

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> str:
        """``tl.where``, with both arms in the result's storage.  Both arms
        are evaluated, which is sound: expressions are effect-free and the
        GPU does not trap."""
        want = self._storage(e)
        arms = self._weak([self._emit_as(e.ift, want), self._emit_as(e.iff, want)],
                          [want, want])
        return f'tl.where({self.emit(e.cond)}, {arms[0]}, {arms[1]})'

    # -- expressions ---------------------------------------------------

    def emit(self, e: Expr) -> str:
        """*e* as Triton source."""
        return self._visit_expr(e, None)

    def _visit_var(self, e: Var, ctx: None) -> str:
        bound = self.subst.get(e.name)
        if bound is not None:
            return bound
        if self._tile_of(e) is not None:
            raise TritonEmitError(
                f'`{e.name}` is a list held in registers, which has a value '
                'only element by element'
            )
        if self._lane is not None and str(e.name) in self._out.rows:
            return f'{e.name}[:, None]'
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: None) -> str:
        return 'True' if e.val else 'False'

    def _visit_integer(self, e: Integer, ctx: None) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_decnum(self, e: Decnum, ctx: None) -> str:
        return self._emit_real_literal(e)

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> str:
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
        """An exact literal as itself, where Triton holds it; else a refusal,
        since `num / denom` would be an operation where FPy has a constant."""
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

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> str:
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

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> str:
        if isinstance(e, Mul):
            scaled = self._emit_ldexp(e)
            if scaled is not None:
                return scaled
        return self._dispatch(e, self.op_table.binary, [
            (self.emit(e.first), e.first),
            (self.emit(e.second), e.second),
        ])

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> str:
        return self._dispatch(e, self.op_table.ternary, [
            (self.emit(e.first), e.first),
            (self.emit(e.second), e.second),
            (self.emit(e.third), e.third),
        ])

    def _visit_naryop(self, e: NaryOp, ctx: None) -> str:
        if isinstance(e, (And, Or)):
            return self._emit_connective(e)
        if isinstance(e, (Max, Min)):
            return self._emit_select_op(e)
        raise TritonEmitError(f'no Triton spelling for `{type(e).__name__}`')

    def _visit_rational(self, e: Rational, ctx: None) -> str:
        """`FreeVarElim` materializes a captured `2.5` as
        `fp.rational(5, 2)`."""
        return self._emit_numeric_literal(e.as_rational())

    def _visit_digits(self, e: Digits, ctx: None) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> str:
        """`nan` and `inf`, as Python floats Triton types where used.

        `fp.nan()` is `C.round(nan)`, which aborts under a context holding no
        NaN; a kernel cannot, so that is refused.  The other constants are
        transcendental, with no exact value to emit.
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

    # -- expressions with no Triton spelling ---------------------------

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> str:
        raise TritonEmitError(
            'a foreign value is not a Triton value; fold it first'
        )

    def _visit_call(self, e: Call, ctx: None) -> str:
        raise TritonEmitError(
            'a call survives only if inlining failed; this backend inlines '
            'everything'
        )

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> str:
        raise TritonEmitError('a tuple has no Triton storage')

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> str:
        raise TritonEmitError(
            'a list literal has no Triton value; only its elements do'
        )

    def _visit_list_comp(self, e: ListComp, ctx: None) -> str:
        raise TritonEmitError(
            'a comprehension has no Triton spelling; lower it to a loop first'
        )

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> str:
        raise TritonEmitError('a slice has no Triton spelling')

    def _visit_attribute(self, e: Attribute, ctx: None) -> str:
        raise TritonEmitError('an attribute has no Triton spelling')

    # -- statements ----------------------------------------------------

    def _visit_assign(self, stmt: Assign, ctx: _IndentedWriter) -> None:
        if isinstance(stmt.target, TupleBinding):
            return self._emit_destructure(stmt, ctx)
        if not isinstance(stmt.target, NamedId):
            raise TritonEmitError(
                f'a `{type(stmt.target).__name__}` assignment target has no '
                'Triton spelling'
            )
        if self.def_use.find_def_from_site(stmt.target, stmt) in self._unmaterialized:
            return      # its rounds compute it
        if isinstance(stmt.expr, Empty):
            return self._allocate(stmt, ctx)
        if isinstance(stmt.expr, ListSlice) and (
            tile := self._tile_of(stmt.expr.value)
        ) is not None:
            return self._slice_tile(stmt, tile, ctx)
        if isinstance(stmt.expr, ListSlice):
            bound = self._slice_base(stmt.expr)
            if bound is not None:
                self.slices[stmt.target] = bound
                return
        elems = self._elements(stmt.expr)
        if elems is not None:
            # held as one name per element
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
                raise TritonEmitError(
                    f'`{stmt.target}` is a row of a slice, whose offset this '
                    'backend does not keep'
                )
            self.rows[stmt.target] = (base, self._pinned(indices))
            return
        match stmt.expr:
            case Range1():
                self.ranges[stmt.target] = ('0', '1')
                return
            case Range3():
                self.ranges[stmt.target] = (
                    self._pin(self.emit(stmt.expr.first)),
                    self._pin(self.emit(stmt.expr.third)),
                )
                return
        code = self._into_class(stmt.target, stmt, stmt.expr)
        if stmt.target in self._carried:
            ctx.add_line(f'{stmt.target} = {self._as_row(code, self._carried[stmt.target])}', 'row')
            return
        if code.isdigit():
            self.consts[str(stmt.target)] = int(code)
        elif isinstance(stmt.expr, Var):
            self.copies[str(stmt.target)] = self._root(code)
        ctx.add_line(f'{stmt.target} = {code}')

    def _into_class(self, target: NamedId, stmt: Assign, e: Expr) -> str:
        """*e*, which *stmt* assigns *target*, emitted in its class storage."""
        code = self.emit(e)
        d = self.def_use.find_def_from_site(target, stmt)
        if not isinstance(self.types.by_def.get(d), RealType):
            return code
        want = self._class_storage(d)
        if want is None:
            # a name is held in one storage, as a declaration has one type
            raise TritonEmitError(
                f'no storage holds every value `{target}` is assigned'
            )
        return self._typed(self._maybe_cast(
            code, self._storage(e), want, self.format_info.by_expr.get(e),
        ), want)

    def _emit_destructure(self, stmt: Assign, ctx: _IndentedWriter) -> None:
        """`a, b = (x, y)` of a literal tuple, as one assignment per element;
        through temporaries where a target is read on the right, since the
        assignment is simultaneous."""
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

        codes = [
            self._into_class(n, stmt, e) if isinstance(n, NamedId)
            else self.emit(e)
            for n, e in zip(names, stmt.expr.elts)
        ]
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

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _IndentedWriter) -> None:
        if (tile := self._tile_of(Var(stmt.var, None))) is not None:
            return self._assign_element(tile, stmt, ctx)
        base, indices, extra = self._resolve(stmt.var, list(stmt.indices))
        addr = f'{base}_ptr + {self._offset(base, indices, extra)}'
        val = self.emit(stmt.expr)
        # `tl.store` converts to the pointer's type, which may round
        want = self._arg_storage(self._root(str(base)))
        have, bound = self._storage(stmt.expr), self.format_info.by_expr.get(stmt.expr)
        if not self._value_fits(bound, have, want):
            raise TritonEmitError(
                f'storing {have.format()} through a {want.format()} pointer '
                'would round'
            )
        val = self._typed(self._maybe_cast(val, have, want, bound), want)
        mask = '' if self.mask is None else f', mask={self.mask}'
        ctx.add_line(f'tl.store({addr}, {val}{mask})')
        self.written.add(f'{self._root(str(base))}_ptr')

    def _assign_element(
        self, tile: NamedId, stmt: IndexedAssign, ctx: _IndentedWriter,
    ) -> None:
        """An element of a local list, set: across the lanes at the lane
        loop's own index, else one element selected."""
        if len(stmt.indices) != 1:
            raise TritonEmitError(f'`{tile}` has one dimension')
        idx = stmt.indices[0]
        elt = self.tiles[tile][2]
        val = self._typed(self._emit_as(stmt.expr, elt), elt)
        if self._at_lane(tile, idx):
            mask = self.mask if self._branches else 'True'
            ctx.add_line(f'{tile} = tl.where({mask}, {val}, {tile})', 'wide')
        elif self._lane is not None:
            raise TritonEmitError(
                f'a lane loop writes `{tile}` other than at its own index'
            )
        else:
            self._insert(tile, self.emit(idx), val)

    def _visit_return(self, stmt: ReturnStmt, ctx: _IndentedWriter) -> None:
        if self._branches:
            raise TritonEmitError(
                'a `return` in a branch has no Triton spelling; a flattened '
                'branch runs on every row'
            )
        ctx.add_line(f'return {self.emit(stmt.expr)}')

    def _visit_context(self, stmt: ContextStmt, ctx: _IndentedWriter) -> None:
        # the dispatch reads each expression's context
        self._visit_block(stmt.body, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: _IndentedWriter) -> None:
        if not any(stmt is g for g in self.guards):
            return self._emit_branch(stmt, stmt.body, None, ctx)
        # a tile's guard only drops the over-run: nothing merges out of it
        prev, prev_guard = self.mask, self._guard_mask
        self.mask = self._guard_mask = self.emit(stmt.cond)
        self._visit_block(stmt.body, ctx)
        self.mask, self._guard_mask = prev, prev_guard

    def _visit_if(self, stmt: IfStmt, ctx: _IndentedWriter) -> None:
        self._emit_branch(stmt, stmt.ift, stmt.iff, ctx)

    def _bind(self, code: str, shape: str | None = None) -> str:
        """*code*, evaluated once into a temporary ahead of the line being
        built; *shape* as for `add_line`."""
        name = f'__t{self._next_tmp}'
        self._next_tmp += 1
        self._out.add_line(f'{name} = {code}', shape)
        return name

    def _class_storage(self, d: Definition) -> TritonScalar | None:
        """The storage of *d*'s class, or `None` where it holds no scalar or
        no storage holds it."""
        if isinstance(self.types.by_def.get(d), BoolType):
            return TritonScalar.BOOL
        c = self._class_of[d]
        if c not in self._class_ty:
            self._class_ty[c] = _joined(
                [self.format_info.by_def.get(m) for m in self._members[c]])
        return self._class_ty[c]

    def _emit_branch(
        self, stmt: IfStmt | If1Stmt, ift: StmtBlock, iff: StmtBlock | None,
        ctx: _IndentedWriter,
    ) -> None:
        """An `if`, flattened.

        Each arm runs on every row, under the enclosing mask and its guard;
        each name the `if` merges is then chosen by `tl.where`.  The arms are
        emitted with their own names -- the analyses key on nodes, so the AST
        is not renamed -- and a name the first arm overwrites is saved before
        it and restored after, for the second arm and the merge to read.
        """
        # a list behind a pointer or in registers merges by its masked
        # writes alone
        phis = []
        lists = set()
        for p in self.def_use.phis[stmt]:
            if not isinstance(self.types.by_def.get(p), ListType):
                phis.append(p)
            elif p.name in self.seqs:
                raise TritonEmitError(
                    f'`{p.name}` is a list chosen by a branch, which has no '
                    'Triton value'
                )
            else:
                lists.add(p.name)
        merged = {p.name for p in phis}
        named: tuple[dict, ...] = (self.consts, self.copies, self.ranges,
                                   self.rows, self.slices, self.seqs)
        saved_named = [dict(m) for m in named]

        def restore(drop: set[NamedId]) -> None:
            # what an arm bound is gone, and a merged name is no constant
            for m, prev in zip(named, saved_named):
                m.clear()
                m.update(prev)
                for v in drop:
                    m.pop(v, None)
                    m.pop(str(v), None)

        outer = self.mask

        cond = self._bind(self.emit(stmt.cond))
        saved = {
            v: self._bind(str(v))
            for v in sorted(self.def_use.mutated_in(ift)) if v in merged
        }
        self._branches += 1
        prev_merging, self._merging = self._merging, self._merging | lists
        self.mask = cond if outer is None else f'({outer} & {cond})'
        self._visit_block(ift, ctx)
        taken = {p.name: self._bind(str(p.name)) for p in phis}
        for v, code in saved.items():
            ctx.add_line(f'{v} = {code}')
        restore(set())
        if iff is not None:
            self.mask = f'(~{cond})' if outer is None else f'({outer} & ~{cond})'
            self._visit_block(iff, ctx)
        self._branches -= 1
        self._merging = prev_merging
        self.mask = outer

        # a phi's sides share its class, so they are in one storage already
        for p in phis:
            ctx.add_line(f'{p.name} = tl.where({cond}, {taken[p.name]}, {p.name})')
        restore(merged)

    def _visit_for(self, stmt: ForStmt, ctx: _IndentedWriter) -> None:
        if any(stmt is t for t in self.tiled):
            return self._emit_tile(stmt, ctx)
        if any(stmt is t for t in self.lane_loops):
            return self._emit_lanes(stmt, ctx)
        if any(stmt is t for t in self.grid_loops):
            return self._emit_grid(stmt, ctx)
        if not isinstance(stmt.target, Id):
            raise TritonEmitError(
                'a destructuring loop target has no Triton spelling'
            )
        # `tl.static_range` yields indices: the loop variable only over a
        # `range`
        if not isinstance(stmt.iterable, (Range1, Range3)):
            raise TritonEmitError(
                f'a `for` over a `{type(stmt.iterable).__name__}` binds an '
                'element, and `tl.static_range` yields an index; iterate a '
                '`range` and subscript instead'
            )
        n = static_trip_count(stmt.iterable, self.sizes)
        if not isinstance(n, int):
            return self._emit_runtime_loop(stmt, ctx)
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

    def _emit_runtime_loop(self, stmt: ForStmt, ctx: _IndentedWriter) -> None:
        """A loop of unproven trip count, as a runtime `range`.

        Triton holds a carried value at one type and shape on every
        iteration.  The type is its class's storage already; in a tile, each
        carried name is broadcast to a row at entry and at every assignment.
        """
        carried = carried_scalars(stmt, self.def_use)
        prev = self._carried
        if self._tile is not None and carried:
            self._carried = {}
            for p in self.def_use.phis.get(stmt, ()):
                if p.name not in carried:
                    continue
                held = self._class_storage(p)
                if held is None:
                    raise TritonEmitError(
                        f'no storage holds every value `{p.name}` carries'
                    )
                self._carried[p.name] = held
                ctx.add_line(f'{p.name} = {self._as_row(str(p.name), held)}', 'row')
        it = stmt.iterable
        assert isinstance(it, (Range1, Range3))
        args = ([self.emit(it.arg)] if isinstance(it, Range1)
                else [self.emit(a) for a in (it.first, it.second, it.third)])
        ctx.add_line(f'for {stmt.target} in range({", ".join(args)}):')
        ctx.indent()
        self._visit_block(stmt.body, ctx)
        ctx.dedent()
        self._carried = prev

    def _as_row(self, code: str, held: TritonScalar) -> str:
        """*code*, broadcast across the tile's rows.  An integer is
        added to zeros, which also widens a name bound to Triton's `int32`;
        a float is not, as `-0.0 + 0.0` is `+0.0`."""
        if held.is_integer():
            return f'({code} + tl.zeros(({self._row_width},), dtype={held.format()}))'
        return f'tl.broadcast_to({self._typed(code, held, True)}, ({self._row_width},))'

    def _arg_storage(self, name: str) -> TritonScalar:
        """The storage of the elements kernel argument *name* points at."""
        arg = next(a for a in self.func.args if str(a.name) == name)
        assert isinstance(arg.name, NamedId)
        d = self.def_use.find_def_from_site(arg.name, arg)
        ty, bound = self.types.by_def.get(d), self.format_info.by_def.get(d)
        while isinstance(ty, ListType):
            ty = ty.elt
        while isinstance(bound, ListFormat):
            bound = bound.elt
        if isinstance(ty, BoolType):
            return TritonScalar.BOOL
        try:
            return choose_storage_scalar(bound)
        except StorageSelectionError:
            raise TritonEmitError(
                f'`{name}` holds no storage Triton has, so it cannot be stored to'
            ) from None

    def _root(self, name: str) -> str:
        """*name* followed through the copies that bound it."""
        seen: set[str] = set()
        while name in self.copies and name not in seen:
            seen.add(name)
            name = self.copies[name]
        return name

    def _emit_tile(self, stmt: ForStmt, ctx: _IndentedWriter) -> None:
        """A tiled loop, as the launch grid plus a tile.

        `tile_loops` leaves `for i in range(0, n, B)` around
        `for j in range(i, i + B)`: the chunk index becomes the program
        instance and the inner index the tile's rows::

            i = tl.program_id(0) * B
            j = i + tl.arange(0, B)

        Any other shape is refused.
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
        self.grid_extent = self._extent(it.second)
        ctx.add_line(f'{outer} = tl.program_id(0) * {width}')
        ctx.add_line(f'{inner.target} = {outer} + tl.arange(0, {width})')
        prev = self._tile, self._row_width
        self._tile, self._row_width = str(inner.target), width
        self._visit_block(inner.body, ctx)
        self._tile, self._row_width = prev

    def _extent(self, stop: Expr) -> int | str | None:
        """How far a grid axis runs: a proven length, or the size parameter
        holding one."""
        if isinstance(stop, Var):
            # a split binds the length to a name; the grid wants the length
            d = self.def_use.find_def_from_use(stop)
            if (isinstance(d, AssignDef) and isinstance(d.site, Assign)
                    and isinstance(d.site.expr, Len)):
                stop = d.site.expr
        bound = self._root(self.emit(stop))
        return (
            int(bound) if bound.isdigit()
            else bound if bound in self.size_params.values()
            else self.consts.get(bound)
        )

    def _emit_grid(self, stmt: ForStmt, ctx: _IndentedWriter) -> None:
        """A loop the grid's second axis takes: one iteration per program."""
        it = stmt.iterable
        if not isinstance(it, Range1) or not isinstance(stmt.target, NamedId):
            raise TritonEmitError('a grid axis should count a `range`')
        self.grid_outer = self._extent(it.arg)
        if self.grid_outer is None:
            raise TritonEmitError('a grid axis needs a length the launcher knows')
        ctx.add_line(f'{stmt.target} = tl.program_id(1)')
        self._visit_block(stmt.body, ctx)

    def _visit_block(self, block: StmtBlock, ctx: _IndentedWriter) -> None:
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    # -- statements with no Triton spelling ----------------------------

    def _visit_while(self, stmt: WhileStmt, ctx: _IndentedWriter) -> None:
        raise TritonEmitError('a `while` has no Triton spelling')

    def _visit_assert(self, stmt: AssertStmt, ctx: _IndentedWriter) -> None:
        if self.drop_asserts:
            return
        raise TritonEmitError(
            'an `assert` has no Triton spelling; a kernel cannot raise. '
            'Pass `drop_asserts` to skip it instead'
        )

    def _visit_effect(self, stmt: EffectStmt, ctx: _IndentedWriter) -> None:
        raise TritonEmitError('an effect has no Triton spelling')

    def _visit_pass(self, stmt: PassStmt, ctx: _IndentedWriter) -> None:
        pass

    def _visit_function(self, func: FuncDef, ctx: _IndentedWriter) -> None:
        raise TritonEmitError('a whole function is emitted by `emit_kernel`')


def emit_expr(e: Expr, func: FuncDef) -> str:
    """*e*, as Triton source.

    *func* is the function it belongs to; its analyses decide the storage each
    operand is held in and the context the operation rounds under.
    """
    if not isinstance(e, Expr):
        raise TypeError(f"Expected an 'Expr', got {e}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    return _Emitter(func).emit(e)


def emit_block(
    block: StmtBlock,
    func: FuncDef,
    tiled: Sequence[ForStmt] = (),
    *,
    drop_asserts: bool = False,
    guards: Sequence[If1Stmt] = (),
    lanes: Sequence[ForStmt] = (),
) -> str:
    """*block*, as Triton source.

    *tiled*, *guards* and *lanes* name the loops carrying a tile, the guards
    on them and the loops across its lanes, as `tile_loops` reported them.
    *drop_asserts* skips an `assert` rather than refusing it.
    """
    if not isinstance(block, StmtBlock):
        raise TypeError(f"Expected a 'StmtBlock', got {block}")
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    emitter = _Emitter(func, tiled, drop_asserts, guards, lanes)
    emitter._visit_block(block, emitter._out)
    return emitter._out.render()


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
    """Whether contracting a multiply-add is unobservable: every product is
    exact.  Triton takes it at launch, not in the source."""

    sizes: tuple[tuple[str, int, int], ...] = ()
    """Each size parameter, after the arguments: its name, and the argument
    position and dimension whose length it is."""

    grid_outer: int | str | None = None
    """How many programs the grid's second axis runs, one per iteration of
    the loop around the tile, as :attr:`grid_extent` is spelled; ``None``
    where the grid has one axis."""

    block: str | None = None
    """The tile width's parameter, a `tl.constexpr` the launcher picks."""

    writes: tuple[str, ...] = ()
    """The pointer parameters the kernel stores through, which tuning runs
    it on more than once and so has to restore."""

    shapes: tuple[tuple[int, tuple[int | str | None, ...]], ...] = ()
    """Each list argument's position and the shape its offsets assume, row
    major: a proven length, the size parameter holding one, or ``None``."""

    dtypes: tuple[tuple[int, str], ...] = ()
    """Each list argument's position and its elements' Triton dtype."""


def _times(a: str, b: str) -> str:
    """``a * b`` as code, folded where both are constants."""
    if a.isdigit() and b.isdigit():
        return str(int(a) * int(b))
    return b if a == '1' else a if b == '1' else f'{a} * {b}'


def _products_are_exact(func: FuncDef, emitter: _Emitter) -> bool:
    """Whether every product in *func* is exact, so fusion cannot be seen."""
    for _, e in walk_exprs(func):
        if isinstance(e, Mul):
            try:
                ctx = emitter._active_ctx(e)
            except TritonEmitError:
                return False
            if not rounds_exactly(e, emitter.format_info.by_expr, ctx):
                return False
    return True


def emit_kernel(
    func: FuncDef,
    tiled: Sequence[ForStmt] = (),
    *,
    block: str | None = None,
    drop_asserts: bool = False,
    guards: Sequence[If1Stmt] = (),
    lanes: Sequence[ForStmt] = (),
    grid: Sequence[ForStmt] = (),
) -> KernelSource:
    """*func* as a ``@triton.jit`` kernel.

    A list argument becomes a pointer, *block* becomes a ``tl.constexpr``, and
    a trailing `return` is dropped -- a kernel writes through its pointers and
    returns nothing, which is why the program it is emitted from takes its
    output as an argument.
    """
    if not isinstance(func, FuncDef):
        raise TypeError(f"Expected a 'FuncDef', got {func}")
    emitter = _Emitter(func, tiled, drop_asserts, guards, lanes, grid)

    params: list[str] = []
    size_params: list[tuple[str, int, int]] = []
    shapes: list[tuple[int, tuple[int | str | None, ...]]] = []
    for pos, arg in enumerate(func.args):
        name = str(arg.name)
        if name == block:
            params.append(f'{name}: tl.constexpr')
        elif isinstance(arg.type, ListTypeAnn):
            params.append(f'{name}_ptr')
            assert isinstance(arg.name, NamedId)
            # an unproven length is a size parameter
            bound = emitter.sizes.by_def.get(emitter.def_use.find_def_from_site(arg.name, arg))
            depth = 0
            dims: list[int | str | None] = []
            while isinstance(bound, ListSize):
                if isinstance(bound.size, NamedId) and bound.size not in emitter.size_params:
                    emitter.size_params[bound.size] = f'{name}_n{depth}'
                    size_params.append((f'{name}_n{depth}', pos, depth))
                dims.append(bound.size if isinstance(bound.size, int)
                            else emitter._size_code(bound.size))
                bound, depth = bound.elt, depth + 1
            shapes.append((pos, tuple(dims)))
        else:
            params.append(name)
    params.extend(name for name, _, _ in size_params)

    body = StmtBlock([
        stmt for stmt in func.body.stmts
        if not isinstance(stmt, ReturnStmt)
    ])
    out = emitter._out
    out.add_line('@triton.jit')
    out.add_line(f'def {func.name}({", ".join(params)}):')
    out.indent()
    emitter._visit_block(body, out)
    out.dedent()
    dtypes = tuple(
        (pos, emitter._arg_storage(str(func.args[pos].name)).format())
        for pos, _ in shapes
    )

    return KernelSource(
        name=func.name,
        source=out.render(),
        params=tuple(params),
        grid_extent=emitter.grid_extent,
        grid_outer=emitter.grid_outer,
        block=block,
        writes=tuple(p for p in params if p in emitter.written),
        shapes=tuple(shapes),
        dtypes=dtypes,
        enable_fp_fusion=_products_are_exact(func, emitter),
        sizes=tuple(size_params),
    )
