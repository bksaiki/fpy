"""
Type-directed Hypothesis generators for FPy programs.

The canonical entry point is :func:`expr` — given a target ``Type`` (real,
bool, list, tuple, context), an environment, and a depth budget, returns a
strategy producing a well-typed FPy expression. :func:`stmt_block` and
:func:`fpy_funcdef` build on top of it to generate full statement blocks
and function definitions whose declared signature is recoverable from the
inferred types. Construction is AST-direct (``FuncDef`` nodes are built
in code and fed to the same passes the ``@fp.fpy`` decorator pipeline
runs) — the parser is **not** exercised here; the target audience is
analysis passes and the interpreter.

Productions per target type are gated by :class:`enum.Flag` enums — one
per target type (:class:`RealProd`, :class:`BoolProd`, :class:`ListProd`,
:class:`TupleProd`, :class:`ContextProd`, :class:`StmtProd`). Pass
``include=RealProd.ARITH | RealProd.LITERAL`` to a helper to narrow what
it emits, or set the corresponding ``*_prods`` field on :class:`Grammar`
to narrow across recursive calls (including cross-type ones). Per-call
overrides for ``range_arg_min``/``range_arg_max`` bound the integer
literals used as ``Range1``/``Range2`` arguments.

Function bodies are ``StmtBlock``s of ``Assign`` + ``ContextStmt``
(``with CTX: ...``) + ``IfStmt`` / ``If1Stmt`` + ``ForStmt`` + ``WhileStmt``
+ a terminating ``ReturnStmt``. Each ``max_*`` cap on :func:`stmt_block` /
:func:`fpy_funcdef` bounds how many of that statement kind appear in a
body. Local-name uniqueness is maintained by a single counter shared
across the entire body (including nested ``with``/``if``/``for``/``while``
bodies). Scoping rules:

- ``Assign``s in a ``with`` body leak out to the enclosing scope (matches
  FPy lexical scoping).
- ``Assign``s in ``if`` branches and in ``for``/``while`` bodies do **not**
  propagate to the outer scope — a deliberate simplification (sound
  per-branch merging would require env intersection on shared defs).
- For-loop targets are body-scoped (FPy rejects post-loop refs).
- ``WhileStmt`` is emitted as a counter-driven template ``c = 0; while c < N:
  <body>; c = c + 1`` to guarantee termination; only the body is fuzzed.

Known-deferred surface (see inline comments for the why):

- ``Cast`` — needs per-context exact-representable tracking.
- ``IsNormal`` — workaround for an FPy ``ops.isnormal`` bug on context-less
  integer Floats.
- ``ListRef`` / ``ListSlice`` — need array-size tracking to stay in bounds.
- ``ListComp``, ``IndexedAssign``, tuple-unpacking ``Assign``.
- ``Range3`` and ``fp.REAL`` as a ``with``-context (math ops not implemented
  under REAL).
"""

import dataclasses
from enum import Flag, auto
from typing import TypeAlias, TypeVar

from hypothesis import strategies as st

import fpy2 as fp

from fpy2.ast.fpyast import (
    Abs,
    Add,
    And,
    Argument,
    Assign,
    BoolTypeAnn,
    BoolVal,
    Compare,
    ContextStmt,
    ContextTypeAnn,
    Cos,
    Decnum,
    Div,
    Enumerate,
    Exp,
    Expr,
    Hexnum,
    ForeignVal,
    ForStmt,
    FuncDef,
    FuncMeta,
    If1Stmt,
    IfExpr,
    IfStmt,
    Integer,
    IsFinite,
    IsInf,
    IsNan,
    Len,
    ListExpr,
    ListTypeAnn,
    Log,
    Mul,
    Neg,
    NamedUnaryOp,
    Not,
    Or,
    Range1,
    Range2,
    Range3,
    Rational,
    RealTypeAnn,
    ReturnStmt,
    Round,
    Signbit,
    Sin,
    Sqrt,
    Stmt,
    StmtBlock,
    Sub,
    TupleExpr,
    TupleTypeAnn,
    TypeAnn,
    Var,
    WhileStmt,
    Zip,
)
from fpy2.types import BoolType, ContextType, ListType, RealType, TupleType, Type
from fpy2.env import ForeignEnv
from fpy2.utils import CompareOp, NamedId, UnderscoreId


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TypeEnv: TypeAlias = dict[NamedId, Type]
"""Mapping from in-scope names to their FPy types."""


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

_LITERAL_RANGE = (-1_000, 1_000)

# Default inclusive bounds for ``Integer`` literals used as ``Range1`` /
# ``Range2`` arguments. Per-call overrides on the public strategies.
_DEFAULT_RANGE_ARG_MIN = -5
_DEFAULT_RANGE_ARG_MAX = 8


def _var_of_type(
    env: TypeEnv,
    target_type: Type,
) -> st.SearchStrategy[Expr] | None:
    """Strategy for a ``Var`` of exactly ``target_type``, or ``None`` if env has none.

    Compares by value (``Type.__eq__``), so ``list[real]`` and ``list[bool]``
    are distinguished — important now that compound-typed locals are
    allowed.
    """
    names = sorted((n for n, t in env.items() if t == target_type), key=str)
    if not names:
        return None
    return st.sampled_from(names).map(lambda n: Var(n, None))


def _func_sym(name: str) -> Var:
    """Construct the ``FuncSymbol`` used in ``Named*Op``.

    NamedUnaryOp etc. carry a ``func`` symbol that names the surface-syntax
    callable (e.g. ``sqrt``). Type inference doesn't look at this name (the
    type is determined by the AST class), but some passes do.
    """
    return Var(NamedId(name), None)


# Concrete rounding contexts the generator can drop into ``with`` blocks.
# ``fp.REAL`` is excluded — ``sqrt``/``sin``/``log``/``div`` raise under it
# (no closed-form rational result), so ``with REAL: t = sqrt(0)`` would crash.
_DEFAULT_CONTEXTS: list = [fp.FP64, fp.FP32, fp.MX_E5M2, fp.MX_E4M3]


# ---------------------------------------------------------------------------
# Productions — :class:`enum.Flag` grammar surface
# ---------------------------------------------------------------------------
# One ``Flag`` per target type names every production this generator can
# emit (including currently-deferred ones).  Aliases such as
# ``RealProd.ARITH`` are conventional ``Flag`` composites (e.g.
# ``ADD | SUB | …``).


class RealProd(Flag):
    """Production set for ``real``-typed expressions."""
    INTEGER = auto()       # ``Integer`` literal (alias: LITERAL)
    DECNUM = auto()        # ``Decnum`` decimal-string literal
    HEXNUM = auto()        # ``Hexnum`` hex-string literal (e.g. ``0x1.8p+3``)
    RATIONAL = auto()      # ``Rational(p, q)`` literal
    VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    ABS = auto()
    SQRT = auto()
    SIN = auto()
    COS = auto()
    LOG = auto()
    EXP = auto()
    LEN = auto()
    IF_EXPR = auto()
    ROUND = auto()
    CAST = auto()          # deferred; raises if generated

    # Aliases.
    LITERAL = INTEGER      # back-compat alias for callers that pre-dated the split
    NUMERIC_LITERAL = INTEGER | DECNUM | HEXNUM | RATIONAL
    ARITH = ADD | SUB | MUL | DIV | NEG | ABS
    NAMED_UNARY = SQRT | SIN | COS | LOG | EXP
    LEAVES = NUMERIC_LITERAL | VAR
    ALL = (LEAVES | ARITH | NAMED_UNARY | LEN | IF_EXPR | ROUND | CAST)


class BoolProd(Flag):
    """Production set for ``bool``-typed expressions."""
    LITERAL = auto()
    VAR = auto()
    COMPARE = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    ISFINITE = auto()
    ISINF = auto()
    ISNAN = auto()
    SIGNBIT = auto()
    ISNORMAL = auto()  # deferred (``ops.isnormal`` crashes on int Floats)

    AND_OR = AND | OR
    PREDICATE = ISFINITE | ISINF | ISNAN | SIGNBIT | ISNORMAL
    LEAVES = LITERAL | VAR
    ALL = LEAVES | COMPARE | NOT | AND_OR | PREDICATE


class ListProd(Flag):
    """Production set for ``list[T]``-typed expressions."""
    LITERAL = auto()
    VAR = auto()
    RANGE1 = auto()
    RANGE2 = auto()
    RANGE3 = auto()  # deferred
    ZIP = auto()
    ENUMERATE = auto()
    LIST_COMP = auto()  # deferred

    RANGE = RANGE1 | RANGE2 | RANGE3
    LEAVES = LITERAL | VAR
    ALL = LEAVES | RANGE | ZIP | ENUMERATE | LIST_COMP


class TupleProd(Flag):
    """Production set for ``tuple[...]``-typed expressions."""
    LITERAL = auto()
    VAR = auto()

    LEAVES = LITERAL | VAR
    ALL = LEAVES


class ContextProd(Flag):
    """Production set for ``context``-typed expressions."""
    LITERAL = auto()

    LEAVES = LITERAL
    ALL = LEAVES


class StmtProd(Flag):
    """Statement kinds emitted inside a generated :class:`StmtBlock`."""
    ASSIGN = auto()
    WITH = auto()
    IF = auto()
    FOR = auto()
    WHILE = auto()
    INDEXED_ASSIGN = auto()       # deferred
    TUPLE_UNPACK_ASSIGN = auto()  # deferred

    # Aliases.
    LEAVES = ASSIGN
    CONTROL_FLOW = IF | FOR | WHILE
    LOOPS = FOR | WHILE
    ALL = (ASSIGN | WITH | IF | FOR | WHILE
           | INDEXED_ASSIGN | TUPLE_UNPACK_ASSIGN)


_F = TypeVar('_F', bound=Flag)


def _check_flag(include: _F, expected: type[_F]) -> _F:
    """Assert that ``include`` is a :class:`Flag` of the expected type and
    return it unchanged; raise :class:`TypeError` otherwise.  Callers
    handle the ``None`` case (fall back to the grammar default) before
    invoking this."""
    if not isinstance(include, expected):
        raise TypeError(
            f'expected {expected.__name__} for include, '
            f'got {type(include).__name__}'
        )
    return include


# ---------------------------------------------------------------------------
# Grammar bundle
# ---------------------------------------------------------------------------
# Programmable filter over every production in the grammar.  Threading a
# single :class:`Grammar` through every recursive call means a per-type
# narrowing (e.g. ``bool_prods=BoolProd.LEAVES``) survives crossing into a
# different generator, which a per-call ``include=`` cannot.

# Productions deferred because their generators aren't implemented yet —
# enabling one raises :class:`NotImplementedError` in the corresponding
# helper.  Re-enable by implementing the production and dropping its
# entry here.
_DEFERRED_REAL: RealProd = RealProd.CAST
_DEFERRED_BOOL: BoolProd = BoolProd.ISNORMAL
_DEFERRED_LIST: ListProd = ListProd.LIST_COMP
_DEFERRED_STMT: StmtProd = StmtProd.INDEXED_ASSIGN | StmtProd.TUPLE_UNPACK_ASSIGN

# Productions whose generators are implemented but noticeably slow to
# draw — opt in explicitly when the test needs them (typically by
# OR-ing into a narrowed ``real_prods``).
_SLOW_REAL: RealProd = RealProd.DECNUM | RealProd.HEXNUM | RealProd.RATIONAL


@dataclasses.dataclass(frozen=True)
class Grammar:
    """Production filters + literal bounds + ``with``-block context list.

    Each ``*_prods`` field gates the corresponding generator's output;
    defaults mask :data:`_DEFERRED_*` (productions whose generators
    aren't implemented yet) and :data:`_SLOW_*` (productions whose
    Hypothesis strategies materially slow draws — currently ``DECNUM``
    and ``RATIONAL``, which roughly triple ``stmt_block`` draw time).
    Opt either back in via :meth:`narrow`.
    """
    real_prods:    RealProd    = RealProd.ALL    & ~_DEFERRED_REAL & ~_SLOW_REAL
    bool_prods:    BoolProd    = BoolProd.ALL    & ~_DEFERRED_BOOL
    list_prods:    ListProd    = ListProd.ALL    & ~_DEFERRED_LIST
    tuple_prods:   TupleProd   = TupleProd.ALL
    context_prods: ContextProd = ContextProd.ALL
    stmt_prods:    StmtProd    = StmtProd.ALL    & ~_DEFERRED_STMT

    contexts:          tuple = tuple(_DEFAULT_CONTEXTS)
    int_literal_range: tuple[int, int] = _LITERAL_RANGE
    range_arg_range:   tuple[int, int] = (_DEFAULT_RANGE_ARG_MIN, _DEFAULT_RANGE_ARG_MAX)

    def __post_init__(self):
        # Hard preconditions that downstream helpers would otherwise
        # raise on (or crash with a less helpful message).  Catching
        # here points the stack trace at the test that built the
        # grammar, not at the strategy that consumed it.
        if StmtProd.ASSIGN not in self.stmt_prods:
            raise ValueError(
                'Grammar.stmt_prods must include StmtProd.ASSIGN — '
                'assignments are the only statement kind that fires at '
                'sub_depth=0 and serves as the body-block base case'
            )
        if not self.contexts:
            raise ValueError(
                'Grammar.contexts must be non-empty so context_expr can '
                'sample at least one with-block context'
            )

    def narrow(self, **kw) -> 'Grammar':
        """Return a copy of this grammar with the given fields replaced."""
        return dataclasses.replace(self, **kw)


DEFAULT_GRAMMAR = Grammar()
"""Default grammar: every documented production except those marked deferred
in the per-type :class:`Flag` enums (:class:`RealProd`, :class:`BoolProd`, …)."""


# ---------------------------------------------------------------------------
# Real-typed expressions
# ---------------------------------------------------------------------------

# (Production flag, AST class, surface-syntax name) — name matters for any
# pass that resolves the symbol back to a callable; the type-checker only
# looks at the class.
_REAL_NAMED_UNARY_FLAG: list[tuple[RealProd, type[NamedUnaryOp], str]] = [
    (RealProd.SQRT, Sqrt, 'sqrt'),
    (RealProd.SIN,  Sin,  'sin'),
    (RealProd.COS,  Cos,  'cos'),
    (RealProd.LOG,  Log,  'log'),
    (RealProd.EXP,  Exp,  'exp'),
]


def _real_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: RealProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``real`` under ``env``.

    ``include`` is an optional :class:`RealProd` flag that overrides
    ``grammar.real_prods`` for this helper only.  ``grammar`` carries
    the cross-type production filter that threads through recursive
    calls.
    """
    use = _check_flag(include, RealProd) if include is not None else grammar.real_prods
    rmin = range_arg_min if range_arg_min is not None else grammar.range_arg_range[0]
    rmax = range_arg_max if range_arg_max is not None else grammar.range_arg_range[1]

    leaves: list[st.SearchStrategy[Expr]] = []
    if RealProd.INTEGER in use:
        lo, hi = grammar.int_literal_range
        leaves.append(st.integers(lo, hi).map(lambda v: Integer(v, None)))
    if RealProd.DECNUM in use:
        lo, hi = grammar.int_literal_range
        # Decnums carry the decimal as a string.  Emit ``<int>.<frac>``
        # form (no exponent) — covers the non-dyadic-fraction path
        # (e.g. ``0.1``) without growing magnitudes via ``e±N``.
        leaves.append(
            st.tuples(st.integers(lo, hi), st.integers(0, 9999))
            .map(lambda pf: Decnum(f'{pf[0]}.{pf[1]:04d}', None))
        )
    if RealProd.HEXNUM in use:
        # Hexnums are dyadic-by-construction.  Emit ``[-]0x<i>.<f>p±E``
        # with at most 4 hex digits per part and a small binary exponent
        # to keep magnitudes manageable.
        hex_sym = _func_sym('hexnum')
        hex_digits = st.integers(0, 0xffff).map(lambda n: f'{n:x}')
        leaves.append(
            st.tuples(
                st.booleans(), hex_digits, hex_digits, st.integers(-8, 8),
            ).map(lambda t: Hexnum(
                hex_sym,
                f'{"-" if t[0] else ""}0x{t[1]}.{t[2]}p{t[3]:+d}',
                None,
            ))
        )
    if RealProd.RATIONAL in use:
        lo, hi = grammar.int_literal_range
        rat_sym = _func_sym('rational')
        # Bound the denominator separately to keep magnitudes manageable
        # — int_literal_range alone over both p and q would let the value
        # explode (``1000 / 1`` = 1000) or vanish (``1 / 1000`` ≈ 0).
        leaves.append(
            st.tuples(st.integers(lo, hi), st.integers(1, 100))
            .map(lambda pq: Rational(rat_sym, pq[0], pq[1], None))
        )
    if RealProd.VAR in use:
        var_strat = _var_of_type(env, RealType())
        if var_strat is not None:
            leaves.append(var_strat)

    inner: list[st.SearchStrategy[Expr]] = []
    if depth > 0:
        sub_real = _real_expr(
            env, depth - 1, include=use, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )
        sub_bool = _bool_expr(
            env, depth - 1, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )

        def _binop(cls):
            return st.tuples(sub_real, sub_real).map(lambda ab: cls(ab[0], ab[1], None))

        def _unop(cls):
            return sub_real.map(lambda x: cls(x, None))

        if RealProd.ADD in use: inner.append(_binop(Add))
        if RealProd.SUB in use: inner.append(_binop(Sub))
        if RealProd.MUL in use: inner.append(_binop(Mul))
        if RealProd.DIV in use: inner.append(_binop(Div))
        if RealProd.NEG in use: inner.append(_unop(Neg))
        if RealProd.ABS in use: inner.append(_unop(Abs))

        def _named_unary(cls: type[NamedUnaryOp], name: str) -> st.SearchStrategy[Expr]:
            sym = _func_sym(name)
            return sub_real.map(lambda x: cls(sym, x, None))

        for prod, cls, name in _REAL_NAMED_UNARY_FLAG:
            if prod in use:
                inner.append(_named_unary(cls, name))

        if RealProd.IF_EXPR in use:
            inner.append(st.tuples(sub_bool, sub_real, sub_real).map(
                lambda cab: IfExpr(cab[0], cab[1], cab[2], None)
            ))
        if RealProd.LEN in use:
            sub_real_list = _list_expr(
                RealType(), env, depth - 1, grammar=grammar,
                range_arg_min=rmin, range_arg_max=rmax,
            )
            inner.append(sub_real_list.map(
                lambda xs: Len(_func_sym('len'), xs, None)
            ))
        if RealProd.ROUND in use:
            # ``Cast`` is deliberately not generated: it asserts the
            # rounded value is exact, which fails whenever the value
            # isn't already representable in the active context (e.g.
            # ``cast(9)`` under E5M2). Sound generation would need a
            # per-context exact-representable check per literal.
            round_sym = _func_sym('round')
            inner.append(sub_real.map(lambda x: Round(round_sym, x, None)))
        if RealProd.CAST in use:
            raise NotImplementedError(
                'RealProd.CAST generation is deferred — see _real_expr docstring'
            )

    if not leaves and not inner:
        raise ValueError(
            f'real_expr produces nothing under include={use!r}; '
            'at minimum enable INTEGER, DECNUM, or RATIONAL '
            '(or VAR with a real-typed env)'
        )
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# Bool-typed expressions
# ---------------------------------------------------------------------------

_COMPARE_OPS = list(CompareOp)

# (Production flag, AST class, surface-syntax name) for the real-valued
# predicates dispatched by ``_bool_expr``.
_REAL_PREDICATES_FLAG: list[tuple[BoolProd, type[NamedUnaryOp], str]] = [
    (BoolProd.ISFINITE, IsFinite, 'isfinite'),
    (BoolProd.ISINF,    IsInf,    'isinf'),
    (BoolProd.ISNAN,    IsNan,    'isnan'),
    (BoolProd.SIGNBIT,  Signbit,  'signbit'),
    # ``IsNormal`` is intentionally absent from the dispatch table: enabling
    # ``BoolProd.ISNORMAL`` raises below because ``ops.isnormal`` crashes on
    # context-less integer Floats (e.g. ``isnormal(735)``).  Restore once
    # that path handles plain integers.
]


def _bool_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: BoolProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``bool`` under ``env``.

    ``include`` is an optional :class:`BoolProd` flag that overrides
    ``grammar.bool_prods`` for this helper only.
    """
    use = _check_flag(include, BoolProd) if include is not None else grammar.bool_prods
    rmin = range_arg_min if range_arg_min is not None else grammar.range_arg_range[0]
    rmax = range_arg_max if range_arg_max is not None else grammar.range_arg_range[1]

    leaves: list[st.SearchStrategy[Expr]] = []
    if BoolProd.LITERAL in use:
        leaves.append(st.booleans().map(lambda v: BoolVal(v, None)))
    if BoolProd.VAR in use:
        var_strat = _var_of_type(env, BoolType())
        if var_strat is not None:
            leaves.append(var_strat)

    inner: list[st.SearchStrategy[Expr]] = []
    if depth > 0:
        sub_bool = _bool_expr(
            env, depth - 1, include=use, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )
        sub_real = _real_expr(
            env, depth - 1, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )

        if BoolProd.COMPARE in use:
            # Restrict to simple binary comparisons (chain length 1).
            inner.append(st.tuples(
                st.sampled_from(_COMPARE_OPS), sub_real, sub_real,
            ).map(lambda oab: Compare([oab[0]], [oab[1], oab[2]], None)))
        if BoolProd.NOT in use:
            inner.append(sub_bool.map(lambda x: Not(x, None)))
        and_or_args = None
        if BoolProd.AND in use or BoolProd.OR in use:
            and_or_args = st.lists(sub_bool, min_size=2, max_size=3)
        if BoolProd.AND in use:
            assert and_or_args is not None
            inner.append(and_or_args.map(lambda xs: And(xs, None)))
        if BoolProd.OR in use:
            assert and_or_args is not None
            inner.append(and_or_args.map(lambda xs: Or(xs, None)))
        def _predicate(cls: type[NamedUnaryOp], name: str) -> st.SearchStrategy[Expr]:
            sym = _func_sym(name)
            return sub_real.map(lambda x: cls(sym, x, None))

        for prod, cls, name in _REAL_PREDICATES_FLAG:
            if prod in use:
                inner.append(_predicate(cls, name))
        if BoolProd.ISNORMAL in use:
            raise NotImplementedError(
                'BoolProd.ISNORMAL generation is deferred — see '
                '_REAL_PREDICATES_FLAG comment'
            )

    if not leaves and not inner:
        raise ValueError(
            f'bool_expr produces nothing under include={use!r}; '
            'at minimum enable LITERAL (or VAR with a bool-typed env)'
        )
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# List- and tuple-typed expressions
# ---------------------------------------------------------------------------

_LIST_LITERAL_LEN = (1, 4)
"""Inclusive size range for ``ListExpr`` literals. Min size 1 — an empty
``ListExpr`` has no element to anchor inference to the requested elt type."""


def _list_expr(
    elt_type: Type,
    env: TypeEnv,
    depth: int,
    *,
    include: ListProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``list[elt_type]`` expression under ``env``.

    ``include`` is an optional :class:`ListProd` flag.
    ``RANGE1``/``RANGE2`` only fire for ``RealType`` elements; ``ZIP``
    requires a ``TupleType`` element with arity ≥ 1; ``ENUMERATE``
    requires a 2-tuple with a ``RealType`` first component.
    """
    use = _check_flag(include, ListProd) if include is not None else grammar.list_prods
    rmin = range_arg_min if range_arg_min is not None else grammar.range_arg_range[0]
    rmax = range_arg_max if range_arg_max is not None else grammar.range_arg_range[1]

    productions: list[st.SearchStrategy[Expr]] = []
    if ListProd.LITERAL in use:
        elt_strat = expr(
            elt_type, env, max(0, depth - 1), grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )
        productions.append(st.lists(
            elt_strat, min_size=_LIST_LITERAL_LEN[0], max_size=_LIST_LITERAL_LEN[1],
        ).map(lambda xs: ListExpr(xs, None)))

    if ListProd.VAR in use:
        var_strat = _var_of_type(env, ListType(elt_type))
        if var_strat is not None:
            productions.append(var_strat)

    if depth > 0 and isinstance(elt_type, RealType):
        # Args restricted to ``Integer`` literals: the interpreter rejects
        # non-integer ``range`` args (e.g. ``range(log(0)) == range(-inf)``).
        range_sym = _func_sym('range')
        small_int = st.integers(rmin, rmax).map(lambda n: Integer(n, None))
        if ListProd.RANGE1 in use:
            productions.append(small_int.map(
                lambda n: Range1(range_sym, n, None)
            ))
        if ListProd.RANGE2 in use:
            productions.append(st.tuples(small_int, small_int).map(
                lambda ab: Range2(range_sym, ab[0], ab[1], None)
            ))
        if ListProd.RANGE3 in use:
            # ``range(start, stop, step)`` — step must be non-zero
            # (the interpreter, like Python's ``range``, rejects step 0).
            step_int = (
                st.integers(rmin, rmax)
                .filter(lambda n: n != 0)
                .map(lambda n: Integer(n, None))
            )
            productions.append(st.tuples(small_int, small_int, step_int).map(
                lambda abc: Range3(range_sym, abc[0], abc[1], abc[2], None)
            ))

    if (ListProd.ZIP in use and depth > 0
            and isinstance(elt_type, TupleType) and elt_type.elts):
        # FPy's ``zip`` is length-strict (unlike Python's default), so every
        # arg is a ``ListExpr`` literal of a shared drawn length.
        zip_sym = _func_sym('zip')
        elt_strats = [
            expr(
                sub_t, env, max(0, depth - 1), grammar=grammar,
                range_arg_min=rmin, range_arg_max=rmax,
            )
            for sub_t in elt_type.elts
        ]

        def _zip_of_len(n: int, _strats=elt_strats):
            sub_lists = [
                st.lists(s, min_size=n, max_size=n).map(lambda xs: ListExpr(xs, None))
                for s in _strats
            ]
            return st.tuples(*sub_lists).map(
                lambda xs: Zip(zip_sym, list(xs), None)
            )

        productions.append(st.integers(1, 4).flatmap(_zip_of_len))

    if (ListProd.ENUMERATE in use and depth > 0
            and isinstance(elt_type, TupleType)
            and len(elt_type.elts) == 2
            and isinstance(elt_type.elts[0], RealType)):
        # enumerate(xs : list[t]) : list[tuple[real, t]] — index first.
        _, val_t = elt_type.elts
        enum_sym = _func_sym('enumerate')
        sub_list = _list_expr(
            val_t, env, depth - 1, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )
        productions.append(sub_list.map(
            lambda xs: Enumerate(enum_sym, xs, None)
        ))

    if ListProd.LIST_COMP in use:
        raise NotImplementedError(
            'ListProd.LIST_COMP generation is deferred'
        )

    if not productions:
        raise ValueError(
            f'list_expr produces nothing under include={use!r}; '
            'enable LITERAL (always available) or RANGE1/RANGE2 '
            '(real elt + depth > 0)'
        )
    return st.one_of(*productions)


def _tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    include: TupleProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``tuple[*elt_types]`` expression under ``env``.

    ``include`` is an optional :class:`TupleProd` flag.
    """
    use = _check_flag(include, TupleProd) if include is not None else grammar.tuple_prods
    rmin = range_arg_min if range_arg_min is not None else grammar.range_arg_range[0]
    rmax = range_arg_max if range_arg_max is not None else grammar.range_arg_range[1]

    productions: list[st.SearchStrategy[Expr]] = []

    if TupleProd.VAR in use:
        var_strat = _var_of_type(env, TupleType(*elt_types))
        if var_strat is not None:
            productions.append(var_strat)

    if TupleProd.LITERAL not in use:
        if not productions:
            raise ValueError(
                f'tuple_expr produces nothing under include={use!r}; '
                'enable LITERAL (always available) or VAR (with a '
                'matching-type binding in env)'
            )
        return st.one_of(*productions)

    sub_depth = max(0, depth - 1)
    elt_strats = [
        expr(
            t, env, sub_depth, grammar=grammar,
            range_arg_min=rmin, range_arg_max=rmax,
        )
        for t in elt_types
    ]
    if not elt_strats:
        # tuple[] — degenerate case; emit an empty TupleExpr.
        productions.append(st.just(TupleExpr([], None)))
    else:
        productions.append(
            st.tuples(*elt_strats).map(lambda elts: TupleExpr(list(elts), None))
        )
    return st.one_of(*productions)


# ---------------------------------------------------------------------------
# Context-typed expressions
# ---------------------------------------------------------------------------

def _ctx_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: ContextProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``context``.

    Only production: ``ForeignVal`` of a value from ``grammar.contexts``
    (default :data:`_DEFAULT_CONTEXTS`).  ``env``/``depth``/``range_arg_*``
    accepted for API symmetry.
    """
    del env, depth, range_arg_min, range_arg_max
    use = _check_flag(include, ContextProd) if include is not None else grammar.context_prods
    if ContextProd.LITERAL not in use:
        raise ValueError(
            f'ctx_expr has no productions under include={use!r}; '
            'enable LITERAL'
        )
    contexts = grammar.contexts or _DEFAULT_CONTEXTS
    return st.sampled_from(list(contexts)).map(lambda c: ForeignVal(c, None))


# ---------------------------------------------------------------------------
# Type-dispatched expression strategy
# ---------------------------------------------------------------------------

def expr(
    target: Type,
    env: TypeEnv,
    depth: int,
    *,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Strategy for an expression of type ``target`` under ``env``.

    This is the canonical type-directed entry point. ``depth`` bounds the
    nesting depth of compound operations; at ``depth == 0`` only leaves
    (literals and variable references for scalars, or single-level
    constructors for compound types) are produced.

    ``grammar`` carries per-type production filters, ``with``-block
    contexts, and literal-range bounds; see :class:`Grammar`.
    ``range_arg_min`` / ``range_arg_max`` (when given) override
    ``grammar.range_arg_range`` for this call only.

    Raises :class:`NotImplementedError` for type targets not yet supported.
    """
    if isinstance(target, RealType):
        return _real_expr(
            env, depth, grammar=grammar,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
    if isinstance(target, BoolType):
        return _bool_expr(
            env, depth, grammar=grammar,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
    if isinstance(target, ListType):
        return _list_expr(
            target.elt, env, depth, grammar=grammar,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
    if isinstance(target, TupleType):
        return _tuple_expr(
            target.elts, env, depth, grammar=grammar,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
    if isinstance(target, ContextType):
        return _ctx_expr(
            env, depth, grammar=grammar,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
    raise NotImplementedError(f'no generator for target type {target.format()}')


def real_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: RealProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(RealType(), env, depth)``.

    ``include`` is an optional :class:`RealProd` flag.
    """
    return _real_expr(
        env, depth, include=include, grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def bool_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: BoolProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(BoolType(), env, depth)``.

    ``include`` is an optional :class:`BoolProd` flag.
    """
    return _bool_expr(
        env, depth, include=include, grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def list_expr(
    elt_type: Type,
    env: TypeEnv,
    depth: int,
    *,
    include: ListProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ListType(elt_type), env, depth)``.

    ``include`` is an optional :class:`ListProd` flag.
    """
    return _list_expr(
        elt_type, env, depth, include=include, grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    include: TupleProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(TupleType(*elt_types), env, depth)``.

    ``include`` is an optional :class:`TupleProd` flag.
    """
    return _tuple_expr(
        elt_types, env, depth, include=include, grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def context_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: ContextProd | None = None,
    grammar: Grammar = DEFAULT_GRAMMAR,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ContextType(), env, depth)``.

    ``include`` is an optional :class:`ContextProd` flag.  Contexts
    sampled from ``grammar.contexts``.
    """
    return _ctx_expr(env, depth, include=include, grammar=grammar)


# ---------------------------------------------------------------------------
# Statement blocks
# ---------------------------------------------------------------------------

# Default type strategy for ``Assign`` locals: scalars + small compound
# types (depth-1 lists/tuples of scalars). Overridable via ``local_types``.
def _default_local_types() -> st.SearchStrategy[Type]:
    return arbitrary_type(max_depth=1)


@st.composite
def _stmt_block(
    draw,
    env: TypeEnv,
    return_type: Type,
    depth: int,
    max_assigns: int = 3,
    max_contexts: int = 0,
    max_ifs: int = 0,
    max_loops: int = 0,
    max_whiles: int = 0,
    local_prefix: str = 't',
    local_types: st.SearchStrategy[Type] | None = None,
    *,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> StmtBlock:
    """Generate a ``StmtBlock`` ending in a ``ReturnStmt``.

    Before the return, the block contains up to ``max_assigns`` ``Assign``,
    ``max_contexts`` ``ContextStmt``, ``max_ifs`` ``IfStmt``/``If1Stmt``,
    ``max_loops`` ``ForStmt`` (over ``range(small_int)`` with a fresh
    real-typed loop var), and ``max_whiles`` ``WhileStmt`` (counter-driven
    template ``c = 0; while c < N: <body>; c = c + 1``).

    ``grammar`` carries the production filters and context list threaded
    through every expression sub-strategy.  ``grammar.stmt_prods`` also
    gates which statement kinds can appear: a kind is generated only when
    *both* its corresponding :class:`StmtProd` flag is set and its
    ``max_*`` budget is non-zero (so ``max_*`` bounds quantity, and
    ``stmt_prods`` bounds existence).  ``StmtProd.ASSIGN`` is required —
    it is the only kind that fires at ``sub_depth=0`` and at the base of
    any ``min_n=1`` body block.  Per-call ``range_arg_min``/
    ``range_arg_max`` override ``grammar.range_arg_range`` for this call
    only.

    Scoping (see module docstring for the rationale): ``with``-body assigns
    leak out, ``if`` / ``for`` / ``while`` body assigns don't; for-loop
    targets and while-body locals are body-scoped, while counters leak.
    Local names are issued from a single counter shared across all nested
    bodies. ``local_types`` overrides the Assign-local type strategy
    (defaults to :func:`arbitrary_type` at depth 1).
    """
    # ``StmtProd.ASSIGN in grammar.stmt_prods`` is validated at
    # ``Grammar.__post_init__``, so this helper can assume it.
    if local_types is None:
        local_types = _default_local_types()
    counter = [0]

    def fresh_local() -> NamedId:
        i = counter[0]
        counter[0] += 1
        return NamedId(f'{local_prefix}{i}')

    rmin = range_arg_min if range_arg_min is not None else grammar.range_arg_range[0]
    rmax = range_arg_max if range_arg_max is not None else grammar.range_arg_range[1]

    range_sym = _func_sym('range')

    def gen_stmts(
        env_local: TypeEnv,
        sub_depth: int,
        max_n: int,
        ctx_budget: int,
        if_budget: int,
        loop_budget: int,
        while_budget: int,
        min_n: int = 0,
    ) -> list[Stmt]:
        """Returns a list of statements; mutates ``env_local`` for new top-level bindings."""
        n = draw(st.integers(min_n, max(min_n, max_n)))
        stmts: list[Stmt] = []
        for _ in range(n):
            kinds: list[str] = []
            if StmtProd.ASSIGN in grammar.stmt_prods:
                kinds.append('assign')
            if (StmtProd.WITH in grammar.stmt_prods
                    and ctx_budget > 0 and sub_depth > 0):
                kinds.append('context')
            if (StmtProd.IF in grammar.stmt_prods
                    and if_budget > 0 and sub_depth > 0):
                kinds.append('if')
            if (StmtProd.FOR in grammar.stmt_prods
                    and loop_budget > 0 and sub_depth > 0):
                kinds.append('for')
            if (StmtProd.WHILE in grammar.stmt_prods
                    and while_budget > 0 and sub_depth > 0):
                kinds.append('while')
            kind = draw(st.sampled_from(kinds))

            if kind == 'context':
                ctx_value_expr = draw(_ctx_expr(
                    env_local, sub_depth, grammar=grammar,
                ))
                inner_env = dict(env_local)
                inner_stmts = gen_stmts(
                    inner_env, sub_depth - 1,
                    max(1, max_n // 2),
                    ctx_budget - 1, if_budget, loop_budget, while_budget,
                    min_n=1,
                )
                # With-body assigns leak out (FPy lexical scoping).
                for k, v in inner_env.items():
                    if k not in env_local:
                        env_local[k] = v
                stmts.append(ContextStmt(
                    UnderscoreId(), ctx_value_expr,
                    StmtBlock(inner_stmts), None,
                ))
                ctx_budget -= 1

            elif kind == 'if':
                cond = draw(_bool_expr(
                    env_local, sub_depth, grammar=grammar,
                    range_arg_min=rmin, range_arg_max=rmax,
                ))
                two_armed = draw(st.booleans())
                # Branch locals do not propagate out; the shared counter
                # still prevents cross-branch collisions.
                ift_env = dict(env_local)
                ift_stmts = gen_stmts(
                    ift_env, sub_depth - 1,
                    max(1, max_n // 2),
                    ctx_budget, if_budget - 1, loop_budget, while_budget,
                    min_n=1,
                )
                if two_armed:
                    iff_env = dict(env_local)
                    iff_stmts = gen_stmts(
                        iff_env, sub_depth - 1,
                        max(1, max_n // 2),
                        ctx_budget, if_budget - 1, loop_budget, while_budget,
                        min_n=1,
                    )
                    stmts.append(IfStmt(
                        cond, StmtBlock(ift_stmts), StmtBlock(iff_stmts), None,
                    ))
                else:
                    stmts.append(If1Stmt(cond, StmtBlock(ift_stmts), None))
                if_budget -= 1

            elif kind == 'for':
                bound = draw(st.integers(rmin, rmax))
                iterable = Range1(range_sym, Integer(bound, None), None)
                loop_var = fresh_local()
                loop_env = dict(env_local)
                loop_env[loop_var] = RealType()
                loop_stmts = gen_stmts(
                    loop_env, sub_depth - 1,
                    max(1, max_n // 2),
                    ctx_budget, if_budget, loop_budget - 1, while_budget,
                    min_n=1,
                )
                # Loop var and body locals are body-scoped — don't leak.
                stmts.append(ForStmt(
                    loop_var, iterable, StmtBlock(loop_stmts), None,
                ))
                loop_budget -= 1

            elif kind == 'while':
                # Counter-driven template guarantees termination.
                bound = draw(st.integers(0, 4))
                counter = fresh_local()
                stmts.append(Assign(counter, None, Integer(0, None), None))
                env_local[counter] = RealType()
                body_env = dict(env_local)
                body_stmts = gen_stmts(
                    body_env, sub_depth - 1,
                    max(1, max_n // 2),
                    ctx_budget, if_budget, loop_budget, while_budget - 1,
                    min_n=0,
                )
                inc = Assign(
                    counter, None,
                    Add(Var(counter, None), Integer(1, None), None), None,
                )
                cond = Compare(
                    [CompareOp.LT],
                    [Var(counter, None), Integer(bound, None)],
                    None,
                )
                stmts.append(WhileStmt(cond, StmtBlock(body_stmts + [inc]), None))
                while_budget -= 1

            else:  # 'assign'
                name = fresh_local()
                local_type = draw(local_types)
                rhs = draw(expr(
                    local_type, env_local, sub_depth, grammar=grammar,
                    range_arg_min=rmin, range_arg_max=rmax,
                ))
                stmts.append(Assign(name, None, rhs, None))
                env_local[name] = local_type

        return stmts

    env = dict(env)
    body_stmts = gen_stmts(
        env, depth, max_assigns,
        max_contexts, max_ifs, max_loops, max_whiles,
    )
    ret_expr = draw(expr(
        return_type, env, depth, grammar=grammar,
        range_arg_min=rmin, range_arg_max=rmax,
    ))
    return StmtBlock(body_stmts + [ReturnStmt(ret_expr, None)])


def stmt_block(
    env: TypeEnv,
    return_type: Type,
    depth: int,
    max_assigns: int = 3,
    max_contexts: int = 0,
    max_ifs: int = 0,
    max_loops: int = 0,
    max_whiles: int = 0,
    local_prefix: str = 't',
    local_types: st.SearchStrategy[Type] | None = None,
    *,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[StmtBlock]:
    """Public wrapper around the statement-block strategy."""
    return _stmt_block(
        env, return_type, depth,
        max_assigns, max_contexts, max_ifs, max_loops, max_whiles,
        local_prefix, local_types,
        grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


# ---------------------------------------------------------------------------
# Type → TypeAnn / random-type / value-for-type
# ---------------------------------------------------------------------------

def _type_to_typeann(t: Type) -> TypeAnn:
    """Map an FPy ``Type`` to its surface-syntax ``TypeAnn``."""
    if isinstance(t, RealType):
        return RealTypeAnn(None, None)
    if isinstance(t, BoolType):
        return BoolTypeAnn(None)
    if isinstance(t, TupleType):
        return TupleTypeAnn([_type_to_typeann(e) for e in t.elts], None)
    if isinstance(t, ListType):
        return ListTypeAnn(_type_to_typeann(t.elt), None)
    if isinstance(t, ContextType):
        return ContextTypeAnn(None)
    raise NotImplementedError(f'no TypeAnn for type {t.format()}')


def arbitrary_type(
    max_depth: int = 2,
    *,
    scalar_only: bool = False,
) -> st.SearchStrategy[Type]:
    """Strategy for an arbitrary FPy ``Type`` up to nesting depth ``max_depth``.

    With ``scalar_only=True``, only ``RealType`` / ``BoolType`` are produced —
    useful for generating an arg signature whose values are easy to feed
    into the interpreter.
    """
    scalar = st.one_of(st.just(RealType()), st.just(BoolType()))
    if max_depth <= 0 or scalar_only:
        return scalar
    inner = arbitrary_type(max_depth - 1, scalar_only=scalar_only)
    list_t = inner.map(ListType)
    tuple_t = st.lists(inner, min_size=1, max_size=3).map(
        lambda elts: TupleType(*elts)
    )
    return st.one_of(scalar, list_t, tuple_t)


def value_for_type(t: Type) -> st.SearchStrategy:
    """Strategy for a Python value of type ``t``, suitable as a function input.

    - ``RealType`` ⇒ a small ``RealFloat``
    - ``BoolType`` ⇒ a ``bool``
    - ``TupleType`` ⇒ a ``tuple`` of element values
    - ``ListType`` ⇒ a ``list`` (length 0–4) of element values
    """
    if isinstance(t, RealType):
        # Import locally to avoid a top-level dep cycle with this module.
        from .number import real_floats
        return real_floats(prec_max=8, exp_min=-4, exp_max=4)
    if isinstance(t, BoolType):
        return st.booleans()
    if isinstance(t, TupleType):
        return st.tuples(*[value_for_type(e) for e in t.elts])
    if isinstance(t, ListType):
        return st.lists(value_for_type(t.elt), min_size=0, max_size=4)
    raise NotImplementedError(f'no value strategy for type {t.format()}')


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

def _make_arg_names(n: int) -> list[NamedId]:
    return [NamedId(f'x{i}') for i in range(n)]


def _make_funcdef(
    name: str,
    arg_names: list[NamedId],
    arg_types: tuple[Type, ...],
    body: StmtBlock,
) -> FuncDef:
    """Wrap a ``StmtBlock`` body in a ``def name(arg_names: ...): ...`` ``FuncDef``."""
    args = [
        Argument(n, _type_to_typeann(t), None)
        for n, t in zip(arg_names, arg_types)
    ]
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef(name, args, body, meta)


@st.composite
def fpy_funcdef(
    draw,
    arg_types: tuple[Type, ...],
    return_type: Type,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    max_contexts: st.SearchStrategy[int] | None = None,
    max_ifs: st.SearchStrategy[int] | None = None,
    max_loops: st.SearchStrategy[int] | None = None,
    max_whiles: st.SearchStrategy[int] | None = None,
    *,
    name: str = 'f',
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> FuncDef:
    """Generate a well-typed FPy function with the given signature.

    ``arg_types`` and ``return_type`` may be any types accepted by :func:`expr`.
    The body is a sequence of ``Assign`` / ``ContextStmt`` / ``IfStmt`` /
    ``If1Stmt`` / ``ForStmt`` / ``WhileStmt`` statements followed by a
    ``ReturnStmt`` of type ``return_type``. See :func:`stmt_block` for
    ``max_*`` semantics; ``grammar`` controls the expression productions.
    """
    if max_depth is None:
        max_depth = st.integers(0, 3)
    if max_assigns is None:
        max_assigns = st.integers(0, 3)
    if max_contexts is None:
        max_contexts = st.integers(0, 2)
    if max_ifs is None:
        max_ifs = st.integers(0, 2)
    if max_loops is None:
        max_loops = st.integers(0, 2)
    if max_whiles is None:
        max_whiles = st.integers(0, 2)

    arg_names = _make_arg_names(len(arg_types))
    env: TypeEnv = dict(zip(arg_names, arg_types))
    body = draw(_stmt_block(
        env, return_type, draw(max_depth),
        max_assigns=draw(max_assigns), max_contexts=draw(max_contexts),
        max_ifs=draw(max_ifs), max_loops=draw(max_loops),
        max_whiles=draw(max_whiles),
        grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ))
    return _make_funcdef(name, arg_names, arg_types, body)


def fpy_function(
    arg_types: tuple[Type, ...],
    return_type: Type,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    max_contexts: st.SearchStrategy[int] | None = None,
    max_ifs: st.SearchStrategy[int] | None = None,
    max_loops: st.SearchStrategy[int] | None = None,
    max_whiles: st.SearchStrategy[int] | None = None,
    *,
    name: str = 'f',
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_funcdef(
        arg_types, return_type,
        max_depth=max_depth, max_assigns=max_assigns,
        max_contexts=max_contexts, max_ifs=max_ifs,
        max_loops=max_loops, max_whiles=max_whiles,
        name=name, grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ).map(fp.Function)


# ---------------------------------------------------------------------------
# All-real-signature convenience wrappers
# ---------------------------------------------------------------------------

@st.composite
def fpy_real_funcdef(
    draw,
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    max_contexts: st.SearchStrategy[int] | None = None,
    max_ifs: st.SearchStrategy[int] | None = None,
    max_loops: st.SearchStrategy[int] | None = None,
    max_whiles: st.SearchStrategy[int] | None = None,
    *,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> FuncDef:
    """Generate ``f(x0..xN: Real) -> Real``.

    Thin convenience wrapper around :func:`fpy_funcdef` for the
    common all-real signature.
    """
    if num_args is None:
        num_args = st.integers(0, 3)
    n = draw(num_args)
    arg_types: tuple[Type, ...] = tuple(RealType() for _ in range(n))
    return draw(fpy_funcdef(
        arg_types, RealType(),
        max_depth=max_depth, max_assigns=max_assigns,
        max_contexts=max_contexts, max_ifs=max_ifs,
        max_loops=max_loops, max_whiles=max_whiles,
        grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ))


def fpy_real_function(
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    max_contexts: st.SearchStrategy[int] | None = None,
    max_ifs: st.SearchStrategy[int] | None = None,
    max_loops: st.SearchStrategy[int] | None = None,
    max_whiles: st.SearchStrategy[int] | None = None,
    *,
    grammar: Grammar = DEFAULT_GRAMMAR,
    range_arg_min: int | None = None,
    range_arg_max: int | None = None,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_real_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_real_funcdef(
        num_args=num_args, max_depth=max_depth, max_assigns=max_assigns,
        max_contexts=max_contexts, max_ifs=max_ifs,
        max_loops=max_loops, max_whiles=max_whiles,
        grammar=grammar,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ).map(fp.Function)
