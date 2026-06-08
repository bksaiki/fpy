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

Productions per target type are gated by string tags (see :data:`REAL_TAGS`
etc.). Pass ``include={...}`` to narrow what a helper emits; tags
propagate through same-type recursion but not across types. Per-call
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

from typing import TypeAlias

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
    Div,
    Enumerate,
    Exp,
    Expr,
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


# ---------------------------------------------------------------------------
# Production tag sets — narrowing escape hatch
# ---------------------------------------------------------------------------
# Each per-target generator gates its productions by tag membership. Pass
# ``include={'literal', 'arith'}`` to a helper to narrow what it emits.
# Tags propagate through same-type recursion but not across types
# (a narrowed ``_real_expr`` still calls ``_bool_expr`` with bool defaults).

REAL_TAGS: frozenset[str] = frozenset({
    'literal', 'var', 'arith', 'if_expr', 'len', 'named_unary', 'round',
})
"""All production tags supported by :func:`real_expr`. ``round`` enables
``Round`` only — ``Cast`` is deferred (see ``_real_expr``)."""

BOOL_TAGS: frozenset[str] = frozenset({
    'literal', 'var', 'compare', 'predicate', 'not', 'and_or',
})
"""All production tags supported by :func:`bool_expr`."""

LIST_TAGS: frozenset[str] = frozenset({
    'literal', 'var', 'range', 'zip', 'enumerate',
})
"""All production tags supported by :func:`list_expr`. ``zip`` and
``enumerate`` only fire when the element type is a ``TupleType`` of the
appropriate shape (``zip``: arity ≥ 1; ``enumerate``: arity 2 with the
first element ``RealType``)."""

TUPLE_TAGS: frozenset[str] = frozenset({'literal', 'var'})
"""All production tags supported by :func:`tuple_expr`."""

CONTEXT_TAGS: frozenset[str] = frozenset({'literal'})
"""All production tags supported by :func:`context_expr`."""

# Concrete rounding contexts the generator can drop into ``with`` blocks.
# ``fp.REAL`` is excluded — ``sqrt``/``sin``/``log``/``div`` raise under it
# (no closed-form rational result), so ``with REAL: t = sqrt(0)`` would crash.
_DEFAULT_CONTEXTS: list = [fp.FP64, fp.FP32, fp.MX_E5M2, fp.MX_E4M3]


def _resolve_include(include: set[str] | None, all_tags: frozenset[str]) -> frozenset[str]:
    """``None`` ⇒ everything; otherwise intersect with the helper's tag set."""
    if include is None:
        return all_tags
    unknown = set(include) - all_tags
    if unknown:
        raise ValueError(
            f'unknown production tags: {sorted(unknown)}; '
            f'valid tags for this generator are {sorted(all_tags)}'
        )
    return frozenset(include)


# ---------------------------------------------------------------------------
# Real-typed expressions
# ---------------------------------------------------------------------------

def _real_literal_strategy() -> st.SearchStrategy[Expr]:
    return st.integers(*_LITERAL_RANGE).map(lambda v: Integer(v, None))


# (AST class, surface-syntax name) — name matters for any pass that resolves
# the symbol back to a callable; the type-checker only looks at the class.
_REAL_NAMED_UNARY: list[tuple[type[NamedUnaryOp], str]] = [
    (Sqrt, 'sqrt'),
    (Sin, 'sin'),
    (Cos, 'cos'),
    (Log, 'log'),
    (Exp, 'exp'),
]


def _real_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``real`` under ``env``.

    See :data:`REAL_TAGS` for the supported ``include`` values.
    """
    use = _resolve_include(include, REAL_TAGS)

    leaves: list[st.SearchStrategy[Expr]] = []
    if 'literal' in use:
        leaves.append(_real_literal_strategy())
    if 'var' in use:
        var_strat = _var_of_type(env, RealType())
        if var_strat is not None:
            leaves.append(var_strat)

    inner: list[st.SearchStrategy[Expr]] = []
    if depth > 0:
        sub_real = _real_expr(
            env, depth - 1, include=include,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
        sub_bool = _bool_expr(
            env, depth - 1,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )

        def _binop(cls):
            return st.tuples(sub_real, sub_real).map(lambda ab: cls(ab[0], ab[1], None))

        def _unop(cls):
            return sub_real.map(lambda x: cls(x, None))

        def _named_unary(cls, name):
            sym = _func_sym(name)
            return sub_real.map(lambda x: cls(sym, x, None))

        if 'arith' in use:
            inner.extend([
                _binop(Add), _binop(Sub), _binop(Mul), _binop(Div),
                _unop(Neg), _unop(Abs),
            ])
        if 'if_expr' in use:
            inner.append(st.tuples(sub_bool, sub_real, sub_real).map(
                lambda cab: IfExpr(cab[0], cab[1], cab[2], None)
            ))
        if 'len' in use:
            sub_real_list = _list_expr(
                RealType(), env, depth - 1,
                range_arg_min=range_arg_min, range_arg_max=range_arg_max,
            )
            inner.append(sub_real_list.map(
                lambda xs: Len(_func_sym('len'), xs, None)
            ))
        if 'named_unary' in use:
            inner.extend(_named_unary(cls, name) for cls, name in _REAL_NAMED_UNARY)
        if 'round' in use:
            # ``Cast`` is deliberately not generated: it asserts the
            # rounded value is exact, which fails whenever the value
            # isn't already representable in the active context (e.g.
            # ``cast(9)`` under E5M2). Sound generation would need a
            # per-context exact-representable check per literal.
            round_sym = _func_sym('round')
            inner.append(sub_real.map(lambda x: Round(round_sym, x, None)))

    if not leaves and not inner:
        raise ValueError(
            'real_expr produces nothing under the given include set; '
            'at minimum include "literal" (or "var" with a real-typed env)'
        )
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# Bool-typed expressions
# ---------------------------------------------------------------------------

_COMPARE_OPS = list(CompareOp)

# ``IsNormal`` is omitted: ``ops.isnormal`` crashes on context-less integer
# Floats (e.g. ``isnormal(735)``) via ``Float.is_normal`` (floats.py:616) —
# the generator surfaced this; restore once that path handles plain integers.
_REAL_PREDICATES: list[tuple[type[NamedUnaryOp], str]] = [
    (IsFinite, 'isfinite'),
    (IsInf, 'isinf'),
    (IsNan, 'isnan'),
    (Signbit, 'signbit'),
]


def _bool_literal_strategy() -> st.SearchStrategy[Expr]:
    return st.booleans().map(lambda v: BoolVal(v, None))


def _bool_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``bool`` under ``env``.

    See :data:`BOOL_TAGS` for the supported ``include`` values.
    """
    use = _resolve_include(include, BOOL_TAGS)

    leaves: list[st.SearchStrategy[Expr]] = []
    if 'literal' in use:
        leaves.append(_bool_literal_strategy())
    if 'var' in use:
        var_strat = _var_of_type(env, BoolType())
        if var_strat is not None:
            leaves.append(var_strat)

    inner: list[st.SearchStrategy[Expr]] = []
    if depth > 0:
        sub_bool = _bool_expr(
            env, depth - 1, include=include,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
        sub_real = _real_expr(
            env, depth - 1,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )

        if 'compare' in use:
            # Restrict to simple binary comparisons (chain length 1).
            inner.append(st.tuples(
                st.sampled_from(_COMPARE_OPS), sub_real, sub_real,
            ).map(lambda oab: Compare([oab[0]], [oab[1], oab[2]], None)))
        if 'not' in use:
            inner.append(sub_bool.map(lambda x: Not(x, None)))
        if 'and_or' in use:
            and_or_args = st.lists(sub_bool, min_size=2, max_size=3)
            inner.append(and_or_args.map(lambda xs: And(xs, None)))
            inner.append(and_or_args.map(lambda xs: Or(xs, None)))
        if 'predicate' in use:
            def _predicate(cls, name):
                sym = _func_sym(name)
                return sub_real.map(lambda x: cls(sym, x, None))
            inner.extend(_predicate(cls, name) for cls, name in _REAL_PREDICATES)

    if not leaves and not inner:
        raise ValueError(
            'bool_expr produces nothing under the given include set; '
            'at minimum include "literal" (or "var" with a bool-typed env)'
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
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``list[elt_type]`` expression under ``env``.

    See :data:`LIST_TAGS` for the supported ``include`` values.

    ``range_arg_min`` / ``range_arg_max`` (inclusive) bound the integer
    literals used as ``Range1`` / ``Range2`` arguments when ``elt_type`` is
    ``RealType``.
    """
    use = _resolve_include(include, LIST_TAGS)

    productions: list[st.SearchStrategy[Expr]] = []
    if 'literal' in use:
        elt_strat = expr(
            elt_type, env, max(0, depth - 1),
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
        productions.append(st.lists(
            elt_strat, min_size=_LIST_LITERAL_LEN[0], max_size=_LIST_LITERAL_LEN[1],
        ).map(lambda xs: ListExpr(xs, None)))

    if 'var' in use:
        var_strat = _var_of_type(env, ListType(elt_type))
        if var_strat is not None:
            productions.append(var_strat)

    if 'range' in use and depth > 0 and isinstance(elt_type, RealType):
        # Args restricted to ``Integer`` literals: the interpreter rejects
        # non-integer ``range`` args (e.g. ``range(log(0)) == range(-inf)``).
        range_sym = _func_sym('range')
        small_int = st.integers(range_arg_min, range_arg_max).map(
            lambda n: Integer(n, None)
        )
        productions.append(small_int.map(lambda n: Range1(range_sym, n, None)))
        productions.append(st.tuples(small_int, small_int).map(
            lambda ab: Range2(range_sym, ab[0], ab[1], None)
        ))

    if 'zip' in use and depth > 0 and isinstance(elt_type, TupleType) and elt_type.elts:
        # FPy's ``zip`` is length-strict (unlike Python's default), so every
        # arg is a ``ListExpr`` literal of a shared drawn length.
        zip_sym = _func_sym('zip')
        elt_strats = [
            expr(
                sub_t, env, max(0, depth - 1),
                range_arg_min=range_arg_min, range_arg_max=range_arg_max,
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

    if ('enumerate' in use and depth > 0 and isinstance(elt_type, TupleType)
            and len(elt_type.elts) == 2 and isinstance(elt_type.elts[0], RealType)):
        # enumerate(xs : list[t]) : list[tuple[real, t]] — index first.
        _, val_t = elt_type.elts
        enum_sym = _func_sym('enumerate')
        sub_list = _list_expr(
            val_t, env, depth - 1,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
        )
        productions.append(sub_list.map(
            lambda xs: Enumerate(enum_sym, xs, None)
        ))

    if not productions:
        raise ValueError(
            'list_expr produces nothing under the given include set; '
            'include "literal" (always available) or "range" (real elt + depth > 0)'
        )
    return st.one_of(*productions)


def _tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``tuple[*elt_types]`` expression under ``env``.

    See :data:`TUPLE_TAGS` for the supported ``include`` values.
    """
    use = _resolve_include(include, TUPLE_TAGS)

    productions: list[st.SearchStrategy[Expr]] = []

    if 'var' in use:
        var_strat = _var_of_type(env, TupleType(*elt_types))
        if var_strat is not None:
            productions.append(var_strat)

    if 'literal' not in use:
        if not productions:
            raise ValueError(
                'tuple_expr produces nothing under the given include set; '
                'include "literal" (always available) or "var" (with a '
                'matching-type binding in env)'
            )
        return st.one_of(*productions)

    sub_depth = max(0, depth - 1)
    elt_strats = [
        expr(
            t, env, sub_depth,
            range_arg_min=range_arg_min, range_arg_max=range_arg_max,
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
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``context``.

    Only production: ``ForeignVal`` of a value from :data:`_DEFAULT_CONTEXTS`.
    ``env``/``depth``/``range_arg_*`` are accepted for API symmetry.
    """
    del env, depth, range_arg_min, range_arg_max
    _resolve_include(include, CONTEXT_TAGS)  # validation only
    return st.sampled_from(_DEFAULT_CONTEXTS).map(lambda c: ForeignVal(c, None))


# ---------------------------------------------------------------------------
# Type-dispatched expression strategy
# ---------------------------------------------------------------------------

def expr(
    target: Type,
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an expression of type ``target`` under ``env``.

    This is the canonical type-directed entry point. ``depth`` bounds the
    nesting depth of compound operations; at ``depth == 0`` only leaves
    (literals and variable references for scalars, or single-level
    constructors for compound types) are produced.

    ``range_arg_min`` / ``range_arg_max`` (inclusive) bound the integer
    literals used as ``Range1`` / ``Range2`` arguments. Affects only
    list-typed sub-strategies (reachable directly via a ``ListType``
    target, or indirectly through ``Len(list[real])`` in real expressions).

    Raises :class:`NotImplementedError` for type targets not yet supported
    (e.g. ``ContextType``, ``FunctionType``).
    """
    kw = dict(range_arg_min=range_arg_min, range_arg_max=range_arg_max)
    if isinstance(target, RealType):
        return _real_expr(env, depth, **kw)
    if isinstance(target, BoolType):
        return _bool_expr(env, depth, **kw)
    if isinstance(target, ListType):
        return _list_expr(target.elt, env, depth, **kw)
    if isinstance(target, TupleType):
        return _tuple_expr(target.elts, env, depth, **kw)
    if isinstance(target, ContextType):
        return _ctx_expr(env, depth, **kw)
    raise NotImplementedError(f'no generator for target type {target.format()}')


def real_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(RealType(), env, depth)``.

    See :data:`REAL_TAGS` for the supported ``include`` values.
    """
    return _real_expr(
        env, depth, include=include,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def bool_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(BoolType(), env, depth)``.

    See :data:`BOOL_TAGS` for the supported ``include`` values.
    """
    return _bool_expr(
        env, depth, include=include,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def list_expr(
    elt_type: Type,
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ListType(elt_type), env, depth)``.

    See :data:`LIST_TAGS` for the supported ``include`` values.
    """
    return _list_expr(
        elt_type, env, depth, include=include,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(TupleType(*elt_types), env, depth)``.

    See :data:`TUPLE_TAGS` for the supported ``include`` values.
    """
    return _tuple_expr(
        elt_types, env, depth, include=include,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def context_expr(
    env: TypeEnv,
    depth: int,
    *,
    include: set[str] | None = None,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ContextType(), env, depth)``.

    See :data:`CONTEXT_TAGS` for the supported ``include`` values.
    """
    return _ctx_expr(env, depth, include=include)


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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> StmtBlock:
    """Generate a ``StmtBlock`` ending in a ``ReturnStmt``.

    Before the return, the block contains up to ``max_assigns`` ``Assign``,
    ``max_contexts`` ``ContextStmt``, ``max_ifs`` ``IfStmt``/``If1Stmt``,
    ``max_loops`` ``ForStmt`` (over ``range(small_int)`` with a fresh
    real-typed loop var), and ``max_whiles`` ``WhileStmt`` (counter-driven
    template ``c = 0; while c < N: <body>; c = c + 1``).

    Scoping (see module docstring for the rationale): ``with``-body assigns
    leak out, ``if`` / ``for`` / ``while`` body assigns don't; for-loop
    targets and while-body locals are body-scoped, while counters leak.
    Local names are issued from a single counter shared across all nested
    bodies. ``local_types`` overrides the Assign-local type strategy
    (defaults to :func:`arbitrary_type` at depth 1).
    """
    if local_types is None:
        local_types = _default_local_types()
    counter = [0]

    def fresh_local() -> NamedId:
        i = counter[0]
        counter[0] += 1
        return NamedId(f'{local_prefix}{i}')

    kw_expr = dict(range_arg_min=range_arg_min, range_arg_max=range_arg_max)

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
            kinds = ['assign']
            if ctx_budget > 0 and sub_depth > 0:
                kinds.append('context')
            if if_budget > 0 and sub_depth > 0:
                kinds.append('if')
            if loop_budget > 0 and sub_depth > 0:
                kinds.append('for')
            if while_budget > 0 and sub_depth > 0:
                kinds.append('while')
            kind = draw(st.sampled_from(kinds))

            if kind == 'context':
                ctx_value_expr = draw(_ctx_expr(env_local, sub_depth))
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
                cond = draw(_bool_expr(env_local, sub_depth, **kw_expr))
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
                bound = draw(st.integers(range_arg_min, range_arg_max))
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
                rhs = draw(expr(local_type, env_local, sub_depth, **kw_expr))
                stmts.append(Assign(name, None, rhs, None))
                env_local[name] = local_type

        return stmts

    env = dict(env)
    body_stmts = gen_stmts(
        env, depth, max_assigns,
        max_contexts, max_ifs, max_loops, max_whiles,
    )
    ret_expr = draw(expr(return_type, env, depth, **kw_expr))
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[StmtBlock]:
    """Public wrapper around the statement-block strategy."""
    return _stmt_block(
        env, return_type, depth,
        max_assigns, max_contexts, max_ifs, max_loops, max_whiles,
        local_prefix, local_types,
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> FuncDef:
    """Generate a well-typed FPy function with the given signature.

    ``arg_types`` and ``return_type`` may be any types accepted by :func:`expr`.
    The body is a sequence of ``Assign`` / ``ContextStmt`` / ``IfStmt`` /
    ``If1Stmt`` / ``ForStmt`` / ``WhileStmt`` statements followed by a
    ``ReturnStmt`` of type ``return_type``. See :func:`stmt_block` for
    ``max_*`` semantics.
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_funcdef(
        arg_types, return_type,
        max_depth=max_depth, max_assigns=max_assigns,
        max_contexts=max_contexts, max_ifs=max_ifs,
        max_loops=max_loops, max_whiles=max_whiles,
        name=name,
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_real_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_real_funcdef(
        num_args=num_args, max_depth=max_depth, max_assigns=max_assigns,
        max_contexts=max_contexts, max_ifs=max_ifs,
        max_loops=max_loops, max_whiles=max_whiles,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ).map(fp.Function)
