"""
Type-directed Hypothesis generators for FPy programs.

Phase 4 scope:
- Four expression target types: ``RealType``, ``BoolType``, ``ListType``,
  ``TupleType``, dispatched through a single :func:`expr` entry point.
- Function bodies are now ``StmtBlock``s: zero or more ``Assign`` statements
  introducing scalar locals (``real`` / ``bool``), followed by a
  ``ReturnStmt`` of the declared return type. Locals are visible to
  subsequent statements via ``Var`` references in the threaded ``TypeEnv``.
- Real ops: arithmetic (``Add`` / ``Sub`` / ``Mul`` / ``Div`` / ``Neg`` /
  ``Abs``), named unary math (``Sqrt`` / ``Sin`` / ``Cos`` / ``Log`` /
  ``Exp``), ``IfExpr``, and ``Len`` (list[real] → real).
- Bool ops: ``BoolVal``, ``Compare``, real classification predicates
  (``IsFinite`` / ``IsNan`` / ``IsInf`` / ``Signbit`` — ``IsNormal`` is
  held back; see comment on ``_REAL_PREDICATES``), and ``Not`` / ``And`` /
  ``Or``.
- List ops: ``ListExpr`` constructor (1..4 elements of the target element
  type); ``Range1`` / ``Range2`` for list[real].
- Tuple ops: ``TupleExpr`` constructor; one sub-expression per declared
  element type.

Deferred to later phases:
- Control-flow statements (``IfStmt`` / ``ForStmt`` / ``WhileStmt``) — need
  per-branch ``env`` merging for definitions that survive both branches.
- List/tuple locals — need a more precise ``_var_of_type`` lookup that
  matches element types, not just the outer constructor.
- ``ListRef`` / ``ListSlice`` — need array-size tracking to stay in bounds.
- ``ListComp``, tuple unpacking — need binding-aware generation.
- ``Zip`` / ``Enumerate`` / ``Range3`` — straightforward incremental adds.
- Empty ``ListExpr`` — would need a type annotation to disambiguate the
  element type at the parser/inference boundary.

Construction is AST-direct (we build ``FuncDef`` nodes and feed them into
the same passes the ``@fp.fpy`` decorator pipeline runs), so the parser is
*not* exercised here — the generator targets analyses and the interpreter.
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
    BoolVal,
    Compare,
    Cos,
    Div,
    Exp,
    Expr,
    FuncDef,
    FuncMeta,
    IfExpr,
    Integer,
    IsFinite,
    IsInf,
    IsNan,
    Len,
    ListExpr,
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
    Signbit,
    Sin,
    Sqrt,
    StmtBlock,
    Sub,
    TupleExpr,
    Var,
)
from fpy2.types import BoolType, ListType, RealType, TupleType, Type
from fpy2.env import ForeignEnv
from fpy2.utils import CompareOp, NamedId


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
# ``Range2`` arguments. Per-call overrides are accepted on the public
# strategies (``expr`` / ``list_expr`` / etc.); see the module docstring on
# the planned ``GeneratorConfig`` migration once more knobs accumulate.
_DEFAULT_RANGE_ARG_MIN = -5
_DEFAULT_RANGE_ARG_MAX = 8


def _var_of_type(
    env: TypeEnv,
    target_cls: type[Type],
) -> st.SearchStrategy[Expr] | None:
    """Strategy for a ``Var`` of the given type class, or ``None`` if env has none."""
    names = sorted((n for n, t in env.items() if isinstance(t, target_cls)), key=str)
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``real`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_real_literal_strategy()]
    var_strat = _var_of_type(env, RealType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_real = _real_expr(
        env, depth - 1,
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

    if_expr = st.tuples(sub_bool, sub_real, sub_real).map(
        lambda cab: IfExpr(cab[0], cab[1], cab[2], None)
    )

    # Len(list[real]) → real. Other list-elt types would also typecheck, but
    # list[real] keeps the recursion shape simple; widen later if needed.
    sub_real_list = _list_expr(
        RealType(), env, depth - 1,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )
    len_strat = sub_real_list.map(
        lambda xs: Len(_func_sym('len'), xs, None)
    )

    inner: list[st.SearchStrategy[Expr]] = [
        _binop(Add),
        _binop(Sub),
        _binop(Mul),
        _binop(Div),
        _unop(Neg),
        _unop(Abs),
        if_expr,
        len_strat,
    ]
    inner.extend(_named_unary(cls, name) for cls, name in _REAL_NAMED_UNARY)
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# Bool-typed expressions
# ---------------------------------------------------------------------------

_COMPARE_OPS = list(CompareOp)

# TODO: re-enable ``IsNormal`` once ``ops.isnormal`` handles context-less
# integer Floats. Currently ``isnormal(<int literal>)`` raises ``ValueError``
# from ``Float.is_normal`` (floats.py:616) because the Float has no context;
# the other four predicates handle the same input fine. Generator-side fuzz
# initially surfaced this from a generated function like ``isnormal(735)``.
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
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``bool`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_bool_literal_strategy()]
    var_strat = _var_of_type(env, BoolType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_bool = _bool_expr(
        env, depth - 1,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )
    sub_real = _real_expr(
        env, depth - 1,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )

    # Compare: a chain of (n+1) reals connected by n comparison ops; restrict
    # to simple binary comparisons here (n = 1) — chains add little coverage
    # for what we're trying to fuzz.
    compare = st.tuples(
        st.sampled_from(_COMPARE_OPS),
        sub_real,
        sub_real,
    ).map(lambda oab: Compare([oab[0]], [oab[1], oab[2]], None))

    def _predicate(cls, name):
        sym = _func_sym(name)
        return sub_real.map(lambda x: cls(sym, x, None))

    not_ = sub_bool.map(lambda x: Not(x, None))
    and_or_args = st.lists(sub_bool, min_size=2, max_size=3)
    and_ = and_or_args.map(lambda xs: And(xs, None))
    or_ = and_or_args.map(lambda xs: Or(xs, None))

    inner: list[st.SearchStrategy[Expr]] = [compare, not_, and_, or_]
    inner.extend(_predicate(cls, name) for cls, name in _REAL_PREDICATES)
    return st.one_of(*leaves, *inner)


# ---------------------------------------------------------------------------
# List- and tuple-typed expressions
# ---------------------------------------------------------------------------

_LIST_LITERAL_LEN = (1, 4)
"""Inclusive size range for ``ListExpr`` literals.

Min size 1 because an empty ``ListExpr`` has no element to anchor inference
to the requested element type. (Once we generate type annotations on
``Assign`` we can lift this.)
"""


def _list_expr(
    elt_type: Type,
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``list[elt_type]`` expression under ``env``.

    ``range_arg_min`` / ``range_arg_max`` (inclusive) bound the integer
    literals used as ``Range1`` / ``Range2`` arguments when ``elt_type`` is
    ``RealType``. Reasonable values for the typical fuzz target are small
    (the bounds map to list length under FP64, so widening them risks
    interpreter-time slowdowns from giant ranges).
    """
    elt_strat = expr(
        elt_type, env, max(0, depth - 1),
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )
    list_literal = st.lists(
        elt_strat, min_size=_LIST_LITERAL_LEN[0], max_size=_LIST_LITERAL_LEN[1],
    ).map(lambda xs: ListExpr(xs, None))

    if depth <= 0:
        return list_literal

    inner: list[st.SearchStrategy[Expr]] = [list_literal]
    if isinstance(elt_type, RealType):
        # Range1/Range2 always produce list[real]; only enable them when the
        # element target matches. The interpreter rejects non-integer arguments
        # to ``range`` (e.g. ``range(log(0)) == range(-inf)`` raises), so we
        # restrict the arguments to small ``Integer`` literals. A future phase
        # can introduce an integer-valued real-expression sub-generator that
        # tracks integrality through ``Integer`` arithmetic and ``floor`` /
        # ``nearbyint`` / etc.
        range_sym = _func_sym('range')
        small_int = st.integers(range_arg_min, range_arg_max).map(
            lambda n: Integer(n, None)
        )
        inner.append(small_int.map(lambda n: Range1(range_sym, n, None)))
        inner.append(st.tuples(small_int, small_int).map(
            lambda ab: Range2(range_sym, ab[0], ab[1], None)
        ))
    return st.one_of(*inner)


def _tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``tuple[*elt_types]`` expression under ``env``.

    One sub-expression per declared element type; element types may differ.
    """
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
        return st.just(TupleExpr([], None))
    return st.tuples(*elt_strats).map(lambda elts: TupleExpr(list(elts), None))


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
    raise NotImplementedError(f'no generator for target type {target.format()}')


def real_expr(
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(RealType(), env, depth)``."""
    return _real_expr(
        env, depth,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def bool_expr(
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(BoolType(), env, depth)``."""
    return _bool_expr(
        env, depth,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def list_expr(
    elt_type: Type,
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ListType(elt_type), env, depth)``."""
    return _list_expr(
        elt_type, env, depth,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


def tuple_expr(
    elt_types: tuple[Type, ...],
    env: TypeEnv,
    depth: int,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(TupleType(*elt_types), env, depth)``."""
    return _tuple_expr(
        elt_types, env, depth,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


# ---------------------------------------------------------------------------
# Statement blocks
# ---------------------------------------------------------------------------

# Scalar types eligible for ``Assign`` locals in phase 4. List/tuple locals
# require a finer-grained ``_var_of_type`` lookup that tracks element types,
# so they're held back until that's in place.
_SCALAR_TYPES: list[Type] = [RealType(), BoolType()]


@st.composite
def _stmt_block(
    draw,
    env: TypeEnv,
    return_type: Type,
    depth: int,
    max_assigns: int = 3,
    local_prefix: str = 't',
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> StmtBlock:
    """Generate ``Assign``* ``ReturnStmt`` blocks under ``env``.

    Each ``Assign`` introduces a fresh ``{local_prefix}{i}`` local of a
    scalar type, extending ``env`` for subsequent statements. The block
    ends with a ``ReturnStmt`` whose expression has type ``return_type``.

    ``depth`` is the depth budget for individual rhs expressions; ``max_assigns``
    caps how many locals the block introduces. ``range_arg_min`` / ``range_arg_max``
    are forwarded to every rhs ``expr(...)`` call.
    """
    env = dict(env)
    n_assigns = draw(st.integers(0, max_assigns))
    stmts: list = []
    kw = dict(range_arg_min=range_arg_min, range_arg_max=range_arg_max)
    for i in range(n_assigns):
        name = NamedId(f'{local_prefix}{i}')
        local_type = draw(st.sampled_from(_SCALAR_TYPES))
        rhs = draw(expr(local_type, env, depth, **kw))
        stmts.append(Assign(name, None, rhs, None))
        env[name] = local_type

    ret_expr = draw(expr(return_type, env, depth, **kw))
    stmts.append(ReturnStmt(ret_expr, None))
    return StmtBlock(stmts)


def stmt_block(
    env: TypeEnv,
    return_type: Type,
    depth: int,
    max_assigns: int = 3,
    local_prefix: str = 't',
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[StmtBlock]:
    """Public wrapper around the statement-block strategy."""
    return _stmt_block(
        env, return_type, depth, max_assigns, local_prefix,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    )


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

def _make_arg_names(n: int) -> list[NamedId]:
    return [NamedId(f'x{i}') for i in range(n)]


def _make_funcdef(name: str, arg_names: list[NamedId], body: StmtBlock) -> FuncDef:
    """Wrap a ``StmtBlock`` body in a ``f(arg_names...: Real) -> ...`` ``FuncDef``."""
    args = [Argument(n, RealTypeAnn(None, None), None) for n in arg_names]
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef(name, args, body, meta)


@st.composite
def fpy_real_funcdef(
    draw,
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> FuncDef:
    """Generate a well-typed FPy function ``f(x0..xN: Real) -> Real``.

    The body is a sequence of zero or more ``Assign`` statements
    (introducing scalar locals ``t0..tM``) followed by a ``ReturnStmt``
    whose expression has type ``real``.

    ``range_arg_min`` / ``range_arg_max`` are forwarded to the body's
    expression strategies (see :func:`expr`).
    """
    if num_args is None:
        num_args = st.integers(0, 3)
    if max_depth is None:
        max_depth = st.integers(0, 3)
    if max_assigns is None:
        max_assigns = st.integers(0, 3)

    n = draw(num_args)
    depth = draw(max_depth)
    k = draw(max_assigns)

    arg_names = _make_arg_names(n)
    env: TypeEnv = {name: RealType() for name in arg_names}
    body = draw(_stmt_block(
        env, RealType(), depth, max_assigns=k,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ))
    return _make_funcdef('f', arg_names, body)


def fpy_real_function(
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
    max_assigns: st.SearchStrategy[int] | None = None,
    *,
    range_arg_min: int = _DEFAULT_RANGE_ARG_MIN,
    range_arg_max: int = _DEFAULT_RANGE_ARG_MAX,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_real_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_real_funcdef(
        num_args=num_args, max_depth=max_depth, max_assigns=max_assigns,
        range_arg_min=range_arg_min, range_arg_max=range_arg_max,
    ).map(fp.Function)
