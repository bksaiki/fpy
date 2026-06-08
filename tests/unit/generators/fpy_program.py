"""
Type-directed Hypothesis generators for FPy programs.

Phase 3 scope:
- Four target types: ``RealType``, ``BoolType``, ``ListType``, ``TupleType``,
  dispatched through a single :func:`expr` entry point.
- One construction shape: a ``FuncDef`` with N real-typed parameters and a
  single ``ReturnStmt`` whose expression has type ``real``. The list/tuple
  generators are reachable as direct targets (for testing the dispatcher)
  and indirectly through ``Len`` in the real-expression path.
- Real ops: arithmetic (``Add`` / ``Sub`` / ``Mul`` / ``Div`` / ``Neg`` /
  ``Abs``), named unary math (``Sqrt`` / ``Sin`` / ``Cos`` / ``Log`` /
  ``Exp``), ``IfExpr``, and ``Len`` (list[real] → real).
- Bool ops: ``BoolVal``, ``Compare``, real classification predicates
  (``IsFinite`` / ``IsNan`` / ``IsInf`` / ``IsNormal`` / ``Signbit``), and
  ``Not`` / ``And`` / ``Or``.
- List ops: ``ListExpr`` constructor (1..4 elements of the target element
  type); ``Range1`` / ``Range2`` for list[real].
- Tuple ops: ``TupleExpr`` constructor; one sub-expression per declared
  element type.

Deferred to later phases:
- ``ListRef`` / ``ListSlice`` — need array-size tracking to stay in bounds.
- ``ListComp``, tuple unpacking — need binding-aware generation / statements.
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
    IsNormal,
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


def _real_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``real`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_real_literal_strategy()]
    var_strat = _var_of_type(env, RealType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_real = _real_expr(env, depth - 1)
    sub_bool = _bool_expr(env, depth - 1)

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
    sub_real_list = _list_expr(RealType(), env, depth - 1)
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

_REAL_PREDICATES: list[tuple[type[NamedUnaryOp], str]] = [
    (IsFinite, 'isfinite'),
    (IsInf, 'isinf'),
    (IsNan, 'isnan'),
    (IsNormal, 'isnormal'),
    (Signbit, 'signbit'),
]


def _bool_literal_strategy() -> st.SearchStrategy[Expr]:
    return st.booleans().map(lambda v: BoolVal(v, None))


def _bool_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an FPy expression of type ``bool`` under ``env``."""
    leaves: list[st.SearchStrategy[Expr]] = [_bool_literal_strategy()]
    var_strat = _var_of_type(env, BoolType)
    if var_strat is not None:
        leaves.append(var_strat)

    if depth <= 0:
        return st.one_of(*leaves)

    sub_bool = _bool_expr(env, depth - 1)
    sub_real = _real_expr(env, depth - 1)

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
    elt_type: Type, env: TypeEnv, depth: int
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``list[elt_type]`` expression under ``env``."""
    elt_strat = expr(elt_type, env, max(0, depth - 1))
    list_literal = st.lists(
        elt_strat, min_size=_LIST_LITERAL_LEN[0], max_size=_LIST_LITERAL_LEN[1],
    ).map(lambda xs: ListExpr(xs, None))

    if depth <= 0:
        return list_literal

    inner: list[st.SearchStrategy[Expr]] = [list_literal]
    if isinstance(elt_type, RealType):
        # Range1/Range2 always produce list[real]; only enable them when the
        # element target matches.
        sub_real = _real_expr(env, depth - 1)
        range_sym = _func_sym('range')
        inner.append(sub_real.map(lambda n: Range1(range_sym, n, None)))
        inner.append(st.tuples(sub_real, sub_real).map(
            lambda ab: Range2(range_sym, ab[0], ab[1], None)
        ))
    return st.one_of(*inner)


def _tuple_expr(
    elt_types: tuple[Type, ...], env: TypeEnv, depth: int
) -> st.SearchStrategy[Expr]:
    """Strategy for a ``tuple[*elt_types]`` expression under ``env``.

    One sub-expression per declared element type; element types may differ.
    """
    sub_depth = max(0, depth - 1)
    elt_strats = [expr(t, env, sub_depth) for t in elt_types]
    if not elt_strats:
        # tuple[] — degenerate case; emit an empty TupleExpr.
        return st.just(TupleExpr([], None))
    return st.tuples(*elt_strats).map(lambda elts: TupleExpr(list(elts), None))


# ---------------------------------------------------------------------------
# Type-dispatched expression strategy
# ---------------------------------------------------------------------------

def expr(target: Type, env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Strategy for an expression of type ``target`` under ``env``.

    This is the canonical type-directed entry point. ``depth`` bounds the
    nesting depth of compound operations; at ``depth == 0`` only leaves
    (literals and variable references for scalars, or single-level
    constructors for compound types) are produced.

    Raises :class:`NotImplementedError` for type targets not yet supported
    (e.g. ``ContextType``, ``FunctionType``).
    """
    if isinstance(target, RealType):
        return _real_expr(env, depth)
    if isinstance(target, BoolType):
        return _bool_expr(env, depth)
    if isinstance(target, ListType):
        return _list_expr(target.elt, env, depth)
    if isinstance(target, TupleType):
        return _tuple_expr(target.elts, env, depth)
    raise NotImplementedError(f'no generator for target type {target.format()}')


def real_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(RealType(), env, depth)``."""
    return _real_expr(env, depth)


def bool_expr(env: TypeEnv, depth: int) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(BoolType(), env, depth)``."""
    return _bool_expr(env, depth)


def list_expr(
    elt_type: Type, env: TypeEnv, depth: int
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(ListType(elt_type), env, depth)``."""
    return _list_expr(elt_type, env, depth)


def tuple_expr(
    elt_types: tuple[Type, ...], env: TypeEnv, depth: int
) -> st.SearchStrategy[Expr]:
    """Convenience wrapper for ``expr(TupleType(*elt_types), env, depth)``."""
    return _tuple_expr(elt_types, env, depth)


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

def _make_arg_names(n: int) -> list[NamedId]:
    return [NamedId(f'x{i}') for i in range(n)]


def _make_funcdef(name: str, arg_names: list[NamedId], body_expr: Expr) -> FuncDef:
    """Wrap an expression in a ``f(arg_names...) -> Real: return body_expr``."""
    args = [Argument(n, RealTypeAnn(None, None), None) for n in arg_names]
    body = StmtBlock([ReturnStmt(body_expr, None)])
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef(name, args, body, meta)


@st.composite
def fpy_real_funcdef(
    draw,
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
) -> FuncDef:
    """Generate a well-typed FPy function ``f(x0..xN: Real) -> Real``.

    The body is a single ``return <expr>`` where ``<expr>`` is generated by
    :func:`real_expr` with the given depth budget.
    """
    if num_args is None:
        num_args = st.integers(0, 3)
    if max_depth is None:
        max_depth = st.integers(0, 3)

    n = draw(num_args)
    depth = draw(max_depth)

    arg_names = _make_arg_names(n)
    env: TypeEnv = {name: RealType() for name in arg_names}
    body_expr = draw(real_expr(env, depth))
    return _make_funcdef('f', arg_names, body_expr)


def fpy_real_function(
    num_args: st.SearchStrategy[int] | None = None,
    max_depth: st.SearchStrategy[int] | None = None,
) -> st.SearchStrategy[fp.Function]:
    """Same as :func:`fpy_real_funcdef` but wraps the AST in :class:`fp.Function`."""
    return fpy_real_funcdef(num_args=num_args, max_depth=max_depth).map(fp.Function)
